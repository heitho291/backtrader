import hashlib
from pathlib import Path

import pandas as pd
import pytest

from tools import xauusd_prefilter_in_detail as prefilter


def _rows():
    return [
        {
            "candidate_key": "cand_1",
            "stable_candidate_key": "a|>=|1",
            "coarse_single_pos_hits": 1,
            "coarse_single_mask_count": 20,
            "coarse_lift": 1.0,
        },
        {
            "candidate_key": "cand_2",
            "stable_candidate_key": "b|>=|2",
            "coarse_single_pos_hits": 5,
            "coarse_single_mask_count": 10,
            "coarse_lift": 2.0,
        },
    ]


def _current_coarse_frame(ctx_sig="ctx"):
    rows = []
    for i, row in enumerate(_rows(), start=1):
        rows.append(
            {
                **row,
                "col": f"c{i}",
                "op": ">=",
                "value": float(i),
                "family": "f",
                "coarse_single_neg_hits": 1,
                "coarse_single_ratio": float(i),
                "binary": 0,
                "__stage": "coarse",
                "__schema_version": prefilter.CANDIDATE_CACHE_SCHEMA_VERSION,
                "__ctx_sig": ctx_sig,
            }
        )
    return pd.DataFrame(rows)


def test_runtime_filters_do_not_mutate_inventory():
    inventory = _rows()
    strict = prefilter._filter_candidate_inventory_rows(inventory, 5, 1.5, 10, None)
    wide = prefilter._filter_candidate_inventory_rows(inventory, 0, 0.0, 0, None)
    assert [r["candidate_key"] for r in strict] == ["cand_2"]
    assert len(wide) == len(inventory) == 2


def test_empty_allowlist_only_empties_runtime_scope():
    inventory = _rows()
    assert prefilter._filter_candidate_inventory_rows(inventory, 0, 0.0, 0, set()) == []
    assert len(inventory) == 2


def test_tick_scope_semantics_are_independent_of_metric_status():
    inventory = _rows()
    filtered = inventory[1:]
    fam_top = inventory[:1]
    inventory[0]["tick_metric_status"] = "full"
    assert prefilter._tick_scope_keys("all_coarse", inventory, filtered, fam_top) == {"a|>=|1", "b|>=|2"}
    assert prefilter._tick_scope_keys("filtered", inventory, filtered, fam_top) == {"b|>=|2"}
    assert prefilter._tick_scope_keys("fam_top", inventory, filtered, fam_top) == {"a|>=|1"}


def test_full_tick_metrics_do_not_require_tick_lift():
    metrics = {name: 1.0 for name in prefilter.REQUIRED_TICK_METRIC_COLUMNS}
    assert "tick_lift" not in prefilter.REQUIRED_TICK_METRIC_COLUMNS
    assert prefilter._has_full_tick_metrics(metrics)
    metrics["tick_single_ratio"] = float("nan")
    assert not prefilter._has_full_tick_metrics(metrics)


def test_legacy_coarse_requests_fresh_rebuild(tmp_path):
    path = tmp_path / "coarse.csv"
    _current_coarse_frame().drop(columns=["__schema_version"]).to_csv(path, index=False)
    assert prefilter._load_stage_csv_if_match(
        path,
        "coarse",
        "ctx",
        prefilter.CANDIDATE_CACHE_SCHEMA_VERSION,
        rebuild_legacy=True,
    ) is None


def test_current_coarse_context_mismatch_does_not_modify_file(tmp_path):
    path = tmp_path / "coarse.csv"
    _current_coarse_frame("old").to_csv(path, index=False)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="context mismatch"):
        prefilter._load_stage_csv_if_match(
            path,
            "coarse",
            "new",
            prefilter.CANDIDATE_CACHE_SCHEMA_VERSION,
            rebuild_legacy=True,
        )
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_refined_context_mismatch_does_not_modify_file(tmp_path):
    path = tmp_path / "refined.csv"
    row = _current_coarse_frame("old").iloc[0].to_dict()
    row.pop("candidate_key")
    row.update(
        {
            "candidate_key_refined": "cand_1",
            "tick_metric_status": "full",
            "__stage": "refined",
            "__schema_version": prefilter.REFINED_CACHE_SCHEMA_VERSION,
            **{name: 1.0 for name in prefilter.REQUIRED_TICK_METRIC_COLUMNS},
        }
    )
    pd.DataFrame([row]).to_csv(path, index=False)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="context mismatch"):
        prefilter._load_stage_csv_if_match(
            path,
            "refined",
            "new",
            prefilter.REFINED_CACHE_SCHEMA_VERSION,
            rebuild_legacy=False,
        )
    assert path.read_bytes() == before


def test_context_signature_uses_numeric_quantiles_and_normalized_datetime_column():
    def signature(quantiles, datetime_column):
        return prefilter._ctx_sig(
            {
                "quantiles": [float(x) for x in quantiles.split(",") if x.strip()],
                "tick_datetime_column": datetime_column.strip().lower(),
                "path": "/Case/Sensitive",
            }
        )

    assert signature("0.05,0.10", " DateTime ") == signature("0.05, 0.10", "datetime")
    assert signature("0.10,0.05", "datetime") != signature("0.05,0.10", "datetime")


def test_production_source_has_no_tick_lift_or_build_width_fields():
    source = Path(prefilter.__file__).read_text(encoding="utf-8")
    assert "tick_lift" not in source
    assert "build_min_single_pos_hits" not in source
    assert "build_min_single_lift" not in source
    assert "build_max_single_mask_count" not in source
