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
                "coarse_single_mask_keep_ratio": 1.0,
                "coarse_single_ratio_change": 1.0,
                "binary": 0,
                "__stage": "coarse",
                "__schema_version": prefilter.CANDIDATE_CACHE_SCHEMA_VERSION,
                "__ctx_sig": ctx_sig,
            }
        )
    return pd.DataFrame(rows)


def _legacy_coarse_frame(stage="coarse"):
    frame = _current_coarse_frame().drop(columns=["__schema_version"])
    frame["__stage"] = stage
    frame["kept_after_family_topn"] = 1
    return frame


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


def test_full_tick_metrics_require_only_intrinsic_columns():
    metrics = {name: 1.0 for name in prefilter.REQUIRED_TICK_METRIC_COLUMNS}
    assert prefilter._has_full_tick_metrics(metrics)
    metrics["tick_single_ratio"] = float("nan")
    assert not prefilter._has_full_tick_metrics(metrics)


def test_legacy_coarse_requests_fresh_rebuild(tmp_path):
    path = tmp_path / "coarse.csv"
    _legacy_coarse_frame().to_csv(path, index=False)
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


def test_fresh_resume_inventory_runtime_family_and_scope_parity():
    fresh_inventory = [
        {**_rows()[0], "_family": "f", "lift": 1.0, "_single_pos_hits": 1, "_single_mask_count": 20},
        {**_rows()[1], "_family": "f", "lift": 2.0, "_single_pos_hits": 5, "_single_mask_count": 10},
    ]
    resume_inventory = [dict(row) for row in fresh_inventory]
    fresh_runtime = prefilter._filter_candidate_inventory_rows(fresh_inventory, 2, 1.5, 0, None)
    resume_runtime = prefilter._filter_candidate_inventory_rows(resume_inventory, 2, 1.5, 0, None)
    fresh_family = prefilter._family_top_rows(fresh_runtime, 1)
    resume_family = prefilter._family_top_rows(resume_runtime, 1)
    assert {r["stable_candidate_key"] for r in fresh_inventory} == {r["stable_candidate_key"] for r in resume_inventory}
    assert {r["stable_candidate_key"] for r in fresh_runtime} == {r["stable_candidate_key"] for r in resume_runtime}
    assert {r["stable_candidate_key"] for r in fresh_family} == {r["stable_candidate_key"] for r in resume_family}
    for scope in ("fam_top", "filtered", "all_coarse"):
        assert prefilter._tick_scope_keys(scope, fresh_inventory, fresh_runtime, fresh_family) == prefilter._tick_scope_keys(
            scope, resume_inventory, resume_runtime, resume_family
        )


def test_narrow_then_wide_reuses_same_inventory_without_merge():
    inventory = _rows()
    narrow = prefilter._filter_candidate_inventory_rows(inventory, 5, 1.5, 10, None)
    wide = prefilter._filter_candidate_inventory_rows(inventory, 0, 0.0, 0, None)
    assert [row["stable_candidate_key"] for row in narrow] == ["b|>=|2"]
    assert {row["stable_candidate_key"] for row in wide} == {"a|>=|1", "b|>=|2"}
    assert inventory == _rows()


@pytest.mark.parametrize(
    ("scope", "expected"),
    [
        ("fam_top", {"a|>=|1"}),
        ("filtered", {"b|>=|2"}),
        ("all_coarse", {"a|>=|1", "b|>=|2"}),
    ],
)
def test_mask_reconstruction_keys_follow_actual_tick_scope(scope, expected):
    inventory = _rows()
    filtered = inventory[1:]
    fam_top = inventory[:1]
    scope_keys = prefilter._tick_scope_keys(scope, inventory, filtered, fam_top)
    assert prefilter._mask_required_keys(scope_keys, True) == expected
    assert prefilter._mask_required_keys(scope_keys, False) == set()


def test_family_selection_uses_persisted_scalar_metrics_only():
    rows = [
        {**_rows()[0], "_family": "f", "lift": 1.0, "_single_pos_hits": 1, "_single_mask_count": 20},
        {**_rows()[1], "_family": "f", "lift": 2.0, "_single_pos_hits": 5, "_single_mask_count": 10},
    ]
    assert all("mask" not in row for row in rows)
    assert [row["candidate_key"] for row in prefilter._family_top_rows(rows, 1)] == ["cand_2"]


def test_refined_rows_exactly_follow_inventory_keys():
    inventory = _rows()
    rows_by_key = {
        "a|>=|1": {"stable_candidate_key": "a|>=|1"},
        "b|>=|2": {"stable_candidate_key": "b|>=|2"},
        "stale": {"stable_candidate_key": "stale"},
    }
    result = prefilter._refined_rows_for_inventory(inventory, rows_by_key)
    assert [row["stable_candidate_key"] for row in result] == ["a|>=|1", "b|>=|2"]


def test_refined_full_metrics_survive_narrower_scope_status():
    old = {
        "stable_candidate_key": "a|>=|1",
        "tick_metric_status": "full",
        **{name: 1.0 for name in prefilter.REQUIRED_TICK_METRIC_COLUMNS},
    }
    new = {"stable_candidate_key": "a|>=|1", "tick_metric_status": "out_of_scope"}
    kept = prefilter._prefer_refined_row(old, new, "ctx", prefilter.REFINED_CACHE_SCHEMA_VERSION)
    assert kept["tick_metric_status"] == "full"
    assert prefilter._has_full_tick_metrics(kept)


def test_broader_scope_identifies_only_missing_tick_metrics():
    rows = []
    for i in range(4):
        row = {"stable_candidate_key": str(i)}
        if i < 2:
            row.update({name: float(i) for name in prefilter.REQUIRED_TICK_METRIC_COLUMNS})
        rows.append(row)
    full, missing = prefilter._partition_tick_metric_rows(rows)
    assert [row["stable_candidate_key"] for row in full] == ["0", "1"]
    assert [row["stable_candidate_key"] for row in missing] == ["2", "3"]


def test_textual_full_status_does_not_override_missing_metric_columns():
    assert not prefilter._has_full_tick_metrics({"tick_metric_status": "full"})


def test_500_250_250_partition_keeps_all_candidates_in_replay_scope():
    scope = []
    for i in range(500):
        row = {"stable_candidate_key": str(i)}
        if i < 250:
            row.update({name: float(i) for name in prefilter.REQUIRED_TICK_METRIC_COLUMNS})
        scope.append(row)
    full, missing = prefilter._partition_tick_metric_rows(scope)
    replay_keys = prefilter._mask_required_keys({row["stable_candidate_key"] for row in scope}, True)
    assert len(full) == 250
    assert len(missing) == 250
    assert len(replay_keys) == 500
    assert {row["stable_candidate_key"] for row in full}.issubset(replay_keys)


@pytest.mark.parametrize(
    ("scope_rows", "replay", "missing", "entries", "expected"),
    [
        (0, False, 0, 0, "empty_scope"),
        (2, False, 2, 0, "no_replay_configured"),
        (2, True, 0, 4, "complete"),
        (2, True, 2, 0, "null_critical_entries_existing_semantics"),
        (2, True, 1, 4, "incomplete"),
    ],
)
def test_refinement_states_preserve_block1_null_critical_semantics(scope_rows, replay, missing, entries, expected):
    assert prefilter._refinement_state(scope_rows, replay, missing, entries) == expected


def test_phase_d_base_is_current_scope_intersection_full_metrics():
    scope = [{"stable_candidate_key": "full", **{name: 1.0 for name in prefilter.REQUIRED_TICK_METRIC_COLUMNS}}, {"stable_candidate_key": "missing"}]
    full, _ = prefilter._partition_tick_metric_rows(scope)
    assert [row["stable_candidate_key"] for row in full] == ["full"]


def test_unknown_coarse_schema_version_fails_without_modifying_file(tmp_path):
    path = tmp_path / "coarse.csv"
    frame = _current_coarse_frame()
    frame["__schema_version"] = "future_schema"
    frame.to_csv(path, index=False)
    before = path.read_bytes()
    with pytest.raises(ValueError, match=r"found=\['future_schema'\]"):
        prefilter._load_stage_csv_if_match(
            path, "coarse", "ctx", prefilter.CANDIDATE_CACHE_SCHEMA_VERSION, rebuild_legacy=True
        )
    assert path.read_bytes() == before


@pytest.mark.parametrize("missing_column", ["coarse_single_mask_keep_ratio", "coarse_single_ratio_change", "binary"])
def test_current_coarse_schema_requires_all_semantic_scalar_fields(tmp_path, missing_column):
    path = tmp_path / "coarse.csv"
    frame = _current_coarse_frame()
    frame["coarse_single_mask_keep_ratio"] = 1.0
    frame["coarse_single_ratio_change"] = 1.0
    frame.drop(columns=[missing_column]).to_csv(path, index=False)
    before = path.read_bytes()
    with pytest.raises(ValueError, match=missing_column):
        prefilter._load_stage_csv_if_match(
            path, "coarse", "ctx", prefilter.CANDIDATE_CACHE_SCHEMA_VERSION, rebuild_legacy=True
        )
    assert path.read_bytes() == before


def test_mixed_schema_stage_and_context_values_report_all_found_values(tmp_path):
    path = tmp_path / "coarse.csv"
    frame = _current_coarse_frame()
    frame["coarse_single_mask_keep_ratio"] = 1.0
    frame["coarse_single_ratio_change"] = 1.0
    frame.loc[0, "__schema_version"] = "other"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="candidate_inventory_v2.*other|other.*candidate_inventory_v2"):
        prefilter._load_stage_csv_if_match(
            path, "coarse", "ctx", prefilter.CANDIDATE_CACHE_SCHEMA_VERSION, rebuild_legacy=True
        )


def test_context_signature_keeps_paths_and_other_strings_case_sensitive():
    assert prefilter._ctx_sig({"path": "/Data/A", "value": "Case"}) != prefilter._ctx_sig(
        {"path": "/data/a", "value": "case"}
    )


def test_external_candidate_key_and_allowlist_semantics_remain_candidate_key_based():
    inventory = _rows()
    selected = prefilter._filter_candidate_inventory_rows(inventory, 0, 0.0, 0, {"cand_2"})
    assert [row["candidate_key"] for row in selected] == ["cand_2"]
    assert inventory[1]["stable_candidate_key"] == "b|>=|2"


def test_single_candidate_masks_are_released_before_search_setup():
    source = Path(prefilter.__file__).read_text(encoding="utf-8")
    early_release = source.index("inventory_mask_items.clear()")
    search_setup = source.index("rank_sort_t0 = time.perf_counter()")
    assert early_release < search_setup
    assert "if str(it[\"stable_candidate_key\"]) in mask_required_keys" in source
    assert "by_mask.clear()" in source
    assert "x = m = raw_train_m = m_sel = cur = None" in source


def test_schema_less_refined_stage_is_not_rebuilt_as_legacy_coarse(tmp_path):
    path = tmp_path / "wrong-stage.csv"
    _legacy_coarse_frame(stage="refined").to_csv(path, index=False)
    before = path.read_bytes()
    with pytest.raises(ValueError, match=r"found=\['refined'\].*expected=coarse"):
        prefilter._load_stage_csv_if_match(
            path, "coarse", "ctx", prefilter.CANDIDATE_CACHE_SCHEMA_VERSION, rebuild_legacy=True
        )
    assert path.read_bytes() == before


def test_current_coarse_reload_preserves_inventory_runtime_family_and_tick_scope(tmp_path):
    path = tmp_path / "coarse.csv"
    frame = _current_coarse_frame()
    frame.to_csv(path, index=False)
    loaded = prefilter._load_stage_csv_if_match(
        path, "coarse", "ctx", prefilter.CANDIDATE_CACHE_SCHEMA_VERSION, rebuild_legacy=True
    )
    inventory = loaded.to_dict("records")
    for row in inventory:
        row.update(
            {
                "_family": row["family"],
                "lift": row["coarse_lift"],
                "_single_pos_hits": row["coarse_single_pos_hits"],
                "_single_mask_count": row["coarse_single_mask_count"],
            }
        )
    runtime = prefilter._filter_candidate_inventory_rows(inventory, 2, 1.5, 0, None)
    family = prefilter._family_top_rows(runtime, 1)
    assert {row["stable_candidate_key"] for row in inventory} == {"a|>=|1", "b|>=|2"}
    assert [row["stable_candidate_key"] for row in runtime] == ["b|>=|2"]
    assert [row["stable_candidate_key"] for row in family] == ["b|>=|2"]
    assert prefilter._tick_scope_keys("all_coarse", inventory, runtime, family) == {"a|>=|1", "b|>=|2"}


def test_strict_then_loose_filters_leave_same_current_coarse_cache_unchanged(tmp_path):
    path = tmp_path / "coarse.csv"
    _current_coarse_frame().to_csv(path, index=False)
    before = path.read_bytes()
    strict_loaded = prefilter._load_stage_csv_if_match(
        path, "coarse", "ctx", prefilter.CANDIDATE_CACHE_SCHEMA_VERSION, rebuild_legacy=True
    ).to_dict("records")
    strict = prefilter._filter_candidate_inventory_rows(strict_loaded, 5, 1.5, 10, None)
    loose_loaded = prefilter._load_stage_csv_if_match(
        path, "coarse", "ctx", prefilter.CANDIDATE_CACHE_SCHEMA_VERSION, rebuild_legacy=True
    ).to_dict("records")
    loose = prefilter._filter_candidate_inventory_rows(loose_loaded, 0, 0.0, 0, None)
    assert len(strict) == 1
    assert len(loose) == 2
    assert {row["stable_candidate_key"] for row in strict_loaded} == {
        row["stable_candidate_key"] for row in loose_loaded
    }
    assert path.read_bytes() == before


def test_current_coarse_duplicate_stable_keys_fail_fast_and_preserve_file(tmp_path):
    path = tmp_path / "coarse.csv"
    frame = _current_coarse_frame()
    frame.loc[1, "stable_candidate_key"] = frame.loc[0, "stable_candidate_key"]
    frame.to_csv(path, index=False)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="duplicate_stable_candidate_keys"):
        prefilter._load_stage_csv_if_match(
            path, "coarse", "ctx", prefilter.CANDIDATE_CACHE_SCHEMA_VERSION, rebuild_legacy=True
        )
    assert path.read_bytes() == before
