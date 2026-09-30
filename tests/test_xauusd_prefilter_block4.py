import sys

import pandas as pd
import pytest

from tools import xauusd_prefilter_in_detail as prefilter


def _candidate(key, family="f", lift=1.0, ratio=1.0, pos=3, mask_count=5):
    col, op, value = key.split("|")
    return {
        "candidate_key": f"cand_{col}_{value}",
        "stable_candidate_key": key,
        "col": col,
        "op": op,
        "value": float(value),
        "_family": family,
        "family": family,
        "lift": float(lift),
        "coarse_lift": float(lift),
        "ratio": float(ratio),
        "_single_ratio": float(ratio),
        "_single_pos_hits": int(pos),
        "_single_neg_hits": 1,
        "_single_mask_count": int(mask_count),
    }


def _rule(ratio, combo=(0,), pos=5, neg=1):
    return {
        "ratio": float(ratio),
        "pos_hits": int(pos),
        "neg_hits": int(neg),
        "_combo": tuple(combo),
        "conds": [],
    }


def _required_argv():
    return ["prog", "--features", "f", "--binned-features", "b", "--binned-metadata", "m"]


def test_block4_cli_defaults_and_removed_names(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", _required_argv())
    args = prefilter.parse_args()
    assert args.phase_ab_family_top_n == 9
    assert args.phase_c_family_top_n == 9
    assert args.phase_ab_max_path_conds == 8
    assert args.phase_c_max_conds == 9
    assert args.phase_d_max_conds == 9
    assert not hasattr(args, "family_top_n")
    assert not hasattr(args, "max_path_conds")

    monkeypatch.setattr(sys, "argv", ["prog", "--help"])
    with pytest.raises(SystemExit) as help_exit:
        prefilter.parse_args()
    assert help_exit.value.code == 0
    help_text = capsys.readouterr().out
    assert "--phase-ab-family-top-n" in help_text
    assert "--phase-c-family-top-n" in help_text
    assert "--phase-ab-max-path-conds" in help_text
    assert "candidates retained per family for A/B (default: 9)" in help_text
    assert "candidates retained per family for C (default: 9)" in help_text
    assert "maximum A/B condition depth (default: 8)" in help_text
    assert "--phase-c-max-conds" in help_text
    assert "--phase-d-max-conds" in help_text
    assert "--family-top-n" not in help_text
    assert "--max-path-conds" not in help_text

    for old_name in ("--family-top-n", "--max-path-conds"):
        monkeypatch.setattr(sys, "argv", _required_argv() + [old_name, "3"])
        with pytest.raises(SystemExit):
            prefilter.parse_args()


def test_phase_specific_family_pools_and_rankings_are_independent():
    rows = [
        _candidate("a|>=|1", lift=3.0, ratio=0.8, pos=4, mask_count=8),
        _candidate("b|>=|2", lift=2.0, ratio=1.3, pos=2, mask_count=3),
        _candidate("c|>=|3", family="g", lift=1.0, ratio=1.1, pos=7, mask_count=4),
    ]
    ab_family = prefilter._family_top_rows(rows, 1)
    c_family = prefilter._family_top_rows(rows, 2)
    assert [row["stable_candidate_key"] for row in ab_family] == ["a|>=|1", "c|>=|3"]
    assert [row["stable_candidate_key"] for row in c_family] == ["a|>=|1", "b|>=|2", "c|>=|3"]

    ab_lift = prefilter._rank_family_pool(ab_family, False, "lift")
    c_tick = prefilter._rank_family_pool(c_family, True, "ratio")
    assert [row["stable_candidate_key"] for row in ab_lift] == ["a|>=|1", "c|>=|3"]
    assert [row["stable_candidate_key"] for row in c_tick] == ["b|>=|2", "c|>=|3", "a|>=|1"]


def test_tick_scope_fam_top_is_stable_key_union_and_other_scopes_are_preserved():
    a = _candidate("a|>=|1")
    b = _candidate("b|>=|2")
    c = _candidate("c|>=|3")
    inventory = [a, b, c]
    filtered = [b, c]
    assert prefilter._tick_scope_keys("fam_top", inventory, filtered, [a, b], [b, c]) == {
        "a|>=|1",
        "b|>=|2",
        "c|>=|3",
    }
    assert prefilter._tick_scope_keys("filtered", inventory, filtered, [a], [c]) == {"b|>=|2", "c|>=|3"}
    assert prefilter._tick_scope_keys("all_coarse", inventory, filtered, [a], [c]) == {
        "a|>=|1",
        "b|>=|2",
        "c|>=|3",
    }


def test_family_membership_output_replaces_legacy_column():
    fields = prefilter._family_membership_fields("a|>=|1", {"a|>=|1"}, {"b|>=|2"})
    assert fields == {
        "kept_after_phase_ab_family_topn": 1,
        "kept_after_phase_c_family_topn": 0,
    }
    row = {
        "stable_candidate_key": "a|>=|1",
        "kept_after_family_topn": 1,
        **fields,
    }
    frame = prefilter._candidate_output_frame(
        [row],
        [
            "stable_candidate_key",
            "kept_after_phase_ab_family_topn",
            "kept_after_phase_c_family_topn",
        ],
    )
    assert list(frame.columns) == [
        "stable_candidate_key",
        "kept_after_phase_ab_family_topn",
        "kept_after_phase_c_family_topn",
    ]


def test_current_schema_with_legacy_family_diagnostic_remains_loadable(tmp_path):
    path = tmp_path / "coarse.csv"
    pd.DataFrame(
        [
            {
                "candidate_key": "cand_a",
                "stable_candidate_key": "a|>=|1",
                "col": "a",
                "op": ">=",
                "value": 1.0,
                "family": "f",
                "coarse_single_pos_hits": 3,
                "coarse_single_neg_hits": 1,
                "coarse_single_mask_count": 4,
                "coarse_single_ratio": 3.0,
                "coarse_single_mask_keep_ratio": 1.0,
                "coarse_single_ratio_change": 1.0,
                "coarse_lift": 2.0,
                "binary": 0,
                "kept_after_family_topn": 1,
                "__stage": "coarse",
                "__schema_version": prefilter.CANDIDATE_CACHE_SCHEMA_VERSION,
                "__ctx_sig": "ctx",
            }
        ]
    ).to_csv(path, index=False)
    loaded = prefilter._load_stage_csv_if_match(
        path,
        "coarse",
        "ctx",
        prefilter.CANDIDATE_CACHE_SCHEMA_VERSION,
        rebuild_legacy=True,
    )
    assert len(loaded) == 1
    migrated = loaded.to_dict("records")
    migrated[0].update(prefilter._family_membership_fields("a|>=|1", {"a|>=|1"}, set()))
    output = prefilter._candidate_output_frame(
        migrated,
        ["stable_candidate_key", "kept_after_phase_ab_family_topn", "kept_after_phase_c_family_topn"],
    )
    assert "kept_after_family_topn" not in output.columns


def test_current_refined_schema_reuses_metrics_and_drops_legacy_diagnostic(tmp_path):
    path = tmp_path / "refined.csv"
    row = {
        "candidate_key_refined": "cand_a",
        "stable_candidate_key": "a|>=|1",
        "col": "a",
        "op": ">=",
        "value": 1.0,
        "family": "f",
        "coarse_single_pos_hits": 3,
        "coarse_single_neg_hits": 1,
        "coarse_single_mask_count": 4,
        "coarse_single_ratio": 3.0,
        "coarse_single_mask_keep_ratio": 1.0,
        "coarse_single_ratio_change": 1.0,
        "coarse_lift": 2.0,
        "binary": 0,
        "kept_after_family_topn": 1,
        "tick_metric_status": "full",
        "__stage": "refined",
        "__schema_version": prefilter.REFINED_CACHE_SCHEMA_VERSION,
        "__ctx_sig": "ctx",
        **{name: 1.0 for name in prefilter.REQUIRED_TICK_METRIC_COLUMNS},
    }
    pd.DataFrame([row]).to_csv(path, index=False)
    loaded = prefilter._load_stage_csv_if_match(
        path,
        "refined",
        "ctx",
        prefilter.REFINED_CACHE_SCHEMA_VERSION,
        rebuild_legacy=False,
    )
    assert len(loaded) == 1
    assert prefilter._has_full_tick_metrics(loaded.iloc[0].to_dict())
    migrated = loaded.to_dict("records")
    migrated[0].update(prefilter._family_membership_fields("a|>=|1", {"a|>=|1"}, set()))
    output = prefilter._candidate_output_frame(
        migrated,
        [
            "stable_candidate_key",
            "kept_after_phase_ab_family_topn",
            "kept_after_phase_c_family_topn",
            *prefilter.REQUIRED_TICK_METRIC_COLUMNS,
        ],
    )
    assert "kept_after_family_topn" not in output.columns
    assert all(float(output.iloc[0][name]) == 1.0 for name in prefilter.REQUIRED_TICK_METRIC_COLUMNS)


def test_real_new_block_boundary_drives_free_combinations_and_rejects_reordering():
    previous = [_candidate(f"c{i}|>=|{i}") for i in range(1, 16)]
    current = previous + [_candidate("c16|>=|16"), _candidate("c17|>=|17")]
    new_idxs = prefilter._new_block_indices(previous, current)
    assert new_idxs == [15, 16]
    combos = list(prefilter._iter_combinations_with_new(list(range(17)), 2, new_idxs[0]))
    assert combos
    assert all(any(idx in new_idxs for idx in combo) for combo in combos)
    assert prefilter._new_block_indices(current, current) == []
    with pytest.raises(ValueError, match="stable-identity prefix"):
        prefilter._new_block_indices(previous, [current[1], current[0], *current[2:]])


def test_ab_parent_extensions_add_only_from_new_block_and_allow_multiple_adds():
    final_parent = _rule(1.2, combo=(0, 1))
    search_parent = _rule(0.9, combo=(0, 1))
    seeds = prefilter._ab_parent_seed_snapshot([search_parent, final_parent], 2)
    assert seeds == [final_parent, search_parent]
    extensions = list(prefilter._iter_all_parent_extensions(seeds, [4, 5], 4))
    combos = [combo for combo, _parent in extensions]
    assert (0, 1, 4) in combos
    assert (0, 1, 5) in combos
    assert (0, 1, 4, 5) in combos
    assert all(set(combo).difference({0, 1}).issubset({4, 5}) for combo in combos)
    assert all(len(combo) <= 4 for combo in combos)
    assert prefilter._parent_score_allows(_rule(1.3), final_parent)
    assert not prefilter._parent_score_allows(_rule(1.1), final_parent)


def test_phase_a_transition_state_uses_exact_next_block_for_b_or_moves_to_c():
    previous = [_candidate("a|>=|1"), _candidate("b|>=|2")]
    next_pool = previous + [_candidate("c|>=|3")]
    survivor = _rule(0.9)
    transition = prefilter._phase_a_transition_state(
        "phase_a_no_improvement", 10, previous, [survivor], next_pool
    )
    assert transition == {
        "unlocked": 10,
        "processed_pool": previous,
        "next_phase": "B",
        "reason": "phase_a_no_improvement",
    }
    assert prefilter._new_block_indices(transition["processed_pool"], next_pool) == [2]
    early_stop = prefilter._phase_a_transition_state(
        "early_stop_no_improvement", 10, previous, [survivor], next_pool
    )
    assert early_stop["next_phase"] == "B"
    assert early_stop["reason"] == "early_stop_no_improvement"

    no_survivors = prefilter._phase_a_transition_state(
        "phase_a_no_improvement", 10, previous, [], next_pool
    )
    assert no_survivors["next_phase"] == "C"
    no_real_next_block = prefilter._phase_a_transition_state(
        "phase_a_no_improvement", 10, previous, [survivor], previous
    )
    assert no_real_next_block["next_phase"] == "C"
    full_pool_early_stop = prefilter._phase_a_transition_state(
        "early_stop_no_improvement", 10, previous, [survivor], previous
    )
    assert full_pool_early_stop["next_phase"] == "C"
    assert full_pool_early_stop["processed_pool"] == previous


def test_empty_intermediate_a_unlock_advances_to_later_real_growth():
    class Miner:
        @staticmethod
        def candidate_anchor_columns(binary_col, _all_candidate_cols):
            return ["missing_anchor"] if binary_col == "hidden" else []

    first = _candidate("first|>=|1")
    hidden = _candidate("hidden|>=|2")
    hidden["binary"] = True
    later = _candidate("later|>=|3")
    rank = [first, hidden, later]
    previous = prefilter._build_unlocked_pool(Miner, [rank], ["first", "hidden", "later"], 1, 1, 10, 0, 10)
    intermediate = prefilter._build_unlocked_pool(Miner, [rank], ["first", "hidden", "later"], 2, 1, 10, 0, 10)
    action, new_idxs = prefilter._ab_unlock_decision(previous, intermediate, True, 2, len(rank))
    assert action == "advance_a"
    assert new_idxs == []

    later_pool = prefilter._build_unlocked_pool(Miner, [rank], ["first", "hidden", "later"], 3, 1, 10, 0, 10)
    action, new_idxs = prefilter._ab_unlock_decision(intermediate, later_pool, True, 3, len(rank))
    assert action == "run"
    assert new_idxs == [1]
    assert [row["stable_candidate_key"] for row in later_pool] == ["first|>=|1", "later|>=|3"]

    b_action, b_new_idxs = prefilter._ab_unlock_decision(previous, intermediate, False, 2, len(rank))
    assert b_action == "to_c"
    assert b_new_idxs == []


def test_parent_usage_diagnostics_run_once_per_debug_round_and_do_no_work_when_disabled(monkeypatch, capsys):
    final_parent = _rule(1.2)
    search_parent = _rule(0.9)
    disabled = prefilter._new_ab_parent_usage_stats([final_parent, search_parent], 1.0, False)
    assert disabled is None
    monkeypatch.setattr(
        prefilter,
        "_record_ab_parent_usage",
        lambda *_args, **_kwargs: pytest.fail("disabled diagnostics performed per-evaluation counting"),
    )
    prefilter._record_ab_parent_usage_batch(disabled, [final_parent], [_rule(1.3)], 1.0)
    assert not prefilter._print_ab_parent_usage_once(disabled, 5, "A")
    assert capsys.readouterr().out == ""

    monkeypatch.undo()
    stats = prefilter._new_ab_parent_usage_stats([final_parent, search_parent], 1.0, True)
    assert stats is not None
    prefilter._record_ab_parent_usage_batch(
        stats,
        [None, final_parent, search_parent, search_parent],
        [_rule(2.0), _rule(1.3), _rule(1.1), None],
        1.0,
    )
    expected = {
        "ab_survivors_final_valid": 1,
        "ab_survivors_search_only": 1,
        "parent_ext_evaluated_from_final_valid": 1,
        "parent_ext_evaluated_from_search_only": 2,
        "parent_ext_final_valid_children_from_final_valid": 1,
        "parent_ext_final_valid_children_from_search_only": 1,
    }
    assert {key: stats[key] for key in expected} == expected
    assert prefilter._print_ab_parent_usage_once(stats, 10, "B")
    assert not prefilter._print_ab_parent_usage_once(stats, 10, "B")
    output_lines = capsys.readouterr().out.splitlines()
    assert len(output_lines) == 1
    assert output_lines[0].startswith("[prefilter-ab-parent-usage] round=10 phase=B ")
    for key, value in expected.items():
        assert output_lines[0].count(f"{key}={value}") == 1


def test_c_mapping_uses_its_own_pool_and_preserves_partial_and_zero_fallback():
    pool_c = [_candidate("a|>=|1"), _candidate("b|>=|2")]
    mapped = _rule(1.2)
    mapped["conds"] = [{"col": "b", "op": ">=", "value": 2.0}]
    missing = _rule(1.3)
    missing["conds"] = [{"col": "outside", "op": ">=", "value": 9.0}]
    nodes, fallback, _next_id = prefilter._map_c_start_nodes([mapped, missing], pool_c, 2, 4)
    assert not fallback
    assert [node["combo"] for node in nodes] == [(1,)]
    nodes, fallback, _next_id = prefilter._map_c_start_nodes([missing], pool_c, 2, 4)
    assert fallback
    assert [node["combo"] for node in nodes] == [(0,), (1,)]


def test_c_and_d_extensions_use_their_own_depth_even_when_ab_max_is_lower():
    nodes = [{"parent_id": 7, "combo": (0, 1, 2), "root_ids": (0,), "train_rank": (1.0, 3, -1)}]
    # An A/B max of 3 is deliberately not passed to the C/D production helper.
    c_extensions = list(prefilter._iter_beam_extensions(nodes, [0, 1, 2, 3, 4], 1, 2, 5))
    d_extensions = list(prefilter._iter_beam_extensions(nodes, [0, 1, 2, 3, 4, 5], 1, 2, 6))
    assert ((0, 1, 2, 3, 4), 7) in c_extensions
    assert ((0, 1, 2, 3, 4), 7) in d_extensions
    assert all(len(combo) <= 5 for combo, _parent in c_extensions)
    assert all(len(combo) <= 6 for combo, _parent in d_extensions)

    max_depth_rule = _rule(1.1, combo=(0, 1, 2, 3, 4))
    archive, expandable = prefilter._partition_level_roles([max_depth_rule], 1.0, 5)
    assert archive == [max_depth_rule]
    assert expandable == []


def test_d_start_depth_uses_only_d_max_and_pool_size():
    assert prefilter._phase_d_initial_depth(1, 9, 4) == 1
    assert prefilter._phase_d_initial_depth(2, 9, 4) == 2
    assert prefilter._phase_d_initial_depth(2, 1, 4) is None
    assert prefilter._phase_d_initial_depth(3, 9, 4) == 3
    assert prefilter._phase_d_initial_depth(10, 9, 4) == 4
    assert prefilter._phase_d_initial_depth(0, 9, 4) == 1
