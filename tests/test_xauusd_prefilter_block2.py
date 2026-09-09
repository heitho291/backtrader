from pathlib import Path

from tools import xauusd_prefilter_in_detail as prefilter


def _row(ratio, pos, neg, mask_hash="m", combo=(0,)):
    return {
        "ratio": float(ratio),
        "pos_hits": int(pos),
        "neg_hits": int(neg),
        "test_pos_hits": 0,
        "test_neg_hits": 0,
        "test_ratio": 0.0,
        "_mask_hash": mask_hash,
        "_combo": tuple(combo),
        "conds": [],
    }


def test_min_main_classifies_archive_eligibility_not_beam_eligibility():
    search_only = _row(0.99, 10, 2)
    final_valid = _row(1.0, 10, 2)
    assert not prefilter._is_final_valid_result(search_only, 1.0)
    assert prefilter._is_final_valid_result(final_valid, 1.0)
    assert prefilter._final_valid_results([search_only, final_valid], 1.0) == [final_valid]


def test_common_ab_survivors_share_ranking_cap_and_final_archive_filter():
    rows = [
        _row(1.0, 9, 1, "a"),
        _row(0.9, 100, 0, "b"),
        _row(1.0, 10, 3, "c"),
        _row(1.0, 10, 2, "d"),
    ]
    dedupe_calls = []

    def dedupe(items):
        dedupe_calls.append(list(items))
        return list(items)

    survivors, archive, final_batch = prefilter._update_ab_search_pools([], [], rows, 1.0, 4, dedupe)
    assert dedupe_calls == [rows, rows[:1] + rows[2:]]
    assert survivors == [rows[3], rows[2], rows[0], rows[1]]
    assert archive == survivors[:3]
    assert final_batch == [rows[0], rows[2], rows[3]]
    assert prefilter._bounded_train_survivors(rows, 3, lambda items: items) == survivors[:3]
    assert prefilter._ab_parent_seed_snapshot(survivors, 2) == survivors[:2]


def test_phase_b_parent_decision_and_c_seed_mapping_use_survivors_with_single_fallback():
    pool_c = [
        {"col": "a", "op": ">=", "value": 1.0},
        {"col": "b", "op": "<=", "value": 2.0},
    ]
    survivor = _row(0.9, 4, 1)
    survivor["conds"] = [{"col": "b", "op": "<=", "value": 2.0}]
    assert prefilter._phase_b_has_parent_survivors([survivor])
    assert not prefilter._phase_b_has_parent_survivors([])
    mapped, fallback = prefilter._map_c_start_beam([survivor], pool_c, 1, 3)
    assert mapped == [(1,)]
    assert not fallback
    unmappable = _row(0.8, 3, 1)
    unmappable["conds"] = [{"col": "missing", "op": ">=", "value": 1.0}]
    mapped, fallback = prefilter._map_c_start_beam([unmappable], pool_c, 2, 3)
    assert mapped == [(0,), (1,)]
    assert fallback


def test_structure_binary_cap_reject_is_not_a_beam_or_archive_candidate():
    conditions = [
        {"col": "a", "op": "==", "value": 1.0, "binary": True},
        {"col": "b", "op": "==", "value": 1.0, "binary": True},
    ]
    evaluated_calls = []
    evaluated, reason = prefilter._run_structure_gate(
        conditions,
        1,
        lambda _triplets: (True, {}),
        lambda valid: evaluated_calls.append(valid) or _row(2.0, 2, 1),
    )
    assert evaluated is None
    assert reason == "binary_cap"
    assert evaluated_calls == []
    beam_candidates = [] if evaluated is None else [evaluated]
    archive, expandable = prefilter._partition_level_roles(beam_candidates, 1.0, 3)
    assert archive == []
    assert expandable == []


def test_raw_and_post_clm_min_pos_remain_beam_rejects_but_filters_can_be_disabled():
    calls = []

    def select_entries():
        calls.append("clm")
        return "selected", 4, 1

    rejected, reason = prefilter._run_min_pos_gate_pipeline(1, 2.0, 1.0, True, select_entries, lambda _mask: (9, 0))
    assert rejected is None
    assert reason == "raw_min_pos"
    assert calls == []

    rejected, reason = prefilter._run_min_pos_gate_pipeline(2, 2.0, 1.0, True, select_entries, lambda _mask: (1, 1))
    assert rejected is None
    assert reason == "post_clm_min_pos"
    assert calls == ["clm"]

    accepted, reason = prefilter._run_min_pos_gate_pipeline(0, 2.0, 1.0, False, select_entries, lambda _mask: (0, 1))
    assert reason is None
    assert accepted == ("selected", 4, 1, 0, 1)


def test_parent_strict_improvement_is_preserved_for_ab_and_disabled_for_none():
    parent = _row(2.0, 10, 1)
    child = _row(1.5, 10, 1)
    assert not prefilter._parent_score_allows(child, parent)
    assert prefilter._parent_score_allows(child, None)


def test_search_only_ab_survivor_is_a_real_parent_seed_but_not_archived():
    search_only = _row(0.9, 10, 1, combo=(0,))
    survivors, archive, final_batch = prefilter._update_ab_search_pools(
        [], [], [search_only], 1.0, 5, lambda rows: rows
    )
    seeds = prefilter._ab_parent_seed_snapshot(survivors, 5)
    extensions = list(prefilter._iter_all_parent_extensions(seeds, [0, 1, 2], 2))
    assert archive == []
    assert final_batch == []
    assert seeds == [search_only]
    assert extensions == [((0, 1), search_only), ((0, 2), search_only)]


def test_c_and_d_paths_preserve_parent_none_without_immediate_parent_gate():
    parent = _row(2.0, 10, 1)
    child = _row(1.5, 10, 1)
    calls = []

    def evaluate(combo, pool, immediate_parent, source="A"):
        calls.append((combo, pool, immediate_parent, source))
        return child if prefilter._parent_score_allows(child, immediate_parent) else None

    assert evaluate((0,), [], parent, source="A") is None
    assert prefilter._evaluate_without_immediate_parent(evaluate, (0,), ["c"], "C") is child
    assert prefilter._evaluate_without_immediate_parent(evaluate, (0, 1), ["d"], "D") is child
    assert calls[-2][2:] == (None, "C")
    assert calls[-1][2:] == (None, "D")
    source = Path(prefilter.__file__).read_text(encoding="utf-8")
    assert '_evaluate_without_immediate_parent(evaluate_combo, cb, pool_c, "C")' in source
    assert '_evaluate_without_immediate_parent(evaluate_combo, cb, tick_pool, "D")' in source
    assert 'lambda cb: _evaluate_without_immediate_parent(evaluate_combo, cb, tick_pool, "D")' in source


def test_search_ranking_is_ratio_then_positive_then_negative_hits():
    rows = [
        _row(1.0, 10, 4),
        _row(1.1, 1, 99),
        _row(1.0, 11, 9),
        _row(1.0, 11, 2),
    ]
    assert sorted(rows, key=prefilter._train_search_sort_key) == [rows[1], rows[3], rows[2], rows[0]]


def test_c_beam_preserves_no_functional_mask_dedupe_and_reports_diagnostics():
    first = _row(1.1, 5, 1, mask_hash="same", combo=(0, 1))
    second = _row(0.9, 4, 1, mask_hash="same", combo=(0, 2))
    archive, beam, diagnostics = prefilter._select_level_results([first, second], 1.0, 4, 5)
    assert archive == [first]
    assert beam == [first, second]
    assert diagnostics == {
        "beam_candidates": 2,
        "beam_kept": 2,
        "beam_kept_final_valid": 1,
        "beam_kept_search_only": 1,
    }


def test_d_archive_tie_break_and_level_diagnostics_are_preserved():
    shorter = _row(1.1, 5, 1, combo=(0, 1))
    longer = _row(1.1, 5, 1, combo=(0, 1, 2))
    assert sorted([longer, shorter], key=prefilter._d_archive_sort_key) == [shorter, longer]
    archive, beam, diagnostics = prefilter._select_level_results(
        [longer, shorter], 1.0, 4, 5, lambda rows: rows
    )
    assert archive == [longer, shorter]
    assert beam == [longer, shorter]
    assert diagnostics["beam_candidates"] == 2
    assert diagnostics["beam_kept"] == 2
    assert diagnostics["beam_kept_final_valid"] == 2
    assert diagnostics["beam_kept_search_only"] == 0


def test_phase_d_start_depth_changes_only_explicit_one_and_two_semantics():
    assert prefilter._phase_d_initial_depth(1, 9, 8, 4) == 1
    assert prefilter._phase_d_initial_depth(2, 9, 8, 4) == 2
    assert prefilter._phase_d_initial_depth(2, 1, 8, 4) is None
    assert prefilter._phase_d_initial_depth(2, 9, 8, 1) is None
    # Other raw values preserve the base max(1, min(raw, limits, pool)) behavior.
    assert prefilter._phase_d_initial_depth(3, 9, 8, 4) == 3
    assert prefilter._phase_d_initial_depth(3, 2, 8, 4) == 2
    assert prefilter._phase_d_initial_depth(10, 9, 8, 4) == 4
    assert prefilter._phase_d_initial_depth(0, 9, 8, 4) == 1
    assert prefilter._phase_d_initial_depth(-3, 9, 8, 4) == 1


def test_phase_d_start_dispatch_behavior_and_preservation():
    def execute(raw_start, eligible_by_depth, phase_max=4, path_max=4, pool_size=4):
        stages = []
        logs = []

        def run(depth):
            stages.append(("run", depth))
            enabled = bool(eligible_by_depth.get(depth, []))
            rows, _generated, _evaluated, _truncated = prefilter._run_global_d_stage(
                range(pool_size),
                depth,
                64,
                0,
                lambda combo: _row(0.9, 2, 1, combo=combo) if enabled else None,
            )
            return rows

        def consume(depth, tag, rows):
            stages.append(("consume", depth, tag, tuple(rows)))
            return bool(rows)

        result_depth = prefilter._dispatch_phase_d_start(raw_start, phase_max, path_max, pool_size, run, consume, logs.append)
        return result_depth, stages, logs

    depth, stages, logs = execute(1, {1: ["search-only"]})
    assert depth == 1
    assert [entry[1] for entry in stages if entry[0] == "run"] == [1]
    assert logs == ["start1 pair fallback not needed: beam-eligible singles exist"]

    depth, stages, logs = execute(1, {1: [], 2: ["pair"]})
    assert depth == 2
    assert [entry[1] for entry in stages if entry[0] == "run"] == [1, 2]
    assert logs == ["start1 pair fallback triggered"]

    depth, stages, logs = execute(1, {1: [], 2: []})
    assert depth == 2
    assert [entry[1] for entry in stages if entry[0] == "run"] == [1, 2]
    assert logs[-1] == "start1 pair fallback ended without beam-eligible pairs"

    depth, stages, _logs = execute(2, {2: []})
    assert depth == 2
    assert [entry[1] for entry in stages if entry[0] == "run"] == [2]
    depth, stages, logs = execute(2, {}, pool_size=1)
    assert depth is None
    assert stages == []
    assert logs == ["start2 pair stage unavailable: pool_size or effective max depth is below 2"]

    depth, stages, logs = execute(1, {}, phase_max=1)
    assert depth == 1
    assert [entry[1] for entry in stages if entry[0] == "run"] == [1]
    assert logs == ["start1 pair fallback unavailable: pool_size or effective max depth is below 2"]

    depth, stages, _logs = execute(3, {3: []})
    assert depth == 3
    assert [entry[1] for entry in stages if entry[0] == "run"] == [3]
    depth, stages, _logs = execute(3, {2: []}, phase_max=2)
    assert depth == 2
    assert [entry[1] for entry in stages if entry[0] == "run"] == [2]
    depth, stages, _logs = execute(0, {1: []})
    assert depth == 1
    assert [entry[1] for entry in stages if entry[0] == "run"] == [1]


def test_global_d_pair_stage_enumerates_all_pairs_and_preserves_generation_cap():
    seen = []

    def evaluate(combo):
        seen.append(combo)
        return _row(1.1, 3, 1, combo=combo)

    out, generated, evaluated, truncated = prefilter._run_global_d_stage(range(4), 2, 64, 0, evaluate)
    assert seen == [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    assert [row["_combo"] for row in out] == seen
    assert (generated, evaluated, truncated) == (6, 6, False)

    seen.clear()
    out, generated, evaluated, truncated = prefilter._run_global_d_stage(range(4), 2, 64, 3, evaluate)
    assert seen == [(0, 1), (0, 2), (0, 3)]
    assert [row["_combo"] for row in out] == seen
    assert (generated, evaluated, truncated) == (3, 3, True)


def test_tick_override_record_validation_is_shared_and_strict():
    valid = {"y": 1, "pnl": 0.1, "t_exit": 2, "t_qual": 1, "tp_hits": 1}
    assert prefilter._valid_tick_override_record(0, valid, 10, 5)
    assert not prefilter._valid_tick_override_record(0, {**valid, "y": -1}, 10, 5)
    assert not prefilter._valid_tick_override_record(0, {**valid, "t_exit": 0}, 10, 5)
    missing_t_qual = dict(valid)
    missing_t_qual.pop("t_qual")
    assert not prefilter._valid_tick_override_record(0, missing_t_qual, 10, 5)


def test_min_main_is_not_a_score_reject_and_le_off_stays_unfiltered():
    def select_entries():
        return "selected", 3, 1

    def finish_score(_selected, _raw_count, _clusters, pos, neg):
        return _row(pos / max(1, neg), pos, neg)

    under_min_main, reason = prefilter._run_scoring_gate_pipeline(
        2, 1.0, 1.0, True, select_entries, lambda _selected: (2, 3), finish_score
    )
    assert reason is None
    assert under_min_main is not None
    assert under_min_main["ratio"] < 1.0
    assert not prefilter._is_final_valid_result(under_min_main, 1.0)

    le_off, reason = prefilter._run_scoring_gate_pipeline(
        0, 1.0, 99.0, False, select_entries, lambda _selected: (0, 2), finish_score
    )
    assert reason is None
    assert le_off is not None
    assert le_off["pos_hits"] == 0

    # Supplementary call-site guards only; behavioral assertions are above.
    source = Path(prefilter.__file__).read_text(encoding="utf-8")
    assert "ratio < float(args.min_main_score)" not in source
    assert "rejected_min_main_score" not in source
    assert "_score_from_mask(le_mask, only_lower_entry=False, enforce_filters=False)" in source
    assert "_is_final_valid_result(best_sc, float(args.min_main_score))" in source


def test_neighbor_merge_behavior_rejects_search_only_and_accepts_better_final_valid():
    current = _row(1.1, 10, 2)
    search_only_better = _row(0.99, 100, 0)
    final_valid_better = _row(1.2, 10, 2)
    final_valid_worse = _row(1.0, 9, 2)
    assert prefilter._neighbor_merge_trial(current, search_only_better, 1.0) is current
    assert prefilter._neighbor_merge_trial(current, final_valid_better, 1.0) is final_valid_better
    assert prefilter._neighbor_merge_trial(current, final_valid_worse, 1.0) is current


def test_max_depth_archives_only_final_valid_and_expands_neither_category():
    final_valid = _row(1.1, 10, 2, combo=(0, 1, 2))
    search_only = _row(0.9, 10, 2, combo=(0, 1, 2))
    archive, expandable = prefilter._partition_level_roles([final_valid, search_only], 1.0, 3)
    assert archive == [final_valid]
    assert expandable == []

    shallower_search_only = _row(0.9, 10, 2, combo=(0, 1))
    archive, expandable = prefilter._partition_level_roles([shallower_search_only], 1.0, 3)
    assert archive == []
    assert expandable == [shallower_search_only]
