import sys

from tools import xauusd_prefilter_in_detail as prefilter


def _row(ratio=1.0, pos=3, neg=1, combo=(0,), mask_hash="m"):
    return {
        "ratio": float(ratio),
        "pos_hits": int(pos),
        "neg_hits": int(neg),
        "test_pos_hits": 0,
        "test_neg_hits": 0,
        "test_ratio": 0.0,
        "_combo": tuple(combo),
        "_mask_hash": mask_hash,
        "conds": [],
    }


def _parent(parent_id, combo, roots, ratio, pos, neg):
    return {
        "parent_id": int(parent_id),
        "combo": tuple(combo),
        "root_ids": tuple(roots),
        "train_rank": (float(ratio), int(pos), -int(neg)),
    }


def test_exact_combo_dedupe_same_and_cross_batch_and_level_reset():
    raw = [((0, 1), None), ((0, 1), None), ((0, 2), None)]
    calls = []
    state = {}
    out, generated, evaluated, truncated = prefilter._run_exact_combo_stage(
        raw, 1, 0, lambda combo: calls.append(combo) or _row(combo=combo), state_out=state
    )
    assert calls == [(0, 1), (0, 2)]
    assert [r["_combo"] for r in out] == calls
    assert (generated, evaluated, truncated) == (3, 2, False)
    assert state["raw"] == state["unique"] + state["duplicates"]
    assert state["evaluated"] == state["unique"] == 2
    assert state["duplicates"] == 1

    second_calls = []
    second_state = {}
    prefilter._run_exact_combo_stage(
        [((0, 1), None)], 64, 0, lambda combo: second_calls.append(combo) or _row(combo=combo), state_out=second_state
    )
    assert second_calls == [(0, 1)]
    assert state["seen"] is not second_state["seen"]


def test_exact_combo_seen_state_crosses_real_batch_boundary():
    first = (0, 100)
    raw = [(first, None)] + [((i, i + 1), None) for i in range(1, 64)] + [(first, None)]
    calls = []
    state = {}
    prefilter._run_exact_combo_stage(
        raw, 64, 0, lambda combo: calls.append(combo) or _row(combo=combo), state_out=state
    )
    assert len(raw) == 65
    assert calls.count(first) == 1
    assert state["raw"] == 65
    assert state["unique"] == state["evaluated"] == 64
    assert state["duplicates"] == 1


def test_exact_combo_cap_counts_raw_prefix_and_does_not_backfill():
    a, b, c = (0, 1), (0, 2), (0, 3)
    calls = []
    state = {}
    out, generated, evaluated, truncated = prefilter._run_exact_combo_stage(
        [(a, None), (a, None), (b, None), (c, None)],
        64,
        3,
        lambda combo: calls.append(combo) or _row(combo=combo),
        state_out=state,
    )
    assert calls == [a, b]
    assert [r["_combo"] for r in out] == [a, b]
    assert (generated, evaluated, truncated) == (3, 2, True)
    assert state["duplicates"] == 1


def test_exact_combo_stage_covers_add_one_add_two_and_separate_phase_states():
    base = (0,)
    add_one = [(tuple(sorted(set(base) | {i})), None) for i in (1, 2)]
    add_two = [(tuple(sorted(set(base) | set(pair))), None) for pair in ((1, 2), (1, 3), (2, 3))]
    c_state, d_state = {}, {}
    c_calls, d_calls = [], []
    prefilter._run_exact_combo_stage(
        add_one + add_two, 64, 0, lambda combo: c_calls.append(combo) or _row(combo=combo), state_out=c_state
    )
    prefilter._run_exact_combo_stage(
        add_one, 64, 0, lambda combo: d_calls.append(combo) or _row(combo=combo), state_out=d_state
    )
    assert c_calls == [(0, 1), (0, 2), (0, 1, 2), (0, 1, 3), (0, 2, 3)]
    assert d_calls == [(0, 1), (0, 2)]
    assert c_state["seen"] is not d_state["seen"]


def test_global_d_stage_uses_exact_runner_and_preserves_unique_combinations():
    state = {}
    calls = []
    out, generated, evaluated, truncated = prefilter._run_global_d_stage(
        range(4), 2, 64, 0, lambda combo: calls.append(combo) or _row(combo=combo), state
    )
    assert calls == [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    assert [r["_combo"] for r in out] == calls
    assert (generated, evaluated, truncated) == (6, 6, False)
    assert state["duplicates"] == 0


def test_real_same_reference_reduction_keeps_original_exact_key_and_scores_once():
    pool = [
        {"col": "support_a", "op": ">=", "value": 1.0},
        {"col": "support_b", "op": "<=", "value": 2.0},
    ]
    inner_calls = []
    removed = []

    def evaluate(combo):
        return prefilter._evaluate_with_same_reference(
            combo,
            pool,
            {"support_a": 7, "support_b": 7},
            lambda _col: {"family": "dist_support"},
            lambda conds: inner_calls.append(list(conds)) or {"ratio": 1.0},
            lambda: removed.append(1),
        )

    state = {}
    out, _raw, evaluated, _truncated = prefilter._run_exact_combo_stage(
        [((0, 1), None), ((0, 1), None)], 64, 0, evaluate, state_out=state
    )
    assert evaluated == 1
    assert len(inner_calls) == 1
    assert inner_calls[0] == [pool[0]]
    assert removed == [1]
    assert out[0]["_combo"] == (0, 1)
    assert state["seen"] == {(0, 1)}


def test_provenance_aggregates_parents_roots_and_all_exact_tie_winners():
    parents = {
        1: _parent(1, (0,), (10,), 1.0, 5, 2),
        2: _parent(2, (1,), (10, 20), 1.0, 5, 2),
        3: _parent(3, (2,), (30,), 0.9, 100, 0),
    }
    state = {}
    calls = []
    prefilter._run_exact_combo_stage(
        [((0, 1, 2), 1), ((0, 1, 2), 2), ((0, 1, 2), 3)],
        64,
        0,
        lambda combo: calls.append(combo) or _row(combo=combo),
        parents,
        state,
    )
    prov = state["provenance"][(0, 1, 2)]
    assert calls == [(0, 1, 2)]
    assert prov["parent_ids"] == {1, 2, 3}
    assert prov["root_ids"] == {10, 20, 30}
    assert prov["best_parent_ids"] == {1, 2}

    reordered_state = {}
    prefilter._run_exact_combo_stage(
        [((0, 1, 2), 3), ((0, 1, 2), 2), ((0, 1, 2), 1)],
        64,
        0,
        lambda combo: _row(combo=combo),
        parents,
        reordered_state,
    )
    reordered = reordered_state["provenance"][(0, 1, 2)]
    assert reordered["parent_ids"] == prov["parent_ids"]
    assert reordered["root_ids"] == prov["root_ids"]
    assert reordered["best_train_rank"] == prov["best_train_rank"]
    assert reordered["best_parent_ids"] == prov["best_parent_ids"] == {1, 2}


def test_provenance_three_field_ranking_replaces_ties_only_when_strictly_better():
    parents = {
        1: _parent(1, (0,), (1,), 1.0, 5, 2),
        2: _parent(2, (1,), (2,), 1.0, 5, 2),
        3: _parent(3, (2,), (3,), 1.0, 6, 9),
        4: _parent(4, (3,), (4,), 1.0, 6, 1),
        5: _parent(5, (4,), (5,), 1.1, 1, 99),
        6: _parent(6, (5,), (6,), 1.1, 1, 99),
    }
    state = {}
    prefilter._run_exact_combo_stage(
        [((0, 1), pid) for pid in (1, 2, 3, 4, 5, 6)],
        64,
        0,
        lambda combo: _row(combo=combo),
        parents,
        state,
    )
    prov = state["provenance"][(0, 1)]
    assert prov["best_train_rank"] == (1.1, 1, -99)
    assert prov["best_parent_ids"] == {5, 6}


def test_advance_beam_nodes_propagates_only_aggregated_roots():
    rule = _row(1.2, 8, 2, combo=(0, 1))
    provenance = {(0, 1): {"root_ids": {4, 7}, "parent_ids": {1, 2}, "best_parent_ids": {1, 2}}}
    nodes, next_id = prefilter._advance_beam_nodes([rule], provenance, 9)
    assert nodes == [{"parent_id": 9, "combo": (0, 1), "root_ids": (4, 7), "train_rank": (1.2, 8, -2)}]
    assert next_id == 10


def test_c_start_nodes_map_real_seeds_and_preserve_partial_unmappable_behavior():
    pool = [
        {"col": "a", "op": ">=", "value": 1.0, "_single_ratio": 1.0, "_single_pos_hits": 3, "_single_neg_hits": 1},
        {"col": "b", "op": "<=", "value": 2.0, "_single_ratio": 0.8, "_single_pos_hits": 2, "_single_neg_hits": 1},
    ]
    good = _row(1.2, 6, 2)
    good["conds"] = [{"col": "b", "op": "<=", "value": 2.0}]
    bad = _row(2.0, 9, 1)
    bad["conds"] = [{"col": "missing", "op": ">=", "value": 1.0}]
    nodes, fallback, next_id = prefilter._map_c_start_nodes([good, bad], pool, 2, 3)
    assert not fallback
    assert nodes == [{"parent_id": 0, "combo": (1,), "root_ids": (0,), "train_rank": (1.2, 6, -2)}]
    assert next_id == 1
    assert prefilter._map_c_start_nodes([good, bad], pool, 2, 3) == (nodes, fallback, next_id)


def test_c_start_nodes_create_two_stable_roots_for_two_normally_mapped_seeds():
    pool = [
        {"col": "a", "op": ">=", "value": 1.0},
        {"col": "b", "op": "<=", "value": 2.0},
    ]
    first = _row(1.2, 6, 2)
    first["conds"] = [{"col": "a", "op": ">=", "value": 1.0}]
    second = _row(1.1, 5, 2)
    second["conds"] = [{"col": "b", "op": "<=", "value": 2.0}]
    nodes, fallback, next_id = prefilter._map_c_start_nodes([second, first], pool, 2, 3)
    assert not fallback
    assert nodes == [
        {"parent_id": 0, "combo": (0,), "root_ids": (0,), "train_rank": (1.2, 6, -2)},
        {"parent_id": 1, "combo": (1,), "root_ids": (1,), "train_rank": (1.1, 5, -2)},
    ]
    assert next_id == 2
    assert prefilter._map_c_start_nodes([second, first], pool, 2, 3) == (nodes, fallback, next_id)


def test_c_single_fallback_nodes_are_stable_unique_origins():
    pool = [
        {"col": "a", "op": ">=", "value": 1.0, "_single_ratio": 1.0, "_single_pos_hits": 3, "_single_neg_hits": 1},
        {"col": "b", "op": "<=", "value": 2.0, "_single_ratio": 0.8, "_single_pos_hits": 2, "_single_neg_hits": 1},
    ]
    bad = _row()
    bad["conds"] = [{"col": "missing", "op": ">=", "value": 1.0}]
    nodes, fallback, next_id = prefilter._map_c_start_nodes([bad], pool, 2, 3)
    assert fallback
    assert [node["combo"] for node in nodes] == [(0,), (1,)]
    assert [node["root_ids"] for node in nodes] == [(0,), (1,)]
    assert [node["train_rank"] for node in nodes] == [(1.0, 3, -1), (0.8, 2, -1)]
    assert next_id == 2

    empty, still_fallback, empty_next = prefilter._map_c_start_nodes([bad], pool, 2, 0)
    assert empty == [] and still_fallback and empty_next == 0


def test_d_dispatch_initializes_only_selected_start_roots_for_single_pair_and_special_depth():
    def execute(raw_start, accepted_depths, beam_width=2):
        selected_nodes = []
        stages = []

        def run(depth):
            stages.append(depth)
            rows, _raw, _evaluated, _truncated = prefilter._run_global_d_stage(
                range(4), depth, 64, 0, lambda combo: _row(1.0 + combo[0] / 10.0, 4, 1, combo=combo) if depth in accepted_depths else None
            )
            return rows

        def consume(_depth, _tag, rows):
            _archive, beam, _diag = prefilter._select_level_results(rows, 0.0, 5, beam_width, lambda values: values)
            nodes, _next = prefilter._make_start_nodes(beam)
            selected_nodes[:] = nodes
            return bool(rows)

        result = prefilter._dispatch_phase_d_start(raw_start, 5, 5, 4, run, consume, lambda _message: None)
        return result, stages, selected_nodes

    depth, stages, singles = execute(1, {1})
    assert depth == 1 and stages == [1]
    assert len(singles) == 2 and [n["root_ids"] for n in singles] == [(0,), (1,)]

    depth, stages, pairs = execute(2, {2})
    assert depth == 2 and stages == [2]
    assert len(pairs) == 2 and [n["root_ids"] for n in pairs] == [(0,), (1,)]

    depth, stages, fallback_pairs = execute(1, {2})
    assert depth == 2 and stages == [1, 2]
    assert len(fallback_pairs) == 2 and [n["root_ids"] for n in fallback_pairs] == [(0,), (1,)]

    depth, stages, triples = execute(3, {3})
    assert depth == 3 and stages == [3]
    assert len(triples) == 2 and [n["root_ids"] for n in triples] == [(0,), (1,)]
    assert execute(3, {3})[2] == triples


def test_d_pair_fallback_without_eligible_pairs_creates_no_roots():
    roots = []
    stages = []

    def run(depth):
        stages.append(depth)
        return []

    def consume(_depth, _tag, rows):
        nodes, _next = prefilter._make_start_nodes(rows)
        roots[:] = nodes
        return bool(rows)

    assert prefilter._dispatch_phase_d_start(1, 4, 4, 4, run, consume, lambda _message: None) == 2
    assert stages == [1, 2]
    assert roots == []


def test_c_mask_diagnostic_is_side_effect_free_and_real_c_beam_keeps_same_masks(monkeypatch):
    rows = [
        _row(1.2, 5, 1, combo=(0, 1), mask_hash="x"),
        _row(1.1, 4, 1, combo=(0, 2), mask_hash="x"),
        _row(1.0, 3, 1, combo=(0, 3), mask_hash="y"),
    ]
    before = [dict(row) for row in rows]
    reject_stats = {"rejected_duplicate_mask": 0}
    def forbidden_functional_dedupe(_rows, on_duplicate=None):
        reject_stats["rejected_duplicate_mask"] += 1
        if on_duplicate is not None:
            on_duplicate()
        raise AssertionError("C diagnostic must not call functional mask dedupe")

    monkeypatch.setattr(prefilter, "_dedupe_mask_rows", forbidden_functional_dedupe)
    diag = prefilter._c_mask_duplicate_diagnostics(rows, 4, 2)
    _archive, beam, _stats = prefilter._select_level_results(rows, 0.0, 4, 3)
    assert diag == {
        "c_valid_results_before_mask_dedupe": 3,
        "c_unique_masks": 2,
        "c_mask_duplicate_results": 1,
        "c_hypothetical_beam_unique_masks": 2,
    }
    assert len(beam) == 3
    assert rows == before
    assert reject_stats == {"rejected_duplicate_mask": 0}


def test_production_d_mask_dedupe_seam_remains_functional_and_counts_duplicates():
    longer = _row(1.1, 5, 1, combo=(0, 1), mask_hash="same")
    longer["conds"] = [{"col": "a"}, {"col": "b"}]
    shorter = _row(1.1, 5, 1, combo=(0, 2), mask_hash="same")
    shorter["conds"] = [{"col": "a"}]
    distinct = _row(1.0, 4, 1, combo=(0, 3), mask_hash="other")
    duplicate_calls = []
    deduped = prefilter._dedupe_mask_rows([longer, shorter, distinct], lambda: duplicate_calls.append(1))
    assert deduped == [shorter, distinct]
    assert duplicate_calls == [1]

    _archive, d_beam, _diag = prefilter._select_level_results(
        [longer, shorter, distinct], 0.0, 4, 3, prefilter._dedupe_mask_rows
    )
    assert d_beam == [shorter, distinct]


def test_score_stage_helpers_debug_off_make_no_perf_counter_calls(monkeypatch):
    monkeypatch.setattr(prefilter.time, "perf_counter", lambda: (_ for _ in ()).throw(AssertionError("timed")))
    prefilter._record_score_evaluate(None)
    assert prefilter._run_mask_build(lambda: "mask", None) == "mask"
    score, reason = prefilter._run_score_stages(
        lambda: (2, 3), 1.0, 1.0, True, lambda: ("selected", 3, 1), lambda _selected: (2, 1),
        lambda *_args: {"ratio": 2.0}, None,
    )
    assert reason is None and score == {"ratio": 2.0}


def test_score_stage_profiler_counts_rejects_success_entries_and_nonnegative_times(monkeypatch):
    clock = iter(float(i) for i in range(50))
    monkeypatch.setattr(prefilter.time, "perf_counter", lambda: next(clock))

    raw_stats = prefilter._new_score_stage_stats()
    prefilter._record_score_evaluate(raw_stats)
    prefilter._run_mask_build(lambda: "mask", raw_stats)
    score, reason = prefilter._run_score_stages(
        lambda: (0, 7), 1.0, 1.0, True, lambda: (_ for _ in ()).throw(AssertionError("clm")),
        lambda _selected: (0, 0), lambda *_args: {}, raw_stats,
    )
    assert score is None and reason == "raw_min_pos"
    assert raw_stats["evaluate_calls"] == 1
    assert raw_stats["mask_build_calls"] == 1
    assert raw_stats["mask_build_sec"] >= 0.0
    assert raw_stats["raw_precheck_calls"] == raw_stats["raw_precheck_rejects"] == 1
    assert raw_stats["raw_entries_total"] == 7
    assert raw_stats["clm_calls"] == raw_stats["remaining_score_calls"] == 0

    post_stats = prefilter._new_score_stage_stats()
    prefilter._record_score_evaluate(post_stats)
    score, reason = prefilter._run_score_stages(
        lambda: (2, 5), 1.0, 1.0, True, lambda: ("selected", 5, 2), lambda _selected: (0, 1),
        lambda *_args: {}, post_stats,
    )
    assert score is None and reason == "post_clm_min_pos"
    assert post_stats["evaluate_calls"] == 1
    assert post_stats["clm_calls"] == post_stats["remaining_score_calls"] == 1
    assert post_stats["clm_entries_total"] == 5
    assert post_stats["post_clm_minpos_rejects"] == 1

    ok_stats = prefilter._new_score_stage_stats()
    prefilter._record_score_evaluate(ok_stats)
    prefilter._run_mask_build(lambda: "mask", ok_stats)
    score, reason = prefilter._run_score_stages(
        lambda: (3, 6), 1.0, 1.0, True, lambda: ("selected", 6, 2), lambda _selected: (3, 1),
        lambda *_args: {"ratio": 3.0}, ok_stats,
    )
    summary = prefilter._score_stage_summary(ok_stats, 9.0)
    assert reason is None and score == {"ratio": 3.0}
    assert summary["evaluate_calls"] == 1
    assert summary["mask_build_calls"] == 1
    assert summary["avg_raw_entries"] == summary["avg_clm_entries"] == 6.0
    assert summary["level_wall_sec"] == 9.0
    assert all(float(summary[key]) >= 0 for key in ("mask_build_sec", "raw_precheck_sec", "clm_sec", "remaining_score_sec"))
    required_fields = {
        "evaluate_calls", "mask_build_calls", "mask_build_sec", "raw_precheck_calls", "raw_precheck_sec",
        "raw_precheck_rejects", "raw_entries_total", "avg_raw_entries", "clm_calls", "clm_sec",
        "clm_entries_total", "avg_clm_entries", "post_clm_minpos_rejects", "remaining_score_calls",
        "remaining_score_sec", "level_wall_sec",
    }
    assert required_fields <= summary.keys()
    line = prefilter._format_score_stage_line("C", 2, "extension", ok_stats, 9.0)
    for field in required_fields - {"raw_entries_total", "clm_entries_total"}:
        assert f"{field}=" in line


def test_score_profiler_on_off_preserves_score_and_beam_roles(monkeypatch):
    monkeypatch.setattr(prefilter.time, "perf_counter", lambda: 1.0)

    def run(stats):
        return prefilter._run_score_stages(
            lambda: (3, 4),
            1.0,
            1.0,
            True,
            lambda: ("selected", 4, 1),
            lambda _selected: (3, 2),
            lambda *_args: _row(0.9, 3, 2, combo=(0, 1)),
            stats,
        )

    score_off, reason_off = run(None)
    stats = prefilter._new_score_stage_stats()
    score_on, reason_on = run(stats)
    assert (score_on, reason_on) == (score_off, reason_off)
    assert prefilter._select_level_results([score_on], 1.0, 3, 2) == prefilter._select_level_results([score_off], 1.0, 3, 2)


def test_debug_cli_flags_default_off_and_parse_when_enabled(monkeypatch):
    required = ["prog", "--features", "f", "--binned-features", "b", "--binned-metadata", "m"]
    monkeypatch.setattr(sys, "argv", required)
    args = prefilter.parse_args()
    assert not args.debug_exact_combo_stats
    assert not args.debug_score_stage_timing
    monkeypatch.setattr(sys, "argv", required + ["--debug-exact-combo-stats", "--debug-score-stage-timing"])
    args = prefilter.parse_args()
    assert args.debug_exact_combo_stats
    assert args.debug_score_stage_timing
