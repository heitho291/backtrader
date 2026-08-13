#!/usr/bin/env python3
"""Standalone in-detail prefilter combinatorial search.

Uses feature and binned-feature files for feature/mask construction, builds labels
from AskBid-M1 entry/exit prices, then searches complete rule combinations.
"""
# TEMPORARY PR-BEHAVIOR PROBE 2 - remove after UI test.

from __future__ import annotations

import argparse
import concurrent.futures
import collections
import gc
import hashlib
import importlib.util
import itertools
import json
import math
import os
import random
import tempfile
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

NS_PER_MINUTE = 60_000_000_000
ASKBID_M1_COLUMNS = [
    "datetime",
    "ask_open",
    "ask_entry_latency",
    "ask_high",
    "ask_low",
    "ask_close",
    "bid_open",
    "bid_high",
    "bid_low",
    "bid_close",
    "bid_high_after_entry_latency",
    "bid_low_after_entry_latency",
    "volume_check",
    "ticks_count",
]
ASKBID_M1_PRICE_COLUMNS = [
    "ask_open",
    "ask_entry_latency",
    "ask_high",
    "ask_low",
    "ask_close",
    "bid_open",
    "bid_high",
    "bid_low",
    "bid_close",
    "bid_high_after_entry_latency",
    "bid_low_after_entry_latency",
]
ASKBID_M1_SEMANTICS = "askbid_m1_quote_continuous_feature_grid_v1_entry_latency_after_bounds_v1"
ASKBID_M1_MAX_CARRY_GAP_MINUTES = 360
CANDIDATE_CACHE_SCHEMA_VERSION = "candidate_inventory_v2"
REFINED_CACHE_SCHEMA_VERSION = "candidate_refined_intrinsic_metrics_v2"
REQUIRED_TICK_METRIC_COLUMNS = (
    "tick_single_pos_hits",
    "tick_single_neg_hits",
    "tick_single_mask_count",
    "tick_single_ratio",
    "tick_single_mask_keep_ratio",
    "tick_single_ratio_change",
)


def _load_miner_module(path: Path):
    spec = importlib.util.spec_from_file_location("xau_miner_module_for_prefilter", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load miner script: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Standalone in-detail co-occurrence combinatorial search")
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--binned-features", type=Path, required=True)
    p.add_argument("--binned-metadata", type=Path, required=True)
    p.add_argument("--miner-script", type=Path, default=Path("tools/xauusd_miner_ohlc_first_hit_pessimistic.py"))

    p.add_argument("--train-frac", type=float, default=0.7)
    p.add_argument("--wf-folds", type=int, default=4)
    p.add_argument("--quantiles", type=str, default="0.05,0.10,0.90,0.95")

    p.add_argument("--tps", type=str, default="0.0015,0.0025,0.0035,0.0045")
    p.add_argument("--tp-weights", type=str, default="")
    p.add_argument("--use-multi-tp", action="store_true", default=True)
    p.add_argument("--no-use-multi-tp", dest="use_multi_tp", action="store_false")
    p.add_argument("--no-tp", action="store_true", default=False, help="Disable TP exits (allowed only with --trail)")
    p.add_argument("--sl", type=float, default=0.0015)
    p.add_argument("--hold", type=int, default=90)
    p.add_argument("--slippage-bps", type=float, default=0.0)
    p.add_argument("--spread-bps", type=float, default=0.0)
    p.add_argument("--trail", action="store_true", default=True)
    p.add_argument("--no-trail", dest="trail", action="store_false")
    p.add_argument("--trail-activate", type=float, default=0.0010)
    p.add_argument("--trail-offset", type=float, default=0.0006)
    p.add_argument("--trail-factor", type=float, default=0.5)
    p.add_argument("--trail-min-level", type=float, default=0.0)
    p.add_argument("--include-unrealized-at-test-end", action="store_true", default=True)
    p.add_argument("--no-include-unrealized-at-test-end", dest="include_unrealized_at_test_end", action="store_false")
    p.add_argument("--cluster-gap-minutes", type=int, default=1)
    p.add_argument("--max-entries-per-cluster", type=int, default=15)
    p.add_argument("--max-open-trades", type=int, default=15)
    p.add_argument("--cluster-only-lower-entry", dest="cluster_only_lower_entry", action="store_true", default=True)
    p.add_argument("--no-cluster-only-lower-entry", dest="cluster_only_lower_entry", action="store_false")

    p.add_argument("--tick-data", type=Path, default=None)
    p.add_argument("--tick-cache-parquet", type=Path, default=None)
    p.add_argument("--tick-entry-cache-npz", type=Path, default=None)
    p.add_argument("--askbid-m1-parquet", type=Path, default=None)
    p.add_argument("--entry-latency-ms", type=int, default=250)
    p.add_argument("--tick-datetime-column", type=str, default="datetime")
    p.add_argument("--tick-price-column", type=str, default="auto")
    p.add_argument("--tick-sep", type=str, default=",")
    p.add_argument("--tick-refine-scope", type=str, default="fam_top", choices=["fam_top", "filtered", "all_coarse"])
    p.add_argument("--tick-chunk-size", type=str, default="auto")
    p.add_argument("--debug-tick-cache-scope-counts", action="store_true", default=False)
    p.add_argument("--tick-minute-load-mode", type=str, default="critical", choices=["critical", "full_window"])

    p.add_argument("--step-size", type=int, default=5)
    p.add_argument("--top-paths", type=int, default=24)
    p.add_argument("--max-path-conds", type=int, default=8)
    p.add_argument("--workers", type=str, default="auto")
    p.add_argument("--batch-size", type=int, default=50000)
    p.add_argument("--memory-soft-limit-gb", type=float, default=20.0)
    p.add_argument("--max-valids", type=int, default=1000)
    p.add_argument("--early-stop-top-k", type=int, default=100)
    p.add_argument("--early-stop-window-combos", type=int, default=150000)
    p.add_argument("--early-stop-avg-improve-pct", type=float, default=0.25)
    p.add_argument("--batch-random-seed", type=int, default=42)
    p.add_argument("--debug-reject-stats", action="store_true", default=False)
    p.add_argument("--debug-timing-breakdown", action="store_true", default=False)
    p.add_argument("--debug-atr-candidates", action="store_true", default=False)
    p.add_argument("--label-cache-npz", type=Path, default=None)
    p.add_argument("--min-single-pos-hits", type=int, default=2)
    p.add_argument("--min-single-lift", type=float, default=1.01)
    p.add_argument("--max-single-mask-count", type=int, default=0)
    p.add_argument("--family-top-n", type=int, default=40)
    p.add_argument("--family-split-delta-window", action="store_true", default=False)
    p.add_argument("--include-candidates-file", type=Path, default=None)
    p.add_argument("--out-candidates-coarse-csv", type=Path, default=None)
    p.add_argument("--out-candidates-refined-csv", type=Path, default=None)
    p.add_argument("--out-candidates-csv", type=Path, default=None, help="Deprecated (forbidden): use coarse/refined candidate CSV paths")

    p.add_argument("--min-pos-per-week", type=float, default=1.0)
    p.add_argument("--min-main-score", type=float, default=1.0)
    p.add_argument("--binary-cap-per-block", type=int, default=6)
    p.add_argument("--binary-cap-per-list-block", type=int, default=2)
    p.add_argument("--binary-anchor-lookahead-blocks", type=int, default=2)

    p.add_argument("--out-rules-json", type=Path, default=Path("prefilter_in_detail_rules.json"))
    p.add_argument("--out-rules-csv", type=Path, default=Path("prefilter_in_detail_rules.csv"))
    p.add_argument("--control-file", type=Path, default=None)
    p.add_argument("--phase-c-beam-width", type=int, default=64)
    p.add_argument("--phase-c-add-min", type=int, default=1)
    p.add_argument("--phase-c-add-max", type=int, default=2)
    p.add_argument("--phase-c-max-conds", type=int, default=9)
    p.add_argument("--phase-c-max-generated-per-level", type=int, default=0)
    p.add_argument("--phase-d-beam-width", type=int, default=64)
    p.add_argument("--phase-d-start-conds", type=int, default=2)
    p.add_argument("--phase-d-add-min", type=int, default=1)
    p.add_argument("--phase-d-add-max", type=int, default=2)
    p.add_argument("--phase-d-max-conds", type=int, default=9)
    p.add_argument("--phase-d-max-generated-per-level", type=int, default=0)
    return p.parse_args()


def _batched(iterable: Iterable[Tuple[int, ...]], n: int):
    buf = []
    for x in iterable:
        buf.append(x)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf


def _chunk_list(xs: list, n: int) -> list[list]:
    return [xs[i:i + n] for i in range(0, len(xs), n)]


def _next_batch(it: Iterable[Tuple[int, ...]], n: int) -> list[Tuple[int, ...]]:
    return list(itertools.islice(it, max(1, int(n))))


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(path.parent), encoding="utf-8") as tf:
        tf.write(text)
        tmp = Path(tf.name)
    os.replace(tmp, path)


def _disabled_output_path(value) -> bool:
    if value is None:
        return False
    txt = str(value).strip().lower()
    return txt in {"", "-", "0", "false", "none", "null"}


def _safe_ratio_or_nan(num, den) -> float:
    try:
        n = float(num)
        d = float(den)
    except Exception:
        return float("nan")
    if not np.isfinite(n) or not np.isfinite(d) or d == 0.0:
        return float("nan")
    out = n / d
    return float(out) if np.isfinite(out) else float("nan")


def _atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(path.parent), encoding="utf-8", newline="") as tf:
        frame.to_csv(tf.name, index=False)
        tmp = Path(tf.name)
    os.replace(tmp, path)


def _write_csv_if_changed(path: Path, frame: pd.DataFrame, key_cols: list[str] | None = None) -> tuple[bool, str]:
    def _canonicalize_frame_for_compare(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if {"col", "op", "value"}.issubset(set(out.columns)):
            vals = []
            for _, r in out.iterrows():
                vals.append(_canonicalize_candidate_value(str(r.get("col", "")), str(r.get("op", "")), float(r.get("value", np.nan))))
            out["value"] = vals
        if "stable_candidate_key" in out.columns:
            out["stable_candidate_key"] = out["stable_candidate_key"].astype(str).str.strip()
        return out
    if path.exists():
        try:
            old = pd.read_csv(path)
            new_cmp = _canonicalize_frame_for_compare(frame)
            old_cmp = _canonicalize_frame_for_compare(old)
            if key_cols:
                use_cols = [c for c in key_cols if c in new_cmp.columns and c in old_cmp.columns]
                if use_cols:
                    new_cmp = new_cmp.sort_values(use_cols, kind="mergesort").reset_index(drop=True)
                    old_cmp = old_cmp.sort_values(use_cols, kind="mergesort").reset_index(drop=True)
                else:
                    new_cmp = new_cmp.reset_index(drop=True)
                    old_cmp = old_cmp.reset_index(drop=True)
            else:
                new_cmp = new_cmp.reset_index(drop=True)
                old_cmp = old_cmp.reset_index(drop=True)
            if new_cmp.equals(old_cmp):
                return False, "rows_unchanged"
        except Exception:
            pass
    _atomic_write_csv(path, frame)
    return True, "rows_changed"




def _ordered_frame(rows: list[dict], preferred_cols: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=preferred_cols)
    extra = [c for c in frame.columns if c not in preferred_cols]
    return frame[[c for c in preferred_cols if c in frame.columns] + extra]


def _atomic_write_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", delete=False, dir=str(path.parent), suffix=".npz") as tf:
        tmp = Path(tf.name)
    try:
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except Exception:
                pass


TICK_ENTRY_CACHE_SEMANTICS_VERSION = "tick_entry_cache_askbid_m1_critical_replay_c1_tp_weights_trail_order_final_stop_v1"


def _load_tick_entry_cache(path: Path | None, cache_key: str) -> dict[int, dict] | None:
    if path is None:
        return None
    if not path.exists():
        return {}
    try:
        with np.load(path, allow_pickle=False) as z:
            stored = str(np.asarray(z["cache_key"]).item()) if "cache_key" in z.files else ""
            if stored != str(cache_key):
                print("[prefilter-tick-cache] cache context mismatch; ignoring old cache.")
                return {}
            stored_version = str(np.asarray(z["cache_semantics_version"]).item()) if "cache_semantics_version" in z.files else ""
            if stored_version != TICK_ENTRY_CACHE_SEMANTICS_VERSION:
                print("[prefilter-tick-cache] cache semantics version mismatch; ignoring old cache.")
                return {}
            entries = np.asarray(z["entry_idx"], dtype=np.int64)
            tick_y = np.asarray(z["tick_y"])
            tick_pnl = np.asarray(z["tick_pnl"])
            tick_t_exit = np.asarray(z["tick_t_exit"])
            tick_t_qual = np.asarray(z["tick_t_qual"])
            tick_tp_hits = np.asarray(z["tick_tp_hits"])
        out: dict[int, dict] = {}
        for pos, idx in enumerate(entries.tolist()):
            out[int(idx)] = {
                "y": int(tick_y[pos]),
                "pnl": float(tick_pnl[pos]),
                "t_exit": int(tick_t_exit[pos]),
                "t_qual": int(tick_t_qual[pos]),
                "tp_hits": int(tick_tp_hits[pos]),
            }
        print(f"[prefilter-tick-cache] loaded entries={len(out)} from {path}")
        return out
    except Exception as e:
        print(f"[prefilter-tick-cache] failed to load cache; ignoring old cache: {e}")
        return {}


def _write_tick_entry_cache(path: Path | None, cache_key: str, cache: dict[int, dict]) -> None:
    if path is None:
        return
    idxs = np.asarray(sorted(int(k) for k in cache.keys()), dtype=np.int64)
    def arr(name: str, dtype, default):
        return np.asarray([cache[int(i)].get(name, default) for i in idxs.tolist()], dtype=dtype)
    _atomic_write_npz(
        path,
        cache_key=np.asarray(str(cache_key)),
        cache_semantics_version=np.asarray(TICK_ENTRY_CACHE_SEMANTICS_VERSION),
        entry_idx=idxs,
        tick_y=arr("y", np.int8, -1),
        tick_pnl=arr("pnl", np.float32, np.nan),
        tick_t_exit=arr("t_exit", np.int32, -1),
        tick_t_qual=arr("t_qual", np.int32, -1),
        tick_tp_hits=arr("tp_hits", np.int8, 0),
    )
    print(f"[prefilter-tick-cache] wrote entries={len(idxs)} to {path}")



def _print_tick_cache_diagnostics(
    cache_entries: dict[int, dict] | None,
    tick_results: dict[int, dict] | None,
    expected_entries: np.ndarray | None,
    label_y: np.ndarray,
    label_t_exit: np.ndarray,
    hold: int,
) -> None:
    cache_entries = cache_entries or {}
    tick_results = tick_results or {}
    if not cache_entries and not tick_results:
        return
    expected_set = set(int(i) for i in np.asarray(expected_entries if expected_entries is not None else [], dtype=np.int64).tolist())
    diagnostic_entries = cache_entries if cache_entries else tick_results
    expected_present = {int(i): tick_results[int(i)] for i in expected_set if int(i) in tick_results}
    compare_entries = expected_present if expected_set else tick_results

    def _vals(entries: dict[int, dict], name: str, default) -> np.ndarray:
        return np.asarray([rec.get(name, default) for rec in entries.values()])

    y_vals = _vals(diagnostic_entries, "y", -1).astype(np.int64, copy=False)
    t_exit_vals = _vals(diagnostic_entries, "t_exit", -1).astype(np.int64, copy=False)
    pnl_vals = _vals(diagnostic_entries, "pnl", np.nan).astype(np.float64, copy=False)
    y01 = (y_vals == 0) | (y_vals == 1)
    hold_count = int(np.sum(t_exit_vals == int(hold)))
    hold_share = hold_count / max(1, int(np.sum(y01)))
    invalid_t_exit = int(np.sum(y01 & ((t_exit_vals < 1) | (t_exit_vals > int(hold)))))
    pnl_bad = int(np.sum(~np.isfinite(pnl_vals)))
    missing_expected = int(len(expected_set - set(int(k) for k in tick_results.keys()))) if expected_set else 0
    print(
        "[prefilter-tick-cache-diagnostics] "
        f"cache_entries_total={len(cache_entries)} current_tick_entries={len(tick_results)} "
        f"expected_entries={len(expected_set)} tick_missing={missing_expected}"
    )
    print(
        "[prefilter-tick-cache-diagnostics] "
        f"tick_y_1={int(np.sum(y_vals == 1))} tick_y_0={int(np.sum(y_vals == 0))} "
        f"tick_y_minus1={int(np.sum(y_vals == -1))} tick_t_exit_hold={hold_count} "
        f"tick_t_exit_hold_share_y01={hold_share:.6f} tick_t_exit_invalid_y01={invalid_t_exit} "
        f"tick_pnl_nan_or_inf={pnl_bad}"
    )

    def _print_percentiles(y_value: int) -> None:
        arr = t_exit_vals[(y_vals == y_value) & np.isfinite(t_exit_vals)]
        arr = arr[arr >= 0]
        if arr.size <= 0:
            print(f"[prefilter-tick-cache-diagnostics] tick_t_exit_y{y_value}_count=0")
            return
        qs = np.percentile(arr.astype(np.float64, copy=False), [50, 90, 95, 99])
        print(
            "[prefilter-tick-cache-diagnostics] "
            f"tick_t_exit_y{y_value}_count={int(arr.size)} median={qs[0]:.3f} "
            f"p90={qs[1]:.3f} p95={qs[2]:.3f} p99={qs[3]:.3f}"
        )

    _print_percentiles(1)
    _print_percentiles(0)

    y_equal = y_different = t_equal = t_different = same_y_diff_t = hold_vs_label_exit = 0
    for idx_i, rec in compare_entries.items():
        ii = int(idx_i)
        if ii < 0 or ii >= len(label_y) or ii >= len(label_t_exit):
            continue
        ty = int(rec.get("y", -1))
        tt = int(rec.get("t_exit", -1))
        ly = int(label_y[ii])
        lt = int(label_t_exit[ii])
        if ty == ly:
            y_equal += 1
        else:
            y_different += 1
        if tt == lt:
            t_equal += 1
        else:
            t_different += 1
        if ty == ly and tt >= 0 and lt >= 0 and abs(tt - lt) > 1:
            same_y_diff_t += 1
        if tt == int(hold) and 0 <= lt < int(hold):
            hold_vs_label_exit += 1
    print(
        "[prefilter-tick-cache-diagnostics] "
        f"label_compare_entries={len(compare_entries)} label_y_equal={y_equal} "
        f"label_y_different={y_different} label_y_tick_missing={missing_expected} "
        f"label_t_exit_equal={t_equal} label_t_exit_different={t_different} "
        f"label_t_exit_tick_missing={missing_expected} same_y_t_exit_diff_gt1={same_y_diff_t} "
        f"tick_hold_label_exit_before_hold={hold_vs_label_exit}"
    )

def _write_control_none(control_file: Path | None) -> None:
    if control_file is None:
        return
    _atomic_write_text(control_file, "none\n")


def _read_control_command(control_file: Path | None) -> str:
    if control_file is None:
        return "none"
    try:
        txt = control_file.read_text(encoding="utf-8").strip().lower()
    except Exception:
        return "none"
    return txt or "none"


def _iter_combinations_with_new(idxs: list[int], r: int, new_start: int):
    if r < 1 or r > len(idxs):
        return
    old = [x for x in idxs if x < new_start]
    new = [x for x in idxs if x >= new_start]
    if not new:
        return
    for k_new in range(1, min(r, len(new)) + 1):
        k_old = r - k_new
        if k_old > len(old):
            continue
        for n_part in itertools.combinations(new, k_new):
            if k_old == 0:
                yield tuple(sorted(n_part))
            else:
                for o_part in itertools.combinations(old, k_old):
                    yield tuple(sorted(o_part + n_part))


def _iter_parent_extensions(parent: dict, idxs: list[int], max_path_conds: int):
    cset = set(int(x) for x in parent.get("_combo", ()))
    remaining = int(max_path_conds) - len(cset)
    if remaining <= 0:
        return
    add_candidates = [x for x in idxs if x not in cset]
    for add_k in range(1, remaining + 1):
        for adds in itertools.combinations(add_candidates, add_k):
            combo = tuple(sorted(cset | set(adds)))
            if 2 <= len(combo) <= int(max_path_conds):
                yield combo, parent


def _iter_all_parent_extensions(seeds: list[dict], idxs: list[int], max_path_conds: int):
    for p in seeds:
        yield from _iter_parent_extensions(p, idxs, max_path_conds)



def _is_atr_candidate_col(col: str) -> bool:
    c = str(col).lower()
    if c.startswith("delta_"):
        parts = c.split("_", 2)
        c = parts[2] if len(parts) == 3 and parts[1].isdigit() else c[6:]
    return bool(c.startswith("atr") or "_atr" in c)


def _canonicalize_candidate_value(col: str, op: str, value: float) -> float:
    try:
        v = float(value)
    except Exception:
        return float("nan")
    if not np.isfinite(v):
        return v
    col_s = str(col).lower()
    if str(op) == "==":
        if abs(v - round(v)) <= 1e-9:
            return float(int(round(v)))
        return float(round(v, 6))
    if col_s.startswith("dist_") or col_s.startswith("delta_"):
        return float(round(v, 6))
    return float(round(v, 6))


def _stable_candidate_key(col: str, op: str, value: float) -> str:
    v = _canonicalize_candidate_value(col, op, value)
    if np.isfinite(v) and abs(v - round(v)) <= 1e-9:
        v_str = str(int(round(v)))
    else:
        v_str = f"{v:.6f}" if np.isfinite(v) else "nan"
    return f"{str(col)}|{str(op)}|{v_str}"

def _file_sig(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        st = path.stat()
        return f"{path.resolve()}|{int(st.st_mtime_ns)}|{int(st.st_size)}"
    except Exception:
        return str(path)


def _ctx_sig(obj: dict) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True).encode("utf-8")).hexdigest()


def _load_stage_csv_if_match(
    path: Path | None,
    expected_stage: str,
    expected_ctx_sig: str,
    expected_schema_version: str,
    *,
    rebuild_legacy: bool,
) -> pd.DataFrame | None:
    if path is None or (not path.exists()):
        return None
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"{expected_stage} CSV is empty and was not modified: {path}. Use a different output path.")
    if "__stage" not in df.columns:
        raise ValueError(
            f"Cannot identify {expected_stage} CSV stage at {path}: missing __stage. "
            "The existing file was not modified."
        )
    stage_vals = {str(x) for x in df["__stage"].dropna().unique().tolist()}
    if stage_vals != {expected_stage}:
        raise ValueError(
            f"Invalid {expected_stage} CSV stage at {path}: found={sorted(stage_vals)} expected={expected_stage}. "
            "The existing file was not modified."
        )
    if "__schema_version" not in df.columns:
        if rebuild_legacy:
            legacy_coarse_cols = {
                "candidate_key",
                "stable_candidate_key",
                "col",
                "op",
                "value",
                "coarse_single_pos_hits",
                "coarse_single_neg_hits",
                "coarse_single_mask_count",
                "coarse_single_ratio",
                "coarse_single_mask_keep_ratio",
                "coarse_single_ratio_change",
                "coarse_lift",
                "family",
                "binary",
                "kept_after_family_topn",
                "__ctx_sig",
            }
            refined_specific_cols = {"candidate_key_refined", "tick_metric_status", *REQUIRED_TICK_METRIC_COLUMNS}
            missing_legacy_cols = sorted(legacy_coarse_cols.difference(df.columns))
            found_refined_cols = sorted(refined_specific_cols.intersection(df.columns))
            if missing_legacy_cols or found_refined_cols:
                raise ValueError(
                    f"Cannot identify legacy coarse CSV at {path}: missing legacy columns={missing_legacy_cols} "
                    f"refined-specific columns={found_refined_cols}. "
                    "The existing file was not modified."
                )
            print(f"[prefilter-resume] legacy {expected_stage} CSV schema detected; rebuilding from source data without reusing rows: {path}")
            return None
        raise ValueError(
            f"Legacy {expected_stage} CSV schema at {path}; the existing file was not modified. "
            "Use a different output path for the current refined schema."
        )
    schema_vals = {str(x) for x in df["__schema_version"].dropna().unique().tolist()}
    if schema_vals != {expected_schema_version}:
        raise ValueError(
            f"Incompatible {expected_stage} CSV schema at {path}: found={sorted(schema_vals)} "
            f"expected={expected_schema_version}. The existing file was not modified; use a different output path."
        )
    if "__ctx_sig" not in df.columns:
        raise ValueError(
            f"Invalid current-schema {expected_stage} CSV at {path}: missing __ctx_sig. "
            "The existing file was not modified."
        )
    required_cols = {
        "stable_candidate_key",
        "col",
        "op",
        "value",
        "family",
        "coarse_single_pos_hits",
        "coarse_single_neg_hits",
        "coarse_single_mask_count",
        "coarse_single_ratio",
        "coarse_single_mask_keep_ratio",
        "coarse_single_ratio_change",
        "coarse_lift",
        "binary",
    }
    if expected_stage == "refined":
        required_cols.update({"candidate_key_refined", "tick_metric_status", *REQUIRED_TICK_METRIC_COLUMNS})
    else:
        required_cols.add("candidate_key")
    missing_cols = sorted(required_cols.difference(df.columns))
    if missing_cols:
        raise ValueError(
            f"Invalid current-schema {expected_stage} CSV at {path}: missing columns={missing_cols}. "
            "The existing file was not modified."
        )
    sig_vals = {str(x) for x in df["__ctx_sig"].dropna().unique().tolist()}
    if sig_vals != {expected_ctx_sig}:
        raise ValueError(
            f"{expected_stage} CSV context mismatch at {path}: csv_ctx_sigs={sorted(sig_vals)} "
            f"expected_ctx_sig={expected_ctx_sig}. The existing file was not modified because it belongs "
            "to a different semantic context; use a different output path."
        )
    duplicate_stable_keys = df["stable_candidate_key"].astype(str).str.strip().duplicated(keep=False)
    empty_stable_keys = df["stable_candidate_key"].isna() | (df["stable_candidate_key"].astype(str).str.strip() == "")
    if bool(duplicate_stable_keys.any()) or bool(empty_stable_keys.any()):
        duplicate_values = sorted(set(df.loc[duplicate_stable_keys, "stable_candidate_key"].astype(str).tolist()))
        raise ValueError(
            f"Invalid {expected_stage} CSV at {path}: duplicate_stable_candidate_keys={duplicate_values} "
            f"empty_stable_candidate_keys={int(empty_stable_keys.sum())}. The existing file was not modified."
        )
    return df


def _has_full_tick_metrics(obj: dict) -> bool:
    try:
        return all(np.isfinite(float(obj.get(k, np.nan))) for k in REQUIRED_TICK_METRIC_COLUMNS)
    except Exception:
        return False


def _filter_candidate_inventory_rows(
    rows: list[dict],
    min_pos: int,
    min_lift: float,
    max_mask: int,
    allow_keys: set[str] | None,
) -> list[dict]:
    out = list(rows)
    if int(min_pos) > 0:
        out = [r for r in out if int(r.get("coarse_single_pos_hits", r.get("_single_pos_hits", 0))) >= int(min_pos)]
    if float(min_lift) > 0:
        out = [r for r in out if float(r.get("coarse_lift", r.get("lift", 0.0))) >= float(min_lift)]
    if int(max_mask) > 0:
        out = [r for r in out if int(r.get("coarse_single_mask_count", r.get("_single_mask_count", 0))) <= int(max_mask)]
    if allow_keys is not None:
        out = [r for r in out if str(r.get("candidate_key", "")) in allow_keys]
    return out


def _tick_scope_keys(scope: str, inventory: list[dict], filtered: list[dict], fam_top: list[dict]) -> set[str]:
    selected = fam_top if scope == "fam_top" else filtered if scope == "filtered" else inventory
    return {str(r.get("stable_candidate_key", "")) for r in selected}


def _family_top_rows(rows: list[dict], family_top_n: int) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(str(row["_family"]), []).append(row)
    out: list[dict] = []
    for group in groups.values():
        ranked = sorted(group, key=lambda z: (-float(z["lift"]), -int(z["_single_pos_hits"]), int(z["_single_mask_count"])))
        out.extend(ranked[: int(family_top_n)])
    return out


def _mask_required_keys(tick_scope_keys: set[str], replay_enabled: bool) -> set[str]:
    return set(tick_scope_keys) if replay_enabled else set()


def _partition_tick_metric_rows(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    full = [row for row in rows if _has_full_tick_metrics(row)]
    missing = [row for row in rows if not _has_full_tick_metrics(row)]
    return full, missing


def _prefer_refined_row(old: dict | None, new: dict, ctx_sig: str, schema_version: str) -> dict:
    if old is None:
        return new
    status_rank = {"full": 3, "missing": 2, "out_of_scope": 1}
    old_status = "full" if _has_full_tick_metrics(old) else str(old.get("tick_metric_status", "out_of_scope"))
    new_status = "full" if _has_full_tick_metrics(new) else str(new.get("tick_metric_status", "out_of_scope"))
    if int(status_rank.get(new_status, 0)) >= int(status_rank.get(old_status, 0)):
        return new
    kept = dict(old)
    kept["__ctx_sig"] = ctx_sig
    kept["__stage"] = "refined"
    kept["__schema_version"] = schema_version
    kept["tick_metric_status"] = "full" if _has_full_tick_metrics(kept) else old_status
    return kept


def _refined_rows_for_inventory(inventory: list[dict], rows_by_key: dict[str, dict]) -> list[dict]:
    return [rows_by_key[str(row["stable_candidate_key"])] for row in inventory]


def _refinement_state(scope_rows: int, replay_enabled: bool, missing_after: int, replay_entries: int) -> str:
    if int(scope_rows) == 0:
        return "empty_scope"
    if not replay_enabled:
        return "no_replay_configured"
    if int(missing_after) > 0 and int(replay_entries) == 0:
        return "null_critical_entries_existing_semantics"
    if int(missing_after) > 0:
        return "incomplete"
    return "complete"


def _load_allowlist(path: Path | None) -> tuple[set[str] | None, bool]:
    if path is None:
        return None, False
    txt = path.read_text(encoding="utf-8").strip()
    if not txt:
        return set(), False
    if path.suffix.lower() != ".txt":
        raise ValueError(f"{path}: include-candidates-file must be a .txt file")
    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    if not lines:
        return set(), False
    header = lines[0]
    if header not in {"candidate_key", "candidate_key_refined"}:
        raise ValueError(f"{path}: first line must be exactly candidate_key or candidate_key_refined")
    out = []
    for ln in lines[1:]:
        s = ln.strip()
        if not s:
            continue
        out.append(s)
    return set(out), (header == "candidate_key_refined")


def _candidate_family(col: str, split_delta_window: bool) -> str:
    c = str(col)
    if c.startswith("delta_"):
        parts = c.split("_")
        if split_delta_window and len(parts) > 2 and parts[1].isdigit():
            return f"delta_{parts[1]}_{parts[2]}"
        if "rsi" in c:
            return "delta_rsi"
        if "macd" in c:
            return "delta_macd"
        if "adx" in c:
            return "delta_adx"
        if "plus_di" in c:
            return "delta_plus_di"
        if "minus_di" in c:
            return "delta_minus_di"
        if "dx" in c:
            return "delta_dx"
        if "mfi" in c:
            return "delta_mfi"
        if "kdj" in c:
            return "delta_kdj"
        if "vol_z" in c:
            return "delta_vol_z"
        if "atr" in c:
            return "delta_atr"
    if "dist_ema" in c:
        return "dist_ema"
    if c.startswith("ema"):
        return "ema"
    if c.startswith("rsi"):
        return "rsi"
    if c.startswith("macd"):
        return "macd"
    if c.startswith("adx"):
        return "adx"
    if c.startswith("plus_di"):
        return "plus_di"
    if c.startswith("minus_di"):
        return "minus_di"
    if c.startswith("dx"):
        return "dx"
    if c.startswith("mfi"):
        return "mfi"
    if c.startswith("kdj"):
        return "kdj"
    if "orderblock" in c:
        return "orderblock"
    if "support" in c:
        return "support"
    if "resist" in c:
        return "resistance"
    if c.startswith("break_up"):
        return "break_up"
    if c.startswith("break_dn"):
        return "break_dn"
    if c.startswith("candle_"):
        return "candle"
    if "vol_z" in c:
        return "vol_z"
    if c.startswith("fvg_"):
        return "fvg"
    if c.startswith("liq_sweep"):
        return "liq_sweep"
    if c.startswith("bos_") or c.startswith("choch_") or c.startswith("ms_"):
        return "market_structure"
    if c.startswith("atr"):
        return "atr"
    return c.split("_")[0]


def _datetime_index_to_minute_ns(index_like) -> np.ndarray:
    idx = pd.DatetimeIndex(index_like)
    idx_ns = idx.astype("datetime64[ns]")
    return np.asarray(idx_ns.floor("min").view("int64"), dtype=np.int64)


def _ns_to_utc_iso(ns: int) -> str:
    return pd.Timestamp(int(ns), unit="ns", tz="UTC").isoformat()


def _feature_grid_sig(feature_minute_ns: np.ndarray) -> str:
    arr = np.asarray(feature_minute_ns, dtype=np.int64)
    return hashlib.sha1(arr.tobytes()).hexdigest()


def _parse_datetime_series_robust_local(raw) -> pd.Series:
    direct = pd.to_datetime(raw, errors="coerce", utc=True)
    base = pd.Series(raw)
    as_num = pd.to_numeric(base, errors="coerce")
    if not bool(as_num.notna().any()):
        return direct
    candidates = [direct]
    for unit in ("ns", "us", "ms", "s"):
        candidates.append(pd.to_datetime(as_num, errors="coerce", unit=unit, utc=True))

    def _score(dt: pd.Series) -> tuple[int, int]:
        valid = dt.notna()
        valid_count = int(valid.sum())
        if valid_count <= 0:
            return (0, 0)
        years = dt.loc[valid].dt.year
        return (int(((years >= 2000) & (years <= 2100)).sum()), valid_count)

    return max(candidates, key=_score)


def _ensure_plausible_minute_grid(feature_minute_ns: np.ndarray, *, context: str) -> np.ndarray:
    arr = np.asarray(feature_minute_ns, dtype=np.int64)
    if arr.size <= 0:
        raise ValueError(f"{context}: feature datetime grid is empty")
    if np.any(np.diff(arr) <= 0):
        raise ValueError(f"{context}: feature datetime grid must be strictly increasing and unique")
    idx = pd.to_datetime(arr, errors="coerce", utc=True)
    if bool(pd.isna(idx).any()):
        raise ValueError(f"{context}: feature datetime grid contains invalid datetimes")
    min_year = int(idx.min().year)
    max_year = int(idx.max().year)
    if min_year < 2000 or max_year > 2100:
        raise ValueError(
            f"{context}: implausible datetime range min={idx.min().isoformat()} max={idx.max().isoformat()} "
            "(expected roughly years 2000..2100); check timestamp unit (ns/us/ms/s)."
        )
    return arr


def _load_feature_minute_grid_minimal(path: Path) -> np.ndarray:
    """Load only the feature datetime/index grid without materializing feature columns."""
    path = Path(path)
    if not path.exists():
        raise ValueError(f"features file does not exist: {path}")
    context = "load_feature_minute_grid_minimal"
    if path.suffix.lower() in {".parquet", ".pq"}:
        try:
            import pyarrow.parquet as pq  # type: ignore
        except Exception as exc:
            raise RuntimeError("pyarrow is required for minimal feature datetime grid loading from parquet") from exc
        pf = pq.ParquetFile(path)
        schema_cols = [str(c) for c in pf.schema_arrow.names]
        lower_cols = {c.strip().lower(): c for c in schema_cols}
        dt_col = lower_cols.get("datetime") or lower_cols.get("date_time") or lower_cols.get("timestamp")
        if dt_col is None:
            raw_meta = pf.metadata.metadata or {}
            pandas_meta_raw = raw_meta.get(b"pandas")
            if pandas_meta_raw:
                try:
                    pandas_meta = json.loads(pandas_meta_raw.decode("utf-8"))
                    for idx_col in pandas_meta.get("index_columns", []):
                        if isinstance(idx_col, str) and idx_col in schema_cols:
                            dt_col = idx_col
                            break
                        if isinstance(idx_col, dict):
                            name = idx_col.get("name") or idx_col.get("field_name")
                            if isinstance(name, str) and name in schema_cols:
                                dt_col = name
                                break
                except Exception:
                    dt_col = None
        if dt_col is None and "__index_level_0__" in schema_cols:
            dt_col = "__index_level_0__"
        if dt_col is None:
            raise ValueError(
                f"{context}: could not determine feature datetime/index column minimally from {path}; "
                "refusing to load the full feature file. Ensure the feature parquet stores a datetime column/index."
            )
        parts: list[np.ndarray] = []
        for batch in pf.iter_batches(columns=[dt_col], batch_size=250_000):
            # Arrow may restore a stored pandas index as the DataFrame index when using
            # to_pandas(), so read the selected physical column directly.
            raw_dt = batch.column(0).to_pandas()
            dt = _parse_datetime_series_robust_local(raw_dt)
            dt_ns = (
                dt.dt.tz_convert("UTC")
                .dt.tz_localize(None)
                .to_numpy(dtype="datetime64[ns]")
                .astype("int64")
            )
            parts.append(((dt_ns // NS_PER_MINUTE) * NS_PER_MINUTE).astype(np.int64, copy=False))
        if not parts:
            raise ValueError(f"{context}: no datetime rows found in {path}")
        arr = np.sort(np.concatenate(parts).astype(np.int64, copy=False))
        return _ensure_plausible_minute_grid(arr, context=context)

    header = pd.read_csv(path, compression="infer", nrows=0)
    cols = [str(c) for c in header.columns]
    lower_cols = {c.strip().lower(): c for c in cols}
    dt_col = lower_cols.get("datetime") or lower_cols.get("date_time") or lower_cols.get("timestamp")
    if dt_col is None and cols and cols[0].lower().startswith("unnamed"):
        dt_col = cols[0]
    if dt_col is None:
        raise ValueError(
            f"{context}: could not determine feature datetime/index column minimally from {path}; "
            "refusing to load the full feature file. Ensure the feature CSV has a datetime column or unnamed datetime index."
        )
    raw = pd.read_csv(path, compression="infer", usecols=[dt_col])[dt_col]
    dt = _parse_datetime_series_robust_local(raw)
    dt_ns = (
        dt.dt.tz_convert("UTC")
        .dt.tz_localize(None)
        .to_numpy(dtype="datetime64[ns]")
        .astype("int64")
    )
    arr = np.sort(((dt_ns // NS_PER_MINUTE) * NS_PER_MINUTE).astype(np.int64, copy=False))
    return _ensure_plausible_minute_grid(arr, context=context)


def _askbid_m1_arrow_schema():
    try:
        import pyarrow as pa  # type: ignore
    except Exception as exc:
        raise RuntimeError("pyarrow is required for AskBid-M1 Parquet streaming") from exc
    fields = []
    for col in ASKBID_M1_COLUMNS:
        if col == "datetime":
            fields.append(pa.field(col, pa.timestamp("ns")))
        elif col == "ticks_count":
            fields.append(pa.field(col, pa.int64()))
        else:
            fields.append(pa.field(col, pa.float64()))
    return pa.schema(fields)


def _datetime_series_to_ns(dt_full: pd.Series) -> np.ndarray:
    return (
        dt_full.dt.tz_convert("UTC")
        .dt.tz_localize(None)
        .to_numpy(dtype="datetime64[ns]")
        .astype("int64")
    )


def _tick_datetime_ns_from_chunk(ch: pd.DataFrame, dt_col: str, cols_lower: dict[str, str] | None = None) -> np.ndarray:
    if cols_lower is None:
        cols_lower = {str(c).strip().lower(): str(c) for c in ch.columns}
    dt_raw = ch[dt_col]
    date_src = None
    time_src = None
    date_candidates = ("date", "<date>", "<dtyyyymmdd>", "dtyyyymmdd")
    time_candidates = ("time", "<time>", "time_msc", "timestamp_time")
    for dname in date_candidates:
        if dname in cols_lower and cols_lower[dname] != dt_col:
            cand = ch[cols_lower[dname]].astype(str).str.strip()
            if bool((cand.str.fullmatch(r"\d{8}", na=False)).any()) or bool((cand.str.fullmatch(r"\d{4}-\d{2}-\d{2}", na=False)).any()):
                date_src = cand
                break
    for tname in time_candidates:
        if tname in cols_lower and cols_lower[tname] != dt_col:
            cand = ch[cols_lower[tname]].astype(str).str.strip()
            if bool((cand.str.fullmatch(r"\d{2}:\d{2}:\d{2}(?:\.\d{1,9})?", na=False)).any()):
                time_src = cand
                break
    time_s = dt_raw.astype(str).str.strip()
    looks_time_like = bool((time_s.str.fullmatch(r"\d{2}:\d{2}:\d{2}(?:\.\d{1,9})?", na=False)).any())
    looks_date_like = bool((time_s.str.fullmatch(r"\d{8}", na=False)).any()) or bool((time_s.str.fullmatch(r"\d{4}-\d{2}-\d{2}", na=False)).any())
    if date_src is not None and looks_time_like:
        dt_join = date_src.astype(str).str.strip() + " " + time_s
        dt_full = pd.to_datetime(dt_join, format="%Y%m%d %H:%M:%S.%f", errors="coerce", utc=True)
        if not bool(dt_full.notna().any()):
            dt_full = pd.to_datetime(dt_join, format="%Y%m%d %H:%M:%S", errors="coerce", utc=True)
        if not bool(dt_full.notna().any()):
            dt_full = pd.to_datetime(dt_join, errors="coerce", utc=True)
    elif looks_date_like and time_src is not None:
        dt_join = time_s + " " + time_src.astype(str).str.strip()
        dt_full = pd.to_datetime(dt_join, format="%Y%m%d %H:%M:%S.%f", errors="coerce", utc=True)
        if not bool(dt_full.notna().any()):
            dt_full = pd.to_datetime(dt_join, format="%Y%m%d %H:%M:%S", errors="coerce", utc=True)
        if not bool(dt_full.notna().any()):
            dt_full = pd.to_datetime(dt_join, errors="coerce", utc=True)
    else:
        dt_full = _parse_datetime_series_robust_local(dt_raw)
    if bool(dt_full.notna().any()):
        years = dt_full.dropna().dt.year
        if int(years.max()) <= 1971:
            raise ValueError("AskBid-M1 streaming build parsed implausible tick datetimes; check DateTime/Date/Time columns")
    return _datetime_series_to_ns(dt_full)


def _iter_askbid_tick_chunks(
    path: Path,
    *,
    datetime_col: str,
    sep: str,
    tick_chunk_size: int | str = "auto",
):
    chunk_size_eff = 150_000 if str(tick_chunk_size).lower() == "auto" else max(10_000, int(tick_chunk_size))
    nat_ns = np.iinfo(np.int64).min
    prev_ns: int | None = None

    def _prepare_chunk(ch: pd.DataFrame, source: str):
        nonlocal prev_ns
        cols_lower = {str(c).strip().lower(): str(c) for c in ch.columns}
        dt_col = cols_lower.get(str(datetime_col).strip().lower()) if datetime_col else None
        if dt_col is None:
            for cand in ("datetime", "date_time", "timestamp", "time", "date"):
                if cand in cols_lower:
                    dt_col = cols_lower[cand]
                    break
        bid_col = cols_lower.get("bid")
        ask_col = cols_lower.get("ask")
        vol_col = cols_lower.get("volume") or cols_lower.get("vol")
        if dt_col is None or bid_col is None or ask_col is None:
            raise ValueError(f"AskBid-M1 streaming build requires DateTime, Bid and Ask columns in {source}")
        dt_ns = _tick_datetime_ns_from_chunk(ch, dt_col, cols_lower)
        bid = pd.to_numeric(ch[bid_col], errors="coerce").to_numpy(dtype=np.float64)
        ask = pd.to_numeric(ch[ask_col], errors="coerce").to_numpy(dtype=np.float64)
        if vol_col is not None:
            volume = pd.to_numeric(ch[vol_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        else:
            volume = np.zeros(len(ch), dtype=np.float64)
        valid = (dt_ns != nat_ns) & np.isfinite(bid) & np.isfinite(ask) & (bid > 0.0) & (ask > 0.0)
        if not np.any(valid):
            return None
        dt_ns = dt_ns[valid].astype(np.int64, copy=False)
        bid = bid[valid].astype(np.float64, copy=False)
        ask = ask[valid].astype(np.float64, copy=False)
        volume = volume[valid].astype(np.float64, copy=False)
        if dt_ns.size > 1 and np.any(np.diff(dt_ns) < 0):
            raise ValueError(
                f"AskBid-M1 streaming build requires a chronologically sorted tick source; "
                f"datetime goes backwards within {source}. Pre-sort/build the tick parquet first."
            )
        if prev_ns is not None and dt_ns.size and int(dt_ns[0]) < int(prev_ns):
            raise ValueError(
                "AskBid-M1 streaming build requires a chronologically sorted tick source; "
                "datetime goes backwards across batch/row-group boundaries. Pre-sort/build the tick parquet first."
            )
        if dt_ns.size:
            prev_ns = int(dt_ns[-1])
        return dt_ns, bid, ask, volume

    if path.suffix.lower() in {".parquet", ".pq"}:
        try:
            import pyarrow.parquet as pq  # type: ignore
        except Exception as exc:
            raise RuntimeError("pyarrow is required for AskBid-M1 streaming build from tick parquet") from exc
        pf = pq.ParquetFile(path)
        schema_cols = set(str(c) for c in pf.schema_arrow.names)
        missing = {"DateTime", "Bid", "Ask"} - schema_cols
        if missing:
            raise ValueError(f"tick parquet missing columns for AskBid-M1 streaming build: {sorted(missing)}")
        cols = ["DateTime", "Bid", "Ask"] + (["Volume"] if "Volume" in schema_cols else [])
        for batch in pf.iter_batches(columns=cols, batch_size=chunk_size_eff):
            prepared = _prepare_chunk(batch.to_pandas(), "tick parquet")
            if prepared is not None:
                yield prepared
    else:
        for ch in pd.read_csv(path, sep=sep, chunksize=chunk_size_eff):
            prepared = _prepare_chunk(ch, "tick CSV")
            if prepared is not None:
                yield prepared


def _build_askbid_m1_parquet_streaming(
    *,
    output_path: Path,
    feature_minute_ns: np.ndarray,
    tick_source_path: Path,
    tick_datetime_column: str,
    tick_sep: str,
    entry_latency_ms: int,
    metadata: dict[str, str],
    tick_chunk_size: int | str = "auto",
) -> dict[str, int]:
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
    except Exception as exc:
        raise RuntimeError("pyarrow is required for AskBid-M1 streaming parquet build") from exc
    if tick_source_path is None:
        raise ValueError("--tick-data is required to build missing --askbid-m1-parquet")
    if not tick_source_path.exists():
        raise ValueError(f"tick source does not exist: {tick_source_path}")
    feature_minute_ns = _ensure_plausible_minute_grid(feature_minute_ns, context="build_askbid_m1_streaming")
    entry_latency_ns = int(entry_latency_ms) * 1_000_000
    max_carry_ns = int(ASKBID_M1_MAX_CARRY_GAP_MINUTES) * NS_PER_MINUTE
    schema = _askbid_m1_arrow_schema()
    encoded_meta = {str(k).encode("utf-8"): str(v).encode("utf-8") for k, v in metadata.items()}
    schema = schema.with_metadata({**(schema.metadata or {}), **encoded_meta})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    chunk_iter = iter(_iter_askbid_tick_chunks(
        tick_source_path,
        datetime_col=tick_datetime_column,
        sep=tick_sep,
        tick_chunk_size=tick_chunk_size,
    ))
    cur_dt = cur_bid = cur_ask = cur_vol = None
    cur_pos = 0
    no_more_ticks = False

    def _load_next_chunk() -> bool:
        nonlocal cur_dt, cur_bid, cur_ask, cur_vol, cur_pos, no_more_ticks
        try:
            cur_dt, cur_bid, cur_ask, cur_vol = next(chunk_iter)
            cur_pos = 0
            return True
        except StopIteration:
            cur_dt = cur_bid = cur_ask = cur_vol = None
            cur_pos = 0
            no_more_ticks = True
            return False

    def _has_tick() -> bool:
        nonlocal cur_dt, cur_pos
        while not no_more_ticks and (cur_dt is None or cur_pos >= len(cur_dt)):
            if not _load_next_chunk():
                break
        return cur_dt is not None and cur_pos < len(cur_dt)

    last_bid = float("nan")
    last_ask = float("nan")
    last_tick_ns: int | None = None
    allowed_feature_gaps = 0
    session_open_from_first_tick_count = 0
    rows_written = 0
    missing_price_minutes: list[int] = []
    batch_cols: dict[str, list] = {c: [] for c in ASKBID_M1_COLUMNS}
    writer = None

    def _append_row(row: dict) -> None:
        for col in ASKBID_M1_COLUMNS:
            batch_cols[col].append(row[col])

    def _flush() -> None:
        nonlocal writer, batch_cols, rows_written
        if not batch_cols["datetime"]:
            return
        arrays = []
        for col in ASKBID_M1_COLUMNS:
            if col == "datetime":
                arrays.append(pa.array(batch_cols[col], type=pa.timestamp("ns")))
            elif col == "ticks_count":
                arrays.append(pa.array(batch_cols[col], type=pa.int64()))
            else:
                arrays.append(pa.array(batch_cols[col], type=pa.float64()))
        table = pa.Table.from_arrays(arrays, schema=schema)
        if writer is None:
            writer = pq.ParquetWriter(output_path, schema)
        writer.write_table(table)
        rows_written += len(batch_cols["datetime"])
        batch_cols = {c: [] for c in ASKBID_M1_COLUMNS}

    try:
        for row_pos, minute_ns_raw in enumerate(feature_minute_ns.tolist()):
            minute_ns = int(minute_ns_raw)
            next_minute_ns = minute_ns + NS_PER_MINUTE
            if row_pos > 0 and minute_ns - int(feature_minute_ns[row_pos - 1]) > NS_PER_MINUTE:
                allowed_feature_gaps += 1
            while _has_tick() and int(cur_dt[cur_pos]) < minute_ns:
                last_bid = float(cur_bid[cur_pos])
                last_ask = float(cur_ask[cur_pos])
                last_tick_ns = int(cur_dt[cur_pos])
                cur_pos += 1

            minute_dt_parts: list[np.ndarray] = []
            minute_bid_parts: list[np.ndarray] = []
            minute_ask_parts: list[np.ndarray] = []
            minute_vol_parts: list[np.ndarray] = []
            while _has_tick() and int(cur_dt[cur_pos]) < next_minute_ns:
                end = int(np.searchsorted(cur_dt, next_minute_ns, side="left", sorter=None))
                if end <= cur_pos:
                    end = cur_pos + 1
                minute_dt_parts.append(cur_dt[cur_pos:end])
                minute_bid_parts.append(cur_bid[cur_pos:end])
                minute_ask_parts.append(cur_ask[cur_pos:end])
                minute_vol_parts.append(cur_vol[cur_pos:end])
                cur_pos = end
            if minute_dt_parts:
                dt_slice = np.concatenate(minute_dt_parts).astype(np.int64, copy=False)
                bid_slice = np.concatenate(minute_bid_parts).astype(np.float64, copy=False)
                ask_slice = np.concatenate(minute_ask_parts).astype(np.float64, copy=False)
                vol_slice = np.concatenate(minute_vol_parts).astype(np.float64, copy=False)
            else:
                dt_slice = np.asarray([], dtype=np.int64)
                bid_slice = np.asarray([], dtype=np.float64)
                ask_slice = np.asarray([], dtype=np.float64)
                vol_slice = np.asarray([], dtype=np.float64)

            exact_start = len(dt_slice) > 0 and int(dt_slice[0]) == minute_ns
            carry_valid = last_tick_ns is not None and (minute_ns - int(last_tick_ns)) <= max_carry_ns
            if exact_start:
                ask_open = float(ask_slice[0]); bid_open = float(bid_slice[0])
            elif carry_valid:
                ask_open = float(last_ask); bid_open = float(last_bid)
            elif len(dt_slice) > 0:
                ask_open = float(ask_slice[0]); bid_open = float(bid_slice[0])
                session_open_from_first_tick_count += 1
            else:
                missing_price_minutes.append(minute_ns)
                if len(missing_price_minutes) >= 10:
                    break
                continue

            if len(dt_slice) > 0:
                ask_high = float(max(float(np.max(ask_slice)), ask_open))
                ask_low = float(min(float(np.min(ask_slice)), ask_open))
                bid_high = float(max(float(np.max(bid_slice)), bid_open))
                bid_low = float(min(float(np.min(bid_slice)), bid_open))
                ask_close = float(ask_slice[-1])
                bid_close = float(bid_slice[-1])
            else:
                ask_high = ask_low = ask_close = ask_open
                bid_high = bid_low = bid_close = bid_open

            latency_ns = minute_ns + entry_latency_ns
            le_idx = int(np.searchsorted(dt_slice, latency_ns, side="right")) if len(dt_slice) else 0
            if le_idx > 0:
                ask_entry_latency = float(ask_slice[le_idx - 1])
                bid_at_latency = float(bid_slice[le_idx - 1])
            else:
                ask_entry_latency = ask_open
                bid_at_latency = bid_open
            after_idx = int(np.searchsorted(dt_slice, latency_ns, side="left")) if len(dt_slice) else 0
            after_bid = bid_slice[after_idx:] if len(dt_slice) else np.asarray([], dtype=np.float64)
            if len(after_bid) > 0:
                bid_high_after = float(max(float(np.max(after_bid)), bid_at_latency))
                bid_low_after = float(min(float(np.min(after_bid)), bid_at_latency))
            else:
                bid_high_after = bid_low_after = bid_at_latency

            _append_row({
                "datetime": np.datetime64(minute_ns, "ns"),
                "ask_open": ask_open,
                "ask_entry_latency": ask_entry_latency,
                "ask_high": ask_high,
                "ask_low": ask_low,
                "ask_close": ask_close,
                "bid_open": bid_open,
                "bid_high": bid_high,
                "bid_low": bid_low,
                "bid_close": bid_close,
                "bid_high_after_entry_latency": bid_high_after,
                "bid_low_after_entry_latency": bid_low_after,
                "volume_check": float(np.nansum(vol_slice)) if len(vol_slice) else 0.0,
                "ticks_count": int(len(dt_slice)),
            })
            if len(dt_slice) > 0:
                last_bid = float(bid_slice[-1])
                last_ask = float(ask_slice[-1])
                last_tick_ns = int(dt_slice[-1])
            if len(batch_cols["datetime"]) >= 50_000:
                _flush()
        if missing_price_minutes:
            sample = ",".join(_ns_to_utc_iso(x) for x in missing_price_minutes[:5])
            raise ValueError(
                "AskBid-M1 streaming build could not derive quote-continuous prices for "
                f"{len(missing_price_minutes)} feature minutes; sample={sample}. "
                "This usually indicates Feature-DF minutes inside an offmarket/data gap or missing initial quote."
            )
        _flush()
        if writer is not None:
            writer.close(); writer = None
    except Exception:
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
        if output_path.exists():
            try:
                output_path.unlink()
            except Exception:
                pass
        raise
    return {
        "allowed_feature_gaps": int(allowed_feature_gaps),
        "session_open_from_first_tick_count": int(session_open_from_first_tick_count),
        "rows_written": int(rows_written),
    }


def _validate_askbid_m1_parquet_against_grid(
    path: Path,
    *,
    feature_minute_ns: np.ndarray,
    entry_latency_ms: int,
    expected_source_tick_path: Path | None,
) -> None:
    try:
        import pyarrow.parquet as pq  # type: ignore
    except Exception as exc:
        raise RuntimeError("pyarrow is required for streaming AskBid-M1 Parquet validation") from exc
    metadata = _parquet_metadata(path)
    expected_meta = _askbid_m1_metadata(
        entry_latency_ms=int(entry_latency_ms),
        feature_minute_ns=np.asarray(feature_minute_ns, dtype=np.int64),
        source_tick_path=expected_source_tick_path,
    )
    meta_errors = _askbid_m1_required_metadata_errors(metadata, expected_meta)
    if meta_errors:
        raise ValueError("AskBid-M1 metadata mismatch: " + "; ".join(meta_errors[:5]))
    pf = pq.ParquetFile(path)
    schema_cols = [str(c) for c in pf.schema_arrow.names]
    if schema_cols != ASKBID_M1_COLUMNS:
        raise ValueError(f"AskBid-M1 schema mismatch: expected={ASKBID_M1_COLUMNS} actual={schema_cols}")
    if int(pf.metadata.num_rows) != int(len(feature_minute_ns)):
        raise ValueError(f"AskBid-M1 row count mismatch: got={int(pf.metadata.num_rows)} expected={int(len(feature_minute_ns))}")
    offset = 0
    expected_ns = np.asarray(feature_minute_ns, dtype=np.int64)
    for batch in pf.iter_batches(columns=ASKBID_M1_COLUMNS, batch_size=100_000):
        ch = batch.to_pandas()
        dt_ns = pd.to_datetime(ch["datetime"], errors="coerce", utc=True).astype("int64").to_numpy(dtype=np.int64)
        exp = expected_ns[offset:offset + len(dt_ns)]
        if len(dt_ns) != len(exp) or not np.array_equal(dt_ns, exp):
            bad = int(np.flatnonzero(dt_ns != exp)[0]) if len(dt_ns) == len(exp) and np.any(dt_ns != exp) else 0
            raise ValueError(
                "AskBid-M1 datetime alignment mismatch during streaming validation: "
                f"row={offset + bad} got={_ns_to_utc_iso(int(dt_ns[bad])) if len(dt_ns) else 'n/a'} "
                f"expected={_ns_to_utc_iso(int(exp[bad])) if len(exp) else 'n/a'}"
            )
        price_values = ch[ASKBID_M1_PRICE_COLUMNS].to_numpy(dtype=np.float64, copy=False)
        if not np.isfinite(price_values).all() or not (price_values > 0).all():
            bad = np.argwhere((~np.isfinite(price_values)) | ~(price_values > 0))[:5].tolist()
            raise ValueError(f"AskBid-M1 contains invalid price fields during streaming validation; sample_positions={bad}")
        offset += len(dt_ns)
    if offset != len(expected_ns):
        raise ValueError(f"AskBid-M1 validation row count mismatch after scan: got={offset} expected={len(expected_ns)}")


def _ensure_askbid_m1_parquet_ready(
    *,
    path: Path,
    feature_minute_ns: np.ndarray,
    entry_latency_ms: int,
    tick_source_path: Path | None,
    tick_datetime_column: str,
    tick_sep: str,
    tick_chunk_size: int | str = "auto",
) -> None:
    feature_minute_ns = _ensure_plausible_minute_grid(feature_minute_ns, context="ensure_askbid_m1_parquet_ready")
    if path.exists():
        print(f"[prefilter-askbid-m1] early_validate=existing path={path}")
        _validate_askbid_m1_parquet_against_grid(
            path,
            feature_minute_ns=feature_minute_ns,
            entry_latency_ms=int(entry_latency_ms),
            expected_source_tick_path=tick_source_path,
        )
        return
    if tick_source_path is None:
        raise ValueError("--tick-data is required to build missing --askbid-m1-parquet")
    metadata = _askbid_m1_metadata(
        entry_latency_ms=int(entry_latency_ms),
        feature_minute_ns=feature_minute_ns,
        source_tick_path=tick_source_path,
    )
    tmp_path = path.with_name(f"{path.stem}.tmp{path.suffix}")
    invalid_path = path.with_name(f"{path.stem}.invalid{path.suffix}")
    if tmp_path.exists():
        tmp_path.unlink()
    print(f"[prefilter-askbid-m1] streaming_build=start path={path} tmp_path={tmp_path} tick_source={tick_source_path}")
    build_stats = _build_askbid_m1_parquet_streaming(
        output_path=tmp_path,
        feature_minute_ns=feature_minute_ns,
        tick_source_path=tick_source_path,
        tick_datetime_column=tick_datetime_column,
        tick_sep=tick_sep,
        entry_latency_ms=int(entry_latency_ms),
        metadata=metadata,
        tick_chunk_size=tick_chunk_size,
    )
    gc.collect()
    try:
        _validate_askbid_m1_parquet_against_grid(
            tmp_path,
            feature_minute_ns=feature_minute_ns,
            entry_latency_ms=int(entry_latency_ms),
            expected_source_tick_path=tick_source_path,
        )
    except Exception:
        if tmp_path.exists():
            os.replace(tmp_path, invalid_path)
            print(f"[prefilter-askbid-m1] streaming_build_invalid_moved path={invalid_path}")
        raise
    os.replace(tmp_path, path)
    if invalid_path.exists():
        try:
            invalid_path.unlink()
        except Exception:
            pass
    print(
        "[prefilter-askbid-m1] "
        f"streaming_build=done path={path} rows={int(build_stats.get('rows_written', 0))} "
        f"feature_minutes={len(feature_minute_ns)} "
        f"allowed_offmarket_gaps={int(build_stats.get('allowed_feature_gaps', 0))} "
        f"session_open_from_first_tick_count={int(build_stats.get('session_open_from_first_tick_count', 0))} "
        f"entry_latency_ms={int(entry_latency_ms)}"
    )


def _askbid_m1_metadata(
    *,
    entry_latency_ms: int,
    feature_minute_ns: np.ndarray,
    source_tick_path: Path | None,
) -> dict[str, str]:
    feature_minute_ns = np.asarray(feature_minute_ns, dtype=np.int64)
    return {
        "price_basis": "askbid_m1",
        "askbid_m1_semantics": ASKBID_M1_SEMANTICS,
        "entry_latency_ms": str(int(entry_latency_ms)),
        "required_columns": json.dumps(ASKBID_M1_COLUMNS, separators=(",", ":")),
        "row_count": str(int(len(feature_minute_ns))),
        "first_datetime_ns": str(int(feature_minute_ns[0])) if len(feature_minute_ns) else "",
        "last_datetime_ns": str(int(feature_minute_ns[-1])) if len(feature_minute_ns) else "",
        "feature_grid_sig": _feature_grid_sig(feature_minute_ns),
        "source_tick_sig": _file_sig(source_tick_path),
        "timezone": "UTC",
        "quote_continuous_mode": "feature_grid_aligned_no_24x7_fill",
        "open_semantics": "last_quote_at_or_before_minute_start_same_session",
        "close_semantics": "last_quote_strictly_before_next_minute_start",
        "entry_latency_semantics": "last_ask_at_or_before_minute_start_plus_entry_latency_ms",
        "after_entry_bounds_semantics": "bid_bounds_from_entry_latency_inclusive_to_next_minute_start_exclusive",
        "max_carry_gap_minutes": str(int(ASKBID_M1_MAX_CARRY_GAP_MINUTES)),
    }


def _parquet_metadata(path: Path) -> dict[str, str]:
    try:
        import pyarrow.parquet as pq  # type: ignore
    except Exception as exc:
        raise RuntimeError("pyarrow is required for AskBid-M1 Parquet metadata validation") from exc
    raw = pq.ParquetFile(path).metadata.metadata or {}
    out: dict[str, str] = {}
    for k, v in raw.items():
        try:
            out[k.decode("utf-8")] = v.decode("utf-8")
        except Exception:
            continue
    return out


def _write_askbid_m1_parquet(path: Path, frame: pd.DataFrame, metadata: dict[str, str]) -> None:
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
    except Exception as exc:
        raise RuntimeError("pyarrow is required to write AskBid-M1 Parquet files") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(frame, preserve_index=False)
    encoded_meta = {str(k).encode("utf-8"): str(v).encode("utf-8") for k, v in metadata.items()}
    existing = table.schema.metadata or {}
    table = table.replace_schema_metadata({**existing, **encoded_meta})
    with tempfile.NamedTemporaryFile("wb", delete=False, dir=str(path.parent), suffix=".parquet") as tf:
        tmp = Path(tf.name)
    try:
        pq.write_table(table, tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except Exception:
                pass


def _askbid_m1_required_metadata_errors(meta: dict[str, str], expected_meta: dict[str, str]) -> list[str]:
    errors: list[str] = []
    for key in [
        "price_basis",
        "askbid_m1_semantics",
        "entry_latency_ms",
        "required_columns",
        "row_count",
        "first_datetime_ns",
        "last_datetime_ns",
        "feature_grid_sig",
        "timezone",
        "quote_continuous_mode",
    ]:
        if str(meta.get(key, "")) != str(expected_meta.get(key, "")):
            errors.append(f"{key} mismatch: got={meta.get(key, '')!r} expected={expected_meta.get(key, '')!r}")
    return errors


def _normalize_tick_datetime_series_for_askbid(raw) -> pd.Series:
    return pd.to_datetime(raw, errors="coerce", utc=True)


def _validate_askbid_m1_frame(
    frame: pd.DataFrame,
    *,
    feature_minute_ns: np.ndarray,
    entry_latency_ms: int,
    metadata: dict[str, str],
    expected_source_tick_path: Path | None,
) -> None:
    expected_meta = _askbid_m1_metadata(
        entry_latency_ms=int(entry_latency_ms),
        feature_minute_ns=np.asarray(feature_minute_ns, dtype=np.int64),
        source_tick_path=expected_source_tick_path,
    )
    if list(frame.columns) != ASKBID_M1_COLUMNS:
        raise ValueError(
            "AskBid-M1 schema mismatch: "
            f"expected={ASKBID_M1_COLUMNS} actual={list(frame.columns)}"
        )
    meta_errors = _askbid_m1_required_metadata_errors(metadata, expected_meta)
    if meta_errors:
        raise ValueError("AskBid-M1 metadata mismatch: " + "; ".join(meta_errors[:5]))
    if str(metadata.get("source_tick_sig", "")) != str(expected_meta.get("source_tick_sig", "")):
        print(
            "[prefilter-askbid-m1] warning=source_tick_sig_differs "
            f"file_source_sig={metadata.get('source_tick_sig', '')} "
            f"current_source_sig={expected_meta.get('source_tick_sig', '')}"
        )
    dt_ns = pd.to_datetime(frame["datetime"], errors="coerce", utc=True).astype("int64").to_numpy(dtype=np.int64)
    expected_ns = np.asarray(feature_minute_ns, dtype=np.int64)
    if len(dt_ns) != len(expected_ns):
        raise ValueError(f"AskBid-M1 row count mismatch: got={len(dt_ns)} expected={len(expected_ns)}")
    if np.any(np.diff(dt_ns) <= 0):
        raise ValueError("AskBid-M1 datetimes must be strictly increasing without duplicates")
    if not np.array_equal(dt_ns, expected_ns):
        mismatch = int(np.flatnonzero(dt_ns != expected_ns)[0]) if len(dt_ns) == len(expected_ns) and np.any(dt_ns != expected_ns) else -1
        got = _ns_to_utc_iso(int(dt_ns[mismatch])) if mismatch >= 0 else "n/a"
        exp = _ns_to_utc_iso(int(expected_ns[mismatch])) if mismatch >= 0 else "n/a"
        raise ValueError(f"AskBid-M1 datetime alignment mismatch at row={mismatch}: got={got} expected={exp}")
    price_values = frame[ASKBID_M1_PRICE_COLUMNS].to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(price_values).all():
        bad = np.argwhere(~np.isfinite(price_values))
        sample = bad[:5].tolist()
        raise ValueError(f"AskBid-M1 contains NaN/Inf in required price fields; sample_positions={sample}")


def _load_askbid_m1_arrays(
    path: Path,
    *,
    feature_minute_ns: np.ndarray,
    entry_latency_ms: int,
    source_tick_path: Path | None,
) -> dict[str, np.ndarray]:
    metadata = _parquet_metadata(path)
    frame = pd.read_parquet(path)
    _validate_askbid_m1_frame(
        frame,
        feature_minute_ns=feature_minute_ns,
        entry_latency_ms=entry_latency_ms,
        metadata=metadata,
        expected_source_tick_path=source_tick_path,
    )
    ticks_count = pd.to_numeric(frame["ticks_count"], errors="coerce").to_numpy(dtype=np.int64)
    print(
        "[prefilter-askbid-m1] "
        f"load=done path={path} rows={len(frame)} "
        f"first={_ns_to_utc_iso(int(feature_minute_ns[0]))} last={_ns_to_utc_iso(int(feature_minute_ns[-1]))} "
        f"entry_latency_ms={int(entry_latency_ms)} "
        f"ticks_count_min={int(np.min(ticks_count)) if len(ticks_count) else 0} "
        f"ticks_count_median={float(np.median(ticks_count)) if len(ticks_count) else 0.0:.3f} "
        f"ticks_count_max={int(np.max(ticks_count)) if len(ticks_count) else 0}"
    )
    out: dict[str, np.ndarray] = {}
    for col in ASKBID_M1_COLUMNS:
        if col == "datetime":
            out[col] = feature_minute_ns.copy()
        elif col == "ticks_count":
            out[col] = ticks_count
        else:
            out[col] = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=np.float64)
    return out


def _build_or_load_askbid_m1(
    *,
    path: Path,
    feature_index=None,
    feature_minute_ns: np.ndarray | None = None,
    entry_latency_ms: int,
    tick_source_path: Path | None,
    tick_datetime_column: str,
    tick_sep: str,
    tick_chunk_size: int | str = "auto",
) -> dict[str, np.ndarray]:
    if feature_minute_ns is None:
        if feature_index is None:
            raise ValueError("feature_index or feature_minute_ns is required for AskBid-M1 load/build")
        feature_minute_ns = _datetime_index_to_minute_ns(feature_index)
    feature_minute_ns = _ensure_plausible_minute_grid(feature_minute_ns, context="build_or_load_askbid_m1")
    _ensure_askbid_m1_parquet_ready(
        path=path,
        feature_minute_ns=feature_minute_ns,
        entry_latency_ms=int(entry_latency_ms),
        tick_source_path=tick_source_path,
        tick_datetime_column=tick_datetime_column,
        tick_sep=tick_sep,
        tick_chunk_size=tick_chunk_size,
    )
    return _load_askbid_m1_arrays(
        path,
        feature_minute_ns=feature_minute_ns,
        entry_latency_ms=entry_latency_ms,
        source_tick_path=tick_source_path,
    )


def _valid_price(value) -> bool:
    try:
        x = float(value)
    except Exception:
        return False
    return bool(np.isfinite(x) and x > 0.0)


def _simulate_multitp_trailing_askbid_m1(
    *,
    ask_entry_latency: np.ndarray,
    bid_high_after_entry_latency: np.ndarray,
    bid_low_after_entry_latency: np.ndarray,
    bid_high: np.ndarray,
    bid_low: np.ndarray,
    bid_close: np.ndarray,
    tps: list[float],
    tp_w: np.ndarray,
    tp_enabled: bool,
    sl: float,
    hold: int,
    slippage_bps: float,
    trail: bool,
    trail_activate: float,
    trail_offset: float,
    trail_factor: float,
    include_unrealized_at_test_end: bool,
    period_end_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = int(len(ask_entry_latency))
    pnl = np.full(n, np.nan, dtype=np.float64)
    y = np.full(n, -1, dtype=np.int8)
    t_exit = np.full(n, -1, dtype=np.int32)
    t_qual = np.full(n, -1, dtype=np.int32)
    tp_hits = np.zeros(n, dtype=np.int8)

    slip = float(slippage_bps) / 10000.0
    tps_arr = np.asarray(tps, dtype=np.float64)
    tp_w_arr = np.asarray(tp_w, dtype=np.float64)
    k_max = int(len(tps_arr))

    for i in range(max(0, n - 1)):
        entry_i = i + 1
        if entry_i >= n:
            continue
        entry_ask = float(ask_entry_latency[entry_i])
        if not _valid_price(entry_ask):
            continue
        entry = entry_ask * (1.0 + slip)
        if not _valid_price(entry):
            continue

        stop_ret = -float(sl)
        stop_level = entry * (1.0 + stop_ret)
        tp_levels = entry * (1.0 + tps_arr)

        remaining = 1.0
        realized = 0.0
        hits = 0
        max_profit_ret = 0.0
        trailing_active = (not bool(trail))

        period_end = int(period_end_indices[i]) if period_end_indices is not None else (n - 1)
        period_end = max(i, min(period_end, n - 1))
        max_k = min(period_end - i, max(1, int(hold)))
        if max_k <= 0:
            continue
        hold_reached = (i + max(1, int(hold))) <= period_end
        qualified = False
        invalid_data = False

        for k in range(1, max_k + 1):
            j = i + k
            if j == entry_i:
                h = float(bid_high_after_entry_latency[j])
                l = float(bid_low_after_entry_latency[j])
            else:
                h = float(bid_high.take(j))
                l = float(bid_low.take(j))
            if not (_valid_price(h) and _valid_price(l)):
                invalid_data = True
                break

            curr_profit_ret = (h / entry) - 1.0
            if curr_profit_ret > max_profit_ret:
                max_profit_ret = curr_profit_ret

            if trail and (not trailing_active) and max_profit_ret >= float(trail_activate):
                trailing_active = True

            if trail and trailing_active:
                cand = (max_profit_ret - float(trail_offset)) * float(trail_factor)
                if cand > stop_ret:
                    stop_ret = cand
                    stop_level = entry * (1.0 + stop_ret)

            if (not qualified) and trail and k <= int(hold) and max_profit_ret >= float(trail_activate):
                qualified = True
                t_qual[i] = k

            if l <= stop_level:
                realized += remaining * stop_ret
                pnl[i] = realized
                y[i] = 1 if qualified else (1 if realized > 0 else 0)
                t_exit[i] = k
                tp_hits[i] = hits
                break

            if tp_enabled and hits < k_max and h >= tp_levels[hits]:
                w = min(float(tp_w_arr[hits]), remaining)
                if w > 0:
                    realized += w * float(tps_arr[hits])
                    remaining -= w
                hits += 1
                if remaining <= 1e-12:
                    pnl[i] = realized
                    y[i] = 1 if qualified else (1 if realized > 0 else 0)
                    t_exit[i] = k
                    tp_hits[i] = hits
                    break

        if invalid_data:
            continue

        if y[i] == -1:
            if hold_reached or include_unrealized_at_test_end:
                j_end = i + max_k
                if trail and trailing_active:
                    endpoint_ret = float(stop_ret)
                else:
                    endpoint_close = float(bid_close[j_end])
                    if not _valid_price(endpoint_close):
                        continue
                    endpoint_ret = (endpoint_close / entry) - 1.0
                final_pnl = realized + remaining * endpoint_ret
                pnl[i] = final_pnl
                y[i] = 1 if qualified else (1 if final_pnl > 0 else 0)
                t_exit[i] = max_k
                tp_hits[i] = hits

    return pnl, y, t_exit, t_qual, tp_hits


def _critical_minutes_for_entries_askbid_m1(
    *,
    entry_indices: np.ndarray,
    askbid_m1_arrays: dict[str, np.ndarray],
    selection_entry_price: np.ndarray,
    bar_time_ns: np.ndarray,
    hold: int,
    tp_mode: str,
    tp_enabled: bool,
    trail: bool,
    trail_activate: float,
    trail_offset: float,
    trail_factor: float,
    tps: np.ndarray,
    tp_w: np.ndarray,
    sl: float,
    period_end_indices: np.ndarray | None,
) -> dict[int, set[int]]:
    critical_by_entry: dict[int, set[int]] = {}
    if entry_indices.size == 0:
        return critical_by_entry
    bid_high_after = np.asarray(askbid_m1_arrays["bid_high_after_entry_latency"], dtype=np.float64)
    bid_low_after = np.asarray(askbid_m1_arrays["bid_low_after_entry_latency"], dtype=np.float64)
    bid_high = np.asarray(askbid_m1_arrays["bid_high"], dtype=np.float64)
    bid_low = np.asarray(askbid_m1_arrays["bid_low"], dtype=np.float64)
    entries = np.asarray(selection_entry_price, dtype=np.float64)
    tps_arr = np.asarray(tps, dtype=np.float64)
    tp_w_arr = np.asarray(tp_w, dtype=np.float64)
    n = int(len(entries))
    eps = 1e-12
    for idx_i in entry_indices.tolist():
        i = int(idx_i)
        entry_i = i + 1
        if i < 0 or entry_i >= n:
            continue
        entry = float(entries[i])
        if not _valid_price(entry):
            continue
        stop_ret = -float(sl)
        stop_level = entry * (1.0 + stop_ret)
        tp_levels = entry * (1.0 + tps_arr)
        remaining = 1.0
        realized = 0.0
        hits = 0
        max_profit_ret = 0.0
        trailing_active = (not bool(trail))
        period_end = int(period_end_indices[i]) if period_end_indices is not None else (n - 1)
        period_end = max(i, min(period_end, n - 1))
        max_k = min(period_end - i, max(1, int(hold)))
        if max_k <= 0:
            continue
        crit: set[int] = set()
        for k in range(1, max_k + 1):
            j = i + k
            if j == entry_i:
                bar_bid_high = float(bid_high_after[j])
                bar_bid_low = float(bid_low_after[j])
            else:
                bar_bid_high = float(bid_high.take(j))
                bar_bid_low = float(bid_low.take(j))
            if not (_valid_price(bar_bid_high) and _valid_price(bar_bid_low)):
                continue
            curr_profit_ret = (bar_bid_high / entry) - 1.0
            max_profit_after_bar = max(max_profit_ret, curr_profit_ret)
            current_stop_hit = bool(bar_bid_low <= stop_level)
            tp_hit = bool(tp_enabled and hits < len(tp_levels) and bar_bid_high >= tp_levels[hits])
            candidate_stop_ret = float(stop_ret)
            if trail:
                trail_can_be_active_after_bar = bool(trailing_active or max_profit_after_bar >= float(trail_activate))
                if trail_can_be_active_after_bar:
                    candidate_stop_ret = max(
                        candidate_stop_ret,
                        (max_profit_after_bar - float(trail_offset)) * float(trail_factor),
                    )
            candidate_stop_level = entry * (1.0 + candidate_stop_ret)
            trail_stop_level_can_increase = bool(trail and candidate_stop_ret > stop_ret + eps)
            trail_stop_order_ambiguous = bool(
                trail
                and trail_stop_level_can_increase
                and bar_bid_low <= candidate_stop_level
            )
            is_critical = bool((tp_hit and current_stop_hit) or trail_stop_order_ambiguous)
            if is_critical:
                crit.add(int(bar_time_ns[j]))
            if current_stop_hit:
                break

            # Keep a conservative M1 state so later minutes can be inspected with
            # approximately current TP/trailing state; replay itself is authoritative.
            if max_profit_after_bar > max_profit_ret:
                max_profit_ret = max_profit_after_bar
            if trail and (not trailing_active) and max_profit_ret >= float(trail_activate):
                trailing_active = True
            if trail and trailing_active:
                cand = (max_profit_ret - float(trail_offset)) * float(trail_factor)
                if cand > stop_ret:
                    stop_ret = cand
                    stop_level = entry * (1.0 + stop_ret)
            if tp_enabled and hits < len(tp_levels) and tp_hit:
                w = min(float(tp_w_arr[hits]), remaining)
                if w > 0:
                    realized += w * float(tps_arr[hits])
                    remaining -= w
                hits += 1
                if remaining <= 1e-12:
                    break
        if crit:
            critical_by_entry[i] = crit
    return critical_by_entry


def _simulate_selected_entries_with_askbid_ticks(
    *,
    entry_indices: np.ndarray,
    critical_by_entry: dict[int, set[int]],
    askbid_m1_arrays: dict[str, np.ndarray],
    selection_entry_price: np.ndarray,
    bar_time_ns: np.ndarray,
    tick_time_ns_all: np.ndarray,
    tick_minute_bounds: dict[int, tuple[int, int]],
    tick_bids_all: np.ndarray,
    tps: list[float],
    tp_w: np.ndarray,
    tp_enabled: bool,
    sl: float,
    hold: int,
    trail: bool,
    trail_activate: float,
    trail_offset: float,
    trail_factor: float,
    include_unrealized_at_test_end: bool,
    period_end_indices: np.ndarray | None,
    entry_latency_ms: int,
) -> dict[int, dict]:
    out: dict[int, dict] = {}
    invalid_rec = {"y": -1, "pnl": float("nan"), "t_exit": -1, "t_qual": -1, "tp_hits": 0}
    entries = np.asarray(selection_entry_price, dtype=np.float64)
    bid_high_after = np.asarray(askbid_m1_arrays["bid_high_after_entry_latency"], dtype=np.float64)
    bid_low_after = np.asarray(askbid_m1_arrays["bid_low_after_entry_latency"], dtype=np.float64)
    bid_high = np.asarray(askbid_m1_arrays["bid_high"], dtype=np.float64)
    bid_low = np.asarray(askbid_m1_arrays["bid_low"], dtype=np.float64)
    bid_close = np.asarray(askbid_m1_arrays["bid_close"], dtype=np.float64)
    tick_time_ns_all = np.asarray(tick_time_ns_all, dtype=np.int64)
    tick_bids_all = np.asarray(tick_bids_all, dtype=np.float64)
    tps_arr = np.asarray(tps, dtype=np.float64)
    tp_w_arr = np.asarray(tp_w, dtype=np.float64)
    n = int(len(entries))
    entry_latency_ns = int(entry_latency_ms) * 1_000_000

    for idx_i in entry_indices.tolist():
        i = int(idx_i)
        entry_i = i + 1
        if i < 0 or entry_i >= n:
            out[i] = dict(invalid_rec)
            continue
        entry = float(entries[i])
        if not _valid_price(entry):
            out[i] = dict(invalid_rec)
            continue
        stop_ret = -float(sl)
        stop_level = entry * (1.0 + stop_ret)
        tp_levels = entry * (1.0 + tps_arr)
        remaining = 1.0
        realized = 0.0
        hits = 0
        max_profit_ret = 0.0
        trailing_active = (not bool(trail))
        qualified = False
        t_qual_val = -1
        critical_minutes = {int(x) for x in critical_by_entry.get(i, set())}
        period_end = int(period_end_indices[i]) if period_end_indices is not None else (n - 1)
        period_end = max(i, min(period_end, n - 1))
        max_k = min(period_end - i, max(1, int(hold)))
        if max_k <= 0:
            out[i] = dict(invalid_rec)
            continue
        hold_reached = (i + max(1, int(hold))) <= period_end
        exited = False
        invalid_data = False

        def finish(exit_k: int, pnl_value: float, hits_value: int) -> None:
            nonlocal exited
            out[i] = {
                "y": int(1 if qualified else (1 if pnl_value > 0 else 0)),
                "pnl": float(pnl_value),
                "t_exit": int(exit_k),
                "t_qual": int(t_qual_val),
                "tp_hits": int(hits_value),
            }
            exited = True

        for k in range(1, max_k + 1):
            j = i + k
            minute_ns = int(bar_time_ns[j])
            is_critical = minute_ns in critical_minutes
            if is_critical:
                bounds = tick_minute_bounds.get(minute_ns)
                if bounds is None:
                    invalid_data = True
                    break
                s_idx, e_idx = bounds
                times = tick_time_ns_all[s_idx:e_idx]
                bids = tick_bids_all[s_idx:e_idx]
                if j == entry_i:
                    entry_time_ns = int(bar_time_ns[entry_i]) + entry_latency_ns
                    keep = times >= entry_time_ns
                    times = times[keep]
                    bids = bids[keep]
                for px in bids.tolist():
                    bid_px = float(px)
                    if not _valid_price(bid_px):
                        continue
                    curr_profit_ret = (bid_px / entry) - 1.0
                    if curr_profit_ret > max_profit_ret:
                        max_profit_ret = curr_profit_ret
                    if trail and (not trailing_active) and max_profit_ret >= float(trail_activate):
                        trailing_active = True
                    if trail and trailing_active:
                        cand = (max_profit_ret - float(trail_offset)) * float(trail_factor)
                        if cand > stop_ret:
                            stop_ret = cand
                            stop_level = entry * (1.0 + stop_ret)
                    if (not qualified) and trail and k <= int(hold) and max_profit_ret >= float(trail_activate):
                        qualified = True
                        t_qual_val = int(k)
                    if bid_px <= stop_level:
                        realized += remaining * stop_ret
                        finish(k, realized, hits)
                        break
                    if tp_enabled and hits < len(tp_levels) and bid_px >= tp_levels[hits]:
                        w = min(float(tp_w_arr[hits]), remaining)
                        if w > 0:
                            realized += w * float(tps_arr[hits])
                            remaining -= w
                        hits += 1
                        if remaining <= 1e-12:
                            finish(k, realized, hits)
                            break
                if exited:
                    break
                continue

            if j == entry_i:
                bar_bid_high = float(bid_high_after[j])
                bar_bid_low = float(bid_low_after[j])
            else:
                bar_bid_high = float(bid_high.take(j))
                bar_bid_low = float(bid_low.take(j))
            if not (_valid_price(bar_bid_high) and _valid_price(bar_bid_low)):
                invalid_data = True
                break
            curr_profit_ret = (bar_bid_high / entry) - 1.0
            if curr_profit_ret > max_profit_ret:
                max_profit_ret = curr_profit_ret
            if trail and (not trailing_active) and max_profit_ret >= float(trail_activate):
                trailing_active = True
            if trail and trailing_active:
                cand = (max_profit_ret - float(trail_offset)) * float(trail_factor)
                if cand > stop_ret:
                    stop_ret = cand
                    stop_level = entry * (1.0 + stop_ret)
            if (not qualified) and trail and k <= int(hold) and max_profit_ret >= float(trail_activate):
                qualified = True
                t_qual_val = int(k)
            if bar_bid_low <= stop_level:
                realized += remaining * stop_ret
                finish(k, realized, hits)
                break
            if tp_enabled and hits < len(tp_levels) and bar_bid_high >= tp_levels[hits]:
                w = min(float(tp_w_arr[hits]), remaining)
                if w > 0:
                    realized += w * float(tps_arr[hits])
                    remaining -= w
                hits += 1
                if remaining <= 1e-12:
                    finish(k, realized, hits)
                    break
        if exited:
            continue
        if invalid_data:
            out[i] = dict(invalid_rec)
            continue
        if hold_reached or include_unrealized_at_test_end:
            j_end = i + max_k
            if trail and trailing_active:
                endpoint_ret = float(stop_ret)
            else:
                endpoint_close = float(bid_close[j_end])
                if not _valid_price(endpoint_close):
                    out[i] = dict(invalid_rec)
                    continue
                endpoint_ret = (endpoint_close / entry) - 1.0
            final_pnl = realized + remaining * endpoint_ret
            finish(max_k, final_pnl, hits)
        else:
            out[i] = dict(invalid_rec)
    return out


def _load_askbid_tick_replay_minutes(
    path: Path,
    datetime_col: str,
    sep: str,
    minute_filter: set[int],
    tick_chunk_size: int | str = "auto",
) -> tuple[np.ndarray, dict[int, tuple[int, int]], np.ndarray, np.ndarray | None, int]:
    if not minute_filter:
        return np.asarray([], dtype=np.int64), {}, np.asarray([], dtype=np.float64), None, 0
    minute_filter_arr = np.asarray(list(minute_filter), dtype=np.int64)
    minute_parts: list[np.ndarray] = []
    time_parts: list[np.ndarray] = []
    bid_parts: list[np.ndarray] = []
    ask_parts: list[np.ndarray] = []
    saw_ask = False
    matched_minute_set: set[int] = set()
    nat_ns = np.iinfo(np.int64).min

    def add_rows(dt_ns: np.ndarray, bid: np.ndarray, ask: np.ndarray | None) -> None:
        nonlocal saw_ask
        minute_ns = ((dt_ns // NS_PER_MINUTE) * NS_PER_MINUTE).astype(np.int64, copy=False)
        valid = (dt_ns != nat_ns) & np.isfinite(bid) & (bid > 0) & np.isin(minute_ns, minute_filter_arr)
        if ask is not None:
            saw_ask = True
            valid = valid & (np.isfinite(ask) | np.isnan(ask))
        if not np.any(valid):
            return
        keep_idx = np.flatnonzero(valid)
        minute_parts.append(minute_ns[keep_idx].astype(np.int64, copy=False))
        time_parts.append(dt_ns[keep_idx].astype(np.int64, copy=False))
        bid_parts.append(bid[keep_idx].astype(np.float64, copy=False))
        if ask is not None:
            ask_parts.append(ask[keep_idx].astype(np.float64, copy=False))
        matched_minute_set.update({int(x) for x in np.unique(minute_ns[keep_idx]).tolist()})

    if path.suffix.lower() in {".parquet", ".pq"}:
        try:
            import pyarrow.parquet as pa_parquet  # type: ignore
        except Exception as exc:
            raise RuntimeError("pyarrow is required for AskBid tick replay parquet loading") from exc
        pf = pa_parquet.ParquetFile(path)
        schema_cols = set(str(c) for c in pf.schema_arrow.names)
        missing = {"DateTime", "Bid"} - schema_cols
        if missing:
            raise ValueError(f"tick parquet missing columns for AskBid replay: {sorted(missing)}")
        cols = ["DateTime", "Bid"] + (["Ask"] if "Ask" in schema_cols else [])
        chunk_size_eff = 150_000 if str(tick_chunk_size).lower() == "auto" else max(10_000, int(tick_chunk_size))
        for batch in pf.iter_batches(columns=cols, batch_size=chunk_size_eff):
            ch = batch.to_pandas()
            dt_full = pd.to_datetime(ch["DateTime"], errors="coerce", utc=True)
            dt_ns = dt_full.astype("int64").to_numpy(dtype=np.int64)
            bid = pd.to_numeric(ch["Bid"], errors="coerce").to_numpy(dtype=np.float64)
            ask = pd.to_numeric(ch["Ask"], errors="coerce").to_numpy(dtype=np.float64) if "Ask" in ch.columns else None
            add_rows(dt_ns, bid, ask)
    else:
        chunk_size_eff = 150_000 if str(tick_chunk_size).lower() == "auto" else max(10_000, int(tick_chunk_size))
        for ch in pd.read_csv(path, sep=sep, chunksize=chunk_size_eff):
            cols_lower = {str(c).lower(): str(c) for c in ch.columns}
            dt_col = cols_lower.get(str(datetime_col).strip().lower())
            if dt_col is None:
                for cand in ("datetime", "time", "timestamp", "date"):
                    if cand in cols_lower:
                        dt_col = cols_lower[cand]
                        break
            if dt_col is None:
                raise ValueError(f"tick-data missing datetime column for AskBid replay: {datetime_col}")
            bid_col = cols_lower.get("bid")
            ask_col = cols_lower.get("ask")
            if bid_col is None:
                raise ValueError("AskBid tick replay requires a Bid column in tick CSV")
            dt_raw = ch[dt_col]
            date_src = None
            time_src = None
            date_candidates = ("date", "<date>", "<dtyyyymmdd>", "dtyyyymmdd")
            time_candidates = ("time", "<time>", "time_msc", "timestamp_time")
            for dname in date_candidates:
                if dname in cols_lower and cols_lower[dname] != dt_col:
                    cand = ch[cols_lower[dname]].astype(str).str.strip()
                    if bool((cand.str.fullmatch(r"\d{8}", na=False)).any()) or bool((cand.str.fullmatch(r"\d{4}-\d{2}-\d{2}", na=False)).any()):
                        date_src = cand
                        break
            for tname in time_candidates:
                if tname in cols_lower and cols_lower[tname] != dt_col:
                    cand = ch[cols_lower[tname]].astype(str).str.strip()
                    if bool((cand.str.fullmatch(r"\d{2}:\d{2}:\d{2}(?:\.\d{1,9})?", na=False)).any()):
                        time_src = cand
                        break
            if date_src is None and isinstance(ch.index, pd.Index):
                idx_s = ch.index.to_series(index=ch.index).astype(str).str.strip()
                if bool((idx_s.str.fullmatch(r"\d{8}", na=False)).any()):
                    date_src = idx_s
            time_s = dt_raw.astype(str).str.strip()
            looks_time_like = bool((time_s.str.fullmatch(r"\d{2}:\d{2}:\d{2}(?:\.\d{1,9})?", na=False)).any())
            looks_date_like = bool((time_s.str.fullmatch(r"\d{8}", na=False)).any()) or bool((time_s.str.fullmatch(r"\d{4}-\d{2}-\d{2}", na=False)).any())
            if date_src is not None and looks_time_like:
                dt_join = date_src.astype(str).str.strip() + " " + time_s
                dt_full = pd.to_datetime(dt_join, format="%Y%m%d %H:%M:%S.%f", errors="coerce", utc=True)
                if not bool(dt_full.notna().any()):
                    dt_full = pd.to_datetime(dt_join, format="%Y%m%d %H:%M:%S", errors="coerce", utc=True)
                if not bool(dt_full.notna().any()):
                    dt_full = pd.to_datetime(dt_join, errors="coerce", utc=True)
            elif looks_date_like and time_src is not None:
                dt_join = time_s + " " + time_src.astype(str).str.strip()
                dt_full = pd.to_datetime(dt_join, format="%Y%m%d %H:%M:%S.%f", errors="coerce", utc=True)
                if not bool(dt_full.notna().any()):
                    dt_full = pd.to_datetime(dt_join, format="%Y%m%d %H:%M:%S", errors="coerce", utc=True)
                if not bool(dt_full.notna().any()):
                    dt_full = pd.to_datetime(dt_join, errors="coerce", utc=True)
            else:
                dt_full = pd.to_datetime(dt_raw, errors="coerce", utc=True)
            if bool(dt_full.notna().any()):
                y_min = int(dt_full.dropna().dt.year.min())
                y_max = int(dt_full.dropna().dt.year.max())
                if y_max <= 1971:
                    raise ValueError(
                        f"tick-data datetime parse is implausible for AskBid replay (range {y_min}-{y_max})"
                    )
            dt_ns = (
                dt_full.dt.tz_convert("UTC")
                .dt.tz_localize(None)
                .to_numpy(dtype="datetime64[ns]")
                .astype("int64")
            )
            bid = pd.to_numeric(ch[bid_col], errors="coerce").to_numpy(dtype=np.float64)
            ask = pd.to_numeric(ch[ask_col], errors="coerce").to_numpy(dtype=np.float64) if ask_col is not None else None
            add_rows(dt_ns, bid, ask)

    if not bid_parts:
        return np.asarray([], dtype=np.int64), {}, np.asarray([], dtype=np.float64), None, 0
    mins = np.concatenate(minute_parts).astype(np.int64, copy=False)
    times = np.concatenate(time_parts).astype(np.int64, copy=False)
    bids = np.concatenate(bid_parts).astype(np.float64, copy=False)
    asks = np.concatenate(ask_parts).astype(np.float64, copy=False) if saw_ask and ask_parts else None
    order = np.lexsort((times, mins))
    mins = mins[order]
    times = times[order]
    bids = bids[order]
    if asks is not None:
        asks = asks[order]
    bounds: dict[int, tuple[int, int]] = {}
    i = 0
    while i < len(mins):
        m = int(mins[i]); j = i + 1
        while j < len(mins) and int(mins[j]) == m:
            j += 1
        bounds[m] = (i, j)
        i = j
    print(
        "[prefilter-askbid-replay-load] "
        f"rows_matched={len(times)} matched_minutes={len(bounds)} ask_loaded={bool(asks is not None)}"
    )
    return times, bounds, bids, asks, int(len(matched_minute_set))


def _mask_hash_arr(mask: np.ndarray) -> str:
    packed = np.packbits(mask.astype(np.uint8, copy=False))
    return hashlib.sha1(packed.tobytes()).hexdigest()


def _with_rank_context(ranked: list[dict], rank_name: str) -> list[dict]:
    out: list[dict] = []
    for i, it in enumerate(ranked):
        z = dict(it)
        z["_rank_name"] = rank_name
        z["_rank_pos"] = int(i)
        out.append(z)
    return out


def _build_unlocked_pool(
    miner_mod,
    rank_lists: list[list[dict]],
    all_candidate_cols: list[str],
    unlocked_next: int,
    step_size: int,
    binary_cap_per_list_block: int,
    binary_anchor_lookahead_blocks: int,
    binary_cap_per_block: int,
) -> list[dict]:
    if unlocked_next <= 0:
        return []
    block_size = max(1, int(step_size))
    unlocked_blocks = int(math.ceil(float(unlocked_next) / float(block_size)))
    lookahead_blocks = max(0, int(binary_anchor_lookahead_blocks))
    merged: dict[tuple[str, str, float], dict] = {}

    blocks_by_rank = [_chunk_list(rank, block_size) for rank in rank_lists]

    def _has_anchor_within_lookahead(binary_col: str, block_idx: int) -> bool:
        anchors = miner_mod.candidate_anchor_columns(binary_col, all_candidate_cols)
        if not anchors:
            return True
        cols_window: set[str] = set()
        b0 = max(0, int(block_idx))
        b1 = b0 + lookahead_blocks
        for blocks in blocks_by_rank:
            for bi in range(b0, min(len(blocks), b1 + 1)):
                cols_window.update(str(x["col"]) for x in blocks[bi])
        return any(a in cols_window for a in anchors)

    def _add_item(it: dict, source_block: int) -> None:
        key = (str(it["col"]), str(it["op"]), float(it["value"]))
        cur = merged.get(key)
        cand = dict(it)
        cand["_source_block"] = int(source_block)
        if cur is None:
            merged[key] = cand
            return
        if int(cand.get("_rank_pos", 10 ** 9)) < int(cur.get("_rank_pos", 10 ** 9)):
            merged[key] = cand

    for blocks in blocks_by_rank:
        for bi in range(min(len(blocks), unlocked_blocks)):
            bin_count = 0
            for it in blocks[bi]:
                if bool(it.get("binary", False)):
                    if not _has_anchor_within_lookahead(str(it["col"]), bi):
                        continue
                    if bin_count >= int(binary_cap_per_list_block):
                        continue
                    bin_count += 1
                _add_item(it, source_block=bi)

    if int(binary_cap_per_block) > 0:
        per_step_cap = int(binary_cap_per_block)
        by_block: dict[int, list[int]] = {}
        pool = list(merged.values())
        for i, it in enumerate(pool):
            bi = int(it.get("_source_block", 0))
            by_block.setdefault(bi, []).append(i)
        keep_idx: set[int] = set()
        for bi, idxs in by_block.items():
            bidx = [i for i in idxs if bool(pool[i].get("binary", False))]
            nidx = [i for i in idxs if not bool(pool[i].get("binary", False))]
            keep_idx.update(nidx)
            keep_idx.update(bidx[:per_step_cap])
        merged = {
            (str(pool[i]["col"]), str(pool[i]["op"]), float(pool[i]["value"])): pool[i]
            for i in sorted(keep_idx)
        }

    return list(merged.values())


def _calc_wf(mask: np.ndarray, y_test: np.ndarray, wf_folds: int) -> tuple[float, float, int]:
    n = len(y_test)
    if n == 0:
        return float("nan"), float("nan"), 0
    edges = np.linspace(0, n, max(1, wf_folds) + 1, dtype=int)
    ratios = []
    total = 0
    for i in range(len(edges) - 1):
        a, b = int(edges[i]), int(edges[i + 1])
        m = mask[a:b]
        if not m.any():
            continue
        pos = int(np.sum(m & (y_test[a:b] == 1)))
        neg = int(np.sum(m & (y_test[a:b] == 0)))
        total += (pos + neg)
        ratios.append(pos / max(1, neg))
    if not ratios:
        return float("nan"), float("nan"), 0
    return float(np.mean(ratios)), float(np.min(ratios)), int(total)


def main() -> None:
    main_t0 = time.perf_counter()
    args = parse_args()
    if _disabled_output_path(args.out_rules_json):
        args.out_rules_json = None
        print("[prefilter-output] rules_json=disabled")
    if int(args.cluster_gap_minutes) < 0:
        raise ValueError("--cluster-gap-minutes must be >= 0")
    if int(args.max_entries_per_cluster) < 1:
        raise ValueError("--max-entries-per-cluster must be >= 1")
    if int(args.max_open_trades) < 1:
        raise ValueError("--max-open-trades must be >= 1")
    if int(args.entry_latency_ms) < 0 or int(args.entry_latency_ms) >= 60000:
        raise ValueError("--entry-latency-ms must be >= 0 and < 60000")
    if int(args.phase_c_max_generated_per_level) < 0:
        raise ValueError("--phase-c-max-generated-per-level must be >= 0")
    if int(args.phase_d_max_generated_per_level) < 0:
        raise ValueError("--phase-d-max-generated-per-level must be >= 0")
    if args.out_candidates_csv is not None:
        raise ValueError("--out-candidates-csv is deprecated. Use --out-candidates-coarse-csv and --out-candidates-refined-csv.")
    miner = _load_miner_module(args.miner_script)
    timing: dict[str, float] = {}
    timing_detail: dict[str, float] = {
        "tick_cache_load_sec": 0.0,
        "tick_scope_build_sec": 0.0,
        "tick_replay_plan_sec": 0.0,
        "tick_raw_load_sec": 0.0,
        "tick_simulate_selected_entries_sec": 0.0,
        "tick_cache_write_sec": 0.0,
        "tick_cache_diagnostics_sec": 0.0,
        "tick_candidate_metrics_sec": 0.0,
        "tick_missing_metrics_scan_sec": 0.0,
        "candidate_csv_write_sec": 0.0,
        "phase_d_pool_build_sec": 0.0,
        "ranking_sort_sec": 0.0,
    }
    timing_detail2: dict[str, float] = {
        "coarse_csv_load_sec": 0.0,
        "coarse_rows_filter_sec": 0.0,
        "coarse_reconstruct_sec": 0.0,
        "allowlist_filter_sec": 0.0,
        "refined_csv_load_sec": 0.0,
        "refined_key_match_sec": 0.0,
        "refined_tick_metric_restore_sec": 0.0,
        "quantile_thresholds_sec": 0.0,
        "build_items_candidate_loop_sec": 0.0,
        "build_items_mask_sec": 0.0,
        "build_items_metadata_sec": 0.0,
        "filtered_items_sec": 0.0,
        "family_top_sec": 0.0,
        "same_reference_groups_sec": 0.0,
        "search_setup_sec": 0.0,
        "phase_a_total_sec": 0.0,
        "phase_b_total_sec": 0.0,
        "phase_c_total_sec": 0.0,
        "phase_d_total_sec": 0.0,
        "rules_export_sec": 0.0,
    }
    phase_timing_counts: dict[str, int] = {
        "phase_a_parent_extensions_total": 0,
        "phase_a_parent_extensions_rounds": 0,
        "phase_b_parent_extensions_total": 0,
    }
    phase_b_started = False
    phase_b_skip_reason = "not_reached"
    timing_counts: dict[str, int] = {
        "tick_cache_loaded_entries_count": 0,
        "tick_cache_diagnostics_entries_count": 0,
        "tick_scope_entries_count": 0,
        "tick_replay_relevant_entries_count": 0,
        "tick_replay_critical_minutes_count": 0,
        "tick_replay_cache_hits_count": 0,
        "tick_replay_cache_missing_entries_count": 0,
        "tick_replay_raw_critical_minutes_requested_count": 0,
        "tick_replay_raw_loaded_minutes_count": 0,
    }
    timing_flags: dict[str, bool] = {
        "tick_raw_load_skipped": True,
        "tick_simulation_skipped": True,
        "tick_cache_write_skipped": True,
        "tick_cache_diagnostics_skipped": True,
    }

    if args.askbid_m1_parquet is None:
        raise ValueError("--askbid-m1-parquet is required for Prefilter labels/outcomes; old OHLC label simulation is disabled")

    t0 = time.perf_counter()
    feature_minute_ns_minimal = _load_feature_minute_grid_minimal(args.features)
    timing["load_feature_minute_grid_minimal_sec"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ensure_askbid_m1_parquet_ready(
        path=args.askbid_m1_parquet,
        feature_minute_ns=feature_minute_ns_minimal,
        entry_latency_ms=int(args.entry_latency_ms),
        tick_source_path=args.tick_data,
        tick_datetime_column=args.tick_datetime_column,
        tick_sep=args.tick_sep,
        tick_chunk_size=args.tick_chunk_size,
    )
    timing["ensure_askbid_m1_sec"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    df = miner.load_features(args.features)
    timing["load_features_sec"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    bdf = miner.load_binned_features(args.binned_features, tail_rows=0).reindex(df.index)
    timing["load_binned_features_sec"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    meta = miner.load_binned_metadata(args.binned_metadata)
    timing["load_binned_metadata_sec"] = time.perf_counter() - t0

    n = len(df)
    train_idx = max(1, min(int(n * float(args.train_frac)), n - 1))

    bar_time_ns = _datetime_index_to_minute_ns(df.index)
    if not np.array_equal(bar_time_ns, feature_minute_ns_minimal):
        raise ValueError("Full feature datetime index differs from the early minimal feature datetime grid")

    askbid_m1_arrays = _load_askbid_m1_arrays(
        args.askbid_m1_parquet,
        feature_minute_ns=bar_time_ns,
        entry_latency_ms=int(args.entry_latency_ms),
        source_tick_path=args.tick_data,
    )
    print(
        "[prefilter-askbid-m1] "
        f"status=active_label_outcome_basis arrays={len(askbid_m1_arrays)}"
    )
    selection_entry_price = np.full(n, np.nan, dtype=np.float64)
    if n > 1:
        ask_entry_for_selection = np.asarray(askbid_m1_arrays["ask_entry_latency"], dtype=np.float64)
        valid_entry = np.isfinite(ask_entry_for_selection[1:]) & (ask_entry_for_selection[1:] > 0.0)
        selection_entry_price[np.flatnonzero(valid_entry)] = (
            ask_entry_for_selection[1:][valid_entry] * (1.0 + float(args.slippage_bps) / 10000.0)
        )
    selection_entry_valid = np.isfinite(selection_entry_price) & (selection_entry_price > 0.0)
    selection_entry_price_invalid_count = int(np.sum(~selection_entry_valid))
    selection_entry_price_invalid_excluding_last_row_count = int(np.sum(~selection_entry_valid[:-1])) if n > 1 else selection_entry_price_invalid_count
    selection_entry_price_invalid_share = float(selection_entry_price_invalid_excluding_last_row_count / max(1, n - 1))
    print(
        "[prefilter-selection-entry-price] "
        f"selection_entry_price_invalid_count={selection_entry_price_invalid_count} "
        f"selection_entry_price_invalid_excluding_last_row_count={selection_entry_price_invalid_excluding_last_row_count} "
        f"selection_entry_price_invalid_share={selection_entry_price_invalid_share:.6f}"
    )

    tps_all = [float(x) for x in str(args.tps).split(",") if x.strip()]
    if bool(args.no_tp):
        tp_mode = "none"
    else:
        tp_mode = "multi" if bool(args.use_multi_tp) else "single"
    if (not bool(args.trail)) and tp_mode == "none":
        raise ValueError("Invalid TP mode: no TP is not allowed when --trail is disabled.")
    if tp_mode == "multi":
        tps = tps_all
        tp_w = miner.parse_tp_weights(tps, str(args.tp_weights))
        tp_enabled = True
    elif tp_mode == "single":
        tps = [tps_all[0]]
        tp_w = np.asarray([1.0], dtype=np.float64)
        tp_enabled = True
    else:
        tps = [tps_all[0]]
        tp_w = np.asarray([1.0], dtype=np.float64)
        tp_enabled = False

    cache_key_obj = {
        "features_sig": _file_sig(args.features),
        "price_basis": "askbid_m1",
        "askbid_m1_sig": _file_sig(args.askbid_m1_parquet),
        "askbid_m1_semantics": ASKBID_M1_SEMANTICS,
        "entry_latency_ms": int(args.entry_latency_ms),
        "entry_indexing": "entry_i_to_minute_i_plus_1",
        "spread_handling": "ask_contains_spread_spread_bps_not_added",
        "slippage_handling": "ask_entry_latency_times_1_plus_slippage_bps",
        "exit_price_basis": "bid",
        "entry_minute_exit_bounds": "bid_high_low_after_entry_latency",
        "following_minute_exit_bounds": "bid_high_low",
        "endpoint_price_basis": "bid_close",
        "tps": [float(x) for x in tps],
        "tp_weights": [float(x) for x in tp_w.tolist()],
        "tp_mode": str(tp_mode),
        "sl": float(args.sl),
        "hold": int(args.hold),
        "slippage_bps": float(args.slippage_bps),
        "spread_bps": float(args.spread_bps),
        "trail": bool(args.trail),
        "trail_activate": float(args.trail_activate),
        "trail_offset": float(args.trail_offset),
        "trail_factor": float(args.trail_factor),
        "trail_min_level": float(args.trail_min_level),
        "include_unrealized_at_test_end": bool(args.include_unrealized_at_test_end),
        "label_semantics_version": "hold_endpoint_v2",
        "train_frac": float(args.train_frac),
        "train_idx": int(train_idx),
        "n_rows": int(n),
        "full_end_idx": int(n - 1),
        "period_end_semantics": "train_test_split_endpoints_v1",
    }
    cache_key = hashlib.sha1(json.dumps(cache_key_obj, sort_keys=True).encode("utf-8")).hexdigest()
    loaded_cache = False
    if args.label_cache_npz is not None and args.label_cache_npz.exists():
        try:
            z = np.load(args.label_cache_npz, allow_pickle=False)
            if str(z["cache_key"].item()) == cache_key:
                pnl = z["pnl"]
                y = z["y"]
                t_exit = z["t_exit"]
                t_qual = z["t_qual"]
                tp_hits = z["tp_hits"]
                loaded_cache = True
                print(f"[prefilter-cache] loaded labels from {args.label_cache_npz}")
        except Exception as e:
            print(f"[prefilter-cache] failed to load cache: {e}")
    t0 = time.perf_counter()
    if not loaded_cache:
        pnl, y, t_exit, t_qual, tp_hits = _simulate_multitp_trailing_askbid_m1(
            ask_entry_latency=askbid_m1_arrays["ask_entry_latency"],
            bid_high_after_entry_latency=askbid_m1_arrays["bid_high_after_entry_latency"],
            bid_low_after_entry_latency=askbid_m1_arrays["bid_low_after_entry_latency"],
            bid_high=askbid_m1_arrays["bid_high"],
            bid_low=askbid_m1_arrays["bid_low"],
            bid_close=askbid_m1_arrays["bid_close"],
            tps=tps,
            tp_w=tp_w,
            tp_enabled=bool(tp_enabled),
            sl=float(args.sl),
            hold=int(args.hold),
            slippage_bps=float(args.slippage_bps),
            trail=bool(args.trail),
            trail_activate=float(args.trail_activate),
            trail_offset=float(args.trail_offset),
            trail_factor=float(args.trail_factor),
            include_unrealized_at_test_end=bool(args.include_unrealized_at_test_end),
            period_end_indices=np.where(np.arange(n) < train_idx, train_idx - 1, n - 1).astype(np.int64),
        )
        if args.label_cache_npz is not None:
            args.label_cache_npz.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                args.label_cache_npz,
                cache_key=np.asarray(cache_key),
                pnl=pnl,
                y=y,
                t_exit=t_exit,
                t_qual=t_qual,
                tp_hits=tp_hits,
            )
            print(f"[prefilter-cache] wrote labels cache to {args.label_cache_npz}")
    timing["simulate_labels_sec"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    cols = miner.build_candidate_features(df, allow_absolute_price=False, max_features=0)
    timing["build_candidate_features_sec"] = time.perf_counter() - t0
    atr_cols_available = sum(1 for c in cols if _is_atr_candidate_col(c))
    atr_cols_sample = [str(c) for c in cols if _is_atr_candidate_col(c)][:30]
    y_train = y[:train_idx]
    y_test = y[train_idx:]
    tradable_train = ((y_train == 0) | (y_train == 1))
    tradable_test = ((y_test == 0) | (y_test == 1))
    score_y = np.asarray(y).copy()
    score_t_exit = np.asarray(t_exit).copy()
    score_pnl = np.asarray(pnl).copy()
    score_tp_hits = np.asarray(tp_hits).copy()
    score_t_qual = np.asarray(t_qual).copy()
    score_tick_source_map: dict[int, dict] = {}
    period_end_indices = np.where(np.arange(n) < train_idx, train_idx - 1, n - 1).astype(np.int64)

    def _exit_abs_idx(entry_idx: int, exit_rel: int) -> int:
        i = int(entry_idx)
        if i < 0 or i >= n:
            return i
        if int(exit_rel) > 0:
            return min(n - 1, i + int(exit_rel))
        period_end = int(period_end_indices[i]) if i < len(period_end_indices) else n - 1
        period_end = max(i, min(period_end, n - 1))
        return min(period_end, i + max(1, int(args.hold)))

    def _select_entries_for_mask(
        raw_mask: np.ndarray,
        y_eval: np.ndarray,
        t_exit_eval: np.ndarray,
        only_lower_entry: bool | None = None,
    ) -> tuple[np.ndarray, int, int]:
        use_lower = bool(args.cluster_only_lower_entry) if only_lower_entry is None else bool(only_lower_entry)
        eval_y = np.asarray(y_eval)
        eval_exit = np.asarray(t_exit_eval)
        raw = np.asarray(raw_mask, dtype=bool)
        evaluable = raw & ((eval_y == 0) | (eval_y == 1))
        raw_idxs = np.flatnonzero(evaluable)
        selected = np.zeros(n, dtype=bool)
        if raw_idxs.size == 0:
            return selected, 0, 0

        gap_ns = int(args.cluster_gap_minutes) * 60 * 1_000_000_000
        open_trades: list[int] = []
        prev_idx: int | None = None
        entries_in_cluster = 0
        cluster_min_entry_price = float("nan")
        selected_cluster_ids: set[int] = set()
        cluster_id = -1

        for idx_i in raw_idxs.tolist():
            i = int(idx_i)
            new_cluster = prev_idx is None
            if prev_idx is not None:
                if bar_time_ns is not None and len(bar_time_ns) > i and len(bar_time_ns) > prev_idx:
                    new_cluster = (int(bar_time_ns[i]) - int(bar_time_ns[prev_idx])) > gap_ns
                else:
                    new_cluster = (i - int(prev_idx)) > int(args.cluster_gap_minutes)
            if new_cluster:
                cluster_id += 1
                entries_in_cluster = 0
                cluster_min_entry_price = float("nan")
            prev_idx = i

            open_trades = [x for x in open_trades if int(x) > i]
            if entries_in_cluster >= int(args.max_entries_per_cluster):
                continue
            entry_price_i = float(selection_entry_price[i]) if i < len(selection_entry_price) else float("nan")
            if use_lower:
                if not np.isfinite(entry_price_i):
                    continue
                if entries_in_cluster > 0:
                    if (not np.isfinite(cluster_min_entry_price)) or not (entry_price_i < cluster_min_entry_price):
                        continue
            if len(open_trades) >= int(args.max_open_trades):
                continue

            exit_idx = _exit_abs_idx(i, int(eval_exit[i]) if i < len(eval_exit) else -1)
            selected[i] = True
            entries_in_cluster += 1
            selected_cluster_ids.add(int(cluster_id))
            if np.isfinite(entry_price_i):
                cluster_min_entry_price = entry_price_i if not np.isfinite(cluster_min_entry_price) else min(cluster_min_entry_price, entry_price_i)
            open_trades.append(int(exit_idx))
        return selected, int(raw_idxs.size), int(len(selected_cluster_ids))

    ranking_prep_t0 = time.perf_counter()
    allow_keys, allow_keys_refined = _load_allowlist(args.include_candidates_file)
    qs = [float(x) for x in str(args.quantiles).split(",") if x.strip()]
    coarse_ctx = {
        "features_sig": _file_sig(args.features),
        "price_basis": "askbid_m1",
        "askbid_m1_sig": _file_sig(args.askbid_m1_parquet),
        "entry_latency_ms": int(args.entry_latency_ms),
        "askbid_m1_semantics": ASKBID_M1_SEMANTICS,
        "binned_sig": _file_sig(args.binned_features),
        "meta_sig": _file_sig(args.binned_metadata),
        "label_cache_key": cache_key,
        "train_frac": float(args.train_frac),
        "quantiles": qs,
        "family_split_delta_window": int(bool(args.family_split_delta_window)),
        "entry_selection_semantics_version": "cluster_gap_lower_maxopen_v1",
        "cluster_gap_minutes": int(args.cluster_gap_minutes),
        "max_entries_per_cluster": int(args.max_entries_per_cluster),
        "max_open_trades": int(args.max_open_trades),
        "cluster_only_lower_entry": int(bool(args.cluster_only_lower_entry)),
        "candidate_inventory_semantics_version": CANDIDATE_CACHE_SCHEMA_VERSION,
    }
    coarse_ctx_sig = _ctx_sig(coarse_ctx)
    coarse_out = args.out_candidates_coarse_csv
    refined_out = args.out_candidates_refined_csv
    coarse_csv_load_t0 = time.perf_counter()
    coarse_resume = _load_stage_csv_if_match(
        coarse_out,
        "coarse",
        coarse_ctx_sig,
        CANDIDATE_CACHE_SCHEMA_VERSION,
        rebuild_legacy=True,
    )
    timing_detail2["coarse_csv_load_sec"] += time.perf_counter() - coarse_csv_load_t0
    items = []
    inventory_mask_items: list[dict] = []
    candidate_inventory_rows: list[dict] = []
    single_rejects = {"min_pos": 0, "min_lift": 0, "mask_count": 0, "allowlist": 0}
    if coarse_resume is not None:
        rows_loaded_coarse = int(len(coarse_resume))
        stable_seen: set[str] = set()
        for _, r in coarse_resume.iterrows():
            key = str(r.get("candidate_key", "")).strip()
            col = str(r["col"]); op = str(r["op"]); val = _canonicalize_candidate_value(col, op, float(r["value"]))
            stable_k = _stable_candidate_key(col, op, val)
            if not key:
                key = stable_k
            if not stable_k or stable_k in stable_seen:
                raise ValueError(f"Invalid coarse CSV {coarse_out}: duplicate/empty stable_candidate_key={stable_k!r}; file was not modified")
            stable_seen.add(stable_k)
            pos_hits = int(r.get("coarse_single_pos_hits", 0)); neg_hits = int(r.get("coarse_single_neg_hits", 0))
            mask_count = int(r.get("coarse_single_mask_count", pos_hits + neg_hits))
            ratio = float(r.get("coarse_single_ratio", pos_hits / max(1, neg_hits)))
            lift = float(r.get("coarse_lift", 0.0))
            candidate_inventory_rows.append({
                "candidate_key": key, "stable_candidate_key": stable_k, "col": col, "op": op, "value": val,
                "binary": bool(int(r.get("binary", 0))), "_family": str(r.get("family", _candidate_family(col, bool(args.family_split_delta_window)))),
                "_single_pos_hits": pos_hits, "_single_neg_hits": neg_hits, "_single_mask_count": mask_count,
                "_single_ratio": ratio, "coarse_single_pos_hits": pos_hits, "coarse_single_neg_hits": neg_hits,
                "coarse_single_mask_count": mask_count, "coarse_single_ratio": ratio,
                "coarse_single_mask_keep_ratio": float(r.get("coarse_single_mask_keep_ratio", np.nan)),
                "coarse_single_ratio_change": float(r.get("coarse_single_ratio_change", np.nan)),
                "coarse_lift": lift, "lift": lift, "ratio": ratio,
            })
        print(f"[prefilter-resume] rows_loaded_coarse={rows_loaded_coarse}")
        print(f"[prefilter-resume] loaded complete coarse inventory from {coarse_out} rows={len(candidate_inventory_rows)}")
    else:
        t0 = time.perf_counter()
        qmap = miner.quantile_thresholds(df.iloc[:train_idx], cols, qs)
        timing["quantile_thresholds_sec"] = time.perf_counter() - t0
        timing_detail2["quantile_thresholds_sec"] = timing["quantile_thresholds_sec"]
        t0 = time.perf_counter()
        for c in cols:
            if c.startswith("dist_"):
                x = pd.to_numeric(df[c], errors="coerce").to_numpy(copy=False)
                for _, thr in qmap.get(c, {}).items():
                    for op in (">=", "<="):
                        m = np.isfinite(x) & ((x >= thr) if op == ">=" else (x <= thr))
                        m_sel, _, _ = _select_entries_for_mask(m, y, t_exit)
                        pos = int(np.sum(m_sel[:train_idx] & (y_train == 1)))
                        neg = int(np.sum(m_sel[:train_idx] & (y_train == 0)))
                        if pos <= 0:
                            continue
                        items.append({"col": c, "op": op, "value": float(thr), "mask": m, "binary": False,
                                      "freq": pos / max(1, int(np.sum(y_train == 1))),
                                      "lift": (pos / max(1, int(np.sum(y_train == 1)))) / max(1e-12, (neg / max(1, int(np.sum(y_train == 0))))),
                                      "ratio": pos / max(1, neg)})
                continue
            miss = int(meta.get(c, {}).get("missing_code", 0))
            b = pd.to_numeric(bdf[c], errors="coerce").fillna(miss).to_numpy()
            vals = np.unique(b[:train_idx]); vals = vals[vals != miss]
            is_binary = str(meta.get(c, {}).get("feature_type", "")) == "binary"
            for v in vals.tolist():
                m = np.isfinite(b) & (np.abs(b - float(v)) <= 1e-6)
                m_sel, _, _ = _select_entries_for_mask(m, y, t_exit)
                pos = int(np.sum(m_sel[:train_idx] & (y_train == 1))); neg = int(np.sum(m_sel[:train_idx] & (y_train == 0)))
                if pos <= 0:
                    continue
                items.append({"col": c, "op": "==", "value": float(v), "mask": m, "binary": bool(is_binary),
                              "freq": pos / max(1, int(np.sum(y_train == 1))),
                              "lift": (pos / max(1, int(np.sum(y_train == 1)))) / max(1e-12, (neg / max(1, int(np.sum(y_train == 0))))),
                              "ratio": pos / max(1, neg)})
        build_items_candidate_loop_elapsed = time.perf_counter() - t0
        timing_detail2["build_items_candidate_loop_sec"] += build_items_candidate_loop_elapsed
        build_items_metadata_t0 = time.perf_counter()
        build_items_mask_before = float(timing_detail2["build_items_mask_sec"])
        for i, it in enumerate(items, start=1):
            x = dict(it); x["candidate_key"] = f"cand_{i:06d}"
            x["stable_candidate_key"] = _stable_candidate_key(str(x["col"]), str(x["op"]), _canonicalize_candidate_value(str(x["col"]), str(x["op"]), float(x["value"])))
            build_items_mask_t0 = time.perf_counter()
            m = np.asarray(x["mask"], dtype=bool)
            raw_train_m = m[:train_idx]
            coarse_raw_pos = int(np.sum(raw_train_m & (y_train == 1)))
            coarse_raw_neg = int(np.sum(raw_train_m & (y_train == 0)))
            coarse_raw_mask_count = int(coarse_raw_pos + coarse_raw_neg)
            coarse_raw_ratio = coarse_raw_pos / max(1, coarse_raw_neg)
            m_sel, _, _ = _select_entries_for_mask(m, y, t_exit)
            timing_detail2["build_items_mask_sec"] += time.perf_counter() - build_items_mask_t0
            pos_hits = int(np.sum(m_sel[:train_idx] & (y_train == 1))); neg_hits = int(np.sum(m_sel[:train_idx] & (y_train == 0)))
            mask_count = int(pos_hits + neg_hits)
            x["_single_pos_hits"] = pos_hits; x["_single_neg_hits"] = neg_hits; x["_single_mask_count"] = mask_count
            x["_single_ratio"] = pos_hits / max(1, neg_hits)
            x["coarse_single_pos_hits"] = int(pos_hits); x["coarse_single_neg_hits"] = int(neg_hits)
            x["coarse_single_mask_count"] = int(mask_count); x["coarse_single_ratio"] = float(x["_single_ratio"]); x["coarse_lift"] = float(x["lift"])
            x["coarse_single_mask_keep_ratio"] = _safe_ratio_or_nan(mask_count, coarse_raw_mask_count)
            x["coarse_single_ratio_change"] = _safe_ratio_or_nan(x["_single_ratio"], coarse_raw_ratio)
            x["_family"] = _candidate_family(str(x["col"]), bool(args.family_split_delta_window))
            inventory_mask_items.append(x)
        build_items_metadata_elapsed = time.perf_counter() - build_items_metadata_t0
        build_items_mask_elapsed = float(timing_detail2["build_items_mask_sec"]) - build_items_mask_before
        timing_detail2["build_items_metadata_sec"] += max(0.0, build_items_metadata_elapsed - build_items_mask_elapsed)
        filtered_items_build_t0 = time.perf_counter()
        by_mask: dict[str, dict] = {}
        for it in inventory_mask_items:
            mh = _mask_hash_arr(np.asarray(it["mask"][:train_idx], dtype=bool))
            cur = by_mask.get(mh)
            if cur is None:
                by_mask[mh] = it; continue
            tf_cur = int(miner._parse_feature_meta(str(cur["col"])).get("tf") or 10**9)
            tf_new = int(miner._parse_feature_meta(str(it["col"])).get("tf") or 10**9)
            if tf_new < tf_cur:
                by_mask[mh] = it
        inventory_mask_items = list(by_mask.values())
        timing_detail2["filtered_items_sec"] += time.perf_counter() - filtered_items_build_t0
        for it in inventory_mask_items:
            candidate_inventory_rows.append({k: v for k, v in it.items() if k != "mask"})

    atr_debug_source = list(items) if items else list(candidate_inventory_rows)
    def _atr_debug_pass_counts(source_items: list[dict]) -> tuple[int, int, int, int]:
        atr_items = [it for it in source_items if _is_atr_candidate_col(str(it.get("col", "")))]
        after_pos = []
        after_lift = []
        after_mask = []
        for it in atr_items:
            mask_obj = it.get("mask")
            fallback_pos = int(np.sum(np.asarray(mask_obj, dtype=bool)[:train_idx] & (y_train == 1))) if mask_obj is not None else 0
            fallback_count = int(np.sum(np.asarray(mask_obj, dtype=bool)[:train_idx])) if mask_obj is not None else 0
            pos_hits = int(it.get("coarse_single_pos_hits", it.get("_single_pos_hits", fallback_pos)))
            mask_count = int(it.get("coarse_single_mask_count", it.get("_single_mask_count", fallback_count)))
            lift_v = float(it.get("coarse_lift", it.get("lift", 0.0)))
            if pos_hits >= int(args.min_single_pos_hits):
                after_pos.append(it)
                if lift_v >= float(args.min_single_lift):
                    after_lift.append(it)
                    if int(args.max_single_mask_count) <= 0 or mask_count <= int(args.max_single_mask_count):
                        after_mask.append(it)
        return len(atr_items), len(after_pos), len(after_lift), len(after_mask)
    atr_candidates_built, atr_candidates_after_min_pos, atr_candidates_after_min_lift, atr_candidates_after_mask_count = _atr_debug_pass_counts(atr_debug_source)
    filtered_items_build_t0 = time.perf_counter()
    allowlist_filter_elapsed = 0.0
    before = len(candidate_inventory_rows)
    after_pos = _filter_candidate_inventory_rows(candidate_inventory_rows, args.min_single_pos_hits, 0.0, 0, None)
    single_rejects["min_pos"] = before - len(after_pos)
    after_lift = _filter_candidate_inventory_rows(after_pos, 0, args.min_single_lift, 0, None)
    single_rejects["min_lift"] = len(after_pos) - len(after_lift)
    after_mask = _filter_candidate_inventory_rows(after_lift, 0, 0.0, args.max_single_mask_count, None)
    single_rejects["mask_count"] = len(after_lift) - len(after_mask)
    allowlist_filter_t0 = time.perf_counter()
    filtered_items = _filter_candidate_inventory_rows(after_mask, 0, 0.0, 0, allow_keys)
    single_rejects["allowlist"] = len(after_mask) - len(filtered_items)
    allowlist_filter_elapsed = time.perf_counter() - allowlist_filter_t0
    timing_detail2["allowlist_filter_sec"] += allowlist_filter_elapsed
    print(f"[prefilter-resume] rows_after_allowlist={len(filtered_items)}")
    print(f"[prefilter-candidates] candidate_inventory_rows={len(candidate_inventory_rows)}")
    print(f"[prefilter-candidates] runtime_single_scope_rows={len(filtered_items)}")
    filtered_elapsed = max(0.0, time.perf_counter() - filtered_items_build_t0 - allowlist_filter_elapsed)
    timing_detail2["filtered_items_sec"] += filtered_elapsed
    fam_top_build_t0 = time.perf_counter()
    fam_top = _family_top_rows(filtered_items, int(args.family_top_n))
    print(f"[prefilter-candidates] current_family_top_rows={len(fam_top)}")
    family_top_elapsed = time.perf_counter() - fam_top_build_t0
    timing_detail2["family_top_sec"] += family_top_elapsed
    askbid_tick_replay_enabled = args.tick_entry_cache_npz is not None
    replay_entry_index_set: set[int] = set()
    tick_refined_mode = False
    refined_ctx = {
        **coarse_ctx,
        "tick_data_sig": _file_sig(args.tick_data),
        "tp_mode": str(tp_mode),
        "trail": int(bool(args.trail)),
        "trail_activate": float(args.trail_activate),
        "hold": int(args.hold),
        "tick_datetime_column": str(args.tick_datetime_column).strip().lower(),
        "tick_sep": str(args.tick_sep),
        "price_basis": "askbid_m1",
        "askbid_m1_sig": _file_sig(args.askbid_m1_parquet),
        "askbid_m1_semantics": ASKBID_M1_SEMANTICS,
        "entry_latency_ms": int(args.entry_latency_ms),
        "tick_replay_price_basis": "askbid_m1",
        "critical_minutes_semantics": "askbid_m1_bounds_order_ambiguity_tp_weights_trail_stop_final_stop_c1_v1",
        "tick_replay_semantics": "chronological_bid_ticks_from_entry_latency_v1",
        "tick_entry_cache_semantics_version": TICK_ENTRY_CACHE_SEMANTICS_VERSION,
        "refined_candidate_metrics_semantics_version": REFINED_CACHE_SCHEMA_VERSION,
    }
    refined_ctx_sig = _ctx_sig(refined_ctx)
    tick_entry_cache_ctx = {
        **refined_ctx,
        "features_sig": _file_sig(args.features),
        "tps": [float(x) for x in tps],
        "tp_weights": [float(x) for x in np.asarray(tp_w, dtype=np.float64).tolist()],
        "use_multi_tp": int(bool(args.use_multi_tp)),
        "no_tp": int(bool(args.no_tp)),
        "sl": float(args.sl),
        "slippage_bps": float(args.slippage_bps),
        "spread_bps": float(args.spread_bps),
        "trail_offset": float(args.trail_offset),
        "trail_factor": float(args.trail_factor),
        "trail_min_level": float(args.trail_min_level),
        "include_unrealized_at_test_end": int(bool(args.include_unrealized_at_test_end)),
        "train_idx": int(train_idx),
        "n_rows": int(n),
        "period_end_semantics": "train_test_split_period_end_v1",
        "price_basis": "askbid_m1",
        "askbid_m1_sig": _file_sig(args.askbid_m1_parquet),
        "askbid_m1_semantics": ASKBID_M1_SEMANTICS,
        "entry_latency_ms": int(args.entry_latency_ms),
        "entry_indexing": "entry_i_to_minute_i_plus_1",
        "entry_price_basis": "ask_entry_latency_plus_slippage",
        "exit_price_basis": "bid_ticks",
        "endpoint_price_basis": "bid_close",
        "critical_minutes_semantics": "askbid_m1_bounds_order_ambiguity_tp_weights_trail_stop_final_stop_c1_v1",
        "tick_replay_semantics": "chronological_bid_ticks_from_entry_latency_v1",
        "spread_handling": "ask_contains_spread_spread_bps_not_added",
        "slippage_handling": "ask_entry_latency_times_1_plus_slippage_bps",
        "tick_entry_cache_semantics_version": TICK_ENTRY_CACHE_SEMANTICS_VERSION,
    }
    tick_entry_cache_sig = _ctx_sig(tick_entry_cache_ctx)
    refined_csv_load_t0 = time.perf_counter()
    refined_resume = _load_stage_csv_if_match(
        refined_out,
        "refined",
        refined_ctx_sig,
        REFINED_CACHE_SCHEMA_VERSION,
        rebuild_legacy=False,
    )
    timing_detail2["refined_csv_load_sec"] += time.perf_counter() - refined_csv_load_t0
    tick_scope_stable_keys = _tick_scope_keys(str(args.tick_refine_scope), candidate_inventory_rows, filtered_items, fam_top)
    inventory_by_stable = {str(it["stable_candidate_key"]): it for it in candidate_inventory_rows}
    mask_required_keys = _mask_required_keys(tick_scope_stable_keys, askbid_tick_replay_enabled)
    fresh_masks = {
        str(it["stable_candidate_key"]): it["mask"]
        for it in inventory_mask_items
        if str(it["stable_candidate_key"]) in mask_required_keys
    }

    def _reconstruct_candidate_mask(it: dict) -> np.ndarray:
        stable_k = str(it["stable_candidate_key"])
        if stable_k in fresh_masks:
            return np.asarray(fresh_masks[stable_k], dtype=bool)
        col = str(it["col"]); op = str(it["op"]); val = float(it["value"])
        xvec = pd.to_numeric(bdf[col] if op == "==" else df[col], errors="coerce").to_numpy(copy=False)
        if op == "==":
            return np.isfinite(xvec) & (np.abs(xvec - val) <= 1e-6)
        if op == ">=":
            return np.isfinite(xvec) & (xvec >= val)
        return np.isfinite(xvec) & (xvec <= val)

    coarse_reconstruct_t0 = time.perf_counter()
    for stable_k in mask_required_keys:
        inventory_by_stable[stable_k]["mask"] = _reconstruct_candidate_mask(inventory_by_stable[stable_k])
    timing_detail2["coarse_reconstruct_sec"] += time.perf_counter() - coarse_reconstruct_t0
    print(f"[prefilter-resume] rows_reconstructed={len(mask_required_keys)}")
    items_total_count = len(items) if items else len(candidate_inventory_rows)
    dist_items_count = sum(1 for x in (items if items else candidate_inventory_rows) if str(x["col"]).startswith("dist_"))
    binary_items_count = sum(1 for x in (items if items else candidate_inventory_rows) if bool(x.get("binary", False)))
    fresh_masks.clear()
    inventory_mask_items.clear()
    items.clear()
    atr_debug_source.clear()
    if coarse_resume is None:
        by_mask.clear()
        x = m = raw_train_m = m_sel = cur = None
    gc.collect()
    tick_scope_items = [it for it in candidate_inventory_rows if str(it["stable_candidate_key"]) in tick_scope_stable_keys]
    print(f"[prefilter-resume] rows_runtime_scope={len(tick_scope_items)} scope={args.tick_refine_scope}")
    all_by_key = {str(it["candidate_key"]): it for it in candidate_inventory_rows}
    req_tick_cols = list(REQUIRED_TICK_METRIC_COLUMNS)

    refined_rows_total = int(len(refined_resume)) if refined_resume is not None else 0
    refined_rows_with_tick = 0
    refined_rows_missing_tick = refined_rows_total
    if refined_resume is not None:
        print(f"[prefilter-resume] rows_loaded_refined={int(len(refined_resume))}")
        by_key = {str(it["candidate_key"]): it for it in tick_scope_items}
        refined_rows_missing_tick = 0
        for _, r in refined_resume.iterrows():
            refined_key_t0 = time.perf_counter()
            k = str(r.get("candidate_key_refined", "")).strip()
            col_r = str(r.get("col", ""))
            op_r = str(r.get("op", ""))
            val_r = float(r.get("value", np.nan)) if pd.notna(r.get("value", np.nan)) else np.nan
            stable_k = ""
            if col_r and op_r and np.isfinite(val_r):
                stable_k = _stable_candidate_key(col_r, op_r, _canonicalize_candidate_value(col_r, op_r, val_r))
            it = all_by_key.get(k)
            if it is None and stable_k:
                it = inventory_by_stable.get(stable_k)
            timing_detail2["refined_key_match_sec"] += time.perf_counter() - refined_key_t0
            if it is None:
                continue
            refined_restore_t0 = time.perf_counter()
            row_values = {c: r.get(c, np.nan) for c in req_tick_cols}
            if not _has_full_tick_metrics(row_values):
                timing_detail2["refined_tick_metric_restore_sec"] += time.perf_counter() - refined_restore_t0
                refined_rows_missing_tick += 1
                continue
            refined_rows_with_tick += 1
            it["tick_single_pos_hits"] = int(r.get("tick_single_pos_hits", 0))
            it["tick_single_neg_hits"] = int(r.get("tick_single_neg_hits", 0))
            it["tick_single_mask_count"] = int(r.get("tick_single_mask_count", 0))
            it["tick_single_ratio"] = float(r.get("tick_single_ratio", 0.0))
            it["tick_single_mask_keep_ratio"] = float(r.get("tick_single_mask_keep_ratio", np.nan))
            it["tick_single_ratio_change"] = float(r.get("tick_single_ratio_change", np.nan))
            if str(it.get("candidate_key")) in by_key:
                it["_single_pos_hits"] = int(it["tick_single_pos_hits"])
                it["_single_neg_hits"] = int(it["tick_single_neg_hits"])
                it["_single_mask_count"] = int(it["tick_single_mask_count"])
                it["_single_ratio"] = float(it["tick_single_ratio"])
                it["ratio"] = float(it["tick_single_ratio"])
            timing_detail2["refined_tick_metric_restore_sec"] += time.perf_counter() - refined_restore_t0
        tick_refined_mode = any(_has_full_tick_metrics(x) for x in by_key.values())
        print(f"[prefilter-resume] loaded refined candidates from {refined_out}")
        print(f"[prefilter-resume] refined_csv_rows_total={refined_rows_total}")
        print(f"[prefilter-resume] refined_rows_with_stored_tick_metrics={refined_rows_with_tick}")
        print(f"[prefilter-resume] refined_rows_missing_tick_metrics={refined_rows_missing_tick}")
        print(f"[prefilter-resume] requested_allowlist_keys={len(allow_keys) if allow_keys is not None else 0}")
        print(f"[prefilter-resume] usable_refined_keys_for_fam_top={sum(1 for it in fam_top if 'tick_single_ratio' in it)}")
        print(f"[prefilter-resume] usable_refined_keys_for_tick_scope={sum(1 for it in tick_scope_items if 'tick_single_ratio' in it)}")
    elif bool(allow_keys_refined):
        if refined_out is None or not refined_out.exists():
            raise ValueError("candidate_key_refined input requires --out-candidates-refined-csv file to reload refined metrics")
        raise ValueError("Refined TXT header provided, but refined candidate CSV context did not match current run.")
    else:
        print("[prefilter-resume] refined_csv_rows_total=0")
        print("[prefilter-resume] refined_rows_with_stored_tick_metrics=0")
    missing_metrics_scan_t0 = time.perf_counter()
    tick_full_items, tick_missing_items = _partition_tick_metric_rows(tick_scope_items)
    timing_detail["tick_missing_metrics_scan_sec"] += time.perf_counter() - missing_metrics_scan_t0
    current_tick_scope_full_before = len(tick_full_items)
    print(f"[prefilter-tick] current_tick_scope_rows={len(tick_scope_items)}")
    print(f"[prefilter-tick] current_tick_scope_full_before={current_tick_scope_full_before}")
    print(f"[prefilter-tick] current_tick_scope_missing_before={len(tick_missing_items)}")
    if askbid_tick_replay_enabled and len(tick_scope_items) > 0:
        print(f"[prefilter-tick] tick_refine_scope={args.tick_refine_scope}")
        print(f"[prefilter-tick] tick_refine_candidates_count={len(tick_scope_items)}")
        print(f"[prefilter-tick] tick_refine_missing_to_compute={len(tick_missing_items)}")
        tick_scope_t0 = time.perf_counter()
        fam_union = np.zeros(n, dtype=bool)
        for it in tick_scope_items:
            fam_union |= np.asarray(it["mask"], dtype=bool)
        entry_indices = np.flatnonzero(fam_union).astype(np.int64, copy=False)
        tick_period_end_indices = np.where(np.arange(n) < train_idx, train_idx - 1, n - 1).astype(np.int64)
        timing_detail["tick_scope_build_sec"] += time.perf_counter() - tick_scope_t0
        timing_counts["tick_scope_entries_count"] = int(len(entry_indices))
        tick_replay_plan_t0 = time.perf_counter()
        critical_by_entry_all = _critical_minutes_for_entries_askbid_m1(
            entry_indices=entry_indices,
            askbid_m1_arrays=askbid_m1_arrays,
            selection_entry_price=selection_entry_price,
            bar_time_ns=bar_time_ns,
            hold=int(args.hold),
            tp_mode=str(tp_mode),
            tp_enabled=bool(tp_enabled),
            trail=bool(args.trail),
            trail_activate=float(args.trail_activate),
            trail_offset=float(args.trail_offset),
            trail_factor=float(args.trail_factor),
            tps=np.asarray(tps, dtype=np.float64),
            tp_w=np.asarray(tp_w, dtype=np.float64),
            sl=float(args.sl),
            period_end_indices=tick_period_end_indices,
        )
        replay_entry_indices = np.asarray(sorted(critical_by_entry_all.keys()), dtype=np.int64)
        replay_entry_index_set = {int(i) for i in replay_entry_indices.tolist()}
        critical_minutes_all = {int(m) for mins in critical_by_entry_all.values() for m in mins}
        timing_detail["tick_replay_plan_sec"] += time.perf_counter() - tick_replay_plan_t0
        timing_counts["tick_replay_relevant_entries_count"] = int(len(replay_entry_indices))
        timing_counts["tick_replay_critical_minutes_count"] = int(len(critical_minutes_all))
        print(f"[prefilter-tick-replay] replay_relevant_entries={len(replay_entry_indices)}")
        print(f"[prefilter-tick-replay] critical_minutes_count={len(critical_minutes_all)}")
        if bool(args.debug_tick_cache_scope_counts):
            entry_minutes = {int(bar_time_ns[int(i) + 1]) for i in replay_entry_indices.tolist() if int(i) + 1 < n}
            print(f"[prefilter-tick-cache] scope_entry_minutes_count={len(entry_minutes)}")
            print(f"[prefilter-tick-cache] scope_critical_minutes_count={len(critical_minutes_all)}")
            print(f"[prefilter-tick-cache] scope_minutes_would_load_count={len(critical_minutes_all)}")
            print("[prefilter-tick-cache] scope_counts_source=askbid_m1_replay_no_raw_tick_load")

        cached_entries: dict[int, dict] = {}
        tick_map: dict[int, dict] = {}
        if replay_entry_indices.size > 0:
            cache_load_t0 = time.perf_counter()
            cached_entries = _load_tick_entry_cache(args.tick_entry_cache_npz, tick_entry_cache_sig) if args.tick_entry_cache_npz is not None else {}
            timing_detail["tick_cache_load_sec"] += time.perf_counter() - cache_load_t0
            cached_entries = cached_entries if cached_entries is not None else {}
            timing_counts["tick_cache_loaded_entries_count"] = int(len(cached_entries))
            cached_present = {int(k) for k in cached_entries.keys() if int(k) in replay_entry_index_set}
            tick_map = {int(i): dict(cached_entries[int(i)]) for i in cached_present if int(i) in cached_entries}
            missing_entry_indices = np.asarray([int(i) for i in replay_entry_indices.tolist() if int(i) not in cached_present], dtype=np.int64)
            timing_counts["tick_replay_cache_hits_count"] = int(len(cached_present))
            timing_counts["tick_replay_cache_missing_entries_count"] = int(len(missing_entry_indices))
            print(f"[prefilter-tick-cache] requested_entries={len(replay_entry_indices)} cached_entries_used={len(cached_present)} missing_entries={len(missing_entry_indices)}")
            if missing_entry_indices.size > 0:
                if args.tick_data is None:
                    raise ValueError(
                        "--tick-entry-cache-npz enables AskBid-M1 critical TickReplay, but replay-relevant entries are missing from cache; "
                        "--tick-data is required to compute them."
                    )
                missing_critical_minutes = {int(m) for i in missing_entry_indices.tolist() for m in critical_by_entry_all.get(int(i), set())}
                timing_counts["tick_replay_raw_critical_minutes_requested_count"] = int(len(missing_critical_minutes))
                if missing_critical_minutes:
                    raw_load_t0 = time.perf_counter()
                    tick_time_ns_all, tick_minute_bounds, tick_bids_all, tick_asks_all, matched_total_minutes_count = _load_askbid_tick_replay_minutes(
                        path=args.tick_data,
                        datetime_col=args.tick_datetime_column,
                        sep=args.tick_sep,
                        minute_filter=missing_critical_minutes,
                        tick_chunk_size=args.tick_chunk_size,
                    )
                    timing_detail["tick_raw_load_sec"] += time.perf_counter() - raw_load_t0
                    timing_flags["tick_raw_load_skipped"] = False
                    missing_minutes = sorted(int(m) for m in missing_critical_minutes if int(m) not in tick_minute_bounds)
                    print(f"[prefilter-tick-raw-load] critical_minutes_count={len(missing_critical_minutes)}")
                    timing_counts["tick_replay_raw_loaded_minutes_count"] = int(matched_total_minutes_count)
                    print(f"[prefilter-tick-raw-load] matched_total_minutes_count={int(matched_total_minutes_count)}")
                    print(f"[prefilter-tick-raw-load] used_real_ticks_source={'parquet_ticks' if args.tick_data.suffix.lower() in {'.parquet', '.pq'} else 'raw_ticks'}")
                    if missing_minutes:
                        sample = ",".join(str(x) for x in missing_minutes[:10])
                        raise ValueError(
                            "AskBid-M1 critical TickReplay requires raw bid ticks for every critical minute; "
                            f"missing_critical_minutes_count={len(missing_minutes)} sample={sample}"
                        )
                    tick_sim_t0 = time.perf_counter()
                    tick_map_new = _simulate_selected_entries_with_askbid_ticks(
                        entry_indices=missing_entry_indices,
                        critical_by_entry=critical_by_entry_all,
                        askbid_m1_arrays=askbid_m1_arrays,
                        selection_entry_price=selection_entry_price,
                        bar_time_ns=bar_time_ns,
                        tick_time_ns_all=tick_time_ns_all,
                        tick_minute_bounds=tick_minute_bounds,
                        tick_bids_all=tick_bids_all,
                        tps=tps,
                        tp_w=tp_w,
                        tp_enabled=bool(tp_enabled),
                        sl=float(args.sl),
                        hold=int(args.hold),
                        trail=bool(args.trail),
                        trail_activate=float(args.trail_activate),
                        trail_offset=float(args.trail_offset),
                        trail_factor=float(args.trail_factor),
                        include_unrealized_at_test_end=bool(args.include_unrealized_at_test_end),
                        period_end_indices=tick_period_end_indices,
                        entry_latency_ms=int(args.entry_latency_ms),
                    )
                    timing_detail["tick_simulate_selected_entries_sec"] += time.perf_counter() - tick_sim_t0
                    timing_flags["tick_simulation_skipped"] = False
                    for idx_i, rec in tick_map_new.items():
                        tick_map[int(idx_i)] = dict(rec)
                        cached_entries[int(idx_i)] = dict(rec)
                    del tick_time_ns_all, tick_minute_bounds, tick_bids_all, tick_asks_all
                    gc.collect()
                    if args.tick_entry_cache_npz is not None:
                        cache_write_t0 = time.perf_counter()
                        _write_tick_entry_cache(args.tick_entry_cache_npz, tick_entry_cache_sig, cached_entries)
                        timing_detail["tick_cache_write_sec"] += time.perf_counter() - cache_write_t0
                        timing_flags["tick_cache_write_skipped"] = False
                else:
                    print("[prefilter-tick-replay] no critical minutes for missing replay entries; raw tick load skipped")
            else:
                print("[prefilter-tick-cache] tick_data_source=entry_cache_only")
                print("[prefilter-tick-cache] raw_tick_load_skipped=True reason=entry_cache_complete")
                print("[prefilter-tick-raw-load] skipped=True reason=entry_cache_complete")

            score_tick_source_map = tick_map
            if tick_map or cached_entries:
                diag_t0 = time.perf_counter()
                _print_tick_cache_diagnostics(cached_entries, tick_map, replay_entry_indices, y, t_exit, int(args.hold))
                timing_detail["tick_cache_diagnostics_sec"] += time.perf_counter() - diag_t0
                timing_counts["tick_cache_diagnostics_entries_count"] = int(len(tick_map) if tick_map else len(cached_entries))
                timing_flags["tick_cache_diagnostics_skipped"] = False
            if tick_map:
                candidate_metrics_t0 = time.perf_counter()
                y_ref = np.asarray(y, dtype=np.int8).copy()
                t_exit_ref = np.asarray(t_exit, dtype=np.int32).copy()
                for idx_i, rec in tick_map.items():
                    if int(idx_i) in replay_entry_index_set:
                        y_ref[int(idx_i)] = np.int8(int(rec.get("y", -1)))
                        t_exit_ref[int(idx_i)] = np.int32(int(rec.get("t_exit", -1)))
                y_ref_train = y_ref[:train_idx]
                metrics_complete = True
                for it in tick_scope_items:
                    if _has_full_tick_metrics(it):
                        continue
                    raw_m = np.asarray(it["mask"], dtype=bool)
                    raw_tick_m = raw_m[:train_idx] & tradable_train & ((y_ref_train == 0) | (y_ref_train == 1))
                    tick_raw_pos = int(np.sum(raw_tick_m & (y_ref_train == 1)))
                    tick_raw_neg = int(np.sum(raw_tick_m & (y_ref_train == 0)))
                    tick_raw_mask_count = int(tick_raw_pos + tick_raw_neg)
                    tick_raw_ratio = tick_raw_pos / max(1, tick_raw_neg)
                    selected_m, raw_evaluable, _ = _select_entries_for_mask(raw_m, y_ref, t_exit_ref)
                    valid_m = selected_m[:train_idx] & tradable_train & ((y_ref_train == 0) | (y_ref_train == 1))
                    requested_entries = sum(1 for idx_i in np.flatnonzero(raw_m).tolist() if int(idx_i) in replay_entry_index_set)
                    cached_for_item = sum(1 for idx_i in np.flatnonzero(raw_m).tolist() if int(idx_i) in tick_map and int(idx_i) in replay_entry_index_set)
                    if cached_for_item < requested_entries:
                        metrics_complete = False
                        continue
                    tick_pos = int(np.sum(valid_m & (y_ref_train == 1)))
                    tick_neg = int(np.sum(valid_m & (y_ref_train == 0)))
                    tick_ratio = tick_pos / max(1, tick_neg)
                    it["tick_single_pos_hits"] = tick_pos
                    it["tick_single_neg_hits"] = tick_neg
                    tick_mask_count = int(tick_pos + tick_neg)
                    it["tick_single_mask_count"] = tick_mask_count
                    it["tick_single_ratio"] = float(tick_ratio)
                    it["tick_single_mask_keep_ratio"] = _safe_ratio_or_nan(tick_mask_count, tick_raw_mask_count)
                    it["tick_single_ratio_change"] = _safe_ratio_or_nan(tick_ratio, tick_raw_ratio)
                    it["_single_pos_hits"] = tick_pos
                    it["_single_neg_hits"] = tick_neg
                    it["_single_mask_count"] = tick_mask_count
                    it["_single_ratio"] = float(tick_ratio)
                    it["ratio"] = float(tick_ratio)
                tick_refined_mode = any(_has_full_tick_metrics(it) for it in tick_scope_items)
                if not metrics_complete:
                    print("[prefilter-tick-cache] warning: some replay-relevant current-scope entries are missing tick cache results; affected candidates remain missing.")
                timing_detail["tick_candidate_metrics_sec"] += time.perf_counter() - candidate_metrics_t0
                print(f"[prefilter] askbid-tick-replayed {args.tick_refine_scope} scope on {len(replay_entry_indices)} replay-relevant entry rows.")
        else:
            print("[prefilter-tick-replay] no replay-relevant entries; AskBid-M1 M1 score basis remains unchanged")
    elif askbid_tick_replay_enabled:
        print(f"[prefilter-tick] skipped: tick_refine_scope={args.tick_refine_scope} produced empty candidate scope")
    wrote_coarse = False
    wrote_refined = False
    coarse_reason = "not_requested"
    refined_reason = "not_requested"
    if coarse_out is not None:
        coarse_csv_t0 = time.perf_counter()
        fam_top_keys = {str(z.get("candidate_key")) for z in fam_top}
        coarse_rows = []
        for it in candidate_inventory_rows:
            stable_k = str(it.get("stable_candidate_key", "")).strip() or _stable_candidate_key(str(it["col"]), str(it["op"]), _canonicalize_candidate_value(str(it["col"]), str(it["op"]), float(it["value"])))
            row = {
                "candidate_key": str(it["candidate_key"]),
                "stable_candidate_key": stable_k,
                "col": str(it["col"]),
                "op": str(it["op"]),
                "value": _canonicalize_candidate_value(str(it["col"]), str(it["op"]), float(it["value"])),
                "family": str(it["_family"]),
                "coarse_single_pos_hits": int(it.get("coarse_single_pos_hits", it.get("_single_pos_hits", 0))),
                "coarse_single_neg_hits": int(it.get("coarse_single_neg_hits", it.get("_single_neg_hits", 0))),
                "coarse_single_mask_count": int(it.get("coarse_single_mask_count", it.get("_single_mask_count", 0))),
                "coarse_single_ratio": float(it.get("coarse_single_ratio", it.get("_single_ratio", it.get("ratio", 0.0)))),
                "coarse_single_mask_keep_ratio": float(it.get("coarse_single_mask_keep_ratio", np.nan)),
                "coarse_single_ratio_change": float(it.get("coarse_single_ratio_change", np.nan)),
                "coarse_lift": float(it.get("coarse_lift", it.get("lift", 0.0))),
                "binary": int(bool(it.get("binary", False))),
                "kept_after_family_topn": int(str(it["candidate_key"]) in fam_top_keys),
                "__stage": "coarse",
                "__schema_version": CANDIDATE_CACHE_SCHEMA_VERSION,
                "__ctx_sig": coarse_ctx_sig,
            }
            coarse_rows.append(row)
        coarse_cols = ["candidate_key", "stable_candidate_key", "col", "op", "value", "family", "coarse_single_pos_hits", "coarse_single_neg_hits", "coarse_single_mask_count", "coarse_single_ratio", "coarse_single_mask_keep_ratio", "coarse_single_ratio_change", "coarse_lift", "binary", "kept_after_family_topn", "__stage", "__schema_version", "__ctx_sig"]
        coarse_frame = _ordered_frame(coarse_rows, coarse_cols).drop(columns=["coarse_single_raw_mask_count", "coarse_single_raw_ratio", "tick_single_raw_mask_count", "tick_single_raw_ratio"], errors="ignore")
        wrote_coarse, coarse_reason = _write_csv_if_changed(coarse_out, coarse_frame, key_cols=["stable_candidate_key"])
        coarse_rows_prev = int(len(coarse_resume)) if coarse_resume is not None else 0
        coarse_rows_after = int(len(coarse_rows))
        coarse_rows_delta = int(coarse_rows_after - coarse_rows_prev)
        print(f"[prefilter-candidates] coarse_write={'done' if wrote_coarse else 'skipped'} reason_for_write={coarse_reason}")
        print(f"[prefilter-candidates] rows_existing_before={coarse_rows_prev} rows_existing_after={coarse_rows_after} rows_delta={coarse_rows_delta}")
        if wrote_coarse:
            print(f"[prefilter-candidates] wrote coarse CSV: {coarse_out} rows={len(coarse_rows)}")
        timing_detail["candidate_csv_write_sec"] += time.perf_counter() - coarse_csv_t0

    if bool(args.debug_atr_candidates):
        atr_candidates_written_to_csv = sum(1 for it in candidate_inventory_rows if _is_atr_candidate_col(str(it.get("col", ""))))
        print(
            "[prefilter-atr-debug] "
            f"atr_cols_available={atr_cols_available} "
            f"atr_candidates_built={atr_candidates_built} "
            f"atr_candidates_after_min_pos={atr_candidates_after_min_pos} "
            f"atr_candidates_after_min_lift={atr_candidates_after_min_lift} "
            f"atr_candidates_after_mask_count={atr_candidates_after_mask_count} "
            f"atr_candidates_written_to_csv={atr_candidates_written_to_csv}"
        )
        if atr_cols_sample:
            print("[prefilter-atr-debug] atr_cols_sample=" + ",".join(atr_cols_sample))

    phase_d_pool_t0 = time.perf_counter()
    phase_d_pool_base, _ = _partition_tick_metric_rows(tick_scope_items)
    timing_detail["phase_d_pool_build_sec"] += time.perf_counter() - phase_d_pool_t0

    if refined_out is not None:
        refined_csv_t0 = time.perf_counter()
        fam_top_keys = {str(z.get("candidate_key")) for z in fam_top}

        def _refined_row_key(row: dict) -> str:
            try:
                col_r = str(row.get("col", ""))
                op_r = str(row.get("op", ""))
                val_r = _canonicalize_candidate_value(col_r, op_r, float(row.get("value", np.nan)))
                if col_r and op_r and np.isfinite(val_r):
                    return _stable_candidate_key(col_r, op_r, val_r)
            except Exception:
                pass
            stable_k = str(row.get("stable_candidate_key", "")).strip()
            if stable_k:
                return stable_k
            return str(row.get("candidate_key_refined", row.get("candidate_key", ""))).strip()

        existing_rows: dict[str, dict] = {}
        if refined_resume is not None:
            for _, r in refined_resume.iterrows():
                stable_k = _refined_row_key({k: r.get(k) for k in r.index})
                if not stable_k or stable_k not in inventory_by_stable:
                    continue
                row = {k: r.get(k) for k in r.index}
                row["stable_candidate_key"] = stable_k
                row["tick_metric_status"] = str(row.get("tick_metric_status", "out_of_scope") or "out_of_scope")
                row["__ctx_sig"] = refined_ctx_sig
                row["__stage"] = "refined"
                row["__schema_version"] = REFINED_CACHE_SCHEMA_VERSION
                row["tick_metric_status"] = "full" if _has_full_tick_metrics(row) else row["tick_metric_status"]
                existing_rows[stable_k] = _prefer_refined_row(
                    existing_rows.get(stable_k), row, refined_ctx_sig, REFINED_CACHE_SCHEMA_VERSION
                )
        for it in candidate_inventory_rows:
            stable_k = _stable_candidate_key(str(it["col"]), str(it["op"]), _canonicalize_candidate_value(str(it["col"]), str(it["op"]), float(it["value"])))
            row = existing_rows.get(stable_k, {})
            row.update({
                "candidate_key_refined": str(it["candidate_key"]),
                "stable_candidate_key": stable_k,
                "col": str(it["col"]),
                "op": str(it["op"]),
                "value": _canonicalize_candidate_value(str(it["col"]), str(it["op"]), float(it["value"])),
                "family": str(it["_family"]),
                "coarse_single_pos_hits": int(it.get("coarse_single_pos_hits", it.get("_single_pos_hits", 0))),
                "coarse_single_neg_hits": int(it.get("coarse_single_neg_hits", it.get("_single_neg_hits", 0))),
                "coarse_single_mask_count": int(it.get("coarse_single_mask_count", it.get("_single_mask_count", 0))),
                "coarse_single_ratio": float(it.get("coarse_single_ratio", it.get("_single_ratio", it.get("ratio", 0.0)))),
                "coarse_single_mask_keep_ratio": float(it.get("coarse_single_mask_keep_ratio", np.nan)),
                "coarse_single_ratio_change": float(it.get("coarse_single_ratio_change", np.nan)),
                "coarse_lift": float(it.get("coarse_lift", it.get("lift", 0.0))),
                "binary": int(bool(it.get("binary", False))),
                "kept_after_family_topn": int(str(it["candidate_key"]) in fam_top_keys),
                "__stage": "refined",
                "__schema_version": REFINED_CACHE_SCHEMA_VERSION,
                "__ctx_sig": refined_ctx_sig,
            })
            for col_tick in req_tick_cols:
                if col_tick in it and np.isfinite(float(it.get(col_tick, np.nan))):
                    row[col_tick] = float(it[col_tick])
                else:
                    row.setdefault(col_tick, np.nan)
            if _has_full_tick_metrics(row):
                row["tick_metric_status"] = "full"
            elif stable_k in tick_scope_stable_keys:
                row["tick_metric_status"] = "missing"
            else:
                row["tick_metric_status"] = "out_of_scope"
            existing_rows[stable_k] = _prefer_refined_row(
                existing_rows.get(stable_k), row, refined_ctx_sig, REFINED_CACHE_SCHEMA_VERSION
            )
        deduped_rows: dict[str, dict] = {}
        for row in existing_rows.values():
            k = _refined_row_key(row)
            if not k:
                continue
            deduped_rows[k] = _prefer_refined_row(
                deduped_rows.get(k), row, refined_ctx_sig, REFINED_CACHE_SCHEMA_VERSION
            )
        cand_rows = _refined_rows_for_inventory(candidate_inventory_rows, deduped_rows)
        refined_cols = ["candidate_key_refined", "stable_candidate_key", "col", "op", "value", "family", "coarse_single_pos_hits", "coarse_single_neg_hits", "coarse_single_mask_count", "coarse_single_ratio", "coarse_single_mask_keep_ratio", "coarse_single_ratio_change", "coarse_lift", "binary", "kept_after_family_topn", "tick_single_pos_hits", "tick_single_neg_hits", "tick_single_mask_count", "tick_single_ratio", "tick_single_mask_keep_ratio", "tick_single_ratio_change", "tick_metric_status", "__stage", "__schema_version", "__ctx_sig"]
        refined_frame = _ordered_frame(cand_rows, refined_cols).drop(columns=["coarse_single_raw_mask_count", "coarse_single_raw_ratio", "tick_single_raw_mask_count", "tick_single_raw_ratio"], errors="ignore")
        prev_refined_rows = int(len(refined_resume)) if refined_resume is not None else 0
        wrote_refined, refined_reason = _write_csv_if_changed(refined_out, refined_frame, key_cols=["stable_candidate_key"])
        refined_rows_after = int(len(cand_rows))
        refined_rows_delta = int(refined_rows_after - prev_refined_rows)
        print(f"[prefilter-candidates] refined_write={'done' if wrote_refined else 'skipped'} reason_for_write={refined_reason}")
        print(f"[prefilter-candidates] rows_existing_before={prev_refined_rows} rows_existing_after={refined_rows_after} rows_delta={refined_rows_delta}")
        if wrote_refined:
            print(f"[prefilter-candidates] wrote refined CSV: {refined_out} rows={len(cand_rows)}")
        timing_detail["candidate_csv_write_sec"] += time.perf_counter() - refined_csv_t0

    tick_candidates_requested = len(tick_scope_items)
    tick_candidates_with_metrics = sum(1 for it in tick_scope_items if _has_full_tick_metrics(it))
    tick_candidates_missing_metrics = max(0, tick_candidates_requested - tick_candidates_with_metrics)
    tick_refined_mode = bool(tick_refined_mode or len(phase_d_pool_base) > 0)
    print(f"[prefilter-tick] tick_candidates_requested={tick_candidates_requested}")
    print(f"[prefilter-tick] tick_candidates_with_metrics={tick_candidates_with_metrics}")
    print(f"[prefilter-tick] tick_candidates_missing_metrics={tick_candidates_missing_metrics}")
    print(f"[prefilter-tick] current_tick_scope_full_after={tick_candidates_with_metrics}")
    print(f"[prefilter-tick] current_tick_scope_missing_after={tick_candidates_missing_metrics}")
    refinement_state = _refinement_state(
        len(tick_scope_items),
        askbid_tick_replay_enabled,
        tick_candidates_missing_metrics,
        timing_counts["tick_replay_relevant_entries_count"],
    )
    print(f"[prefilter-tick] refinement_state={refinement_state}")
    print(f"[prefilter-tick] tick_candidates_used_for_phase_d={len(phase_d_pool_base)}")
    score_tick_overrides = 0
    score_tick_invalid = 0
    for idx_i, rec in score_tick_source_map.items():
        try:
            ii = int(idx_i)
            tick_y = int(rec.get("y", -1))
            tick_pnl = float(rec.get("pnl", float("nan")))
            tick_t_exit = int(rec.get("t_exit", -1))
            tick_tp_hits = int(rec.get("tp_hits", -1))
            if "t_qual" not in rec:
                raise ValueError("missing tick_t_qual")
            tick_t_qual = int(rec.get("t_qual", -2))
        except Exception:
            score_tick_invalid += 1
            continue
        tick_record_valid = (
            0 <= ii < n
            and tick_y in (0, 1)
            and np.isfinite(tick_pnl)
            and 1 <= tick_t_exit <= int(args.hold)
            and tick_tp_hits >= 0
            and -1 <= tick_t_qual <= int(args.hold)
        )
        if not tick_record_valid:
            score_tick_invalid += 1
            continue
        score_y[ii] = tick_y
        score_pnl[ii] = tick_pnl
        score_t_exit[ii] = tick_t_exit
        score_tp_hits[ii] = tick_tp_hits
        score_t_qual[ii] = tick_t_qual
        score_tick_overrides += 1
    score_y_train = score_y[:train_idx]
    score_y_test = score_y[train_idx:]
    score_tradable_train = ((score_y_train == 0) | (score_y_train == 1))
    score_tradable_test = ((score_y_test == 0) | (score_y_test == 1))
    score_basis = "mixed_askbid_m1_tick_replay" if score_tick_overrides > 0 else "askbid_m1"
    score_tick_override_share = float(score_tick_overrides / max(1, n))
    print(
        "[prefilter-score-basis] "
        f"score_basis={score_basis} "
        f"score_tick_overrides={score_tick_overrides} "
        f"score_tick_invalid={score_tick_invalid} "
        f"score_tick_override_share={score_tick_override_share:.6f}"
    )

    released_single_masks = 0
    for it in candidate_inventory_rows:
        if "mask" in it:
            it.pop("mask", None)
            released_single_masks += 1
    gc.collect()
    print(f"[prefilter-candidates] released_single_candidate_masks={released_single_masks}")

    rank_sort_t0 = time.perf_counter()
    if tick_refined_mode:
        rank_lift = _with_rank_context(sorted(fam_top, key=lambda z: (-float(z.get("ratio", 0.0)), -int(z.get("_single_pos_hits", 0)), int(z.get("_single_mask_count", 0)))), "ratio")
    else:
        rank_lift = _with_rank_context(sorted(fam_top, key=lambda z: z["lift"], reverse=True), "lift")
    rank_freq: list[dict] = []
    rank_ratio: list[dict] = []
    timing_detail["ranking_sort_sec"] += time.perf_counter() - rank_sort_t0
    search_setup_t0 = time.perf_counter()
    workers = max(1, int(os.cpu_count() or 1)) if str(args.workers).lower() == "auto" else max(1, int(args.workers))

    best_paths: list[dict] = []
    unlocked = 0
    prev_best = -np.inf
    timing_detail2["search_setup_sec"] += time.perf_counter() - search_setup_t0
    same_ref_t0 = time.perf_counter()
    same_ref = miner._build_same_reference_groups(df.iloc[:train_idx], [(str(c), ">=", 0.0) for c in cols])
    timing_detail2["same_reference_groups_sec"] += time.perf_counter() - same_ref_t0
    search_setup_t0 = time.perf_counter()
    reject_stats: dict[str, int] = {
        "rejected_pre_min_pos_per_week": 0,
        "rejected_min_pos_per_week": 0,
        "rejected_min_main_score": 0,
        "rejected_same_parent_mask": 0,
        "rejected_not_strictly_better_than_parent": 0,
        "rejected_same_reference": 0,
        "rejected_bundle_anchor": 0,
        "rejected_binary_cap": 0,
        "rejected_duplicate_mask": 0,
    }
    timing_detail2["search_setup_sec"] += time.perf_counter() - search_setup_t0

    def _build_mask_from_conds(conds: list[dict]) -> np.ndarray:
        mask = np.ones(n, dtype=bool)
        for c in conds:
            col = str(c["col"])
            op = str(c["op"])
            val = float(c["value"])
            domain = str(c.get("domain", "auto"))
            ftype = str(meta.get(col, {}).get("feature_type", "unknown"))
            use_binned = (domain == "binned") or (domain == "auto" and ftype in {"binary", "discrete", "continuous"})
            if use_binned and op in {"==", ">=", "<="}:
                x = pd.to_numeric(bdf[col], errors="coerce").to_numpy(copy=False)
            else:
                x = pd.to_numeric(df[col], errors="coerce").to_numpy(copy=False)
            if op == "==":
                if "lo_bin" in c and "hi_bin" in c:
                    lo_bin = float(c["lo_bin"])
                    hi_bin = float(c["hi_bin"])
                    mask &= np.isfinite(x) & (x >= lo_bin - 1e-6) & (x <= hi_bin + 1e-6)
                else:
                    mask &= np.isfinite(x) & (np.abs(x - val) <= 1e-6)
            elif op == ">=":
                mask &= np.isfinite(x) & (x >= val)
            elif op == "<=":
                mask &= np.isfinite(x) & (x <= val)
            else:
                return np.zeros(n, dtype=bool)
        return mask

    def _score_from_mask(mask: np.ndarray, only_lower_entry: bool | None = None, enforce_filters: bool = True) -> dict | None:
        days = max(1.0, float((df.index[train_idx - 1] - df.index[0]).total_seconds() / 86400.0))
        weeks = days / 7.0
        if enforce_filters:
            raw = np.asarray(mask, dtype=bool)
            raw_train = raw[:train_idx] & score_tradable_train
            raw_pos_hits = int(np.sum(raw_train & (score_y_train == 1)))
            if raw_pos_hits < float(args.min_pos_per_week) * weeks:
                if bool(args.debug_reject_stats):
                    reject_stats["rejected_pre_min_pos_per_week"] += 1
                return None
        selected_mask, raw_evaluable, clusters_count = _select_entries_for_mask(mask, score_y, score_t_exit, only_lower_entry=only_lower_entry)
        mt = selected_mask[:train_idx] & score_tradable_train
        pos_hits = int(np.sum(mt & (score_y_train == 1)))
        neg_hits = int(np.sum(mt & (score_y_train == 0)))
        if enforce_filters and pos_hits < float(args.min_pos_per_week) * weeks:
            if bool(args.debug_reject_stats):
                reject_stats["rejected_min_pos_per_week"] += 1
            return None
        ratio = pos_hits / max(1, neg_hits)
        if enforce_filters and ratio < float(args.min_main_score):
            if bool(args.debug_reject_stats):
                reject_stats["rejected_min_main_score"] += 1
            return None
        mt_test = selected_mask[train_idx:] & score_tradable_test
        pos_test = int(np.sum(mt_test & (score_y_test == 1)))
        neg_test = int(np.sum(mt_test & (score_y_test == 0)))
        ratio_test = pos_test / max(1, neg_test)
        wf_mean, wf_min, wf_hits = _calc_wf(mt_test, score_y_test, int(args.wf_folds))
        precision = pos_hits / max(1, (pos_hits + neg_hits))
        selected_evaluable = int(pos_hits + neg_hits + pos_test + neg_test)
        return {
            "pos_hits": pos_hits,
            "neg_hits": neg_hits,
            "ratio": float(ratio),
            "precision": float(precision),
            "test_pos_hits": pos_test,
            "test_neg_hits": neg_test,
            "test_ratio": float(ratio_test),
            "wf_mean_ratio": wf_mean,
            "wf_min_ratio": wf_min,
            "wf_hits": wf_hits,
            "selection_ratio": float(selected_evaluable / raw_evaluable) if raw_evaluable > 0 else 0.0,
            "clusters_count": int(clusters_count),
        }

    def _train_key(r: dict) -> tuple[float, int, int]:
        return (float(r["ratio"]), int(r["pos_hits"]), -int(r["neg_hits"]))

    def _test_key(r: dict) -> tuple[float, int, int]:
        return (float(r["test_ratio"]), int(r["test_pos_hits"]), -int(r["test_neg_hits"]))

    def _mask_hash(mask: np.ndarray) -> str:
        return _mask_hash_arr(mask)

    def evaluate_combo(combo: Tuple[int, ...], pool_items: List[dict], parent: dict | None = None, source: str = "A") -> dict | None:
        conds = [pool_items[i] for i in combo]
        dedup = []
        seen_gid = set()
        for c in conds:
            fam = str(miner._parse_feature_meta(str(c["col"])).get("family", ""))
            if fam in {"dist_support", "dist_resist"}:
                gid = int(same_ref.get(str(c["col"]), 0))
                if gid > 0 and gid in seen_gid:
                    if bool(args.debug_reject_stats):
                        reject_stats["rejected_same_reference"] += 1
                    continue
                if gid > 0:
                    seen_gid.add(gid)
            dedup.append(c)
        conds = dedup
        # Binary flood cap per unlocked block
        if sum(1 for c in conds if c["binary"]) > int(args.binary_cap_per_block):
            if bool(args.debug_reject_stats):
                reject_stats["rejected_binary_cap"] += 1
            return None
        # bundle/anchor validity
        cond_triplets = [(str(c["col"]), str(c["op"]), float(c["value"])) for c in conds]
        ok_bundle, _ = miner.validate_binary_anchor_invariant(cond_triplets, cols)
        if not ok_bundle:
            if bool(args.debug_reject_stats):
                reject_stats["rejected_bundle_anchor"] += 1
            return None

        mask = _build_mask_from_conds(
            [{"col": c["col"], "op": c["op"], "value": c["value"], "domain": "auto"} for c in conds]
        )
        sc = _score_from_mask(mask)
        if sc is None:
            return None
        mh = _mask_hash(mask)
        if parent is not None:
            if str(parent.get("_mask_hash", "")) == mh:
                if bool(args.debug_reject_stats):
                    reject_stats["rejected_same_parent_mask"] += 1
                return None
            if _train_key(sc) <= _train_key(parent):
                if bool(args.debug_reject_stats):
                    reject_stats["rejected_not_strictly_better_than_parent"] += 1
                return None
        return {
            "conds": [{"col": str(c["col"]), "op": str(c["op"]), "value": float(c["value"]), "domain": "auto"} for c in conds],
            "_combo": tuple(sorted(int(x) for x in combo)),
            "_mask_hash": mh,
            "search_source": str(source),
            **sc,
        }

    csv_columns = [
        "path_index", "rule_human", "rule_json_id", "decode_type_info", "decode_bin_info", "pos_hits", "neg_hits",
        "remaining_hit_ratio", "precision_info", "test_pos_hits", "test_neg_hits", "test_ratio",
        "wf_mean_ratio", "wf_min_ratio", "wf_hits", "selection_ratio", "clusters_count",
        *( ["le_off_train_ratio", "le_off_test_ratio", "le_off_total_hits"] if bool(args.cluster_only_lower_entry) else [] ),
        "tp", "sl", "hold", "trail",
        "trail_activate", "trail_offset", "trail_factor", "search_source", "is_fallback_export",
    ]

    def _decode_cond_struct(c: dict) -> dict:
        col = str(c.get("col", ""))
        op = str(c.get("op", ""))
        m = meta.get(col, {}) if isinstance(meta, dict) else {}
        ftype = str(m.get("feature_type", "unknown"))
        out = {"feature_type": ftype, "lo_bin": None, "hi_bin": None, "raw_lo": None, "raw_hi": None, "raw_values": None, "bin_range_text": ""}
        if op != "==":
            return out
        try:
            lo = int(c.get("lo_bin", round(float(c.get("value", 0.0)))))
            hi = int(c.get("hi_bin", lo))
        except Exception:
            return out
        missing = int(m.get("missing_code", 0) or 0)
        eff = int(m.get("effective_bin_count", 0) or 0)
        if lo == missing or hi == missing or lo < 1 or hi < lo or (eff > 0 and hi > eff):
            return out
        out["lo_bin"] = int(lo)
        out["hi_bin"] = int(hi)
        out["bin_range_text"] = f"bin[{lo}]" if lo == hi else f"bin[{lo}..{hi}]"
        bin_to_raw = m.get("bin_to_raw", {})
        if ftype in {"binary", "discrete"} and isinstance(bin_to_raw, dict) and bin_to_raw:
            def _raw_for_bin(b: int):
                v = bin_to_raw.get(str(b), bin_to_raw.get(int(b)))
                if v is None:
                    return None
                try:
                    return float(v)
                except Exception:
                    return v

            raw_values = [_raw_for_bin(b) for b in range(lo, hi + 1)]
            raw_values = [v for v in raw_values if v is not None]
            out["raw_values"] = raw_values if raw_values else None
            out["raw_lo"] = _raw_for_bin(lo)
            out["raw_hi"] = _raw_for_bin(hi)
            return out

        edges = m.get("bin_edges", [])
        if isinstance(edges, list) and len(edges) >= hi:
            lo_pair = edges[lo - 1]
            hi_pair = edges[hi - 1]
            if isinstance(lo_pair, list) and isinstance(hi_pair, list) and len(lo_pair) == 2 and len(hi_pair) == 2:
                out["raw_lo"] = float(lo_pair[0])
                out["raw_hi"] = float(hi_pair[1])
        return out

    def _attach_decode_info(rule: dict) -> dict:
        rr = dict(rule)
        conds = []
        decodes = []
        for c in rr.get("conds", []):
            cc = dict(c)
            dec = _decode_cond_struct(cc)
            cc["decode"] = dec
            conds.append(cc)
            decodes.append(dec)
        rr["conds"] = conds
        rr["decode_type_info_structured"] = decodes
        return rr

    def _rules_to_rows(rules: list[dict], is_fallback_export: bool = False) -> list[dict]:
        def _decode_interval(col: str, lo: int, hi: int) -> str:
            m = meta.get(col, {}) if isinstance(meta, dict) else {}
            edges = m.get("bin_edges", [])
            eff = int(m.get("effective_bin_count", 0) or 0)
            if not isinstance(edges, list) or eff <= 0 or lo < 1 or hi > eff or lo > hi:
                raise ValueError(f"Undecodable interval for {col}: lo={lo} hi={hi} eff={eff}")
            lo_pair = edges[lo - 1]
            hi_pair = edges[hi - 1]
            if not (isinstance(lo_pair, list) and isinstance(hi_pair, list) and len(lo_pair) == 2 and len(hi_pair) == 2):
                raise ValueError(f"Invalid bin_edges for {col}")
            lo_raw = float(lo_pair[0])
            hi_raw = float(hi_pair[1])
            return f"{col} in [{lo_raw:.6g}, {hi_raw:.6g}] (bin[{lo}..{hi}])"

        rows: list[dict] = []
        for i, r in enumerate(rules, start=1):
            decoded_parts = []
            type_parts = []
            bin_parts = []
            for c in r["conds"]:
                col = str(c["col"])
                op = str(c["op"])
                val = float(c["value"])
                ftype = str(meta.get(col, {}).get("feature_type", "unknown"))
                if op == "==":
                    if "lo_bin" in c and "hi_bin" in c and int(c["lo_bin"]) != int(c["hi_bin"]):
                        lo_bin = int(c["lo_bin"])
                        hi_bin = int(c["hi_bin"])
                        decoded_parts.append(_decode_interval(col, lo_bin, hi_bin))
                    else:
                        decoded_parts.append(miner.decode_bin_condition(col, val, meta))
                else:
                    decoded_parts.append(f"{col} {op} {val:.6g}")
                dec = c.get("decode") if isinstance(c.get("decode"), dict) else _decode_cond_struct(c)
                if dec.get("bin_range_text"):
                    raw_lo = dec.get("raw_lo")
                    raw_hi = dec.get("raw_hi")
                    raw_txt = "" if raw_lo is None or raw_hi is None else f" raw[{float(raw_lo):.6g}..{float(raw_hi):.6g}]"
                    type_parts.append(f"{col}:{ftype} {dec.get('bin_range_text')}{raw_txt}")
                    bin_parts.append(json.dumps({"col": col, **dec}, ensure_ascii=False, sort_keys=True))
                else:
                    type_parts.append(f"{col}:{ftype}")
            row = {
                "path_index": i,
                "rule_human": " & ".join(decoded_parts),
                "rule_json_id": f"rule_{i}",
                "decode_type_info": " | ".join(type_parts),
                "decode_bin_info": " | ".join(bin_parts),
                "pos_hits": r["pos_hits"],
                "neg_hits": r["neg_hits"],
                "remaining_hit_ratio": r["ratio"],
                "precision_info": r["precision"],
                "test_pos_hits": r["test_pos_hits"],
                "test_neg_hits": r["test_neg_hits"],
                "test_ratio": r["test_ratio"],
                "wf_mean_ratio": r["wf_mean_ratio"],
                "wf_min_ratio": r["wf_min_ratio"],
                "wf_hits": r["wf_hits"],
                "selection_ratio": float(r.get("selection_ratio", 0.0)),
                "clusters_count": int(r.get("clusters_count", 0)),
                "tp": ",".join([f"{x:.8g}" for x in tps]),
                "sl": float(args.sl),
                "hold": int(args.hold),
                "trail": int(args.trail),
                "trail_activate": float(args.trail_activate),
                "trail_offset": float(args.trail_offset),
                "trail_factor": float(args.trail_factor),
                "search_source": str(r.get("search_source", "")),
                "is_fallback_export": int(bool(is_fallback_export)),
            }
            if bool(args.cluster_only_lower_entry):
                row["le_off_train_ratio"] = float(r.get("le_off_train_ratio", np.nan))
                row["le_off_test_ratio"] = float(r.get("le_off_test_ratio", np.nan))
                row["le_off_total_hits"] = int(r.get("le_off_total_hits", 0))
            rows.append(row)
        return rows

    def _validate_rows(rows: list[dict]) -> None:
        if not rows:
            return
        keys0 = set(rows[0].keys())
        for i, r in enumerate(rows, start=1):
            if set(r.keys()) != keys0:
                raise ValueError(f"CSV row key mismatch at row {i}")
            txt = str(r.get("rule_human", ""))
            if txt.count("(") != txt.count(")"):
                raise ValueError(f"rule_human parentheses mismatch at row {i}: {txt!r}")

    def _dedupe_mask(rows: list[dict]) -> list[dict]:
        buckets: dict[tuple[int, int, int, int], dict[str, dict]] = {}
        for r in rows:
            key = (int(r["pos_hits"]), int(r["neg_hits"]), int(r["test_pos_hits"]), int(r["test_neg_hits"]))
            buckets.setdefault(key, {})
            mh = str(r.get("_mask_hash", ""))
            cur = buckets[key].get(mh)
            if cur is None:
                buckets[key][mh] = r
                continue
            if bool(args.debug_reject_stats):
                reject_stats["rejected_duplicate_mask"] += 1
            if len(r.get("conds", [])) < len(cur.get("conds", [])):
                buckets[key][mh] = r
            elif len(r.get("conds", [])) == len(cur.get("conds", [])):
                if _test_key(r) > _test_key(cur) or (_test_key(r) == _test_key(cur) and _train_key(r) > _train_key(cur)):
                    buckets[key][mh] = r
        out: list[dict] = []
        for m in buckets.values():
            out.extend(m.values())
        return out

    def _save_progress(valid_pool: list[dict], state: dict) -> None:
        non_d_rows = [r for r in valid_pool if str(r.get("search_source", "")) != "D"]
        d_rows = [r for r in valid_pool if str(r.get("search_source", "")) == "D"]
        non_d_rows = sorted(_dedupe_mask(non_d_rows), key=lambda z: (-float(z["ratio"]), -int(z["pos_hits"]), int(z["neg_hits"])))[: int(args.top_paths)]
        d_rows = sorted(_dedupe_mask(d_rows), key=lambda z: (-float(z["ratio"]), -int(z["pos_hits"]), int(z["neg_hits"])))[: int(args.top_paths)]
        export_subset = [_attach_decode_info(r) for r in (non_d_rows + d_rows)]
        rules_only = {"version": 2, "rules": export_subset}
        if args.out_rules_json is not None:
            _atomic_write_text(args.out_rules_json, json.dumps(rules_only, ensure_ascii=False, indent=2))
        rows = _rules_to_rows(export_subset, is_fallback_export=False)
        _validate_rows(rows)
        if rows:
            _atomic_write_csv(args.out_rules_csv, pd.DataFrame(rows))
        else:
            _atomic_write_csv(args.out_rules_csv, pd.DataFrame(columns=csv_columns))

    def _log_phase_transition(from_phase: str, to_phase: str, reason: str, **fields) -> None:
        parts = [f"from={from_phase}", f"to={to_phase}", f"reason={reason}"]
        for k, v in fields.items():
            if v is None:
                continue
            parts.append(f"{k}={v}")
        print("[phase-transition] " + " ".join(parts))

    rng = random.Random(int(args.batch_random_seed))
    valid_pool: list[dict] = []
    progress_state: dict = {}
    phase_c_was_active = False
    original_early_stop_window_combos = int(args.early_stop_window_combos)
    stop_after_batch_requested = False
    force_phase_b_requested = False
    phase_b_start_reason = "max_valids_reached"
    completed_phase_b_pool_keys: set[tuple[str, ...]] = set()

    def _phase_b_pool_key(pool_items: list[dict]) -> tuple[str, ...]:
        keys: list[str] = []
        for it in pool_items:
            stable_k = str(it.get("stable_candidate_key", "")).strip()
            if not stable_k:
                try:
                    stable_k = _stable_candidate_key(str(it.get("col", "")), str(it.get("op", "")), _canonicalize_candidate_value(str(it.get("col", "")), str(it.get("op", "")), float(it.get("value", np.nan))))
                except Exception:
                    stable_k = ""
            keys.append(stable_k or str(it.get("candidate_key", "")))
        return tuple(keys)

    timing["ranking_prep_sec"] = time.perf_counter() - ranking_prep_t0
    ranking_prep_accounted_sec = (
        float(timing_detail2.get("coarse_csv_load_sec", 0.0))
        + float(timing_detail2.get("coarse_rows_filter_sec", 0.0))
        + float(timing_detail2.get("coarse_reconstruct_sec", 0.0))
        + float(timing_detail2.get("quantile_thresholds_sec", 0.0))
        + float(timing_detail2.get("build_items_candidate_loop_sec", 0.0))
        + float(timing_detail2.get("build_items_mask_sec", 0.0))
        + float(timing_detail2.get("build_items_metadata_sec", 0.0))
        + float(timing_detail2.get("filtered_items_sec", 0.0))
        + float(timing_detail2.get("allowlist_filter_sec", 0.0))
        + float(timing_detail2.get("family_top_sec", 0.0))
        + float(timing_detail2.get("refined_csv_load_sec", 0.0))
        + float(timing_detail2.get("refined_key_match_sec", 0.0))
        + float(timing_detail2.get("refined_tick_metric_restore_sec", 0.0))
        + float(timing_detail.get("tick_cache_load_sec", 0.0))
        + float(timing_detail.get("tick_scope_build_sec", 0.0))
        + float(timing_detail.get("tick_replay_plan_sec", 0.0))
        + float(timing_detail.get("tick_missing_metrics_scan_sec", 0.0))
        + float(timing_detail.get("tick_raw_load_sec", 0.0))
        + float(timing_detail.get("tick_simulate_selected_entries_sec", 0.0))
        + float(timing_detail.get("tick_cache_write_sec", 0.0))
        + float(timing_detail.get("tick_cache_diagnostics_sec", 0.0))
        + float(timing_detail.get("tick_candidate_metrics_sec", 0.0))
        + float(timing_detail.get("candidate_csv_write_sec", 0.0))
        + float(timing_detail.get("phase_d_pool_build_sec", 0.0))
        + float(timing_detail.get("ranking_sort_sec", 0.0))
        + float(timing_detail2.get("same_reference_groups_sec", 0.0))
        + float(timing_detail2.get("search_setup_sec", 0.0))
    )
    timing_detail["ranking_prep_accounted_sec"] = ranking_prep_accounted_sec
    timing_detail["ranking_prep_unaccounted_sec"] = float(timing.get("ranking_prep_sec", 0.0)) - ranking_prep_accounted_sec

    if bool(args.debug_timing_breakdown):
        print("[prefilter-timing] " + " | ".join([f"{k}={v:.3f}s" for k, v in timing.items()]))
        print(
            "[prefilter-timing-detail] "
            f"tick_cache_load_sec={timing_detail['tick_cache_load_sec']:.3f} "
            f"tick_cache_loaded_entries_count={timing_counts['tick_cache_loaded_entries_count']} "
            f"tick_scope_build_sec={timing_detail['tick_scope_build_sec']:.3f} "
            f"tick_scope_entries_count={timing_counts['tick_scope_entries_count']} "
            f"tick_replay_plan_sec={timing_detail['tick_replay_plan_sec']:.3f} "
            f"tick_replay_relevant_entries_count={timing_counts['tick_replay_relevant_entries_count']} "
            f"tick_replay_critical_minutes_count={timing_counts['tick_replay_critical_minutes_count']} "
            f"tick_replay_cache_hits_count={timing_counts['tick_replay_cache_hits_count']} "
            f"tick_replay_cache_missing_entries_count={timing_counts['tick_replay_cache_missing_entries_count']} "
            f"tick_replay_raw_critical_minutes_requested_count={timing_counts['tick_replay_raw_critical_minutes_requested_count']} "
            f"tick_replay_raw_loaded_minutes_count={timing_counts['tick_replay_raw_loaded_minutes_count']} "
            f"tick_raw_load_sec={timing_detail['tick_raw_load_sec']:.3f} "
            f"tick_raw_load_skipped={bool(timing_flags['tick_raw_load_skipped'])} "
            f"tick_simulate_selected_entries_sec={timing_detail['tick_simulate_selected_entries_sec']:.3f} "
            f"tick_simulation_skipped={bool(timing_flags['tick_simulation_skipped'])} "
            f"tick_missing_entries={int(len(missing_entry_indices)) if 'missing_entry_indices' in locals() else 0} "
            f"tick_cache_write_sec={timing_detail['tick_cache_write_sec']:.3f} "
            f"tick_cache_write_skipped={bool(timing_flags['tick_cache_write_skipped'])} "
            f"tick_cache_diagnostics_sec={timing_detail['tick_cache_diagnostics_sec']:.3f} "
            f"tick_cache_diagnostics_skipped={bool(timing_flags['tick_cache_diagnostics_skipped'])} "
            f"tick_cache_diagnostics_entries_count={timing_counts['tick_cache_diagnostics_entries_count']} "
            f"tick_candidate_metrics_sec={timing_detail['tick_candidate_metrics_sec']:.3f} "
            f"tick_missing_metrics_scan_sec={timing_detail['tick_missing_metrics_scan_sec']:.3f} "
            f"tick_candidates_requested={tick_candidates_requested} "
            f"tick_candidates_with_metrics={tick_candidates_with_metrics} "
            f"tick_candidates_missing_metrics={tick_candidates_missing_metrics} "
            f"tick_candidates_used_for_phase_d={len(phase_d_pool_base)} "
            f"candidate_csv_write_sec={timing_detail['candidate_csv_write_sec']:.3f} "
            f"coarse_csv_written={bool(wrote_coarse)} coarse_csv_reason={coarse_reason} "
            f"refined_csv_written={bool(wrote_refined)} refined_csv_reason={refined_reason} "
            f"phase_d_pool_build_sec={timing_detail['phase_d_pool_build_sec']:.3f} "
            f"ranking_sort_sec={timing_detail['ranking_sort_sec']:.3f} "
            f"ranking_prep_accounted_sec={timing_detail['ranking_prep_accounted_sec']:.3f} "
            f"ranking_prep_unaccounted_sec={timing_detail['ranking_prep_unaccounted_sec']:.3f}"
        )
    if bool(args.debug_timing_breakdown) or bool(args.debug_reject_stats):
        dist_items = dist_items_count
        non_dist_items = items_total_count - dist_items
        bin_items = binary_items_count
        non_bin_items = items_total_count - bin_items
        print(
            f"[prefilter-items] cols={len(cols)} items={items_total_count} "
            f"filtered_items={len(filtered_items)} family_top_pool={len(rank_lift)} "
            f"dist_items={dist_items} non_dist_items={non_dist_items} "
            f"binary_items={bin_items} non_binary_items={non_bin_items}"
        )
        if bool(args.debug_reject_stats):
            print("[prefilter-single-rejects] " + " ".join([f"{k}={v}" for k, v in single_rejects.items()]))

    if args.control_file is not None:
        _write_control_none(args.control_file)
    phase_ab_t0 = time.perf_counter()
    phase_b_t0 = None
    try:
        while True:
            force_phase = "A"
            unlocked_next = unlocked + int(args.step_size)
            pool = _build_unlocked_pool(
                miner_mod=miner,
                rank_lists=[rank_lift],
                all_candidate_cols=cols,
                unlocked_next=unlocked_next,
                step_size=int(args.step_size),
                binary_cap_per_list_block=10 ** 9,
                binary_anchor_lookahead_blocks=int(args.binary_anchor_lookahead_blocks),
                binary_cap_per_block=10 ** 9,
            )
            if not pool:
                _log_phase_transition("A" if not phase_b_started else "B", "C", "unlocked_pool_empty", round=unlocked_next, pool_size=0, valid_pool=len(valid_pool), max_valids=int(args.max_valids))
                break
            idxs = list(range(len(pool)))
            old_pool_size = max(0, len(pool) - int(args.step_size))
            phase_a = (len(valid_pool) < int(args.max_valids)) and (not force_phase_b_requested)
            phase_b_pool_key = _phase_b_pool_key(pool)
            if phase_a:
                print("[prefilter-phase-b] started=False reason=phase_a_active")
            if (not phase_a) and phase_b_pool_key in completed_phase_b_pool_keys:
                phase_b_skip_reason = "completed_pool_key"
                print(f"[prefilter-phase-b] started=False reason=completed_pool_key pool_size={len(pool)}")
                print(f"[prefilter-phase-b] skipped: completed pool key already processed. pool_size={len(pool)}")
                _log_phase_transition("A", "C", "phase_b_completed_pool_key", round=unlocked_next, pool_size=len(pool), valid_pool=len(valid_pool), max_valids=int(args.max_valids))
                force_phase = "C"
                break
            if (not phase_a) and not valid_pool:
                phase_b_skip_reason = "no_parent_seeds"
                print("[prefilter-phase-b] started=False reason=no_parent_seeds")
                print("[prefilter-phase-b] skipped: no parent seeds in valid_pool after phase A.")
                _log_phase_transition("A", "C", "no_expandable_phase_b_parents", round=unlocked_next, pool_size=len(pool), valid_pool=len(valid_pool), max_valids=int(args.max_valids))
                force_phase = "C"
                break
            if not phase_a:
                if not phase_b_started:
                    timing_detail2["phase_a_total_sec"] += time.perf_counter() - phase_ab_t0
                    phase_b_t0 = time.perf_counter()
                phase_b_started = True
                phase_b_skip_reason = "started"
                print(f"[prefilter-phase-b] started=True pool_size={len(pool)} parent_seeds={len(valid_pool)}")
                _log_phase_transition("A", "B", phase_b_start_reason, round=unlocked_next, pool_size=len(pool), valid_pool=len(valid_pool), max_valids=int(args.max_valids))
            shard_specs: list[tuple[int, int, int, int]] = []
            batch_eff = max(256, int(args.batch_size))
            sid = 0
            if phase_a:
                for r in range(2, int(args.max_path_conds) + 1):
                    if r > len(idxs):
                        continue
                    total_r = sum(1 for _ in _iter_combinations_with_new(idxs, r, old_pool_size))
                    for st in range(0, total_r, batch_eff):
                        shard_specs.append((sid, r, st, min(batch_eff, total_r - st)))
                        sid += 1
            else:
                # Phase B streams directly from parent_ext_iter until it is exhausted.
                pass
            rng.shuffle(shard_specs)
            tested = 0
            valid_round = 0
            mut_i = 0
            seeds = sorted(valid_pool, key=lambda z: (-float(z["ratio"]), -int(z["pos_hits"]), int(z["neg_hits"])))[: int(args.max_valids)]
            mut_combos: collections.deque[tuple[tuple[int, ...], dict]] = collections.deque()
            parent_ext_iter = None
            if (not phase_a) or (phase_a and len(seeds) > 0 and unlocked_next > int(args.step_size)):
                parent_ext_iter = _iter_all_parent_extensions(seeds, idxs, int(args.max_path_conds))
            combos_total_free = 0
            rmax = min(int(args.max_path_conds), len(idxs))
            if phase_a:
                for rr in range(2, rmax + 1):
                    combos_total_free += sum(1 for _ in _iter_combinations_with_new(idxs, rr, old_pool_size))
                for _ in range(max(1, batch_eff // 2)):
                    nxt = next(parent_ext_iter, None) if parent_ext_iter is not None else None
                    if nxt is None:
                        break
                    mut_combos.append(nxt)
                if mut_combos:
                    tmp = list(mut_combos)
                    rng.shuffle(tmp)
                    mut_combos = collections.deque(tmp)
            else:
                combos_total_free = -1
            combos_total_parent_extensions = int(len(mut_combos)) if phase_a else -1
            combos_total = int(combos_total_free + combos_total_parent_extensions) if phase_a else -1
            if phase_a:
                phase_timing_counts["phase_a_parent_extensions_total"] += int(len(mut_combos))
                if mut_combos:
                    phase_timing_counts["phase_a_parent_extensions_rounds"] += 1
                print(
                    f"[prefilter-phase-ab-detail] round={unlocked_next} phase=A "
                    f"pool_size={len(pool)} combos_total_free={combos_total_free} "
                    f"combos_total_parent_extensions={combos_total_parent_extensions} "
                    f"combos_total={combos_total} phase_a_parent_extensions_enabled={bool(parent_ext_iter is not None)} "
                    f"parent_extensions_sampled_this_round={len(mut_combos)}"
                )

            hist: list[tuple[int, float, tuple[float, int, int]]] = []
            phase_b_exhausted = False
            executor_cache: dict[int, concurrent.futures.ThreadPoolExecutor] = {}
            for spec in (shard_specs if phase_a else itertools.repeat((-1, -1, 0, 0))):
                est_mem_gb = (batch_eff * max(1, len(pool)) * 8.0) / (1024 ** 3)
                target_mem_gb = max(0.5, float(args.memory_soft_limit_gb) * 0.75)
                while est_mem_gb > target_mem_gb and batch_eff > 256:
                    batch_eff = max(256, batch_eff // 2)
                    est_mem_gb = (batch_eff * max(1, len(pool)) * 8.0) / (1024 ** 3)
                workers_eff = workers if est_mem_gb <= target_mem_gb else max(1, workers // 2)
                if phase_a:
                    _sid, rr, st, cnt = spec
                    new_combos = list(itertools.islice(_iter_combinations_with_new(idxs, rr, old_pool_size), st, st + cnt))
                    combos = list(new_combos)
                    parents = [None] * len(new_combos)
                    # from pool 2 onward in phase A: also expand existing valid rules in parallel
                    if mut_combos:
                        fill_n = max(1, batch_eff // 2)
                        exp_part = []
                        for _ in range(fill_n):
                            if not mut_combos:
                                break
                            exp_part.append(mut_combos.popleft())
                        combos.extend([x[0] for x in exp_part])
                        parents.extend([x[1] for x in exp_part])
                    if parent_ext_iter is not None and len(mut_combos) < max(1, batch_eff // 2):
                        refill = []
                        for _ in range(max(1, batch_eff // 2)):
                            nxt = next(parent_ext_iter, None)
                            if nxt is None:
                                break
                            refill.append(nxt)
                        if refill:
                            rng.shuffle(refill)
                            mut_combos.extend(refill)
                else:
                    combos_par = list(itertools.islice(parent_ext_iter, batch_eff)) if parent_ext_iter is not None else []
                    phase_timing_counts["phase_b_parent_extensions_total"] += int(len(combos_par))
                    combos = [x[0] for x in combos_par]
                    parents = [x[1] for x in combos_par]
                    if not combos:
                        phase_b_exhausted = True
                        break
                out_batch: list[dict] = []
                if workers_eff <= 1:
                    for i, cb in enumerate(combos):
                        rr = evaluate_combo(cb, pool, parents[i], source=("A" if phase_a else "B"))
                        if rr is not None:
                            out_batch.append(rr)
                else:
                    ex = executor_cache.get(int(workers_eff))
                    if ex is None:
                        ex = concurrent.futures.ThreadPoolExecutor(max_workers=int(workers_eff))
                        executor_cache[int(workers_eff)] = ex
                    futs = [ex.submit(evaluate_combo, cb, pool, parents[i], ("A" if phase_a else "B")) for i, cb in enumerate(combos)]
                    for f in futs:
                        rr = f.result()
                        if rr is not None:
                            out_batch.append(rr)
                tested += len(combos)
                valid_round += len(out_batch)
                valid_pool.extend(out_batch)
                valid_pool = _dedupe_mask(valid_pool)
                valid_pool = sorted(valid_pool, key=lambda z: (-float(z["ratio"]), -int(z["pos_hits"]), int(z["neg_hits"])))[: int(args.max_valids)]
                topk = valid_pool[: int(args.early_stop_top_k)]
                a1 = _train_key(topk[0]) if topk else (-np.inf, -1, 1)
                avg_ratio = float(np.mean([float(x["ratio"]) for x in topk])) if topk else float("nan")
                hist.append((len(combos), avg_ratio, a1))
                progress_state = {
                    "round": int(unlocked_next),
                    "pool_size": int(len(pool)),
                    "combos_total_free": int(combos_total_free),
                    "combos_total_parent_extensions": int(combos_total_parent_extensions),
                    "combos_total": int(combos_total) if combos_total_parent_extensions >= 0 else -1,
                    "tested": int(tested),
                    "valid_kept": int(len(valid_pool)),
                    "valid_new_round": int(valid_round),
                    "batch_size": int(batch_eff),
                    "workers": int(workers_eff),
                    "best_a1": list(a1),
                    "topk_summary": [{"ratio": float(x["ratio"]), "pos_hits": int(x["pos_hits"]), "neg_hits": int(x["neg_hits"])} for x in topk[:10]],
                    "phase": "A" if phase_a else "B",
                }
                _save_progress(valid_pool, progress_state)
                if phase_a:
                    print(
                        f"[prefilter-progress] round={unlocked_next} phase=A "
                        f"pool_size={len(pool)} combos_total_free={progress_state['combos_total_free']} "
                        f"combos_total_parent_extensions={progress_state['combos_total_parent_extensions']} "
                        f"combos_total={progress_state['combos_total']} "
                        f"tested={tested} valid_kept={len(valid_pool)} valid_new_round={valid_round} batch_size={batch_eff} workers={workers_eff}"
                    )
                else:
                    print(
                        f"[prefilter-progress] round={unlocked_next} phase=B "
                        f"pool_size={len(pool)} combos_total=unknown/streaming parent_extension_mode=streaming "
                        f"tested={tested} valid_kept={len(valid_pool)} valid_new_round={valid_round} batch_size={batch_eff} workers={workers_eff}"
                    )
                cmd = _read_control_command(args.control_file)
                if cmd == "export_now":
                    print("[prefilter-control] export_now")
                    _save_progress(valid_pool, progress_state)
                    _write_control_none(args.control_file)
                elif cmd == "disable_early_stop":
                    print("[prefilter-control] disable_early_stop")
                    args.early_stop_window_combos = 10 ** 18
                    _write_control_none(args.control_file)
                elif cmd == "enable_early_stop":
                    print("[prefilter-control] enable_early_stop")
                    args.early_stop_window_combos = int(original_early_stop_window_combos)
                    _write_control_none(args.control_file)
                elif cmd == "stop_after_batch":
                    print("[prefilter-control] stop_after_batch requested; will stop cleanly after batch")
                    _save_progress(valid_pool, progress_state)
                    _write_control_none(args.control_file)
                    stop_after_batch_requested = True
                    break
                elif cmd == "force_phase_b":
                    print("[prefilter-control] force_phase_b requested; will end phase A and continue with phase B")
                    force_phase_b_requested = True
                    phase_b_start_reason = "control_force_phase_b"
                    _write_control_none(args.control_file)
                    if phase_a:
                        break
                elif cmd == "force_phase_c":
                    print("[prefilter-control] force_phase_c")
                    force_phase = "C"
                    phase_c_was_active = True
                    _log_phase_transition("A" if phase_a else "B", "C", "control_force_phase_c", round=unlocked_next, pool_size=len(pool), valid_pool=len(valid_pool), max_valids=int(args.max_valids), tested_total=tested, total_free=combos_total_free, total_parent_ext=combos_total_parent_extensions, total_combined=combos_total)
                    _write_control_none(args.control_file)
                    break
                elif cmd == "force_phase_d":
                    if phase_c_was_active:
                        print("[prefilter-control] force_phase_d")
                        force_phase = "D"
                    else:
                        print("[prefilter-control] force_phase_d ignored (phase C not active yet)")
                    _write_control_none(args.control_file)
                if bool(args.debug_reject_stats):
                    print("[prefilter-reject-stats] " + " ".join([f"{k}={v}" for k, v in reject_stats.items()]))
                covered = 0
                win = []
                for h in reversed(hist):
                    win.append(h)
                    covered += int(h[0])
                    if covered >= int(args.early_stop_window_combos):
                        break
                if covered >= int(args.early_stop_window_combos) and len(win) >= 2:
                    old = win[-1][1]
                    new = win[0][1]
                    improve_pct = 0.0 if (not np.isfinite(old) or old == 0.0) else ((new - old) / abs(old) * 100.0)
                    a1_improved = any(h[2] > win[-1][2] for h in win[:-1])
                    if improve_pct < float(args.early_stop_avg_improve_pct) and not a1_improved:
                        if phase_a:
                            print("[prefilter-progress] early stop in phase A; continuing with phase B next.")
                            force_phase_b_requested = True
                            phase_b_start_reason = "early_stop_no_improvement"
                        else:
                            print("[prefilter-progress] early stop in phase B; continuing with phase C/D next.")
                            force_phase = "C"
                            _log_phase_transition("B", "C", "early_stop_no_improvement", round=unlocked_next, pool_size=len(pool), valid_pool=len(valid_pool), max_valids=int(args.max_valids), tested_total=tested, tested_free=tested if phase_a else -1, total_free=combos_total_free, tested_parent_ext=-1 if phase_a else tested, total_parent_ext=combos_total_parent_extensions, total_combined=combos_total, early_stop_avg_improve_pct=improve_pct)
                        break

            for ex in executor_cache.values():
                ex.shutdown(wait=True)

            if (not phase_a) and (phase_b_exhausted or force_phase == "C"):
                completed_phase_b_pool_keys.add(phase_b_pool_key)
            if stop_after_batch_requested:
                print("[prefilter-progress] stopped after batch as requested.")
                break
            if (not phase_a) and phase_b_exhausted:
                print("[prefilter-phase-b] completed: parent_ext_iter exhausted")
                _log_phase_transition("B", "C", "phase_b_exhausted", round=unlocked_next, pool_size=len(pool), valid_pool=len(valid_pool), max_valids=int(args.max_valids), tested_total=tested, tested_parent_ext=tested)
                force_phase = "C"
            if force_phase in {"C", "D"}:
                phase_c_was_active = phase_c_was_active or (force_phase == "C")
                print("[prefilter-progress] force phase switch requested; continuing with implemented post A/B phases.")
                break
            if not valid_pool:
                unlocked = unlocked_next
                if unlocked >= len(rank_lift):
                    _log_phase_transition("A" if phase_a else "B", "C", "phase_a_exhausted" if phase_a else "phase_b_exhausted", round=unlocked_next, pool_size=len(pool), valid_pool=0, max_valids=int(args.max_valids), tested_total=tested, total_free=combos_total_free, total_parent_ext=combos_total_parent_extensions, total_combined=combos_total)
                    break
                continue
            best_paths = list(valid_pool)
            cur_best = float(best_paths[0]["ratio"]) if best_paths else -np.inf
            if phase_a and (not force_phase_b_requested) and cur_best <= prev_best + 1e-12:
                _log_phase_transition("A", "C", "phase_a_no_improvement", round=unlocked_next, pool_size=len(pool), valid_pool=len(valid_pool), max_valids=int(args.max_valids), tested_total=tested, total_free=combos_total_free, total_parent_ext=combos_total_parent_extensions, total_combined=combos_total)
                break
            prev_best = cur_best
            unlocked = unlocked_next
    except KeyboardInterrupt:
        print("[prefilter] interrupted; checkpoint saved.")
        _save_progress(valid_pool, progress_state)
    if phase_b_started and phase_b_t0 is not None:
        timing_detail2["phase_b_total_sec"] += time.perf_counter() - phase_b_t0
    else:
        timing_detail2["phase_a_total_sec"] += time.perf_counter() - phase_ab_t0
        if phase_b_skip_reason == "not_reached":
            phase_b_skip_reason = "phase_a_or_direct_to_c"

    # Phase C: family-top-pool beam search
    phase_c_t0 = time.perf_counter()
    pool_c = _build_unlocked_pool(
        miner_mod=miner,
        rank_lists=[rank_lift],
        all_candidate_cols=cols,
        unlocked_next=len(rank_lift),
        step_size=int(args.step_size),
        binary_cap_per_list_block=10 ** 9,
        binary_anchor_lookahead_blocks=int(args.binary_anchor_lookahead_blocks),
        binary_cap_per_block=10 ** 9,
    )
    if len(pool_c) == 0:
        print("[prefilter-phase-c] skipped: empty family-top pool")
    else:
        idxs_c = list(range(len(pool_c)))
        seed_rules = sorted(valid_pool, key=lambda z: (-float(z["ratio"]), -int(z["pos_hits"]), int(z["neg_hits"])))[: max(1, int(args.phase_c_beam_width))]
        cond_to_idx_c = {(str(c["col"]), str(c["op"]), float(c["value"])): i for i, c in enumerate(pool_c)}
        mapped_beam: list[tuple[int, ...]] = []
        for s in seed_rules:
            mapped = []
            ok = True
            for c in s.get("conds", []):
                key = (str(c["col"]), str(c["op"]), float(c["value"]))
                idx = cond_to_idx_c.get(key)
                if idx is None:
                    ok = False
                    break
                mapped.append(int(idx))
            if ok and mapped:
                mapped_beam.append(tuple(sorted(set(mapped))))
        if not mapped_beam:
            mapped_beam = [tuple([i]) for i in idxs_c[: max(1, min(int(args.phase_c_beam_width), len(idxs_c)))]]
        beam = mapped_beam
        beam = [b for b in beam if 1 <= len(b) <= int(args.phase_c_max_conds)]
        print(f"[prefilter-phase-c] start beam={len(beam)} pool={len(pool_c)}")
        phase_c_was_active = True
        phase_c_level = 0
        force_phase_d_after_c = False
        while beam:
            phase_c_level += 1
            level_t0 = time.perf_counter()
            seeds_at_level_start = len(beam)
            generated_extensions = 0
            evaluated_combos = 0
            truncated = False
            out_c: list[dict] = []
            def _iter_cands_c():
                for base in beam:
                    cset = set(base)
                    add_cands = [x for x in idxs_c if x not in cset]
                    for add_k in range(max(1, int(args.phase_c_add_min)), max(1, int(args.phase_c_add_max)) + 1):
                        if len(base) + add_k > int(args.phase_c_max_conds) or len(base) + add_k > int(args.max_path_conds):
                            continue
                        for adds in itertools.combinations(add_cands, add_k):
                            yield tuple(sorted(cset | set(adds)))
            had_any = False
            for chunk in _batched(_iter_cands_c(), max(64, int(args.batch_size))):
                had_any = True
                cap = int(args.phase_c_max_generated_per_level)
                if cap > 0 and generated_extensions + len(chunk) >= cap:
                    chunk = chunk[: max(0, cap - generated_extensions)]
                    truncated = True
                generated_extensions += len(chunk)
                for cb in chunk:
                    evaluated_combos += 1
                    rr = evaluate_combo(cb, pool_c, None, source="C")
                    if rr is not None:
                        out_c.append(rr)
                if truncated:
                    break
            if not had_any:
                print(f"[prefilter-phase-c] level={phase_c_level} depth_range=add{int(args.phase_c_add_min)}..add{int(args.phase_c_add_max)} seeds={seeds_at_level_start} pool={len(pool_c)} estimated_generated=unknown generated=0 evaluated=0 truncated={truncated} rejected_duplicates=unknown rejected_invalid_rules=0 new_valid=0 beam_kept=0 valid_pool={len(valid_pool)} elapsed={time.perf_counter() - level_t0:.2f}s")
                break
            if not out_c:
                print(f"[prefilter-phase-c] level={phase_c_level} depth_range=add{int(args.phase_c_add_min)}..add{int(args.phase_c_add_max)} seeds={seeds_at_level_start} pool={len(pool_c)} estimated_generated=unknown generated={generated_extensions} evaluated={evaluated_combos} truncated={truncated} rejected_duplicates=unknown rejected_invalid_rules={max(0, evaluated_combos - len(out_c))} new_valid=0 beam_kept=0 valid_pool={len(valid_pool)} elapsed={time.perf_counter() - level_t0:.2f}s")
                break
            valid_before = len(valid_pool)
            valid_pool.extend(out_c)
            valid_pool = _dedupe_mask(valid_pool)
            valid_pool = sorted(valid_pool, key=lambda z: (-float(z["ratio"]), -int(z["pos_hits"]), int(z["neg_hits"])))[: int(args.max_valids)]
            _save_progress(valid_pool, progress_state)
            beam_rules = sorted(out_c, key=lambda z: (-float(z["ratio"]), -int(z["pos_hits"]), int(z["neg_hits"])))[: int(args.phase_c_beam_width)]
            beam = [tuple(sorted(int(x) for x in r.get("_combo", tuple()))) for r in beam_rules]
            best_ratio = float(valid_pool[0]["ratio"]) if valid_pool else float("nan")
            print(f"[prefilter-phase-c] level={phase_c_level} depth_range=add{int(args.phase_c_add_min)}..add{int(args.phase_c_add_max)} seeds={seeds_at_level_start} pool={len(pool_c)} estimated_generated=unknown generated={generated_extensions} evaluated={evaluated_combos} truncated={truncated} rejected_duplicates=unknown rejected_invalid_rules={max(0, evaluated_combos - len(out_c))} new_valid={max(0, len(valid_pool) - valid_before)} beam_kept={len(beam)} valid_pool={len(valid_pool)} elapsed={time.perf_counter() - level_t0:.2f}s best_ratio={best_ratio:.6g}")
            cmd = _read_control_command(args.control_file)
            if cmd == "export_now":
                print("[prefilter-control] export_now (phase C)"); _save_progress(valid_pool, progress_state); _write_control_none(args.control_file)
            elif cmd == "stop_after_batch":
                print("[prefilter-control] stop_after_batch requested in phase C; stopping after beam level"); _save_progress(valid_pool, progress_state); _write_control_none(args.control_file); stop_after_batch_requested = True; break
            elif cmd == "force_phase_c":
                print("[prefilter-control] force_phase_c ignored (already in phase C)"); _write_control_none(args.control_file)
            elif cmd == "force_phase_d":
                print("[prefilter-control] force_phase_d"); force_phase_d_after_c = True; _write_control_none(args.control_file); break
            elif cmd == "disable_early_stop":
                print("[prefilter-control] disable_early_stop"); args.early_stop_window_combos = 10 ** 18; _write_control_none(args.control_file)
            elif cmd == "enable_early_stop":
                print("[prefilter-control] enable_early_stop"); args.early_stop_window_combos = int(original_early_stop_window_combos); _write_control_none(args.control_file)
        print("[prefilter-phase-c] done")
    timing_detail2["phase_c_total_sec"] += time.perf_counter() - phase_c_t0

    # Phase D: full tick beam search
    phase_d_t0 = time.perf_counter()
    d_valid_pool: list[dict] = []
    if stop_after_batch_requested:
        print("[prefilter-phase-d] skipped: stop_after_batch requested before phase D")
    elif not tick_refined_mode:
        print("[prefilter-phase-d] skipped: no tick metrics available")
    else:
        tick_pool_raw = list(phase_d_pool_base)
        tick_pool = [r for r in tick_pool_raw if _has_full_tick_metrics(r)]
        print(f"[prefilter-phase-d] tick_candidates_available={len(tick_pool_raw)} tick_candidates_with_full_metrics={len(tick_pool)} (source=current tick_refine_scope full tick candidates)")
        if not tick_pool:
            print("[prefilter-phase-d] skipped: tick pool empty")
        else:
            tick_pool = sorted(tick_pool, key=lambda z: (-float(z.get("tick_single_ratio", -np.inf)), -int(z.get("tick_single_pos_hits", z.get("_single_pos_hits", 0)))))
            idxs_d = list(range(len(tick_pool)))
            start_depth = max(1, min(int(args.phase_d_start_conds), int(args.phase_d_max_conds), int(args.max_path_conds), len(idxs_d)))
            d_beam: list[tuple[int, ...]] = []
            phase_d_level = 0

            def _phase_d_control() -> bool:
                nonlocal stop_after_batch_requested
                cmd = _read_control_command(args.control_file)
                if cmd == "export_now":
                    print("[prefilter-control] export_now (phase D)"); _save_progress(valid_pool + d_valid_pool, progress_state); _write_control_none(args.control_file)
                elif cmd == "stop_after_batch":
                    print("[prefilter-control] stop_after_batch requested in phase D; stopping after beam level"); _save_progress(valid_pool + d_valid_pool, progress_state); _write_control_none(args.control_file); stop_after_batch_requested = True; return True
                elif cmd == "force_phase_c":
                    print("[prefilter-control] force_phase_c ignored (already in phase D)"); _write_control_none(args.control_file)
                elif cmd == "force_phase_d":
                    print("[prefilter-control] force_phase_d ignored (already in phase D)"); _write_control_none(args.control_file)
                elif cmd == "disable_early_stop":
                    print("[prefilter-control] disable_early_stop"); args.early_stop_window_combos = 10 ** 18; _write_control_none(args.control_file)
                elif cmd == "enable_early_stop":
                    print("[prefilter-control] enable_early_stop"); args.early_stop_window_combos = int(original_early_stop_window_combos); _write_control_none(args.control_file)
                return False

            def _d_sort_key(r: dict) -> tuple[float, int, int, int]:
                return (-float(r["ratio"]), -int(r["pos_hits"]), int(r["neg_hits"]), len(tuple(r.get("_combo", tuple()))))

            def _keep_d_pool(rows: list[dict]) -> list[dict]:
                rows = _dedupe_mask(rows)
                return sorted(rows, key=_d_sort_key)[: int(args.max_valids)]

            # level=0: evaluate all start-depth combinations directly, streamingly.
            level_t0 = time.perf_counter()
            generated_extensions = 0
            evaluated_combos = 0
            truncated = False
            out_d: list[dict] = []
            for chunk in _batched(itertools.combinations(idxs_d, start_depth), max(64, int(args.batch_size))):
                cap = int(args.phase_d_max_generated_per_level)
                if cap > 0 and generated_extensions + len(chunk) >= cap:
                    chunk = chunk[: max(0, cap - generated_extensions)]
                    truncated = True
                generated_extensions += len(chunk)
                for cb in chunk:
                    evaluated_combos += 1
                    rr = evaluate_combo(tuple(sorted(cb)), tick_pool, None, source="D")
                    if rr is not None:
                        out_d.append(rr)
                if truncated:
                    break
            if out_d:
                before = len(d_valid_pool)
                d_valid_pool = _keep_d_pool(d_valid_pool + out_d)
                _save_progress(valid_pool + d_valid_pool, progress_state)
                beam_rules = sorted(_dedupe_mask(out_d), key=_d_sort_key)[: int(args.phase_d_beam_width)]
                d_beam = [tuple(sorted(int(x) for x in r.get("_combo", tuple()))) for r in beam_rules]
                best_ratio = float(d_valid_pool[0]["ratio"]) if d_valid_pool else float("nan")
                print(f"[prefilter-phase-d] level=0 depth={start_depth} seeds=all pool={len(tick_pool)} estimated_generated=unknown generated={generated_extensions} evaluated={evaluated_combos} truncated={truncated} rejected_duplicates=unknown rejected_invalid_rules={max(0, evaluated_combos - len(out_d))} new_valid={max(0, len(d_valid_pool) - before)} beam_kept={len(d_beam)} d_valid_pool={len(d_valid_pool)} elapsed={time.perf_counter() - level_t0:.2f}s best_ratio={best_ratio:.6g}")
            else:
                print(f"[prefilter-phase-d] level=0 depth={start_depth} seeds=all pool={len(tick_pool)} estimated_generated=unknown generated={generated_extensions} evaluated={evaluated_combos} truncated={truncated} rejected_duplicates=unknown rejected_invalid_rules={max(0, evaluated_combos)} new_valid=0 beam_kept=0 d_valid_pool={len(d_valid_pool)} elapsed={time.perf_counter() - level_t0:.2f}s")
                _save_progress(valid_pool + d_valid_pool, progress_state)
            if _phase_d_control():
                d_beam = []

            while d_beam and not stop_after_batch_requested:
                phase_d_level += 1
                level_t0 = time.perf_counter()
                seeds_at_level_start = len(d_beam)
                generated_extensions = 0
                evaluated_combos = 0
                truncated = False
                out_d = []
                current_min_base_len = min((len(b) for b in d_beam), default=0)
                current_max_base_len = max((len(b) for b in d_beam), default=0)
                min_depth = current_min_base_len + max(1, int(args.phase_d_add_min))
                hard_max_depth = min(int(args.phase_d_max_conds), int(args.max_path_conds))
                max_depth = min(current_max_base_len + max(1, int(args.phase_d_add_max)), hard_max_depth)
                if min_depth > max_depth:
                    print("[prefilter-phase-d] done: max depth reached")
                    break

                def _iter_cands_d():
                    for base in d_beam:
                        cset = set(base)
                        add_cands = [x for x in idxs_d if x not in cset]
                        for add_k in range(max(1, int(args.phase_d_add_min)), max(1, int(args.phase_d_add_max)) + 1):
                            if len(base) + add_k > int(args.phase_d_max_conds) or len(base) + add_k > int(args.max_path_conds):
                                continue
                            for adds in itertools.combinations(add_cands, add_k):
                                yield tuple(sorted(cset | set(adds)))

                had_any = False
                for chunk in _batched(_iter_cands_d(), max(64, int(args.batch_size))):
                    had_any = True
                    cap = int(args.phase_d_max_generated_per_level)
                    if cap > 0 and generated_extensions + len(chunk) >= cap:
                        chunk = chunk[: max(0, cap - generated_extensions)]
                        truncated = True
                    generated_extensions += len(chunk)
                    for cb in chunk:
                        evaluated_combos += 1
                        rr = evaluate_combo(cb, tick_pool, None, source="D")
                        if rr is not None:
                            out_d.append(rr)
                    if truncated:
                        break
                depth_txt = f"depth={min_depth}" if min_depth == max_depth else f"depth_range={min_depth}..{max_depth}"
                if (not had_any) or (not out_d):
                    print(f"[prefilter-phase-d] level={phase_d_level} {depth_txt} seeds={seeds_at_level_start} pool={len(tick_pool)} estimated_generated=unknown generated={generated_extensions} evaluated={evaluated_combos} truncated={truncated} rejected_duplicates=unknown rejected_invalid_rules={max(0, evaluated_combos)} new_valid=0 beam_kept=0 d_valid_pool={len(d_valid_pool)} elapsed={time.perf_counter() - level_t0:.2f}s")
                    _save_progress(valid_pool + d_valid_pool, progress_state)
                    _phase_d_control()
                    break
                before = len(d_valid_pool)
                d_valid_pool = _keep_d_pool(d_valid_pool + out_d)
                _save_progress(valid_pool + d_valid_pool, progress_state)
                beam_rules = sorted(_dedupe_mask(out_d), key=_d_sort_key)[: int(args.phase_d_beam_width)]
                d_beam = [tuple(sorted(int(x) for x in r.get("_combo", tuple()))) for r in beam_rules]
                best_ratio = float(d_valid_pool[0]["ratio"]) if d_valid_pool else float("nan")
                print(f"[prefilter-phase-d] level={phase_d_level} {depth_txt} seeds={seeds_at_level_start} pool={len(tick_pool)} estimated_generated=unknown generated={generated_extensions} evaluated={evaluated_combos} truncated={truncated} rejected_duplicates=unknown rejected_invalid_rules={max(0, evaluated_combos - len(out_d))} new_valid={max(0, len(d_valid_pool) - before)} beam_kept={len(d_beam)} d_valid_pool={len(d_valid_pool)} elapsed={time.perf_counter() - level_t0:.2f}s best_ratio={best_ratio:.6g}")
                if _phase_d_control():
                    break
            print("[prefilter-phase-d] done")
    timing_detail2["phase_d_total_sec"] += time.perf_counter() - phase_d_t0

    rules_export_t0 = time.perf_counter()
    non_d_pool = [r for r in valid_pool if str(r.get("search_source", "")) != "D"]
    d_pool = [r for r in d_valid_pool if str(r.get("search_source", "")) == "D"] + [r for r in valid_pool if str(r.get("search_source", "")) == "D"]

    def _is_binned_continuous_eq(c: dict) -> bool:
        col = str(c.get("col"))
        op = str(c.get("op"))
        if op != "==":
            return False
        ftype = str(meta.get(col, {}).get("feature_type", ""))
        return ftype == "continuous"

    def _better_score(a: dict | None, b: dict | None) -> bool:
        if a is None:
            return False
        if b is None:
            return True
        ka = (float(a["ratio"]), int(a["pos_hits"]), -int(a["neg_hits"]))
        kb = (float(b["ratio"]), int(b["pos_hits"]), -int(b["neg_hits"]))
        return ka > kb

    def _mask_cond_to_interval(c: dict) -> tuple[int, int]:
        if "lo_bin" in c and "hi_bin" in c:
            return int(c["lo_bin"]), int(c["hi_bin"])
        v = int(round(float(c.get("value", 0.0))))
        return v, v

    def _apply_neighbor_merge(rule: dict) -> dict:
        conds = [dict(x) for x in rule.get("conds", [])]
        base_mask = _build_mask_from_conds(conds)
        best_sc = _score_from_mask(base_mask)
        if best_sc is None:
            return rule
        changed = True
        while changed:
            changed = False
            for i, c in enumerate(conds):
                if not _is_binned_continuous_eq(c):
                    continue
                col = str(c["col"])
                eff = int(meta.get(col, {}).get("effective_bin_count", 0) or 0)
                if eff <= 1:
                    continue
                lo, hi = _mask_cond_to_interval(c)
                local_best_sc = best_sc
                local_best = None
                if lo > 1:
                    cand = dict(c)
                    cand["lo_bin"] = int(lo - 1)
                    cand["hi_bin"] = int(hi)
                    cand["value"] = float(cand["lo_bin"])
                    trial_conds = [dict(x) for x in conds]
                    trial_conds[i] = cand
                    sc = _score_from_mask(_build_mask_from_conds(trial_conds))
                    if _better_score(sc, local_best_sc):
                        local_best_sc = sc
                        local_best = cand
                if hi < eff:
                    cand = dict(c)
                    cand["lo_bin"] = int(lo)
                    cand["hi_bin"] = int(hi + 1)
                    cand["value"] = float(cand["lo_bin"])
                    trial_conds = [dict(x) for x in conds]
                    trial_conds[i] = cand
                    sc = _score_from_mask(_build_mask_from_conds(trial_conds))
                    if _better_score(sc, local_best_sc):
                        local_best_sc = sc
                        local_best = cand
                if local_best is not None:
                    conds[i] = local_best
                    best_sc = local_best_sc
                    changed = True
        merged_rule = dict(rule)
        merged_rule["conds"] = conds
        merged_rule.update(best_sc)
        final_mask = _build_mask_from_conds(conds)
        merged_rule["_mask_hash"] = _mask_hash(final_mask)
        final_sc = _score_from_mask(final_mask)
        if final_sc is not None:
            merged_rule.update(final_sc)
        return merged_rule

    def _prepare_export_group(rows: list[dict], group_name: str) -> tuple[list[dict], bool]:
        prepared = [_attach_decode_info(_apply_neighbor_merge(r)) for r in rows]
        prepared = _dedupe_mask(prepared)
        train_ranked_group = sorted(prepared, key=lambda z: (-float(z["ratio"]), -int(z["pos_hits"]), int(z["neg_hits"])))
        train_rank_map_group = {id(r): i for i, r in enumerate(train_ranked_group)}
        shortlist_n = max(100, int(args.top_paths) * 4)
        shortlist_group = train_ranked_group[:shortlist_n]
        shortlist_group = [r for r in shortlist_group if (int(r["test_pos_hits"]) + int(r["test_neg_hits"])) > 0]
        shortlist_group = sorted(
            shortlist_group,
            key=lambda z: (
                -float(z["test_ratio"]),
                -int(z["test_pos_hits"]),
                int(z["test_neg_hits"]),
                int(train_rank_map_group.get(id(z), 10 ** 9)),
            ),
        )
        export_group = shortlist_group[: int(args.top_paths)]
        fallback_used = False
        if not export_group and train_ranked_group:
            export_group = train_ranked_group[: int(args.top_paths)]
            fallback_used = True
            print(f"[prefilter-fallback] Exporting {len(export_group)} {group_name} train-valid rules (fallback: valid>0 but test_hits=0).")
        return export_group, fallback_used

    non_d_export, non_d_fallback = _prepare_export_group(non_d_pool, "non_d")
    d_export, d_fallback = _prepare_export_group(d_pool, "D")
    best_paths = non_d_export + d_export
    fallback_export_used = bool(non_d_fallback or d_fallback)
    if bool(args.cluster_only_lower_entry):
        for r in best_paths:
            le_mask = _build_mask_from_conds(r.get("conds", []))
            le_sc = _score_from_mask(le_mask, only_lower_entry=False, enforce_filters=False)
            if le_sc is None:
                r["le_off_train_ratio"] = float("nan")
                r["le_off_test_ratio"] = float("nan")
                r["le_off_total_hits"] = 0
            else:
                r["le_off_train_ratio"] = float(le_sc.get("ratio", np.nan))
                r["le_off_test_ratio"] = float(le_sc.get("test_ratio", np.nan))
                r["le_off_total_hits"] = int(le_sc.get("pos_hits", 0)) + int(le_sc.get("neg_hits", 0)) + int(le_sc.get("test_pos_hits", 0)) + int(le_sc.get("test_neg_hits", 0))
    if not best_paths:
        print("[prefilter-warning] No rule with test hits found for final CSV export.")

    out_json = {
        "version": 2,
        "settings": {
            "tp": tps,
            "tp_weights": tp_w.tolist(),
            "sl": float(args.sl),
            "hold": int(args.hold),
            "trail": int(args.trail),
            "trail_activate": float(args.trail_activate),
            "trail_offset": float(args.trail_offset),
            "trail_factor": float(args.trail_factor),
            "include_unrealized_at_test_end": int(bool(args.include_unrealized_at_test_end)),
            "train_frac": float(args.train_frac),
            "wf_folds": int(args.wf_folds),
        },
        "rules": best_paths,
        "fallback_export_used": int(bool(fallback_export_used)),
    }
    if args.out_rules_json is not None:
        _atomic_write_text(args.out_rules_json, json.dumps(out_json, ensure_ascii=False, indent=2))

    rows = _rules_to_rows(best_paths, is_fallback_export=fallback_export_used)
    _validate_rows(rows)
    if rows:
        _atomic_write_csv(args.out_rules_csv, pd.DataFrame(rows))
    else:
        _atomic_write_csv(args.out_rules_csv, pd.DataFrame(columns=csv_columns))
    timing_detail2["rules_export_sec"] += time.perf_counter() - rules_export_t0
    if bool(args.debug_timing_breakdown):
        print(
            "[prefilter-timing-detail-2] "
            f"coarse_csv_load_sec={timing_detail2['coarse_csv_load_sec']:.3f} "
            f"coarse_rows_filter_sec={timing_detail2['coarse_rows_filter_sec']:.3f} "
            f"coarse_reconstruct_sec={timing_detail2['coarse_reconstruct_sec']:.3f} "
            f"allowlist_filter_sec={timing_detail2['allowlist_filter_sec']:.3f} "
            f"refined_csv_load_sec={timing_detail2['refined_csv_load_sec']:.3f} "
            f"refined_key_match_sec={timing_detail2['refined_key_match_sec']:.3f} "
            f"refined_tick_metric_restore_sec={timing_detail2['refined_tick_metric_restore_sec']:.3f} "
            f"quantile_thresholds_sec={timing_detail2['quantile_thresholds_sec']:.3f} "
            f"build_items_candidate_loop_sec={timing_detail2['build_items_candidate_loop_sec']:.3f} "
            f"build_items_mask_sec={timing_detail2['build_items_mask_sec']:.3f} "
            f"build_items_metadata_sec={timing_detail2['build_items_metadata_sec']:.3f} "
            f"filtered_items_sec={timing_detail2['filtered_items_sec']:.3f} "
            f"family_top_sec={timing_detail2['family_top_sec']:.3f} "
            f"tick_missing_metrics_scan_sec={timing_detail['tick_missing_metrics_scan_sec']:.3f} "
            f"same_reference_groups_sec={timing_detail2['same_reference_groups_sec']:.3f} "
            f"search_setup_sec={timing_detail2['search_setup_sec']:.3f} "
            f"phase_a_total_sec={timing_detail2['phase_a_total_sec']:.3f} "
            f"phase_b_total_sec={timing_detail2['phase_b_total_sec']:.3f} "
            f"phase_b_started={bool(phase_b_started)} "
            f"phase_b_skip_reason={phase_b_skip_reason} "
            f"phase_a_parent_extensions_total={phase_timing_counts['phase_a_parent_extensions_total']} "
            f"phase_a_parent_extensions_rounds={phase_timing_counts['phase_a_parent_extensions_rounds']} "
            f"phase_b_parent_extensions_total={phase_timing_counts['phase_b_parent_extensions_total']} "
            f"phase_c_total_sec={timing_detail2['phase_c_total_sec']:.3f} "
            f"phase_d_total_sec={timing_detail2['phase_d_total_sec']:.3f} "
            f"rules_export_sec={timing_detail2['rules_export_sec']:.3f} "
        )
    if bool(args.debug_reject_stats):
        print("[prefilter-reject-stats-total] " + " ".join([f"{k}={v}" for k, v in reject_stats.items()]))
    rules_json_target = str(args.out_rules_json) if args.out_rules_json is not None else "disabled"
    print(f"Saved {len(rows)} rules -> rules_json={rules_json_target} rules_csv={args.out_rules_csv}")


if __name__ == "__main__":
    main()
