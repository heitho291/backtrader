#!/usr/bin/env python3
"""Standalone in-detail prefilter combinatorial search.

Builds labels via TP/SL/Hold/Trail simulation directly from features OHLC,
then searches complete rule combinations (no greedy extension) over unlocked pools.
"""

from __future__ import annotations

import argparse
import concurrent.futures
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

    p.add_argument("--tick-data", type=Path, default=None)
    p.add_argument("--tick-cache-parquet", type=Path, default=None)
    p.add_argument("--tick-datetime-column", type=str, default="datetime")
    p.add_argument("--tick-price-column", type=str, default="auto")
    p.add_argument("--tick-sep", type=str, default=",")

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
    p.add_argument("--label-cache-npz", type=Path, default=None)
    p.add_argument("--min-single-pos-hits", type=int, default=2)
    p.add_argument("--min-single-lift", type=float, default=1.01)
    p.add_argument("--max-single-mask-count", type=int, default=0)
    p.add_argument("--family-top-n", type=int, default=40)
    p.add_argument("--family-split-delta-window", action="store_true", default=False)
    p.add_argument("--include-candidates-file", type=Path, default=None)
    p.add_argument("--out-candidates-coarse-csv", type=Path, default=None)
    p.add_argument("--out-candidates-refined-csv", type=Path, default=None)
    p.add_argument("--out-candidates-csv", type=Path, default=None, help="Deprecated alias: writes both coarse/refined if stage is available")

    p.add_argument("--min-pos-per-week", type=float, default=1.0)
    p.add_argument("--min-main-score", type=float, default=1.0)
    p.add_argument("--binary-cap-per-block", type=int, default=6)
    p.add_argument("--binary-cap-per-list-block", type=int, default=2)
    p.add_argument("--binary-anchor-lookahead-blocks", type=int, default=2)

    p.add_argument("--out-rules-json", type=Path, default=Path("prefilter_in_detail_rules.json"))
    p.add_argument("--out-rules-csv", type=Path, default=Path("prefilter_in_detail_rules.csv"))
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


def _atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(path.parent), encoding="utf-8", newline="") as tf:
        frame.to_csv(tf.name, index=False)
        tmp = Path(tf.name)
    os.replace(tmp, path)


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


def _load_stage_csv_if_match(path: Path | None, expected_stage: str, expected_ctx_sig: str) -> pd.DataFrame | None:
    if path is None or (not path.exists()):
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    if "__stage" not in df.columns or "__ctx_sig" not in df.columns:
        return None
    stage_vals = {str(x) for x in df["__stage"].dropna().unique().tolist()}
    sig_vals = {str(x) for x in df["__ctx_sig"].dropna().unique().tolist()}
    if stage_vals != {expected_stage}:
        return None
    if sig_vals != {expected_ctx_sig}:
        return None
    return df


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


def _critical_minutes_for_entries(
    entry_indices: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    bar_time_ns: np.ndarray,
    hold: int,
    use_multi_tp: bool,
    trail: bool,
    trail_activate: float,
    tps: np.ndarray,
    sl: float,
    slippage_bps: float,
    spread_bps: float,
) -> set[int]:
    crit: set[int] = set()
    if entry_indices.size == 0:
        return crit
    slip = (slippage_bps + spread_bps) / 10000.0
    n = len(close)
    for i in entry_indices.tolist():
        if i >= n - 1:
            continue
        entry = float(close[i]) * (1.0 + slip)
        stop_level = entry * (1.0 - float(sl))
        end = min(n - 1, i + max(1, int(hold)))
        for j in range(i + 1, end + 1):
            h = float(high[j]); l = float(low[j])
            stop_hit = l <= stop_level
            tp_hit = bool(use_multi_tp and np.any(h >= entry * (1.0 + tps)))
            trail_can_activate = bool(trail and ((h / entry) - 1.0) >= float(trail_activate))
            is_critical = False
            if (not trail) and (not use_multi_tp):
                is_critical = stop_hit and tp_hit
            elif (not trail) and use_multi_tp:
                is_critical = stop_hit and tp_hit
            elif trail and (not use_multi_tp):
                is_critical = trail_can_activate or stop_hit
            else:
                event_count = int(stop_hit) + int(tp_hit) + int(trail_can_activate)
                is_critical = event_count >= 2
            if is_critical:
                crit.add(int(bar_time_ns[j]))
    return crit


def _load_tick_minute_map_partial(path: Path, datetime_col: str, price_col: str, sep: str, minute_filter: set[int]) -> tuple[np.ndarray, dict[int, tuple[int, int]]]:
    if not minute_filter:
        return np.asarray([], dtype=np.float64), {}
    use_price_col = None if str(price_col).lower() == "auto" else str(price_col)
    chunks = pd.read_csv(path, sep=sep, chunksize=2_000_000)
    frames = []
    for ch in chunks:
        if datetime_col not in ch.columns:
            raise ValueError(f"tick-data missing datetime column: {datetime_col}")
        pcol = use_price_col
        if pcol is None:
            for cand in ("price", "bid", "ask", "last", "close"):
                if cand in ch.columns:
                    pcol = cand
                    break
            if pcol is None:
                raise ValueError("tick-data price column not found")
        dt = pd.to_datetime(ch[datetime_col], errors="coerce", utc=True).dt.floor("min")
        minute_ns = dt.view("int64")
        keep = np.isin(minute_ns, np.asarray(list(minute_filter), dtype=np.int64))
        if not np.any(keep):
            continue
        part = pd.DataFrame({"minute_ns": minute_ns[keep], "price": pd.to_numeric(ch.loc[keep, pcol], errors="coerce")}).dropna()
        if not part.empty:
            frames.append(part)
    if not frames:
        return np.asarray([], dtype=np.float64), {}
    ticks = pd.concat(frames, axis=0).sort_values("minute_ns")
    prices = pd.to_numeric(ticks["price"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    mins = ticks["minute_ns"].to_numpy(dtype=np.int64, copy=False)
    bounds: dict[int, tuple[int, int]] = {}
    i = 0
    while i < len(mins):
        m = int(mins[i]); j = i + 1
        while j < len(mins) and int(mins[j]) == m:
            j += 1
        bounds[m] = (i, j)
        i = j
    return prices, bounds


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
    args = parse_args()
    miner = _load_miner_module(args.miner_script)
    timing: dict[str, float] = {}

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

    high = df["high"].to_numpy(dtype=np.float32, copy=False)
    low = df["low"].to_numpy(dtype=np.float32, copy=False)
    close = df["close"].to_numpy(dtype=np.float32, copy=False)
    bar_time_ns = np.asarray(pd.DatetimeIndex(df.index).floor("min").view("int64"), dtype=np.int64)

    tick_prices_all = None
    tick_minute_bounds = None

    tps_all = [float(x) for x in str(args.tps).split(",") if x.strip()]
    tps = tps_all if bool(args.use_multi_tp) else [tps_all[0]]
    tp_w = miner.parse_tp_weights(tps, str(args.tp_weights))

    cache_key_obj = {
        "features_sig": _file_sig(args.features),
        "tps": [float(x) for x in tps],
        "tp_weights": [float(x) for x in tp_w.tolist()],
        "use_multi_tp": bool(args.use_multi_tp),
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
        pnl, y, t_exit, t_qual, tp_hits = miner.simulate_multitp_trailing_pessimistic(
            high=high,
            low=low,
            close=close,
            tps=tps,
            tp_w=tp_w,
            tp_enabled=bool(args.use_multi_tp),
            sl=float(args.sl),
            hold=int(args.hold),
            slippage_bps=float(args.slippage_bps),
            spread_bps=float(args.spread_bps),
            trail=bool(args.trail),
            trail_activate=float(args.trail_activate),
            trail_offset=float(args.trail_offset),
            trail_factor=float(args.trail_factor),
            trail_min_level=float(args.trail_min_level),
            include_unrealized_at_test_end=bool(args.include_unrealized_at_test_end),
            bar_time_ns=bar_time_ns,
            tick_prices_all=None,
            tick_minute_bounds=None,
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
    y_train = y[:train_idx]
    y_test = y[train_idx:]
    tradable_train = ((y_train == 0) | (y_train == 1))
    tradable_test = ((y_test == 0) | (y_test == 1))

    t0 = time.perf_counter()
    allow_keys, allow_keys_refined = _load_allowlist(args.include_candidates_file)
    coarse_ctx = {
        "features_sig": _file_sig(args.features),
        "binned_sig": _file_sig(args.binned_features),
        "meta_sig": _file_sig(args.binned_metadata),
        "label_cache_key": cache_key,
        "train_frac": float(args.train_frac),
        "quantiles": str(args.quantiles),
        "min_single_pos_hits": int(args.min_single_pos_hits),
        "min_single_lift": float(args.min_single_lift),
        "max_single_mask_count": int(args.max_single_mask_count),
        "family_top_n": int(args.family_top_n),
        "family_split_delta_window": int(bool(args.family_split_delta_window)),
    }
    coarse_ctx_sig = _ctx_sig(coarse_ctx)
    coarse_out = args.out_candidates_coarse_csv or args.out_candidates_csv
    refined_out = args.out_candidates_refined_csv or args.out_candidates_csv
    coarse_resume = _load_stage_csv_if_match(coarse_out, "coarse", coarse_ctx_sig)

    items = []
    filtered_items: list[dict] = []
    single_rejects = {"min_pos": 0, "min_lift": 0, "mask_count": 0, "allowlist": 0}
    if coarse_resume is not None:
        for _, r in coarse_resume.iterrows():
            key = str(r.get("candidate_key", "")).strip()
            if not key:
                continue
            if allow_keys is not None and key not in allow_keys:
                continue
            col = str(r["col"]); op = str(r["op"]); val = float(r["value"])
            xvec = pd.to_numeric(bdf[col] if op == "==" else df[col], errors="coerce").to_numpy(copy=False)
            if op == "==":
                m = np.isfinite(xvec) & (np.abs(xvec - val) <= 1e-6)
            elif op == ">=":
                m = np.isfinite(xvec) & (xvec >= val)
            else:
                m = np.isfinite(xvec) & (xvec <= val)
            filtered_items.append({
                "candidate_key": key, "col": col, "op": op, "value": val, "mask": m,
                "binary": bool(int(r.get("binary", 0))), "_family": str(r.get("family", _candidate_family(col, bool(args.family_split_delta_window)))),
                "_single_pos_hits": int(r.get("coarse_single_pos_hits", 0)),
                "_single_neg_hits": int(r.get("coarse_single_neg_hits", 0)),
                "_single_mask_count": int(r.get("coarse_single_mask_count", 0)),
                "_single_ratio": float(r.get("coarse_single_ratio", 0.0)),
                "coarse_single_pos_hits": int(r.get("coarse_single_pos_hits", 0)),
                "coarse_single_neg_hits": int(r.get("coarse_single_neg_hits", 0)),
                "coarse_single_mask_count": int(r.get("coarse_single_mask_count", 0)),
                "coarse_single_ratio": float(r.get("coarse_single_ratio", 0.0)),
                "coarse_lift": float(r.get("coarse_lift", 0.0)),
                "lift": float(r.get("coarse_lift", 0.0)),
                "ratio": float(r.get("coarse_single_ratio", 0.0)),
            })
        print(f"[prefilter-resume] loaded coarse candidates from {coarse_out} rows={len(filtered_items)}")
    else:
        qs = [float(x) for x in str(args.quantiles).split(",") if x.strip()]
        t0 = time.perf_counter()
        qmap = miner.quantile_thresholds(df.iloc[:train_idx], cols, qs)
        timing["quantile_thresholds_sec"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        for c in cols:
            if c.startswith("dist_"):
                x = pd.to_numeric(df[c], errors="coerce").to_numpy(copy=False)
                for _, thr in qmap.get(c, {}).items():
                    for op in (">=", "<="):
                        m = np.isfinite(x) & ((x >= thr) if op == ">=" else (x <= thr))
                        pos = int(np.sum(m[:train_idx] & (y_train == 1)))
                        neg = int(np.sum(m[:train_idx] & (y_train == 0)))
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
                pos = int(np.sum(m[:train_idx] & (y_train == 1))); neg = int(np.sum(m[:train_idx] & (y_train == 0)))
                if pos <= 0:
                    continue
                items.append({"col": c, "op": "==", "value": float(v), "mask": m, "binary": bool(is_binary),
                              "freq": pos / max(1, int(np.sum(y_train == 1))),
                              "lift": (pos / max(1, int(np.sum(y_train == 1)))) / max(1e-12, (neg / max(1, int(np.sum(y_train == 0))))),
                              "ratio": pos / max(1, neg)})
        timing["build_items_sec"] = time.perf_counter() - t0
        for i, it in enumerate(items, start=1):
            x = dict(it); x["candidate_key"] = f"cand_{i:06d}"
            m = np.asarray(x["mask"], dtype=bool)
            pos_hits = int(np.sum(m[:train_idx] & (y_train == 1))); neg_hits = int(np.sum(m[:train_idx] & (y_train == 0)))
            mask_count = int(np.sum(m[:train_idx]))
            if pos_hits < int(args.min_single_pos_hits):
                single_rejects["min_pos"] += 1; continue
            if float(x["lift"]) < float(args.min_single_lift):
                single_rejects["min_lift"] += 1; continue
            if int(args.max_single_mask_count) > 0 and mask_count > int(args.max_single_mask_count):
                single_rejects["mask_count"] += 1; continue
            if allow_keys is not None and str(x["candidate_key"]) not in allow_keys:
                single_rejects["allowlist"] += 1; continue
            x["_single_pos_hits"] = pos_hits; x["_single_neg_hits"] = neg_hits; x["_single_mask_count"] = mask_count
            x["_single_ratio"] = pos_hits / max(1, neg_hits)
            x["coarse_single_pos_hits"] = int(pos_hits); x["coarse_single_neg_hits"] = int(neg_hits)
            x["coarse_single_mask_count"] = int(mask_count); x["coarse_single_ratio"] = float(x["_single_ratio"]); x["coarse_lift"] = float(x["lift"])
            x["_family"] = _candidate_family(str(x["col"]), bool(args.family_split_delta_window))
            filtered_items.append(x)
        by_mask: dict[str, dict] = {}
        for it in filtered_items:
            mh = _mask_hash_arr(np.asarray(it["mask"][:train_idx], dtype=bool))
            cur = by_mask.get(mh)
            if cur is None:
                by_mask[mh] = it; continue
            tf_cur = int(miner._parse_feature_meta(str(cur["col"])).get("tf") or 10**9)
            tf_new = int(miner._parse_feature_meta(str(it["col"])).get("tf") or 10**9)
            if tf_new < tf_cur:
                by_mask[mh] = it
        filtered_items = list(by_mask.values())
    fam_top: list[dict] = []
    fam_groups: dict[str, list[dict]] = {}
    for it in filtered_items:
        fam_groups.setdefault(str(it["_family"]), []).append(it)
    for _, arr in fam_groups.items():
        arr = sorted(arr, key=lambda z: (-float(z["lift"]), -int(z["_single_pos_hits"]), int(z["_single_mask_count"])))
        fam_top.extend(arr[: int(args.family_top_n)])

    tick_refined_mode = False
    refined_ctx = {
        **coarse_ctx,
        "tick_data_sig": _file_sig(args.tick_data),
        "tick_cache_sig": _file_sig(args.tick_cache_parquet),
        "use_multi_tp": int(bool(args.use_multi_tp)),
        "trail": int(bool(args.trail)),
        "trail_activate": float(args.trail_activate),
        "hold": int(args.hold),
    }
    refined_ctx_sig = _ctx_sig(refined_ctx)
    refined_resume = _load_stage_csv_if_match(refined_out, "refined", refined_ctx_sig)
    if refined_resume is not None:
        by_key = {str(it["candidate_key"]): it for it in fam_top}
        for _, r in refined_resume.iterrows():
            k = str(r.get("candidate_key_refined", "")).strip()
            if not k or k not in by_key:
                continue
            it = by_key[k]
            it["tick_single_pos_hits"] = int(r.get("tick_single_pos_hits", 0))
            it["tick_single_neg_hits"] = int(r.get("tick_single_neg_hits", 0))
            it["tick_single_mask_count"] = int(r.get("tick_single_mask_count", 0))
            it["tick_single_ratio"] = float(r.get("tick_single_ratio", 0.0))
            it["tick_lift"] = float(r.get("tick_lift", 0.0))
            it["_single_pos_hits"] = int(it["tick_single_pos_hits"])
            it["_single_neg_hits"] = int(it["tick_single_neg_hits"])
            it["_single_mask_count"] = int(it["tick_single_mask_count"])
            it["_single_ratio"] = float(it["tick_single_ratio"])
            it["ratio"] = float(it["tick_single_ratio"])
            it["lift"] = float(it["tick_lift"])
        tick_refined_mode = True
        print(f"[prefilter-resume] loaded refined candidates from {refined_out}")
    elif bool(allow_keys_refined):
        if refined_out is None or not refined_out.exists():
            raise ValueError("candidate_key_refined input requires --out-candidates-refined-csv (or --out-candidates-csv) file to reload refined metrics")
        raise ValueError("Refined TXT header provided, but refined candidate CSV context did not match current run.")
    elif args.tick_data is not None and len(fam_top) > 0:
        fam_union = np.zeros(n, dtype=bool)
        for it in fam_top:
            fam_union |= np.asarray(it["mask"], dtype=bool)
        entry_indices = np.flatnonzero(fam_union).astype(np.int64, copy=False)
        critical_minutes = _critical_minutes_for_entries(
            entry_indices=entry_indices,
            high=high,
            low=low,
            close=close,
            bar_time_ns=bar_time_ns,
            hold=int(args.hold),
            use_multi_tp=bool(args.use_multi_tp),
            trail=bool(args.trail),
            trail_activate=float(args.trail_activate),
            tps=np.asarray(tps, dtype=np.float64),
            sl=float(args.sl),
            slippage_bps=float(args.slippage_bps),
            spread_bps=float(args.spread_bps),
        )
        if critical_minutes:
            tick_prices_all, tick_minute_bounds = _load_tick_minute_map_partial(
                path=args.tick_data,
                datetime_col=args.tick_datetime_column,
                price_col=args.tick_price_column,
                sep=args.tick_sep,
                minute_filter=critical_minutes,
            )
        else:
            tick_prices_all, tick_minute_bounds = np.asarray([], dtype=np.float64), {}
        t0 = time.perf_counter()
        tick_map = miner.simulate_selected_entries_with_ticks(
            entry_indices=entry_indices,
            high=high,
            low=low,
            close=close,
            tps=tps,
            tp_w=tp_w,
            tp_enabled=bool(args.use_multi_tp),
            sl=float(args.sl),
            hold=int(args.hold),
            slippage_bps=float(args.slippage_bps),
            spread_bps=float(args.spread_bps),
            trail=bool(args.trail),
            trail_activate=float(args.trail_activate),
            trail_offset=float(args.trail_offset),
            trail_factor=float(args.trail_factor),
            include_unrealized_at_test_end=bool(args.include_unrealized_at_test_end),
            bar_time_ns=bar_time_ns,
            tick_prices_all=tick_prices_all,
            tick_minute_bounds=tick_minute_bounds,
        )
        timing["tick_refine_sec"] = time.perf_counter() - t0
        y_ref = np.asarray(y, dtype=np.int8).copy()
        for idx_i, rec in tick_map.items():
            y_ref[int(idx_i)] = np.int8(int(rec.get("y", -1)))
        y_ref_train = y_ref[:train_idx]
        union_train_mask = fam_union[:train_idx] & tradable_train
        union_pos = int(np.sum(union_train_mask & (y_ref_train == 1)))
        union_neg = int(np.sum(union_train_mask & (y_ref_train == 0)))
        for it in fam_top:
            m_train = np.asarray(it["mask"][:train_idx], dtype=bool) & tradable_train
            tick_pos = int(np.sum(m_train & (y_ref_train == 1)))
            tick_neg = int(np.sum(m_train & (y_ref_train == 0)))
            tick_ratio = tick_pos / max(1, tick_neg)
            tick_lift = (tick_pos / max(1, union_pos)) / max(1e-12, (tick_neg / max(1, union_neg)))
            it["tick_single_pos_hits"] = tick_pos
            it["tick_single_neg_hits"] = tick_neg
            it["tick_single_mask_count"] = int(np.sum(m_train))
            it["tick_single_ratio"] = float(tick_ratio)
            it["tick_lift"] = float(tick_lift)
            # downstream combinatorics should use refined single stats
            it["_single_pos_hits"] = tick_pos
            it["_single_neg_hits"] = tick_neg
            it["_single_mask_count"] = int(np.sum(m_train))
            it["_single_ratio"] = float(tick_ratio)
            it["ratio"] = float(tick_ratio)
            it["lift"] = float(tick_lift)
        tick_refined_mode = True
        print(f"[prefilter] tick-refined family-top pool on {len(entry_indices)} union entry rows.")
    if coarse_out is not None:
        coarse_rows = []
        fam_top_keys = {str(z.get("candidate_key")) for z in fam_top}
        for it in filtered_items:
            coarse_rows.append({
                "candidate_key": str(it["candidate_key"]),
                "col": str(it["col"]),
                "op": str(it["op"]),
                "value": float(it["value"]),
                "family": str(it["_family"]),
                "coarse_single_pos_hits": int(it.get("coarse_single_pos_hits", it.get("_single_pos_hits", 0))),
                "coarse_single_neg_hits": int(it.get("coarse_single_neg_hits", it.get("_single_neg_hits", 0))),
                "coarse_single_mask_count": int(it.get("coarse_single_mask_count", it.get("_single_mask_count", 0))),
                "coarse_single_ratio": float(it.get("coarse_single_ratio", it.get("_single_ratio", it.get("ratio", 0.0)))),
                "coarse_lift": float(it.get("coarse_lift", it.get("lift", 0.0))),
                "binary": int(bool(it.get("binary", False))),
                "kept_after_family_topn": int(str(it["candidate_key"]) in fam_top_keys),
                "__stage": "coarse",
                "__ctx_sig": coarse_ctx_sig,
            })
        _atomic_write_csv(coarse_out, pd.DataFrame(coarse_rows))
        print(f"[prefilter-candidates] wrote coarse CSV: {coarse_out} rows={len(coarse_rows)}")

    if refined_out is not None:
        fam_top_keys = {str(z.get("candidate_key")) for z in fam_top}
        cand_rows = []
        for it in filtered_items:
            row = {
                "candidate_key_refined": str(it["candidate_key"]),
                "col": str(it["col"]),
                "op": str(it["op"]),
                "value": float(it["value"]),
                "family": str(it["_family"]),
                "coarse_single_pos_hits": int(it.get("coarse_single_pos_hits", it.get("_single_pos_hits", 0))),
                "coarse_single_neg_hits": int(it.get("coarse_single_neg_hits", it.get("_single_neg_hits", 0))),
                "coarse_single_mask_count": int(it.get("coarse_single_mask_count", it.get("_single_mask_count", 0))),
                "coarse_single_ratio": float(it.get("coarse_single_ratio", it.get("_single_ratio", it.get("ratio", 0.0)))),
                "coarse_lift": float(it.get("coarse_lift", it.get("lift", 0.0))),
                "tick_single_pos_hits": float("nan"),
                "tick_single_neg_hits": float("nan"),
                "tick_single_mask_count": float("nan"),
                "tick_single_ratio": float("nan"),
                "tick_lift": float("nan"),
                "binary": int(bool(it.get("binary", False))),
                "kept_after_family_topn": int(str(it["candidate_key"]) in fam_top_keys),
                "__stage": "refined",
                "__ctx_sig": refined_ctx_sig,
            }
            top_hit = next((z for z in fam_top if str(z.get("candidate_key")) == str(it["candidate_key"])), None)
            if top_hit is not None:
                row["tick_single_pos_hits"] = float(top_hit.get("tick_single_pos_hits", np.nan))
                row["tick_single_neg_hits"] = float(top_hit.get("tick_single_neg_hits", np.nan))
                row["tick_single_mask_count"] = float(top_hit.get("tick_single_mask_count", np.nan))
                row["tick_single_ratio"] = float(top_hit.get("tick_single_ratio", np.nan))
                row["tick_lift"] = float(top_hit.get("tick_lift", np.nan))
            cand_rows.append(row)
        _atomic_write_csv(refined_out, pd.DataFrame(cand_rows))
        print(f"[prefilter-candidates] wrote refined CSV: {refined_out} rows={len(cand_rows)}")

    if tick_refined_mode:
        rank_lift = _with_rank_context(sorted(fam_top, key=lambda z: (-float(z.get("ratio", 0.0)), -int(z.get("_single_pos_hits", 0)), int(z.get("_single_mask_count", 0)))), "ratio")
    else:
        rank_lift = _with_rank_context(sorted(fam_top, key=lambda z: z["lift"], reverse=True), "lift")
    rank_freq: list[dict] = []
    rank_ratio: list[dict] = []
    timing["ranking_prep_sec"] = time.perf_counter() - t0

    if bool(args.debug_timing_breakdown):
        print("[prefilter-timing] " + " | ".join([f"{k}={v:.3f}s" for k, v in timing.items()]))
    if bool(args.debug_timing_breakdown) or bool(args.debug_reject_stats):
        dist_items = sum(1 for x in items if str(x["col"]).startswith("dist_"))
        non_dist_items = len(items) - dist_items
        bin_items = sum(1 for x in items if bool(x.get("binary", False)))
        non_bin_items = len(items) - bin_items
        print(
            f"[prefilter-items] cols={len(cols)} items={len(items)} "
            f"filtered_items={len(filtered_items)} family_top_pool={len(rank_lift)} "
            f"dist_items={dist_items} non_dist_items={non_dist_items} "
            f"binary_items={bin_items} non_binary_items={non_bin_items}"
        )
        if bool(args.debug_reject_stats):
            print("[prefilter-single-rejects] " + " ".join([f"{k}={v}" for k, v in single_rejects.items()]))

    workers = max(1, int(os.cpu_count() or 1)) if str(args.workers).lower() == "auto" else max(1, int(args.workers))

    best_paths: list[dict] = []
    unlocked = 0
    prev_best = -np.inf
    same_ref = miner._build_same_reference_groups(df.iloc[:train_idx], [(str(c), ">=", 0.0) for c in cols])
    reject_stats: dict[str, int] = {
        "rejected_min_pos_per_week": 0,
        "rejected_min_main_score": 0,
        "rejected_same_parent_mask": 0,
        "rejected_not_strictly_better_than_parent": 0,
        "rejected_same_reference": 0,
        "rejected_bundle_anchor": 0,
        "rejected_binary_cap": 0,
        "rejected_duplicate_mask": 0,
    }

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

    def _score_from_mask(mask: np.ndarray) -> dict | None:
        mt = mask[:train_idx] & tradable_train
        pos_hits = int(np.sum(mt & (y_train == 1)))
        neg_hits = int(np.sum(mt & (y_train == 0)))
        days = max(1.0, float((df.index[train_idx - 1] - df.index[0]).total_seconds() / 86400.0))
        weeks = days / 7.0
        if pos_hits < float(args.min_pos_per_week) * weeks:
            if bool(args.debug_reject_stats):
                reject_stats["rejected_min_pos_per_week"] += 1
            return None
        ratio = pos_hits / max(1, neg_hits)
        if ratio < float(args.min_main_score):
            if bool(args.debug_reject_stats):
                reject_stats["rejected_min_main_score"] += 1
            return None
        mt_test = mask[train_idx:] & tradable_test
        pos_test = int(np.sum(mt_test & (y_test == 1)))
        neg_test = int(np.sum(mt_test & (y_test == 0)))
        ratio_test = pos_test / max(1, neg_test)
        wf_mean, wf_min, wf_hits = _calc_wf(mt_test, y_test, int(args.wf_folds))
        precision = pos_hits / max(1, (pos_hits + neg_hits))
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
        }

    def _train_key(r: dict) -> tuple[float, int, int]:
        return (float(r["ratio"]), int(r["pos_hits"]), -int(r["neg_hits"]))

    def _test_key(r: dict) -> tuple[float, int, int]:
        return (float(r["test_ratio"]), int(r["test_pos_hits"]), -int(r["test_neg_hits"]))

    def _mask_hash(mask: np.ndarray) -> str:
        return _mask_hash_arr(mask)

    def evaluate_combo(combo: Tuple[int, ...], pool_items: List[dict], parent: dict | None = None) -> dict | None:
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
            **sc,
        }

    csv_columns = [
        "path_index", "rule_human", "rule_json_id", "decode_type_info", "pos_hits", "neg_hits",
        "remaining_hit_ratio", "precision_info", "test_pos_hits", "test_neg_hits", "test_ratio",
        "wf_mean_ratio", "wf_min_ratio", "wf_hits", "tp", "sl", "hold", "trail",
        "trail_activate", "trail_offset", "trail_factor", "is_fallback_export",
    ]

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
                type_parts.append(f"{col}:{ftype}")
            rows.append({
                "path_index": i,
                "rule_human": " & ".join(decoded_parts),
                "rule_json_id": f"rule_{i}",
                "decode_type_info": " | ".join(type_parts),
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
                "tp": ",".join([f"{x:.8g}" for x in tps]),
                "sl": float(args.sl),
                "hold": int(args.hold),
                "trail": int(args.trail),
                "trail_activate": float(args.trail_activate),
                "trail_offset": float(args.trail_offset),
                "trail_factor": float(args.trail_factor),
                "is_fallback_export": int(bool(is_fallback_export)),
            })
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
        rules_only = {"version": 2, "rules": valid_pool}
        _atomic_write_text(args.out_rules_json, json.dumps(rules_only, ensure_ascii=False, indent=2))
        rows = _rules_to_rows(valid_pool[: int(args.top_paths)], is_fallback_export=False)
        _validate_rows(rows)
        if rows:
            _atomic_write_csv(args.out_rules_csv, pd.DataFrame(rows))
        else:
            _atomic_write_csv(args.out_rules_csv, pd.DataFrame(columns=csv_columns))

    rng = random.Random(int(args.batch_random_seed))
    valid_pool: list[dict] = []
    progress_state: dict = {}
    try:
        while True:
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
                break
            idxs = list(range(len(pool)))
            phase_a = len(valid_pool) < int(args.max_valids)
            shard_specs: list[tuple[int, int, int, int]] = []
            batch_eff = max(256, int(args.batch_size))
            sid = 0
            if phase_a:
                for r in range(2, int(args.max_path_conds) + 1):
                    if r > len(idxs):
                        continue
                    total_r = math.comb(len(idxs), r)
                    for st in range(0, total_r, batch_eff):
                        shard_specs.append((sid, r, st, min(batch_eff, total_r - st)))
                        sid += 1
            else:
                seeds = sorted(valid_pool, key=lambda z: (-float(z["ratio"]), -int(z["pos_hits"]), int(z["neg_hits"])))[: int(args.max_valids)]
                for p in seeds:
                    cset = set(int(x) for x in p.get("_combo", ()))
                    for add in idxs:
                        if add in cset:
                            continue
                        combo = tuple(sorted(cset | {add}))
                        if 2 <= len(combo) <= int(args.max_path_conds):
                            shard_specs.append((sid, -1, 0, 0))
                            sid += 1
                # shuffle mutation order globally
            rng.shuffle(shard_specs)
            tested = 0
            valid_round = 0
            mut_i = 0
            seeds = sorted(valid_pool, key=lambda z: (-float(z["ratio"]), -int(z["pos_hits"]), int(z["neg_hits"])))[: int(args.max_valids)]
            mut_combos: list[tuple[tuple[int, ...], dict]] = []
            if (not phase_a) or (phase_a and len(seeds) > 0 and unlocked_next > int(args.step_size)):
                for p in seeds:
                    cset = set(int(x) for x in p.get("_combo", ()))
                    for add in idxs:
                        if add in cset:
                            continue
                        combo = tuple(sorted(cset | {add}))
                        if 2 <= len(combo) <= int(args.max_path_conds):
                            mut_combos.append((combo, p))
                rng.shuffle(mut_combos)
            combos_total_free = 0
            rmax = min(int(args.max_path_conds), len(idxs))
            for rr in range(2, rmax + 1):
                combos_total_free += math.comb(len(idxs), rr)
            combos_total_parent_extensions = int(len(mut_combos))
            combos_total = int(combos_total_free + combos_total_parent_extensions)

            hist: list[tuple[int, float, tuple[float, int, int]]] = []
            executor_cache: dict[int, concurrent.futures.ThreadPoolExecutor] = {}
            for spec in shard_specs:
                est_mem_gb = (batch_eff * max(1, len(pool)) * 8.0) / (1024 ** 3)
                target_mem_gb = max(0.5, float(args.memory_soft_limit_gb) * 0.75)
                while est_mem_gb > target_mem_gb and batch_eff > 256:
                    batch_eff = max(256, batch_eff // 2)
                    est_mem_gb = (batch_eff * max(1, len(pool)) * 8.0) / (1024 ** 3)
                workers_eff = workers if est_mem_gb <= target_mem_gb else max(1, workers // 2)
                if phase_a:
                    _sid, rr, st, cnt = spec
                    new_combos = list(itertools.islice(itertools.combinations(idxs, rr), st, st + cnt))
                    combos = list(new_combos)
                    parents = [None] * len(new_combos)
                    # from pool 2 onward in phase A: also expand existing valid rules in parallel
                    if mut_i < len(mut_combos):
                        fill_n = max(1, batch_eff // 2)
                        exp_part = mut_combos[mut_i: mut_i + fill_n]
                        mut_i += len(exp_part)
                        combos.extend([x[0] for x in exp_part])
                        parents.extend([x[1] for x in exp_part])
                else:
                    combos_par = mut_combos[mut_i: mut_i + batch_eff]
                    mut_i += len(combos_par)
                    combos = [x[0] for x in combos_par]
                    parents = [x[1] for x in combos_par]
                    if not combos:
                        break
                out_batch: list[dict] = []
                if workers_eff <= 1:
                    for i, cb in enumerate(combos):
                        rr = evaluate_combo(cb, pool, parents[i])
                        if rr is not None:
                            out_batch.append(rr)
                else:
                    ex = executor_cache.get(int(workers_eff))
                    if ex is None:
                        ex = concurrent.futures.ThreadPoolExecutor(max_workers=int(workers_eff))
                        executor_cache[int(workers_eff)] = ex
                    futs = [ex.submit(evaluate_combo, cb, pool, parents[i]) for i, cb in enumerate(combos)]
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
                    "combos_total": int(combos_total),
                    "tested": int(tested),
                    "valid": int(valid_round),
                    "batch_size": int(batch_eff),
                    "workers": int(workers_eff),
                    "best_a1": list(a1),
                    "topk_summary": [{"ratio": float(x["ratio"]), "pos_hits": int(x["pos_hits"]), "neg_hits": int(x["neg_hits"])} for x in topk[:10]],
                    "phase": "A" if phase_a else "B",
                }
                _save_progress(valid_pool, progress_state)
                print(
                    f"[prefilter-progress] round={unlocked_next} phase={'A' if phase_a else 'B'} "
                    f"pool_size={len(pool)} combos_total_free={progress_state['combos_total_free']} "
                    f"combos_total_parent_extensions={progress_state['combos_total_parent_extensions']} "
                    f"combos_total={progress_state['combos_total']} "
                    f"tested={tested} valid={valid_round} batch_size={batch_eff} workers={workers_eff}"
                )
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
                        print("[prefilter-progress] early stop in round.")
                        break

            for ex in executor_cache.values():
                ex.shutdown(wait=True)

            if not valid_pool:
                unlocked = unlocked_next
                if unlocked >= len(rank_lift):
                    break
                continue
            best_paths = list(valid_pool)
            cur_best = float(best_paths[0]["ratio"]) if best_paths else -np.inf
            if cur_best <= prev_best + 1e-12:
                break
            prev_best = cur_best
            unlocked = unlocked_next
    except KeyboardInterrupt:
        print("[prefilter] interrupted; checkpoint saved.")
        _save_progress(valid_pool, progress_state)

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
        return merged_rule

    best_paths = [_apply_neighbor_merge(r) for r in best_paths]
    best_paths = _dedupe_mask(best_paths)
    train_ranked = sorted(best_paths, key=lambda z: (-float(z["ratio"]), -int(z["pos_hits"]), int(z["neg_hits"])))
    train_rank_map = {id(r): i for i, r in enumerate(train_ranked)}
    shortlist_n = max(100, int(args.top_paths) * 4)
    shortlist = train_ranked[:shortlist_n]
    shortlist = [r for r in shortlist if (int(r["test_pos_hits"]) + int(r["test_neg_hits"])) > 0]
    shortlist = sorted(
        shortlist,
        key=lambda z: (
            -float(z["test_ratio"]),
            -int(z["test_pos_hits"]),
            int(z["test_neg_hits"]),
            int(train_rank_map.get(id(z), 10 ** 9)),
        ),
    )
    best_paths = shortlist[: int(args.top_paths)]
    fallback_export_used = False
    if not best_paths:
        print("[prefilter-warning] No rule with test hits found for final CSV export.")
        if len(train_ranked) > 0 and len(valid_pool) > 0:
            best_paths = train_ranked[: int(args.top_paths)]
            fallback_export_used = True
            print(f"[prefilter-fallback] Exporting {len(best_paths)} train-valid rules (fallback: valid>0 but test_hits=0).")

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
    _atomic_write_text(args.out_rules_json, json.dumps(out_json, ensure_ascii=False, indent=2))

    rows = _rules_to_rows(best_paths, is_fallback_export=fallback_export_used)
    _validate_rows(rows)
    if rows:
        _atomic_write_csv(args.out_rules_csv, pd.DataFrame(rows))
    else:
        _atomic_write_csv(args.out_rules_csv, pd.DataFrame(columns=csv_columns))
    if bool(args.debug_reject_stats):
        print("[prefilter-reject-stats-total] " + " ".join([f"{k}={v}" for k, v in reject_stats.items()]))
    print(f"Saved {len(rows)} rules -> {args.out_rules_json} and {args.out_rules_csv}")


if __name__ == "__main__":
    main()
