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
    if args.tick_data is not None:
        tick_prices_all, tick_minute_bounds = miner.load_tick_minute_map(
            path=args.tick_data,
            datetime_col=args.tick_datetime_column,
            price_col=args.tick_price_column,
            sep=args.tick_sep,
            cache_parquet=args.tick_cache_parquet,
        )

    tps_all = [float(x) for x in str(args.tps).split(",") if x.strip()]
    tps = tps_all if bool(args.use_multi_tp) else [tps_all[0]]
    tp_w = miner.parse_tp_weights(tps, str(args.tp_weights))

    t0 = time.perf_counter()
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
        tick_prices_all=tick_prices_all,
        tick_minute_bounds=tick_minute_bounds,
    )
    timing["simulate_labels_sec"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    cols = miner.build_candidate_features(df, allow_absolute_price=False, max_features=0)
    timing["build_candidate_features_sec"] = time.perf_counter() - t0
    y_train = y[:train_idx]
    y_test = y[train_idx:]
    tradable_train = ((y_train == 0) | (y_train == 1))
    tradable_test = ((y_test == 0) | (y_test == 1))

    # Build item pools from three ranked lists.
    items = []
    qs = [float(x) for x in str(args.quantiles).split(",") if x.strip()]
    t0 = time.perf_counter()
    qmap = miner.quantile_thresholds(df.iloc[:train_idx], cols, qs)
    timing["quantile_thresholds_sec"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    for c in cols:
        if c.startswith("dist_"):
            # raw threshold candidates from quantiles
            x = pd.to_numeric(df[c], errors="coerce").to_numpy(copy=False)
            for q, thr in qmap.get(c, {}).items():
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
        vals = np.unique(b[:train_idx])
        vals = vals[vals != miss]
        is_binary = str(meta.get(c, {}).get("feature_type", "")) == "binary"
        for v in vals.tolist():
            m = np.isfinite(b) & (np.abs(b - float(v)) <= 1e-6)
            pos = int(np.sum(m[:train_idx] & (y_train == 1)))
            neg = int(np.sum(m[:train_idx] & (y_train == 0)))
            if pos <= 0:
                continue
            items.append({"col": c, "op": "==", "value": float(v), "mask": m, "binary": bool(is_binary),
                          "freq": pos / max(1, int(np.sum(y_train == 1))),
                          "lift": (pos / max(1, int(np.sum(y_train == 1)))) / max(1e-12, (neg / max(1, int(np.sum(y_train == 0))))),
                          "ratio": pos / max(1, neg)})

    timing["build_items_sec"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    rank_freq = _with_rank_context(sorted(items, key=lambda z: z["freq"], reverse=True), "freq")
    rank_lift = _with_rank_context(sorted(items, key=lambda z: z["lift"], reverse=True), "lift")
    rank_ratio = _with_rank_context(sorted(items, key=lambda z: z["ratio"], reverse=True), "ratio")
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
            f"rank_freq={len(rank_freq)} rank_lift={len(rank_lift)} rank_ratio={len(rank_ratio)} "
            f"dist_items={dist_items} non_dist_items={non_dist_items} "
            f"binary_items={bin_items} non_binary_items={non_bin_items}"
        )

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
        packed = np.packbits(mask.astype(np.uint8, copy=False))
        return hashlib.sha1(packed.tobytes()).hexdigest()

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
        "trail_activate", "trail_offset", "trail_factor",
    ]

    def _rules_to_rows(rules: list[dict]) -> list[dict]:
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
                        lo_bin = float(c["lo_bin"])
                        hi_bin = float(c["hi_bin"])
                        lo_txt, _ = miner.decode_bin_condition_verbose(col, lo_bin, meta)
                        hi_txt, _ = miner.decode_bin_condition_verbose(col, hi_bin, meta)
                        decoded_parts.append(f"{col} in [{int(lo_bin)}, {int(hi_bin)}] ({lo_txt}; {hi_txt})")
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
            })
        return rows

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
        rows = _rules_to_rows(valid_pool[: int(args.top_paths)])
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
                rank_lists=[rank_freq, rank_lift, rank_ratio],
                all_candidate_cols=cols,
                unlocked_next=unlocked_next,
                step_size=int(args.step_size),
                binary_cap_per_list_block=int(args.binary_cap_per_list_block),
                binary_anchor_lookahead_blocks=int(args.binary_anchor_lookahead_blocks),
                binary_cap_per_block=int(args.binary_cap_per_block),
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

            hist: list[tuple[int, float, tuple[float, int, int]]] = []
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
                    with concurrent.futures.ThreadPoolExecutor(max_workers=workers_eff) as ex:
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
                    "combos_total": int(len(shard_specs) * max(1, batch_eff)),
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
                    f"pool_size={len(pool)} combos_total={progress_state['combos_total']} "
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

            if not valid_pool:
                break
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
    }
    _atomic_write_text(args.out_rules_json, json.dumps(out_json, ensure_ascii=False, indent=2))

    rows = _rules_to_rows(best_paths)
    if rows:
        _atomic_write_csv(args.out_rules_csv, pd.DataFrame(rows))
    else:
        _atomic_write_csv(args.out_rules_csv, pd.DataFrame(columns=csv_columns))
    if bool(args.debug_reject_stats):
        print("[prefilter-reject-stats-total] " + " ".join([f"{k}={v}" for k, v in reject_stats.items()]))
    print(f"Saved {len(rows)} rules -> {args.out_rules_json} and {args.out_rules_csv}")


if __name__ == "__main__":
    main()
