#!/usr/bin/env python3
"""Standalone in-detail prefilter combinatorial search.

Builds labels via TP/SL/Hold/Trail simulation directly from features OHLC,
then searches complete rule combinations (no greedy extension) over unlocked pools.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import itertools
import json
import math
import os
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


def _with_rank_context(ranked: list[dict], rank_name: str) -> list[dict]:
    out: list[dict] = []
    for i, it in enumerate(ranked):
        z = dict(it)
        z["_rank_name"] = rank_name
        z["_rank_pos"] = int(i)
        out.append(z)
    return out


def _build_unlocked_pool(
    rank_lists: list[list[dict]],
    unlocked_next: int,
    step_size: int,
    binary_cap_per_list_block: int,
    binary_anchor_lookahead_blocks: int,
) -> list[dict]:
    if unlocked_next <= 0:
        return []
    block_size = max(1, int(step_size))
    unlocked_blocks = int(math.ceil(float(unlocked_next) / float(block_size)))
    lookahead_blocks = max(0, int(binary_anchor_lookahead_blocks))
    merged: dict[tuple[str, str, float], dict] = {}

    def _add_item(it: dict, from_lookahead: bool) -> None:
        key = (str(it["col"]), str(it["op"]), float(it["value"]))
        cur = merged.get(key)
        cand = dict(it)
        cand["_from_lookahead"] = bool(from_lookahead)
        if cur is None:
            merged[key] = cand
            return
        # Neighbor-merge preference: keep non-lookahead and better rank position.
        cur_la = bool(cur.get("_from_lookahead", False))
        cand_la = bool(cand.get("_from_lookahead", False))
        if cur_la and not cand_la:
            merged[key] = cand
            return
        if cur_la == cand_la and int(cand.get("_rank_pos", 10 ** 9)) < int(cur.get("_rank_pos", 10 ** 9)):
            merged[key] = cand

    for rank in rank_lists:
        blocks = _chunk_list(rank, block_size)
        for bi in range(min(len(blocks), unlocked_blocks)):
            bin_count = 0
            for it in blocks[bi]:
                if bool(it.get("binary", False)):
                    if bin_count >= int(binary_cap_per_list_block):
                        continue
                    bin_count += 1
                _add_item(it, from_lookahead=False)
        if lookahead_blocks > 0:
            start = unlocked_blocks
            stop = min(len(blocks), unlocked_blocks + lookahead_blocks)
            for bi in range(start, stop):
                bin_count = 0
                for it in blocks[bi]:
                    if not bool(it.get("binary", False)):
                        continue
                    if bin_count >= int(binary_cap_per_list_block):
                        continue
                    bin_count += 1
                    _add_item(it, from_lookahead=True)

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

    df = miner.load_features(args.features)
    bdf = miner.load_binned_features(args.binned_features, tail_rows=0).reindex(df.index)
    meta = miner.load_binned_metadata(args.binned_metadata)

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

    cols = miner.build_candidate_features(df, allow_absolute_price_features=False, max_features=0)
    y_train = y[:train_idx]
    y_test = y[train_idx:]
    tradable_train = ((y_train == 0) | (y_train == 1))
    tradable_test = ((y_test == 0) | (y_test == 1))

    # Build item pools from three ranked lists.
    items = []
    qs = [float(x) for x in str(args.quantiles).split(",") if x.strip()]
    qmap = miner.quantile_thresholds(df.iloc[:train_idx], cols, qs)

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

    rank_freq = _with_rank_context(sorted(items, key=lambda z: z["freq"], reverse=True), "freq")
    rank_lift = _with_rank_context(sorted(items, key=lambda z: z["lift"], reverse=True), "lift")
    rank_ratio = _with_rank_context(sorted(items, key=lambda z: z["ratio"], reverse=True), "ratio")

    workers = max(1, int(os.cpu_count() or 1)) if str(args.workers).lower() == "auto" else max(1, int(args.workers))

    best_paths: list[dict] = []
    unlocked = 0
    prev_best = -np.inf
    same_ref = miner._build_same_reference_groups(df.iloc[:train_idx], [(str(c), ">=", 0.0) for c in cols])

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
            return None
        ratio = pos_hits / max(1, neg_hits)
        if ratio < float(args.min_main_score):
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

    def evaluate_combo(combo: Tuple[int, ...], pool_items: List[dict]) -> dict | None:
        conds = [pool_items[i] for i in combo]
        dedup = []
        seen_gid = set()
        for c in conds:
            fam = str(miner._parse_feature_meta(str(c["col"])).get("family", ""))
            if fam in {"dist_support", "dist_resist"}:
                gid = int(same_ref.get(str(c["col"]), 0))
                if gid > 0 and gid in seen_gid:
                    continue
                if gid > 0:
                    seen_gid.add(gid)
            dedup.append(c)
        conds = dedup
        # Binary flood cap per unlocked block
        if sum(1 for c in conds if c["binary"]) > int(args.binary_cap_per_block):
            return None
        # bundle/anchor validity
        cond_triplets = [(str(c["col"]), str(c["op"]), float(c["value"])) for c in conds]
        ok_bundle, _ = miner.validate_binary_anchor_invariant(cond_triplets, cols)
        if not ok_bundle:
            return None

        mask = _build_mask_from_conds(
            [{"col": c["col"], "op": c["op"], "value": c["value"], "domain": "auto"} for c in conds]
        )
        sc = _score_from_mask(mask)
        if sc is None:
            return None
        return {
            "conds": [{"col": str(c["col"]), "op": str(c["op"]), "value": float(c["value"]), "domain": "auto"} for c in conds],
            **sc,
        }

    while True:
        unlocked_next = unlocked + int(args.step_size)
        pool = _build_unlocked_pool(
            rank_lists=[rank_freq, rank_lift, rank_ratio],
            unlocked_next=unlocked_next,
            step_size=int(args.step_size),
            binary_cap_per_list_block=int(args.binary_cap_per_list_block),
            binary_anchor_lookahead_blocks=int(args.binary_anchor_lookahead_blocks),
        )
        if not pool:
            break

        unlocked_blocks = max(1, int(math.ceil(float(unlocked_next) / float(max(1, int(args.step_size))))))
        global_binary_cap = int(args.binary_cap_per_block) * unlocked_blocks
        # binary cap over unlocked blocks (lookahead binaries are excluded from this cap)
        bin_idx = [i for i, it in enumerate(pool) if bool(it["binary"])]
        keep_bin_idx = [
            i
            for i in bin_idx
            if not bool(pool[i].get("_from_lookahead", False))
        ]
        if len(keep_bin_idx) > global_binary_cap:
            keep = set(keep_bin_idx[:global_binary_cap])
            keep.update(i for i, it in enumerate(pool) if not bool(it["binary"]))
            keep.update(i for i, it in enumerate(pool) if bool(it["binary"]) and bool(it.get("_from_lookahead", False)))
            pool = [pool[i] for i in sorted(keep)]

        idxs = list(range(len(pool)))
        combos_iter = itertools.chain.from_iterable(
            itertools.combinations(idxs, r) for r in range(2, int(args.max_path_conds) + 1)
        )

        results: List[dict] = []
        for batch in _batched(combos_iter, int(args.batch_size)):
            mem_gb = (len(batch) * max(1, len(pool)) * 8.0) / (1024 ** 3)
            if mem_gb > float(args.memory_soft_limit_gb):
                overload = mem_gb / max(1e-9, float(args.memory_soft_limit_gb))
                workers_eff = max(1, int(workers / max(1.0, overload)))
            else:
                workers_eff = workers
            if workers_eff <= 1:
                for combo in batch:
                    r = evaluate_combo(combo, pool)
                    if r is not None:
                        results.append(r)
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=workers_eff) as ex:
                    futs = [ex.submit(evaluate_combo, combo, pool) for combo in batch]
                    for f in futs:
                        r = f.result()
                        if r is not None:
                            results.append(r)

        if not results:
            break

        results = sorted(results, key=lambda z: (-float(z["ratio"]), -int(z["pos_hits"]), int(z["neg_hits"])))
        best_paths = results[: int(args.top_paths)]
        cur_best = float(best_paths[0]["ratio"]) if best_paths else -np.inf
        if cur_best <= prev_best + 1e-12:
            break
        prev_best = cur_best
        unlocked = unlocked_next

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
    args.out_rules_json.write_text(json.dumps(out_json, ensure_ascii=False, indent=2), encoding="utf-8")

    # Human-readable CSV
    rows = []
    for i, r in enumerate(best_paths, start=1):
        decoded_parts = []
        type_parts = []
        for c in r["conds"]:
            col = str(c["col"])
            op = str(c["op"])
            val = float(c["value"])
            ftype = str(meta.get(col, {}).get("feature_type", "unknown"))
            if op == "==":
                decoded_parts.append(miner.decode_bin_condition(col, val, meta))
            else:
                decoded_parts.append(f"{col} {op} {val:.6g}")
            type_parts.append(f"{col}:{ftype}")
        txt = " & ".join(decoded_parts)
        rows.append({
            "path_index": i,
            "rule_human": txt,
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
    pd.DataFrame(rows).to_csv(args.out_rules_csv, index=False)
    print(f"Saved {len(rows)} rules -> {args.out_rules_json} and {args.out_rules_csv}")


if __name__ == "__main__":
    main()
