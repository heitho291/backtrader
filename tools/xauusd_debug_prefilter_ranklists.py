#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_miner_module(path: Path):
    spec = importlib.util.spec_from_file_location("xau_miner_module_debug_ranklists", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load miner script: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--binned-features", type=Path, required=True)
    p.add_argument("--binned-metadata", type=Path, required=True)
    p.add_argument("--miner-script", type=Path, default=Path("tools/xauusd_miner_ohlc_first_hit_pessimistic.py"))

    p.add_argument("--train-frac", type=float, default=0.7)
    p.add_argument("--quantiles", type=str, default="0.05,0.10,0.90,0.95")
    p.add_argument("--top", type=int, default=300)

    p.add_argument("--tps", type=str, default="0.0025")
    p.add_argument("--tp-weights", type=str, default="")
    p.add_argument("--use-multi-tp", action="store_true", default=False)
    p.add_argument("--sl", type=float, default=0.0025)
    p.add_argument("--hold", type=int, default=90)
    p.add_argument("--slippage-bps", type=float, default=0.1)
    p.add_argument("--spread-bps", type=float, default=0.68)
    p.add_argument("--trail", action="store_true", default=False)
    p.add_argument("--trail-activate", type=float, default=0.0010)
    p.add_argument("--trail-offset", type=float, default=0.0006)
    p.add_argument("--trail-factor", type=float, default=0.5)
    p.add_argument("--trail-min-level", type=float, default=0.0)
    p.add_argument("--include-unrealized-at-test-end", action="store_true", default=True)

    p.add_argument("--out", type=Path, default=Path("debug_prefilter_ranklists.csv"))
    return p.parse_args()


def main():
    args = parse_args()
    miner = load_miner_module(args.miner_script)

    print("[debug] loading features...")
    df = miner.load_features(args.features)

    print("[debug] loading binned features...")
    bdf = miner.load_binned_features(args.binned_features, tail_rows=0).reindex(df.index)

    print("[debug] loading metadata...")
    meta = miner.load_binned_metadata(args.binned_metadata)

    n = len(df)
    train_idx = max(1, min(int(n * args.train_frac), n - 1))

    high = df["high"].to_numpy(dtype=np.float32, copy=False)
    low = df["low"].to_numpy(dtype=np.float32, copy=False)
    close = df["close"].to_numpy(dtype=np.float32, copy=False)
    bar_time_ns = np.asarray(pd.DatetimeIndex(df.index).floor("min").view("int64"), dtype=np.int64)

    tps_all = [float(x) for x in str(args.tps).split(",") if x.strip()]
    tps = tps_all if args.use_multi_tp else [tps_all[0]]
    tp_w = miner.parse_tp_weights(tps, str(args.tp_weights))

    print("[debug] simulating labels...")
    _, y, _, _, _ = miner.simulate_multitp_trailing_pessimistic(
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

    y_train = y[:train_idx]
    total_pos = max(1, int(np.sum(y_train == 1)))
    total_neg = max(1, int(np.sum(y_train == 0)))

    print(f"[debug] train rows={train_idx} total_pos={total_pos} total_neg={total_neg}")

    cols = miner.build_candidate_features(df, allow_absolute_price=False, max_features=0)
    qs = [float(x) for x in str(args.quantiles).split(",") if x.strip()]
    qmap = miner.quantile_thresholds(df.iloc[:train_idx], cols, qs)

    rows = []

    print("[debug] building rank candidates...")
    for c in cols:
        feature_type = str(meta.get(c, {}).get("feature_type", "unknown"))
        is_binary = feature_type == "binary"

        if c.startswith("dist_"):
            x = pd.to_numeric(df[c], errors="coerce").to_numpy(copy=False)

            for q, thr in qmap.get(c, {}).items():
                for op in (">=", "<="):
                    m = np.isfinite(x) & ((x >= thr) if op == ">=" else (x <= thr))
                    mt = m[:train_idx]

                    pos = int(np.sum(mt & (y_train == 1)))
                    neg = int(np.sum(mt & (y_train == 0)))
                    mask_count = int(np.sum(mt))

                    if pos <= 0:
                        continue

                    freq = pos / total_pos
                    neg_freq = neg / total_neg
                    lift = freq / max(1e-12, neg_freq)
                    ratio = pos / max(1, neg)
                    precision = pos / max(1, pos + neg)

                    rows.append({
                        "col": c,
                        "op": op,
                        "value": float(thr),
                        "quantile": q,
                        "feature_type": "raw_passthrough",
                        "is_binary": False,
                        "single_pos_hits": pos,
                        "single_neg_hits": neg,
                        "single_ratio": ratio,
                        "single_precision": precision,
                        "freq": freq,
                        "lift": lift,
                        "mask_count_train": mask_count,
                        "candidate_kind": "dist_quantile",
                    })
            continue

        if c not in bdf.columns:
            continue

        miss = int(meta.get(c, {}).get("missing_code", 0))
        b = pd.to_numeric(bdf[c], errors="coerce").fillna(miss).to_numpy()
        vals = np.unique(b[:train_idx])
        vals = vals[vals != miss]

        for v in vals.tolist():
            m = np.isfinite(b) & (np.abs(b - float(v)) <= 1e-6)
            mt = m[:train_idx]

            pos = int(np.sum(mt & (y_train == 1)))
            neg = int(np.sum(mt & (y_train == 0)))
            mask_count = int(np.sum(mt))

            if pos <= 0:
                continue

            freq = pos / total_pos
            neg_freq = neg / total_neg
            lift = freq / max(1e-12, neg_freq)
            ratio = pos / max(1, neg)
            precision = pos / max(1, pos + neg)

            rows.append({
                "col": c,
                "op": "==",
                "value": float(v),
                "quantile": "",
                "feature_type": feature_type,
                "is_binary": bool(is_binary),
                "single_pos_hits": pos,
                "single_neg_hits": neg,
                "single_ratio": ratio,
                "single_precision": precision,
                "freq": freq,
                "lift": lift,
                "mask_count_train": mask_count,
                "candidate_kind": "binned_value",
            })

    out = pd.DataFrame(rows)

    if out.empty:
        print("[debug] no candidates produced")
        out.to_csv(args.out, index=False)
        return

    out["rank_freq"] = out["freq"].rank(method="first", ascending=False).astype(int)
    out["rank_lift"] = out["lift"].rank(method="first", ascending=False).astype(int)
    out["rank_ratio"] = out["single_ratio"].rank(method="first", ascending=False).astype(int)

    # Mark whether candidate appears in top N of each list.
    top = int(args.top)
    out["in_top_freq"] = out["rank_freq"] <= top
    out["in_top_lift"] = out["rank_lift"] <= top
    out["in_top_ratio"] = out["rank_ratio"] <= top

    # Main export: useful sorted view.
    out = out.sort_values(
        ["rank_ratio", "rank_lift", "rank_freq"],
        ascending=[True, True, True],
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    print(f"[debug] saved {len(out)} candidates -> {args.out}")
    print("[debug] top freq:")
    print(out.sort_values("rank_freq").head(15)[["rank_freq", "col", "op", "value", "single_pos_hits", "single_neg_hits", "single_ratio", "freq", "lift"]].to_string(index=False))
    print("[debug] top lift:")
    print(out.sort_values("rank_lift").head(15)[["rank_lift", "col", "op", "value", "single_pos_hits", "single_neg_hits", "single_ratio", "freq", "lift"]].to_string(index=False))
    print("[debug] top ratio:")
    print(out.sort_values("rank_ratio").head(15)[["rank_ratio", "col", "op", "value", "single_pos_hits", "single_neg_hits", "single_ratio", "freq", "lift"]].to_string(index=False))


if __name__ == "__main__":
    main()