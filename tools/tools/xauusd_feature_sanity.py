#!/usr/bin/env python3
"""Sanity-check a (possibly huge) XAUUSD feature CSV/CSV.GZ before mining rules."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


TARGET_COLS = ("fwd_ret_5", "fwd_ret_15", "fwd_ret_60")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sanity-check wide feature tables for leakage/quality issues")
    p.add_argument("--data", type=Path, required=True, help="Feature CSV(.gz) with datetime + feature columns")
    p.add_argument("--sample-rows", type=int, default=200_000, help="Rows sampled from tail for expensive checks")
    p.add_argument("--chunksize", type=int, default=200_000, help="Chunk size for streaming pass")
    p.add_argument("--out-json", type=Path, default=Path("feature_sanity_summary.json"), help="Output JSON summary")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    total_rows = 0
    bad_datetime = 0
    dup_datetime = 0
    non_monotonic_steps = 0
    prev_dt = None
    datetime_set = set()

    target_non_na = {k: 0 for k in TARGET_COLS}
    target_non_zero = {k: 0 for k in TARGET_COLS}

    first_cols = None
    for chunk in pd.read_csv(args.data, compression="infer", chunksize=args.chunksize):
        if first_cols is None:
            first_cols = list(chunk.columns)
        total_rows += len(chunk)

        if "datetime" not in chunk.columns:
            raise ValueError("Missing required 'datetime' column")

        dt = pd.to_datetime(chunk["datetime"], errors="coerce")
        bad_datetime += int(dt.isna().sum())

        valid_dt = dt.dropna()
        for d in valid_dt:
            if d in datetime_set:
                dup_datetime += 1
            else:
                datetime_set.add(d)
            if prev_dt is not None and d < prev_dt:
                non_monotonic_steps += 1
            prev_dt = d

        for t in TARGET_COLS:
            if t in chunk.columns:
                s = pd.to_numeric(chunk[t], errors="coerce")
                target_non_na[t] += int(s.notna().sum())
                target_non_zero[t] += int((s.fillna(0.0) != 0.0).sum())

    if total_rows == 0:
        raise ValueError("No rows found in input file")

    sample = pd.read_csv(args.data, compression="infer").tail(args.sample_rows)
    numeric = sample.select_dtypes(include=[np.number])
    std = numeric.std(numeric_only=True)
    near_constant_cols = [c for c, v in std.items() if pd.notna(v) and v < 1e-12]

    leakage_checks: dict[str, object] = {}
    for t in TARGET_COLS:
        if t in numeric.columns:
            c = numeric.corrwith(numeric[t]).drop(labels=[t], errors="ignore")
            top = c.reindex(c.abs().sort_values(ascending=False).head(8).index)
            leakage_checks[t] = {k: float(v) for k, v in top.items()}

    obvious_leaks = []
    for t, vals in leakage_checks.items():
        for col, corr in vals.items():
            if abs(corr) >= 0.98:
                obvious_leaks.append({"target": t, "column": col, "corr": corr})

    out = {
        "rows": total_rows,
        "columns": len(first_cols or []),
        "bad_datetime_rows": bad_datetime,
        "duplicate_datetime_rows": dup_datetime,
        "non_monotonic_steps": non_monotonic_steps,
        "has_targets": {k: (k in (first_cols or [])) for k in TARGET_COLS},
        "target_non_na": target_non_na,
        "target_non_zero": target_non_zero,
        "target_non_zero_ratio": {k: (target_non_zero[k] / total_rows) for k in TARGET_COLS},
        "near_constant_cols_count": len(near_constant_cols),
        "near_constant_cols_preview": near_constant_cols[:50],
        "sample_rows_for_corr": int(len(sample)),
        "top_abs_corr_to_targets": leakage_checks,
        "obvious_leak_flags": obvious_leaks,
        "notes": [
            "If duplicate_datetime_rows > 0, enforce unique timestamps before mining.",
            "If non_monotonic_steps > 0, sort by datetime before train/test split.",
            "If obvious_leak_flags non-empty, exclude those columns from miner features.",
            "High signal win-rate with low event diversity often indicates overlap bias.",
        ],
    }

    args.out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Saved sanity summary: {args.out_json}")
    print(f"Rows={out['rows']} Cols={out['columns']} Duplicates={dup_datetime} BadDT={bad_datetime}")


if __name__ == "__main__":
    main()
