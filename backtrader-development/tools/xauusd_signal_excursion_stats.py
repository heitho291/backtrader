#!/usr/bin/env python3
"""Append excursion stats (MAE/MFE) to signals_best_rules.csv.

This is a post-processing step and does NOT affect miner runtime.
For each signal, it computes within `minutes_to_hit`:
- max_drawdown_pct_to_hit (worst low excursion vs entry close)
- max_profit_pct_to_hit (best high excursion vs entry close)
- minute_of_max_drawdown / minute_of_max_profit
- max_profit_before_sl_pct / minute_of_max_profit_before_sl (for SL outcomes, to hit minute)
- max_profit_before_eventual_sl_pct / minute_of_max_profit_before_eventual_sl
  using forward scan until the position would eventually hit SL if TP were ignored
- minute_of_eventual_sl_from_entry

Assumes miner-style long-direction labeling (TP up, SL down).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Enrich signals CSV with MAE/MFE excursion stats")
    p.add_argument("--features", type=Path, required=True, help="Path to feature_scan_features.csv or .csv.gz")
    p.add_argument("--signals", type=Path, required=True, help="Path to signals_best_rules.csv or .csv.gz")
    p.add_argument("--out", type=Path, default=Path("signals_best_rules_enriched.csv"), help="Output CSV path")
    return p.parse_args()


def load_features(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, compression="infer")
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df = df.set_index("datetime")
    elif df.columns[0].lower().startswith("unnamed"):
        df[df.columns[0]] = pd.to_datetime(df[df.columns[0]], errors="coerce")
        df = df.set_index(df.columns[0])
    else:
        raise ValueError("features CSV must contain datetime column or unnamed datetime index")

    df = df.sort_index()
    for c in ("high", "low", "close"):
        if c not in df.columns:
            raise ValueError(f"features missing required column: {c}")
    return df


def load_signals(path: Path) -> pd.DataFrame:
    s = pd.read_csv(path, compression="infer")
    needed = {"datetime", "close", "outcome", "minutes_to_hit", "sl"}
    miss = needed - set(s.columns)
    if miss:
        raise ValueError(f"signals missing required columns: {sorted(miss)}")
    s["datetime"] = pd.to_datetime(s["datetime"], errors="coerce")
    s["minutes_to_hit"] = pd.to_numeric(s["minutes_to_hit"], errors="coerce")
    s["close"] = pd.to_numeric(s["close"], errors="coerce")
    s["outcome"] = s["outcome"].astype(str).str.upper()
    return s


def enrich(signals: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    out = signals.copy()

    dt_to_pos = pd.Series(np.arange(len(features), dtype=np.int64), index=features.index)
    highs = features["high"].to_numpy(dtype=float)
    lows = features["low"].to_numpy(dtype=float)

    max_dd = np.full(len(out), np.nan, dtype=float)
    max_pf = np.full(len(out), np.nan, dtype=float)
    min_dd_t = np.full(len(out), np.nan, dtype=float)
    max_pf_t = np.full(len(out), np.nan, dtype=float)
    max_pf_before_sl = np.full(len(out), np.nan, dtype=float)
    max_pf_before_sl_t = np.full(len(out), np.nan, dtype=float)
    max_pf_before_eventual_sl = np.full(len(out), np.nan, dtype=float)
    max_pf_before_eventual_sl_t = np.full(len(out), np.nan, dtype=float)
    eventual_sl_t = np.full(len(out), np.nan, dtype=float)

    for i, row in out.iterrows():
        dt = row["datetime"]
        if pd.isna(dt) or dt not in dt_to_pos.index:
            continue

        hit_m = row["minutes_to_hit"]
        entry = row["close"]
        sl_frac = row["sl"]
        if pd.isna(hit_m) or pd.isna(entry) or pd.isna(sl_frac) or entry <= 0:
            continue

        hit_m_int = int(hit_m)
        if hit_m_int <= 0:
            continue

        entry_pos = int(dt_to_pos.loc[dt])
        a = entry_pos + 1
        b = min(len(features), entry_pos + hit_m_int + 1)
        if a >= b:
            continue

        hi = highs[a:b]
        lo = lows[a:b]

        dd_series = (lo / entry) - 1.0
        pf_series = (hi / entry) - 1.0

        dd_idx = int(np.argmin(dd_series))
        pf_idx = int(np.argmax(pf_series))

        max_dd[i] = float(dd_series[dd_idx])
        max_pf[i] = float(pf_series[pf_idx])
        min_dd_t[i] = float(dd_idx + 1)
        max_pf_t[i] = float(pf_idx + 1)

        if str(row["outcome"]).upper() == "SL":
            max_pf_before_sl[i] = float(pf_series[pf_idx])
            max_pf_before_sl_t[i] = float(pf_idx + 1)

        # Independent forward scan: how long could profit have run before eventual SL?
        sl_level = entry * (1.0 - float(sl_frac))
        lo_fwd = lows[entry_pos + 1 :]
        hi_fwd = highs[entry_pos + 1 :]
        if len(lo_fwd) == 0:
            continue

        sl_hits = np.flatnonzero(lo_fwd <= sl_level)
        if len(sl_hits) == 0:
            continue

        sl_idx = int(sl_hits[0])  # zero-based from entry+1
        eventual_sl_t[i] = float(sl_idx + 1)

        hi_until_sl = hi_fwd[: sl_idx + 1]
        pf_until_sl = (hi_until_sl / entry) - 1.0
        pf_until_sl_idx = int(np.argmax(pf_until_sl))
        max_pf_before_eventual_sl[i] = float(pf_until_sl[pf_until_sl_idx])
        max_pf_before_eventual_sl_t[i] = float(pf_until_sl_idx + 1)

    out["max_drawdown_pct_to_hit"] = max_dd
    out["minute_of_max_drawdown"] = min_dd_t
    out["max_profit_pct_to_hit"] = max_pf
    out["minute_of_max_profit"] = max_pf_t
    out["max_profit_before_sl_pct"] = max_pf_before_sl
    out["minute_of_max_profit_before_sl"] = max_pf_before_sl_t
    out["max_profit_before_eventual_sl_pct"] = max_pf_before_eventual_sl
    out["minute_of_max_profit_before_eventual_sl"] = max_pf_before_eventual_sl_t
    out["minute_of_eventual_sl_from_entry"] = eventual_sl_t

    return out


def main() -> None:
    args = parse_args()
    features = load_features(args.features)
    signals = load_signals(args.signals)
    enriched = enrich(signals, features)
    enriched.to_csv(args.out, index=False)

    n = len(enriched)
    n_valid = int(enriched["max_drawdown_pct_to_hit"].notna().sum())
    print(f"Saved: {args.out}")
    print(f"Rows: {n} | rows_with_excursions: {n_valid}")


if __name__ == "__main__":
    main()

