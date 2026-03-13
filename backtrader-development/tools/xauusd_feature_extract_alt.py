#!/usr/bin/env python3
"""Extract multi-timeframe indicator features and simple pattern analysis for XAUUSD."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
except ModuleNotFoundError:
    np = None
    pd = None

REQUIRED_COLUMNS = [
    "<TICKER>",
    "<DTYYYYMMDD>",
    "<TIME>",
    "<OPEN>",
    "<HIGH>",
    "<LOW>",
    "<CLOSE>",
    "<VOL>",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract indicator features and analyze patterns")
    p.add_argument("--data", type=Path, required=True, help="Path to XAUUSD.txt")
    p.add_argument("--tail", type=int, default=350000, help="Use last N rows")
    p.add_argument("--tfs", type=str, default="1,5,15,30,60", help="Minute timeframes")
    p.add_argument("--ema-lens", type=str, default="20,50,100,200")
    p.add_argument("--rsi-lens", type=str, default="7,14,21")
    p.add_argument("--adx-lens", type=str, default="14")
    p.add_argument("--support-lookbacks", type=str, default="50,200")
    p.add_argument("--horizons", type=str, default="5,15,60", help="Forward-return horizons in bars on base tf")
    p.add_argument("--out-features", type=Path, default=Path("feature_scan_features.csv"))
    p.add_argument("--out-summary", type=Path, default=Path("feature_scan_summary.json"))
    p.add_argument(
        "--drop-warmup-rows",
        dest="drop_warmup_rows",
        action="store_true",
        default=True,
        help="Drop rows where all computed feature columns are NaN (useful for indicator warmup)",
    )
    p.add_argument(
        "--keep-warmup-rows",
        dest="drop_warmup_rows",
        action="store_false",
        help="Keep early warmup rows even if most features are NaN",
    )
    return p.parse_args()


def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1.0 / period, adjust=False).mean()
    roll_down = down.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
    m = ema(close, fast) - ema(close, slow)
    sig = ema(m, signal)
    hist = m - sig
    return m, sig, hist


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    plus_dm = (high.diff()).where((high.diff() > low.diff().abs()) & (high.diff() > 0), 0.0)
    minus_dm = (-low.diff()).where((low.diff().abs() > high.diff()) & (-low.diff() > 0), 0.0)

    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.ewm(alpha=1.0 / period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1.0 / period, adjust=False).mean() / atr.replace(0, np.nan))
    minus_di = 100 * (minus_dm.ewm(alpha=1.0 / period, adjust=False).mean() / atr.replace(0, np.nan))
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    return dx.ewm(alpha=1.0 / period, adjust=False).mean()


def load_base(path: Path, tail: int) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        # Fallback: file without header row
        df = pd.read_csv(path, header=None, names=REQUIRED_COLUMNS, low_memory=False)
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

    df = df.tail(tail).copy()
    dt = pd.to_datetime(
        df["<DTYYYYMMDD>"].astype(str).str.zfill(8) + df["<TIME>"].astype(str).str.zfill(6),
        format="%Y%m%d%H%M%S",
        errors="coerce",
    )

    def _to_float_series(col: str) -> np.ndarray:
        s = df[col].astype(str).str.strip().str.replace(",", ".", regex=False)
        return pd.to_numeric(s, errors="coerce").to_numpy()

    open_v = _to_float_series("<OPEN>")
    high_v = _to_float_series("<HIGH>")
    low_v = _to_float_series("<LOW>")
    close_v = _to_float_series("<CLOSE>")
    vol_v = _to_float_series("<VOL>")

    valid_mask = (~pd.isna(dt)) & (~pd.isna(open_v)) & (~pd.isna(high_v)) & (~pd.isna(low_v)) & (~pd.isna(close_v)) & (~pd.isna(vol_v))
    dropped_invalid = int((~valid_mask).sum())
    if dropped_invalid:
        print(f"Dropped invalid source rows: {dropped_invalid}")

    dt = dt[valid_mask]
    open_v = open_v[valid_mask]
    high_v = high_v[valid_mask]
    low_v = low_v[valid_mask]
    close_v = close_v[valid_mask]
    vol_v = vol_v[valid_mask]

    # Build with aligned values (not Series alignment) to avoid all-NaN columns
    # when switching from RangeIndex to datetime index.
    out = pd.DataFrame(
        {
            "open": open_v,
            "high": high_v,
            "low": low_v,
            "close": close_v,
            "volume": vol_v,
        },
        index=dt,
    )
    out = out[~out.index.duplicated(keep="last")].sort_index()
    if out.empty:
        raise ValueError("No valid rows after parsing source columns/datetime.")
    return out


def compute_tf_features(base: pd.DataFrame, tf: int, ema_lens: list[int], rsi_lens: list[int], adx_lens: list[int], support_lbs: list[int]) -> pd.DataFrame:
    if tf == 1:
        tf_df = base.copy()
    else:
        tf_df = (
            base.resample(f"{tf}min")
            .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
            .dropna()
        )

    f = pd.DataFrame(index=tf_df.index)
    f[f"close_tf{tf}"] = tf_df["close"]

    for l in ema_lens:
        e = ema(tf_df["close"], l)
        f[f"ema{l}_tf{tf}"] = e
        f[f"dist_ema{l}_tf{tf}"] = (tf_df["close"] / e) - 1.0

    for l in rsi_lens:
        f[f"rsi{l}_tf{tf}"] = rsi(tf_df["close"], l)

    m, s, h = macd(tf_df["close"])
    f[f"macd_tf{tf}"] = m
    f[f"macd_signal_tf{tf}"] = s
    f[f"macd_hist_tf{tf}"] = h

    for l in adx_lens:
        f[f"adx{l}_tf{tf}"] = adx(tf_df, l)

    vol_ma = tf_df["volume"].rolling(50).mean()
    vol_std = tf_df["volume"].rolling(50).std()
    f[f"vol_z_tf{tf}"] = (tf_df["volume"] - vol_ma) / vol_std.replace(0, np.nan)

    for lb in support_lbs:
        sup = tf_df["low"].rolling(lb).min()
        res = tf_df["high"].rolling(lb).max()
        f[f"dist_support{lb}_tf{tf}"] = (tf_df["close"] / sup) - 1.0
        f[f"dist_resist{lb}_tf{tf}"] = (res / tf_df["close"]) - 1.0

    return f


def add_targets(df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    out = df.copy()
    for h in horizons:
        out[f"fwd_ret_{h}"] = out["close"].shift(-h) / out["close"] - 1.0
    return out


def summarize(df: pd.DataFrame, horizons: list[int]) -> dict:
    numeric = df.select_dtypes(include=[np.number]).dropna()
    summary: dict[str, object] = {"rows": int(len(df)), "rows_numeric": int(len(numeric))}

    corrs: dict[str, dict[str, float]] = {}
    for h in horizons:
        tgt = f"fwd_ret_{h}"
        if tgt not in numeric.columns:
            continue
        c = numeric.corrwith(numeric[tgt]).sort_values(ascending=False)
        top = c.drop(labels=[tgt], errors="ignore").head(10)
        bottom = c.drop(labels=[tgt], errors="ignore").tail(10)
        corrs[tgt] = {
            "top_pos": {k: float(v) for k, v in top.items()},
            "top_neg": {k: float(v) for k, v in bottom.items()},
        }

    summary["correlations"] = corrs

    bucket_stats = {}
    for col in [c for c in numeric.columns if c.startswith("rsi") or c.startswith("dist_ema")][:20]:
        try:
            q = pd.qcut(numeric[col], 5, duplicates="drop")
            grouped = numeric.groupby(q)[f"fwd_ret_{horizons[0]}"]
            bucket_stats[col] = {str(k): float(v) for k, v in grouped.mean().items()}
        except Exception:
            continue

    summary["bucket_mean_fwd_ret"] = bucket_stats
    return summary


def main() -> None:
    args = parse_args()

    if np is None or pd is None:
        print(
            "Missing dependencies: please install pandas and numpy in your active environment.\n"
            "Example: python -m pip install pandas numpy",
            file=sys.stderr,
        )
        raise SystemExit(1)
    tfs = [int(x) for x in args.tfs.split(",") if x.strip()]
    ema_lens = [int(x) for x in args.ema_lens.split(",") if x.strip()]
    rsi_lens = [int(x) for x in args.rsi_lens.split(",") if x.strip()]
    adx_lens = [int(x) for x in args.adx_lens.split(",") if x.strip()]
    support_lbs = [int(x) for x in args.support_lookbacks.split(",") if x.strip()]
    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]

    base = load_base(args.data, args.tail)
    print(f"Base rows loaded: {len(base)}")
    feature_table = pd.DataFrame(index=base.index)
    feature_table["close"] = base["close"].to_numpy()
    print(f"Base non-NaN close rows: {int(feature_table['close'].notna().sum())}")

    for tf in tfs:
        tf_features = compute_tf_features(base, tf, ema_lens, rsi_lens, adx_lens, support_lbs)
        aligned = tf_features.reindex(feature_table.index, method="ffill")
        feature_table = feature_table.join(aligned, how="left")

    feature_table = add_targets(feature_table, horizons)
    print(f"Rows before warmup filter: {len(feature_table)}")

    if args.drop_warmup_rows:
        target_cols = {f"fwd_ret_{h}" for h in horizons}
        non_feature_cols = {"close", *target_cols}
        feature_cols = [c for c in feature_table.columns if c not in non_feature_cols]
        if feature_cols:
            keep_mask = feature_table[feature_cols].notna().any(axis=1)
            dropped = int((~keep_mask).sum())
            if keep_mask.any():
                feature_table = feature_table.loc[keep_mask].copy()
                print(f"Dropped warmup rows with empty features: {dropped}")
            else:
                print(
                    "Warning: warmup filter would drop all rows; keeping full dataset. "
                    "Try --keep-warmup-rows and inspect datetime parsing/source columns."
                )

    feature_table.to_csv(args.out_features, index_label="datetime")
    report = summarize(feature_table, horizons)
    args.out_summary.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Saved features: {args.out_features}")
    print(f"Saved summary:  {args.out_summary}")
    print(f"Rows: {len(feature_table)} | Columns: {len(feature_table.columns)}")


if __name__ == "__main__":
    main()
