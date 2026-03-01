#!/usr/bin/env python3
"""
Extract multi-timeframe indicator features for XAUUSD and export to CSV(.gz).
Includes base OHLCV columns so event labeling can use High/Low realistically.

Input format (your XAUUSD.txt):
<TICKER>,<DTYYYYMMDD>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
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
    p = argparse.ArgumentParser(description="Extract indicator features (multi-TF) incl. OHLCV export.")
    p.add_argument("--data", type=Path, required=True, help="Path to XAUUSD.txt or .csv")
    p.add_argument("--tail", type=int, default=240000, help="Use last N rows (increase as you like)")
    p.add_argument("--tfs", type=str, default="1,2,3,5,15,30,60", help="Minute timeframes")
    p.add_argument("--ema-lens", type=str, default="8,13,20,21,34,50,55,89,100,144,200")
    p.add_argument("--rsi-lens", type=str, default="5,7,9,14,21,28")
    p.add_argument("--adx-lens", type=str, default="7,14,21")
    p.add_argument("--support-lookbacks", type=str, default="20,50,100,200")
    p.add_argument("--horizons", type=str, default="5,15,60", help="Forward-return horizons in bars on base tf1 (optional)")
    p.add_argument("--with-vwap", action="store_true", default=True, help="Compute daily VWAP features")
    p.add_argument("--no-vwap", dest="with_vwap", action="store_false", help="Disable VWAP features")
    p.add_argument("--out-features", type=Path, default=Path("feature_scan_features.csv.gz"))
    p.add_argument("--out-summary", type=Path, default=Path("feature_scan_summary.json"))
    p.add_argument("--drop-warmup-rows", action="store_true", default=True)
    p.add_argument("--keep-warmup-rows", dest="drop_warmup_rows", action="store_false")
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


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    m = ema(close, fast) - ema(close, slow)
    sig = ema(m, signal)
    hist = m - sig
    return m, sig, hist


def dmi_components(df: pd.DataFrame, period: int = 14):
    """Returns: plus_di, minus_di, dx, adx"""
    high, low, close = df["high"], df["low"], df["close"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = up_move.where((up_move > down_move.abs()) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move.abs() > up_move) & (down_move > 0), 0.0)

    tr = pd.concat(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)

    atr = tr.ewm(alpha=1.0 / period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1.0 / period, adjust=False).mean() / atr.replace(0, np.nan))
    minus_di = 100 * (minus_dm.ewm(alpha=1.0 / period, adjust=False).mean() / atr.replace(0, np.nan))
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    adx = dx.ewm(alpha=1.0 / period, adjust=False).mean()
    return plus_di, minus_di, dx, adx


def load_base(path: Path, tail: int) -> pd.DataFrame:
    # Your file is comma-separated .txt; read with sep="," 
    df = pd.read_csv(path, sep=",", low_memory=False)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        # Fallback: no header
        df = pd.read_csv(path, sep=",", header=None, names=REQUIRED_COLUMNS, low_memory=False)
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

    df = df.tail(tail).copy()

    dt = pd.to_datetime(
        df["<DTYYYYMMDD>"].astype(str).str.zfill(8) + df["<TIME>"].astype(str).str.zfill(6),
        format="%Y%m%d%H%M%S",
        errors="coerce",
    )

    def _to_float(col: str) -> np.ndarray:
        s = df[col].astype(str).str.strip().str.replace(",", ".", regex=False)
        return pd.to_numeric(s, errors="coerce").to_numpy()

    open_v = _to_float("<OPEN>")
    high_v = _to_float("<HIGH>")
    low_v = _to_float("<LOW>")
    close_v = _to_float("<CLOSE>")
    vol_v = _to_float("<VOL>")

    valid = (
        (~pd.isna(dt))
        & (~pd.isna(open_v))
        & (~pd.isna(high_v))
        & (~pd.isna(low_v))
        & (~pd.isna(close_v))
        & (~pd.isna(vol_v))
    )

    if (~valid).any():
        print(f"Dropped invalid source rows: {int((~valid).sum())}")

    out = pd.DataFrame(
        {
            "open": open_v[valid],
            "high": high_v[valid],
            "low": low_v[valid],
            "close": close_v[valid],
            "volume": vol_v[valid],
        },
        index=dt[valid],
    )

    out = out[~out.index.duplicated(keep="last")].sort_index()
    if out.empty:
        raise ValueError("No valid rows after parsing.")
    return out


def daily_vwap_features(tf_df: pd.DataFrame) -> pd.DataFrame:
    """Daily VWAP using typical price (H+L+C)/3 and provided volume (often tick volume)."""
    tp = (tf_df["high"] + tf_df["low"] + tf_df["close"]) / 3.0
    vol = tf_df["volume"].replace(0, np.nan)
    day = tf_df.index.normalize()

    pv = tp * vol
    cum_pv = pv.groupby(day).cumsum()
    cum_v = vol.groupby(day).cumsum()
    vwap = cum_pv / cum_v

    out = pd.DataFrame(index=tf_df.index)
    out["vwap"] = vwap
    out["dist_vwap"] = (tf_df["close"] / vwap) - 1.0
    out["vwap_d5"] = vwap - vwap.shift(5)
    return out


def compute_tf_features(
    base: pd.DataFrame,
    tf: int,
    ema_lens: list[int],
    rsi_lens: list[int],
    adx_lens: list[int],
    support_lbs: list[int],
    with_vwap: bool,
) -> pd.DataFrame:
    if tf == 1:
        tf_df = base.copy()
    else:
        tf_df = (
            base.resample(f"{tf}min")
            .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
            .dropna()
        )

    f = pd.DataFrame(index=tf_df.index)

    # Export OHLCV per TF too (useful for debugging / future labeling)
    f[f"open_tf{tf}"] = tf_df["open"]
    f[f"high_tf{tf}"] = tf_df["high"]
    f[f"low_tf{tf}"] = tf_df["low"]
    f[f"close_tf{tf}"] = tf_df["close"]
    f[f"volume_tf{tf}"] = tf_df["volume"]

    # EMA distances
    for l in ema_lens:
        e = ema(tf_df["close"], l)
        f[f"ema{l}_tf{tf}"] = e
        f[f"dist_ema{l}_tf{tf}"] = (tf_df["close"] / e) - 1.0

    # RSI
    for l in rsi_lens:
        f[f"rsi{l}_tf{tf}"] = rsi(tf_df["close"], l)

    # MACD
    m, s, h = macd(tf_df["close"])
    f[f"macd_tf{tf}"] = m
    f[f"macd_signal_tf{tf}"] = s
    f[f"macd_hist_tf{tf}"] = h

    # DMI/ADX family
    for l in adx_lens:
        plus_di, minus_di, dx, adx_v = dmi_components(tf_df, l)
        f[f"plus_di{l}_tf{tf}"] = plus_di
        f[f"minus_di{l}_tf{tf}"] = minus_di
        f[f"dx{l}_tf{tf}"] = dx
        f[f"adx{l}_tf{tf}"] = adx_v

    # Volume z-score
    vol_ma = tf_df["volume"].rolling(50).mean()
    vol_std = tf_df["volume"].rolling(50).std()
    f[f"vol_z_tf{tf}"] = (tf_df["volume"] - vol_ma) / vol_std.replace(0, np.nan)

    # Support/resistance distances
    for lb in support_lbs:
        sup = tf_df["low"].rolling(lb).min()
        res = tf_df["high"].rolling(lb).max()
        f[f"dist_support{lb}_tf{tf}"] = (tf_df["close"] / sup) - 1.0
        f[f"dist_resist{lb}_tf{tf}"] = (res / tf_df["close"]) - 1.0

    # VWAP
    if with_vwap:
        v = daily_vwap_features(tf_df)
        f[f"vwap_tf{tf}"] = v["vwap"]
        f[f"dist_vwap_tf{tf}"] = v["dist_vwap"]
        f[f"vwap_d5_tf{tf}"] = v["vwap_d5"]

    return f


def add_targets(df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    out = df.copy()
    for h in horizons:
        out[f"fwd_ret_{h}"] = out["close"].shift(-h) / out["close"] - 1.0
    return out


def summarize(df: pd.DataFrame, horizons: list[int]) -> dict:
    numeric = df.select_dtypes(include=[np.number]).dropna()
    summary: dict[str, object] = {"rows": int(len(df)), "rows_numeric": int(len(numeric))}
    corrs = {}
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
    return summary


def miner_feature_usage(df: pd.DataFrame) -> dict[str, object]:
    """Classify columns by current miner selection rules for transparency."""
    ignored_patterns = {
        "base_ohlcv": re.compile(r"^(open|high|low|close|volume)$"),
        "forward_targets": re.compile(r"^fwd_ret_"),
        "tf_ohlcv": re.compile(r"^(open_tf|high_tf|low_tf|close_tf|volume_tf)"),
        "ema_levels": re.compile(r"^ema"),
        "raw_vwap_levels": re.compile(r"^vwap_tf"),
        "other_numeric_not_whitelisted": re.compile(r".*"),
    }
    kept_patterns = {
        "dist_vwap": re.compile(r"^dist_vwap"),
        "dist_generic": re.compile(r"^dist_"),
        "indicator_family": re.compile(r"^(rsi|adx|plus_di|minus_di|dx|macd|vol_z)"),
    }

    kept: list[str] = []
    ignored: dict[str, list[str]] = {k: [] for k in ignored_patterns.keys()}

    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        is_kept = False
        for _, pat in kept_patterns.items():
            if pat.match(col):
                kept.append(col)
                is_kept = True
                break
        if is_kept:
            continue

        if ignored_patterns["base_ohlcv"].match(col):
            ignored["base_ohlcv"].append(col)
            continue
        if ignored_patterns["forward_targets"].match(col):
            ignored["forward_targets"].append(col)
            continue
        if ignored_patterns["tf_ohlcv"].match(col):
            ignored["tf_ohlcv"].append(col)
            continue
        if ignored_patterns["ema_levels"].match(col):
            ignored["ema_levels"].append(col)
            continue
        if ignored_patterns["raw_vwap_levels"].match(col):
            ignored["raw_vwap_levels"].append(col)
            continue
        ignored["other_numeric_not_whitelisted"].append(col)

    return {
        "miner_usable_feature_count": int(len(kept)),
        "miner_usable_feature_examples": sorted(kept)[:30],
        "miner_ignored_by_reason": {
            k: {
                "count": int(len(v)),
                "examples": sorted(v)[:30],
            }
            for k, v in ignored.items()
        },
    }


def _round_by_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in df.columns:
        # price-like
        if col in {"open", "high", "low", "close"}:
            df[col] = df[col].round(1)
            continue

        if col.startswith(("open_tf", "high_tf", "low_tf", "close_tf")):
            df[col] = df[col].round(1)
            continue

        if col.startswith("volume"):
            # keep integer-ish but as float32; round to 0 decimals
            df[col] = df[col].round(0)
            continue

        if col.startswith("fwd_ret_"):
            df[col] = df[col].round(6)
            continue

        if col.startswith("dist_") or col.startswith("dist_vwap"):
            df[col] = df[col].round(6)
            continue

        if col.startswith("ema") or col.startswith("vwap"):
            if "vwap_d" in col:
                df[col] = df[col].round(2)
            else:
                df[col] = df[col].round(1)
            continue

        if col.startswith(("rsi", "adx", "plus_di", "minus_di", "dx")):
            df[col] = df[col].round(3)
            continue

        if col.startswith("macd"):
            df[col] = df[col].round(6)
            continue

        if col.startswith("vol_z"):
            df[col] = df[col].round(3)
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].round(5)

    num_cols = df.select_dtypes(include=["number"]).columns
    df[num_cols] = df[num_cols].astype(np.float32)
    return df


def main() -> None:
    args = parse_args()

    if np is None or pd is None:
        print("Missing dependencies: python -m pip install pandas numpy", file=sys.stderr)
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

    # Base OHLCV (tf1)
    feature_table["open"] = base["open"].to_numpy()
    feature_table["high"] = base["high"].to_numpy()
    feature_table["low"] = base["low"].to_numpy()
    feature_table["close"] = base["close"].to_numpy()
    feature_table["volume"] = base["volume"].to_numpy()

    tf_frames = []
    for tf in tfs:
        print(f"Computing TF={tf} ...")
        tf_features = compute_tf_features(base, tf, ema_lens, rsi_lens, adx_lens, support_lbs, args.with_vwap)
        aligned = tf_features.reindex(feature_table.index, method="ffill")
        tf_frames.append(aligned)
        print(f"TF={tf} done. cols={aligned.shape[1]}")

    if tf_frames:
        feature_table = pd.concat([feature_table] + tf_frames, axis=1)

    # Reduce memory early (before warmup filtering)
    num_cols = feature_table.select_dtypes(include=["number"]).columns
    feature_table[num_cols] = feature_table[num_cols].astype(np.float32)

    feature_table = add_targets(feature_table, horizons)
    print(f"Rows before warmup filter: {len(feature_table)}")

    # ---------- SAFE WARMUP FILTER (memory friendly) ----------
    if args.drop_warmup_rows:
        target_cols = {f"fwd_ret_{h}" for h in horizons}
        non_feature_cols = {"open", "high", "low", "close", "volume", *target_cols}
        feature_cols = [c for c in feature_table.columns if c not in non_feature_cols]

        if feature_cols:
            # Instead of building a huge temporary frame, compute keep_mask in chunks
            keep_mask = np.zeros(len(feature_table), dtype=bool)

            chunk = 50  # number of columns per chunk; lower if RAM is tight
            for start in range(0, len(feature_cols), chunk):
                cols_chunk = feature_cols[start : start + chunk]
                # bool DataFrame for this chunk only
                keep_mask |= feature_table[cols_chunk].notna().any(axis=1).to_numpy()

            dropped = int((~keep_mask).sum())
            if keep_mask.any():
                # Avoid .copy() here to prevent consolidation spike; defer copying
                feature_table = feature_table.loc[keep_mask]
                print(f"Dropped warmup rows with empty features: {dropped}")
            else:
                print("Warning: warmup filter would drop all rows; keeping full dataset.")

    # Round + float32 (size reduction)
    feature_table = _round_by_column(feature_table)

    out_path = str(args.out_features)
    print("Export starting:", out_path)

    if out_path.lower().endswith(".gz"):
        # fast gzip (compresslevel=1)
        with gzip.open(out_path, "wt", compresslevel=1, encoding="utf-8", newline="") as f:
            feature_table.to_csv(f, index_label="datetime", float_format="%.6f")
    else:
        feature_table.to_csv(out_path, index_label="datetime", float_format="%.6f")

    print("Export finished:", out_path)

    report = summarize(feature_table, horizons)
    report["miner_feature_usage"] = miner_feature_usage(feature_table)
    args.out_summary.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Saved features: {args.out_features}")
    print(f"Saved summary:  {args.out_summary}")
    print(f"Rows: {len(feature_table)} | Columns: {len(feature_table.columns)}")


if __name__ == "__main__":
    main()
