#!/usr/bin/env python3
"""
Extract multi-timeframe indicator features for XAUUSD and export to CSV(.gz).
Includes base OHLCV columns so event labeling can use High/Low realistically.

Input format (your XAUUSD.txt):
<TICKER>,<DTYYYYMMDD>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>
"""

from __future__ import annotations

import argparse
import csv
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
    p.add_argument("--fast-loader", choices=["auto", "pandas", "pyarrow", "polars"], default="auto",
                   help="CSV loader backend preference for large files")
    p.add_argument("--chunk-size", type=int, default=2_000_000,
                   help="Reserved for chunk-based CSV processing tuning")
    p.add_argument("--tfs", type=str, default="1,2,3,5,15,30,60", help="Minute timeframes")
    p.add_argument("--ema-lens", type=str, default="8,13,20,21,34,50,55,89,100,144,200")
    p.add_argument("--rsi-lens", type=str, default="5,7,9,14,21,28")
    p.add_argument("--adx-lens", type=str, default="7,14,21")
    p.add_argument("--support-lookbacks", type=str, default="20,50,100,200")
    p.add_argument("--delta-windows", type=str, default="3,5,10,20", help="Add delta features over N bars for indicator columns")
    p.add_argument("--pattern-tfs", type=str, default="1,5,15", help="TF subset for candle/breakout/FVG features")
    p.add_argument("--breakout-lookbacks", type=str, default="20,50,100", help="Lookbacks for breakout features")
    p.add_argument("--with-candle-patterns", action="store_true", default=True)
    p.add_argument("--no-candle-patterns", dest="with_candle_patterns", action="store_false")
    p.add_argument("--with-breakout-features", action="store_true", default=True)
    p.add_argument("--no-breakout-features", dest="with_breakout_features", action="store_false")
    p.add_argument("--with-fvg-features", action="store_true", default=True)
    p.add_argument("--no-fvg-features", dest="with_fvg_features", action="store_false")
    p.add_argument("--with-session-features", action="store_true", default=True)
    p.add_argument("--no-session-features", dest="with_session_features", action="store_false")
    p.add_argument("--with-session-weekend", action="store_true", default=False,
                   help="Include session_is_weekend feature (default off)")
    p.add_argument("--with-liquidity-sweeps", action="store_true", default=True)
    p.add_argument("--no-liquidity-sweeps", dest="with_liquidity_sweeps", action="store_false")
    p.add_argument("--with-market-structure", action="store_true", default=True)
    p.add_argument("--no-market-structure", dest="with_market_structure", action="store_false")
    p.add_argument("--with-orderblock-proximity", action="store_true", default=True)
    p.add_argument("--no-orderblock-proximity", dest="with_orderblock_proximity", action="store_false")
    p.add_argument("--horizons", type=str, default="5,15,60", help="Forward-return horizons in bars on base tf1 (optional)")
    p.add_argument("--with-vwap", action="store_true", default=True, help="Compute daily VWAP features")
    p.add_argument("--no-vwap", dest="with_vwap", action="store_false", help="Disable VWAP features")
    p.add_argument("--out-features", type=Path, default=Path("feature_scan_features"),
                   help="Output path stem or full filename; extension is adjusted by --output-format")
    p.add_argument("--output-format", choices=["csv_gz", "parquet"], default="csv_gz",
                   help="Write exactly one features file format per run")
    p.add_argument("--out-summary", type=Path, default=Path("feature_scan_summary.json"),
                   help="Summary JSON path; set to false/none to skip summary writing")
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


def _detect_sep(path: Path) -> str:
    sample = path.read_text(encoding="utf-8", errors="ignore")[:8192]
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;").delimiter
    except Exception:
        return ","


def _read_csv_any(path: Path, sep: str, fast_loader: str, chunk_size: int = 2_000_000):
    mode = (fast_loader or "auto").lower()
    if mode in {"auto", "pyarrow"}:
        try:
            import pyarrow.csv as pacsv
            table = pacsv.read_csv(str(path), parse_options=pacsv.ParseOptions(delimiter=sep))
            return table.to_pandas(), "pyarrow"
        except Exception:
            if mode == "pyarrow":
                raise
    if mode in {"auto", "polars"}:
        try:
            import polars as pl
            return pl.scan_csv(str(path), separator=sep).collect().to_pandas(), "polars"
        except Exception:
            if mode == "polars":
                raise
    if chunk_size > 0:
        chunks = pd.read_csv(path, sep=sep, low_memory=False, chunksize=chunk_size)
        return pd.concat(chunks, ignore_index=True), "pandas_chunked"
    return pd.read_csv(path, sep=sep, low_memory=False), "pandas"


def load_ohlc_any_format(
    path: Path,
    tail: int,
    fast_loader: str = "auto",
    chunk_size: int = 2_000_000,
) -> tuple[pd.DataFrame, dict[str, object]]:
    sep = _detect_sep(path)
    df, loader_used = _read_csv_any(path, sep=sep, fast_loader=fast_loader, chunk_size=chunk_size)

    cols_lower = {str(c).strip().lower(): c for c in df.columns}
    rows_read = int(len(df))
    fmt = "unknown"

    if all(c in df.columns for c in REQUIRED_COLUMNS):
        fmt = "mt_header"
        dt_str = df["<DTYYYYMMDD>"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(8)
        tm_str = df["<TIME>"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(6)
        open_s, high_s, low_s, close_s, vol_s = df["<OPEN>"], df["<HIGH>"], df["<LOW>"], df["<CLOSE>"], df["<VOL>"]
    elif all(k in cols_lower for k in ["date", "time", "open", "high", "low", "close", "volume"]):
        fmt = "dukascopy"
        dt_str = df[cols_lower["date"]].astype(str).str.replace(".", "", regex=False).str.replace("-", "", regex=False).str.replace("/", "", regex=False)
        tm_str = df[cols_lower["time"]].astype(str).str.replace(":", "", regex=False).str.zfill(6)
        open_s, high_s, low_s, close_s, vol_s = df[cols_lower["open"]], df[cols_lower["high"]], df[cols_lower["low"]], df[cols_lower["close"]], df[cols_lower["volume"]]
    else:
        df2 = pd.read_csv(path, sep=sep, header=None, names=REQUIRED_COLUMNS, low_memory=False)
        rows_read = int(len(df2))
        fmt = "mt_no_header"
        dt_str = df2["<DTYYYYMMDD>"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(8)
        tm_str = df2["<TIME>"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(6)
        open_s, high_s, low_s, close_s, vol_s = df2["<OPEN>"], df2["<HIGH>"], df2["<LOW>"], df2["<CLOSE>"], df2["<VOL>"]

    if tail > 0:
        dt_str = dt_str.tail(tail)
        tm_str = tm_str.tail(tail)
        open_s = open_s.tail(tail)
        high_s = high_s.tail(tail)
        low_s = low_s.tail(tail)
        close_s = close_s.tail(tail)
        vol_s = vol_s.tail(tail)

    dt = pd.to_datetime(dt_str.astype(str).str.zfill(8) + tm_str.astype(str).str.zfill(6), format="%Y%m%d%H%M%S", errors="coerce")

    def _to_float(series: pd.Series) -> np.ndarray:
        x = series.astype(str).str.strip().str.replace(",", ".", regex=False)
        return pd.to_numeric(x, errors="coerce").to_numpy()

    open_v, high_v, low_v, close_v, vol_v = _to_float(open_s), _to_float(high_s), _to_float(low_s), _to_float(close_s), _to_float(vol_s)
    valid = ((~pd.isna(dt)) & (~pd.isna(open_v)) & (~pd.isna(high_v)) & (~pd.isna(low_v)) & (~pd.isna(close_v)) & (~pd.isna(vol_v)))
    dropped_invalid = int((~valid).sum())

    out = pd.DataFrame({"open": open_v[valid], "high": high_v[valid], "low": low_v[valid], "close": close_v[valid], "volume": vol_v[valid]}, index=dt[valid])
    out = out[~out.index.duplicated(keep="last")].sort_index()
    if out.empty:
        raise ValueError("No valid rows after parsing.")

    stats = {
        "format": fmt,
        "separator": sep,
        "fast_loader_requested": fast_loader,
        "fast_loader_used": loader_used,
        "rows_read": rows_read,
        "rows_after_tail": int(len(dt_str)),
        "rows_dropped_invalid": dropped_invalid,
        "rows_valid": int(len(out)),
    }
    return out, stats


def load_base(
    path: Path,
    tail: int,
    fast_loader: str = "auto",
    chunk_size: int = 2_000_000,
) -> tuple[pd.DataFrame, dict[str, object]]:
    return load_ohlc_any_format(path, tail=tail, fast_loader=fast_loader, chunk_size=chunk_size)

def _bars_since_event(flag: pd.Series) -> pd.Series:
    arr = flag.fillna(False).to_numpy(dtype=bool)
    out = np.full(arr.shape[0], np.nan, dtype=np.float64)
    last = -1
    for i, v in enumerate(arr):
        if v:
            last = i
            out[i] = 0.0
        elif last >= 0:
            out[i] = float(i - last)
    return pd.Series(out, index=flag.index)


def compute_price_action_features(
    tf_df: pd.DataFrame,
    tf: int,
    breakout_lbs: list[int],
    with_candle_patterns: bool,
    with_breakout_features: bool,
    with_fvg_features: bool,
    with_liquidity_sweeps: bool,
    with_market_structure: bool,
    with_orderblock_proximity: bool,
) -> pd.DataFrame:
    f = pd.DataFrame(index=tf_df.index)
    h = tf_df["high"]
    l = tf_df["low"]
    o = tf_df["open"]
    c = tf_df["close"]
    rng = (h - l).replace(0, np.nan)

    if with_candle_patterns:
        f[f"candle_clv_tf{tf}"] = ((c - l) / rng) * 100.0
        f[f"candle_body_ratio_tf{tf}"] = (c - o).abs() / rng
        f[f"candle_body_signed_tf{tf}"] = (c - o) / rng
        f[f"candle_upper_wick_ratio_tf{tf}"] = (h - np.maximum(o, c)) / rng
        f[f"candle_lower_wick_ratio_tf{tf}"] = (np.minimum(o, c) - l) / rng
        f[f"candle_wick_imbalance_tf{tf}"] = f[f"candle_lower_wick_ratio_tf{tf}"] - f[f"candle_upper_wick_ratio_tf{tf}"]

    if with_breakout_features:
        for lb in breakout_lbs:
            if lb <= 1:
                continue
            prev_hi = h.rolling(lb).max().shift(1)
            prev_lo = l.rolling(lb).min().shift(1)
            up = c > prev_hi
            dn = c < prev_lo
            f[f"break_up{lb}_tf{tf}"] = up.astype(float)
            f[f"break_dn{lb}_tf{tf}"] = dn.astype(float)
            f[f"break_up_dist{lb}_tf{tf}"] = (c / prev_hi) - 1.0
            f[f"break_dn_dist{lb}_tf{tf}"] = (prev_lo / c) - 1.0
            f[f"break_up_bars_since{lb}_tf{tf}"] = _bars_since_event(up)
            f[f"break_dn_bars_since{lb}_tf{tf}"] = _bars_since_event(dn)

    if with_fvg_features:
        bull_flag = l > h.shift(2)
        bear_flag = h < l.shift(2)
        bull_size = ((l - h.shift(2)) / c).where(bull_flag, 0.0)
        bear_size = ((l.shift(2) - h) / c).where(bear_flag, 0.0)

        f[f"fvg_bull_flag_tf{tf}"] = bull_flag.astype(float)
        f[f"fvg_bear_flag_tf{tf}"] = bear_flag.astype(float)
        f[f"fvg_bull_size_tf{tf}"] = bull_size
        f[f"fvg_bear_size_tf{tf}"] = bear_size
        f[f"fvg_any_size_tf{tf}"] = bull_size + bear_size
        f[f"fvg_bull_bars_since_tf{tf}"] = _bars_since_event(bull_flag)
        f[f"fvg_bear_bars_since_tf{tf}"] = _bars_since_event(bear_flag)

    if with_liquidity_sweeps:
        prev_hh20 = h.rolling(20).max().shift(1)
        prev_ll20 = l.rolling(20).min().shift(1)
        sweep_high = (h > prev_hh20) & (c < prev_hh20) & (c < o)
        sweep_low = (l < prev_ll20) & (c > prev_ll20) & (c > o)
        f[f"liq_sweep_high20_tf{tf}"] = sweep_high.astype(float)
        f[f"liq_sweep_low20_tf{tf}"] = sweep_low.astype(float)
        f[f"liq_sweep_high20_bars_since_tf{tf}"] = _bars_since_event(sweep_high)
        f[f"liq_sweep_low20_bars_since_tf{tf}"] = _bars_since_event(sweep_low)

    if with_market_structure:
        hh = h > h.rolling(5).max().shift(1)
        ll = l < l.rolling(5).min().shift(1)
        f[f"ms_hh_tf{tf}"] = hh.astype(float)
        f[f"ms_ll_tf{tf}"] = ll.astype(float)
        f[f"ms_hh_bars_since_tf{tf}"] = _bars_since_event(hh)
        f[f"ms_ll_bars_since_tf{tf}"] = _bars_since_event(ll)

    if with_orderblock_proximity:
        bull_ob = l.rolling(20).min().shift(1)
        bear_ob = h.rolling(20).max().shift(1)
        f[f"dist_orderblock_bull_tf{tf}"] = (c / bull_ob) - 1.0
        f[f"dist_orderblock_bear_tf{tf}"] = (bear_ob / c) - 1.0

    return f


def add_session_features(df: pd.DataFrame, with_weekend: bool = False) -> pd.DataFrame:
    idx = df.index
    hour = idx.hour + (idx.minute / 60.0)
    dow = idx.dayofweek

    sess = pd.DataFrame(index=df.index)
    sess["session_hour_sin"] = np.sin((2.0 * np.pi * hour) / 24.0).astype(np.float32)
    sess["session_hour_cos"] = np.cos((2.0 * np.pi * hour) / 24.0).astype(np.float32)
    sess["session_dow"] = dow.astype(np.float32)
    sess["session_is_london"] = ((hour >= 7.0) & (hour < 16.0)).astype(np.float32)
    sess["session_is_ny"] = ((hour >= 13.0) & (hour < 22.0)).astype(np.float32)
    sess["session_is_ldn_ny_overlap"] = ((hour >= 13.0) & (hour < 16.0)).astype(np.float32)
    if with_weekend:
        sess["session_is_weekend"] = (dow >= 5).astype(np.float32)
    return pd.concat([df, sess], axis=1)
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
    breakout_lbs: list[int],
    with_candle_patterns: bool,
    with_breakout_features: bool,
    with_fvg_features: bool,
    with_liquidity_sweeps: bool,
    with_market_structure: bool,
    with_orderblock_proximity: bool,
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

    paf = compute_price_action_features(
        tf_df=tf_df,
        tf=tf,
        breakout_lbs=breakout_lbs,
        with_candle_patterns=with_candle_patterns,
        with_breakout_features=with_breakout_features,
        with_fvg_features=with_fvg_features,
        with_liquidity_sweeps=with_liquidity_sweeps,
        with_market_structure=with_market_structure,
        with_orderblock_proximity=with_orderblock_proximity,
    )
    if not paf.empty:
        f = pd.concat([f, paf], axis=1)

    return f




def add_delta_features(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    base_cols = [
        c for c in df.columns
        if c.startswith((
            "dist_", "dist_vwap", "rsi", "adx", "plus_di", "minus_di", "dx", "macd", "vol_z", "vwap_d"
        ))
    ]

    delta_frames = []
    for w in windows:
        if w <= 0:
            continue
        d = df[base_cols].subtract(df[base_cols].shift(w))
        d.columns = [f"delta_{w}_{c}" for c in base_cols]
        d = d.astype(np.float32)
        delta_frames.append(d)

    if delta_frames:
        return pd.concat([df] + delta_frames, axis=1)
    return df


def add_targets(df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    target_frames = []
    for h in horizons:
        t = (df["close"].shift(-h) / df["close"] - 1.0).astype(np.float32)
        target_frames.append(t.rename(f"fwd_ret_{h}"))

    if target_frames:
        return pd.concat([df] + target_frames, axis=1)
    return df


def summarize(df: pd.DataFrame, horizons: list[int]) -> dict:
    # Avoid creating a huge fully non-NA frame (dropna) on very wide tables.
    numeric = df.select_dtypes(include=[np.number])
    rows_numeric = int((numeric.notna().any(axis=1)).sum()) if not numeric.empty else 0
    summary: dict[str, object] = {"rows": int(len(df)), "rows_numeric": rows_numeric}
    corrs = {}
    for h in horizons:
        tgt = f"fwd_ret_{h}"
        if tgt not in numeric.columns:
            continue
        # pairwise NaN handling inside corrwith; no global dropna materialization.
        c = numeric.corrwith(numeric[tgt]).sort_values(ascending=False)
        c = c[c.notna()]
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
        "price_action_family": re.compile(r"^(candle_|break_|fvg_|session_)"),
        "delta_family": re.compile(r"^delta_"),
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
    # memory-safe rounding without full DataFrame copy
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

        if col.startswith("dist_") or col.startswith("dist_vwap") or col.startswith("delta_"):
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

        if col.startswith(("candle_", "break_", "fvg_", "session_")):
            df[col] = df[col].round(6)
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
    breakout_lbs = [int(x) for x in args.breakout_lookbacks.split(",") if x.strip()]
    pattern_tfs = {int(x) for x in args.pattern_tfs.split(",") if x.strip()}
    delta_windows = [int(x) for x in args.delta_windows.split(",") if x.strip()]
    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]

    base, load_stats = load_base(
        args.data,
        args.tail,
        fast_loader=args.fast_loader,
        chunk_size=args.chunk_size,
    )
    print(f"Source format: {load_stats.get('format')} | sep={load_stats.get('separator')}")
    print(
        "Loader requested/used: "
        f"{load_stats.get('fast_loader_requested')} -> {load_stats.get('fast_loader_used')}"
    )
    print(f"Rows read: {load_stats.get('rows_read')}")
    print(f"Rows valid: {load_stats.get('rows_valid')}")
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
        tf_features = compute_tf_features(
            base, tf, ema_lens, rsi_lens, adx_lens, support_lbs, args.with_vwap,
            breakout_lbs=breakout_lbs,
            with_candle_patterns=args.with_candle_patterns and (tf in pattern_tfs),
            with_breakout_features=args.with_breakout_features and (tf in pattern_tfs),
            with_fvg_features=args.with_fvg_features and (tf in pattern_tfs),
            with_liquidity_sweeps=args.with_liquidity_sweeps and (tf in pattern_tfs),
            with_market_structure=args.with_market_structure and (tf in pattern_tfs),
            with_orderblock_proximity=args.with_orderblock_proximity and (tf in pattern_tfs),
        )
        aligned = tf_features.reindex(feature_table.index, method="ffill")
        tf_frames.append(aligned)
        print(f"TF={tf} done. cols={aligned.shape[1]}")

    if tf_frames:
        feature_table = pd.concat([feature_table] + tf_frames, axis=1)

    if args.with_session_features:
        feature_table = add_session_features(feature_table, with_weekend=args.with_session_weekend)

    feature_table = add_delta_features(feature_table, delta_windows)

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

    out_path = Path(args.out_features)
    out_str = str(out_path)
    if args.output_format == "csv_gz":
        if not out_str.lower().endswith(".csv.gz"):
            out_path = Path(out_str + ".csv.gz")
    else:
        if not out_str.lower().endswith(".parquet"):
            out_path = Path(out_str + ".parquet")

    print("Export starting:", out_path)

    if args.output_format == "csv_gz":
        # fast gzip (compresslevel=1)
        with gzip.open(out_path, "wt", compresslevel=1, encoding="utf-8", newline="") as f:
            feature_table.to_csv(f, index_label="datetime", float_format="%.6f")
    else:
        try:
            feature_table.to_parquet(out_path, index=True)
        except Exception as e:
            raise RuntimeError(
                "Parquet export failed. Install a parquet engine (pyarrow or fastparquet), "
                "or use --output-format csv_gz."
            ) from e

    print("Export finished:", out_path)

    out_summary_str = str(args.out_summary).strip().lower()
    write_summary = out_summary_str not in {"", "false", "none", "null", "0"}

    if write_summary:
        report = summarize(feature_table, horizons)
        report["miner_feature_usage"] = miner_feature_usage(feature_table)
        report["source_load_stats"] = load_stats
        report["rows_after_warmup"] = int(len(feature_table))
        report["features_output_path"] = str(out_path)
        report["features_output_format"] = args.output_format
        args.out_summary.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Saved features: {out_path}")
    if write_summary:
        print(f"Saved summary:  {args.out_summary}")
    else:
        print("Saved summary:  skipped")
    print(f"Rows: {len(feature_table)} | Columns: {len(feature_table.columns)}")


if __name__ == "__main__":
    main()
