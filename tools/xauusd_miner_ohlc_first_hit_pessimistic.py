#!/usr/bin/env python3
"""Fast rule miner with pessimistic first-hit simulation and optional multi-TP/trailing.

Default behavior is backward-compatible with the previous miner pipeline:
- writes best_rules_summary.csv and signals_best_rules.csv
- keeps tp/sl/hold columns in summary and signals
- keeps outcome labels TP/SL and minutes_to_hit in signals

Key upgrades:
- one-trade-at-a-time capital lock (optional, default on)
- objective can be test_ev (mean pnl) or test_win
- precomputed candidate masks and incremental AND in greedy search
- optional feature cap to reduce runtime/temperature on smaller machines
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

Cond = Tuple[str, str, float]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--binned-features", type=Path, default=None,
                   help="Optional parquet/csv with pre-binned candidate features aligned to --features")
    p.add_argument("--prefilter-bins", type=int, default=20,
                   help="Used only when --binned-features is not supplied and bins must be built in-process")
    p.add_argument("--tail-rows", type=int, default=0,
                   help="Optional cap: keep only last N rows from features (0=all)")
    p.add_argument("--hold", type=int, default=90)

    p.add_argument("--tps", type=str, default="0.0015,0.0025,0.0035,0.0045")
    p.add_argument("--tp-weights", type=str, default="", help="Optional weights, comma-separated, same length as --tps")
    p.add_argument("--use-multi-tp", action="store_true", default=True,
                   help="Enable TP-based exits; if disabled, positions are closed only by SL/trailing")
    p.add_argument("--no-multi-tp", dest="use_multi_tp", action="store_false")

    p.add_argument("--sl", type=float, default=0.0015)
    p.add_argument("--sl-equals-tp", action="store_true", default=False, help="If set, run one row per TP and override --sl with tp")
    p.add_argument("--slippage-bps", type=float, default=0.0)
    p.add_argument("--spread-bps", type=float, default=0.0, help="Spread in bps applied as additional long entry cost")

    p.add_argument("--trail", action="store_true", default=True)
    p.add_argument("--no-trail", dest="trail", action="store_false")
    p.add_argument("--trail-activate", type=float, default=0.0010, help="x: profit activation threshold")
    p.add_argument("--trail-offset", type=float, default=0.0006, help="y: offset subtracted from max profit before trailing")
    p.add_argument("--trail-factor", type=float, default=0.5, help="z: proportional trailing factor on (max_profit - offset)")
    p.add_argument("--trail-min-level", type=float, default=0.0, help=argparse.SUPPRESS)

    p.add_argument("--train-frac", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=42, help="Run seed for traceability")
    p.add_argument("--min-test-hits", type=int, default=120, help="Minimum TRAIN trades required for a rule (applied before test scoring)")
    p.add_argument("--min-test-hits-reduce-step", type=float, default=0.10,
                   help="If no feasible rule at min-test-hits (train-gated), reduce by this fraction iteratively (e.g. 0.10 => -10%%)")
    p.add_argument("--min-hits-return-override", type=float, default=3.0,
                   help="If cumulative TRAIN return reaches this level, min-test-hits can be bypassed")
    p.add_argument("--include-unrealized-at-test-end", action="store_true", default=True,
                   help="Mark still-open trades at data end using unrealized PnL for scoring/output")
    p.add_argument("--no-include-unrealized-at-test-end", dest="include_unrealized_at_test_end", action="store_false")
    p.add_argument("--max-conds", type=int, default=5)
    p.add_argument("--min-conds", type=int, default=3)
    p.add_argument("--quantiles", type=str, default="0.05,0.10,0.90,0.95")
    p.add_argument("--wf-folds", type=int, default=4)
    p.add_argument("--objective", choices=["test_ev", "test_win", "test_score"], default="test_score")
    p.add_argument("--two-starts", action="store_true", default=True,
                   help="Also test top-k pair starts in addition to best single-condition start")
    p.add_argument("--no-two-starts", dest="two_starts", action="store_false")
    p.add_argument("--two-starts-topk", type=int, default=32,
                   help="Top-k single conditions used to form candidate pair starts when --two-starts is enabled")
    p.add_argument("--two-starts-family-topn", type=int, default=3,
                   help="Per-feature-family top-N singles used for family-diverse pair seeds")

    p.add_argument("--score-return-bad-test", type=float, default=0.75)
    p.add_argument("--score-return-mid-test", type=float, default=1.5)
    p.add_argument("--score-return-good-test", type=float, default=2.5)
    p.add_argument("--score-return-bad-train", type=float, default=1.0)

    p.add_argument("--score-dd-good", type=float, default=0.075)
    p.add_argument("--score-dd-mid", type=float, default=0.175)
    p.add_argument("--score-dd-bad", type=float, default=0.25)

    p.add_argument("--score-profit-factor-bad", type=float, default=1.00)
    p.add_argument("--score-profit-factor-mid", type=float, default=1.30)
    p.add_argument("--score-profit-factor-good", type=float, default=1.75)
    p.add_argument("--score-runner-efficiency-bad", type=float, default=0.50)
    p.add_argument("--score-runner-efficiency-mid", type=float, default=1.00)
    p.add_argument("--score-runner-efficiency-good", type=float, default=1.75)
    p.add_argument("--one-trade-at-a-time", action="store_true", default=True)
    p.add_argument("--no-one-trade-at-a-time", dest="one_trade_at_a_time", action="store_false")
    p.add_argument("--cluster-gap-minutes", type=int, default=5)
    p.add_argument("--max-entries-per-cluster", type=int, default=1)

    p.add_argument("--account-margin-usd", type=float, default=1000.0)
    p.add_argument("--broker-leverage", type=float, default=20.0)
    p.add_argument("--lot-step", type=float, default=0.01)
    p.add_argument("--contract-units-per-lot", type=float, default=100.0)
    p.add_argument("--lot-run", type=float, default=float("nan"),
                   help="Optional fixed lot for the whole run when one-trade-at-a-time is disabled")
    p.add_argument("--lot-run-min", type=float, default=0.01,
                   help="Minimum lot used by auto lot sampling when one-trade-at-a-time is disabled")
    p.add_argument("--lot-run-choices", type=str, default="",
                   help="Optional comma-separated lot choices; one is selected deterministically per run seed")

    p.add_argument("--prefilter-top-per-family", type=int, default=10)
    p.add_argument("--prefilter-max-candidates", type=int, default=200)
    p.add_argument("--prefilter-min-positive-hits", type=int, default=25)
    p.add_argument("--prefilter-min-pos-rate", type=float, default=0.0)
    p.add_argument("--prefilter-max-neg-rate", type=float, default=1.0)
    p.add_argument("--prefilter-min-lift", type=float, default=0.0)
    p.add_argument("--prefilter-min-coverage", type=float, default=0.001)
    p.add_argument("--prefilter-max-coverage", type=float, default=0.98)

    p.add_argument("--max-features", type=int, default=0, help="Optional cap of candidate features for faster runs (0=all)")
    p.add_argument(
        "--allow-absolute-price-features",
        action="store_true",
        default=False,
        help="If set, allow open_tf*/close_tf*/ema*/vwap* as rule conditions",
    )

    p.add_argument("--disable-same-reference-check", action="store_true", default=False,
                   help="Disable support/resist identical-reference dedup check (faster/less memory)")
    p.add_argument("--out-summary", type=Path, default=Path("best_rules_summary.csv"))
    p.add_argument("--out-signals", type=Path, default=Path("signals_best_rules.csv"))

    p.add_argument("--tick-data", type=Path, default=None,
                   help="Optional tick file (CSV/Parquet) for realistic intrabar sequencing on critical candles")
    p.add_argument("--tick-cache-parquet", type=Path, default=None,
                   help="Optional parquet cache path used when --tick-data points to CSV")
    p.add_argument("--tick-datetime-column", type=str, default="datetime")
    p.add_argument("--tick-price-column", type=str, default="auto",
                   help="Price column in tick data; auto tries price/last/close or bid+ask mid")
    p.add_argument("--tick-sep", type=str, default=",", help="CSV separator for --tick-data when needed")
    p.add_argument("--finalist-tick-validation", action="store_true", default=True,
                   help="Run tick-accurate late validation only on final selected trades when tick data is available")
    p.add_argument("--no-finalist-tick-validation", dest="finalist_tick_validation", action="store_false")
    return p.parse_args()


def load_features(path: Path, tail_rows: int = 0) -> pd.DataFrame:
    if str(path).lower().endswith(".parquet"):
        if tail_rows > 0:
            try:
                import pyarrow as pa
                import pyarrow.parquet as pq

                pf = pq.ParquetFile(path)
                remaining = int(tail_rows)
                chunks = []
                for rg in range(pf.num_row_groups - 1, -1, -1):
                    if remaining <= 0:
                        break
                    rg_rows = int(pf.metadata.row_group(rg).num_rows)
                    if rg_rows <= 0:
                        continue
                    t = pf.read_row_group(rg)
                    take = min(remaining, rg_rows)
                    if take < rg_rows:
                        t = t.slice(rg_rows - take, take)
                    chunks.append(t)
                    remaining -= take

                if chunks:
                    table = pa.concat_tables(list(reversed(chunks)))
                    df = table.to_pandas()
                else:
                    df = pd.DataFrame()
            except Exception:
                # Fallback: normal pandas parquet load when optimized tail-read is unavailable.
                df = pd.read_parquet(path)
        else:
            df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, compression="infer")

    # parquet often already carries datetime index; CSV usually needs explicit parse.
    if isinstance(df.index, pd.DatetimeIndex):
        pass
    elif "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df = df.set_index("datetime")
    elif len(df.columns) and str(df.columns[0]).lower().startswith("unnamed"):
        df[df.columns[0]] = pd.to_datetime(df[df.columns[0]], errors="coerce")
        df = df.set_index(df.columns[0])
    else:
        raise ValueError("Features must provide datetime index/column (e.g. extractor output).")

    df = df.sort_index()
    miss = {"open", "high", "low", "close"} - set(df.columns)
    if miss:
        raise ValueError(f"Missing OHLC columns: {sorted(miss)}")

    # Reduce memory pressure on very wide feature tables without forcing
    # one giant contiguous allocation.
    num_cols = df.select_dtypes(include=[np.number]).columns
    for c in num_cols:
        dt = df[c].dtype
        if dt != np.float32:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)
    return df


def load_binned_features(path: Path, tail_rows: int = 0) -> pd.DataFrame:
    df = load_features(path, tail_rows=tail_rows)
    num_cols = df.select_dtypes(include=[np.number]).columns
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(np.uint8)
    return df


def build_binned_feature_frame(df: pd.DataFrame, cols: List[str], bins: int) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    bins = max(2, int(bins))
    dtype = np.uint8 if bins <= np.iinfo(np.uint8).max else np.uint16

    for c in cols:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        valid = s.notna()
        if not valid.any():
            out[c] = pd.Series(0, index=df.index, dtype=dtype)
            continue
        vals = s[valid]
        uniq = int(vals.nunique(dropna=True))
        if uniq <= 1:
            encoded = pd.Series(1, index=vals.index, dtype=np.int32)
        elif uniq <= bins:
            levels = np.sort(vals.dropna().unique())
            mapper = {float(v): i + 1 for i, v in enumerate(levels.tolist())}
            encoded = vals.map(mapper).astype(np.int32)
        else:
            ranked = vals.rank(method="first")
            encoded = pd.qcut(ranked, q=bins, labels=False, duplicates="drop").astype(np.int32) + 1
        col = pd.Series(0, index=df.index, dtype=np.int32)
        col.loc[encoded.index] = encoded.to_numpy(dtype=np.int32, copy=False)
        out[c] = col.astype(dtype)
    return out


def load_tick_minute_map(
    path: Path,
    datetime_col: str = "datetime",
    price_col: str = "auto",
    sep: str = ",",
    cache_parquet: Optional[Path] = None,
) -> tuple[np.ndarray, Dict[int, tuple[int, int]]]:
    src = path
    if str(path).lower().endswith(".csv") and cache_parquet is not None:
        if cache_parquet.exists():
            src = cache_parquet
        else:
            tmp = cache_parquet.with_suffix(cache_parquet.suffix + ".tmp")
            tdf = pd.read_csv(path, sep=sep, low_memory=False)
            tdf.to_parquet(tmp, index=False)
            tmp.replace(cache_parquet)
            src = cache_parquet

    if str(src).lower().endswith(".parquet"):
        tdf = pd.read_parquet(src)
    else:
        tdf = pd.read_csv(src, sep=sep, low_memory=False)

    lower_cols = {str(c).strip().lower(): c for c in tdf.columns}
    dt_col = lower_cols.get(datetime_col.strip().lower()) if datetime_col else None
    if dt_col is None:
        for cand in ("datetime", "time", "timestamp", "date"):
            if cand in lower_cols:
                dt_col = lower_cols[cand]
                break
    if dt_col is None:
        raise ValueError("tick data needs a datetime column (use --tick-datetime-column)")

    price_series = None
    if price_col.strip().lower() != "auto":
        pc = lower_cols.get(price_col.strip().lower())
        if pc is None:
            raise ValueError(f"tick price column not found: {price_col}")
        price_series = pd.to_numeric(tdf[pc], errors="coerce")
    else:
        for cand in ("price", "last", "close", "mid"):
            if cand in lower_cols:
                price_series = pd.to_numeric(tdf[lower_cols[cand]], errors="coerce")
                break
        if price_series is None and "bid" in lower_cols and "ask" in lower_cols:
            bid = pd.to_numeric(tdf[lower_cols["bid"]], errors="coerce")
            ask = pd.to_numeric(tdf[lower_cols["ask"]], errors="coerce")
            price_series = (bid + ask) / 2.0
    if price_series is None:
        raise ValueError("tick data needs a price column (price/last/close/mid or bid+ask)")

    dt = pd.to_datetime(tdf[dt_col], errors="coerce", utc=False)
    ticks = pd.DataFrame({"datetime": dt, "price": price_series}).dropna(subset=["datetime", "price"]).sort_values("datetime")
    if ticks.empty:
        raise ValueError("tick data has no valid rows")

    minute_ns = ticks["datetime"].dt.floor("min").view("int64")
    ticks["minute_ns"] = minute_ns

    prices = ticks["price"].to_numpy(dtype=np.float64, copy=False)
    minute_arr = ticks["minute_ns"].to_numpy(dtype=np.int64, copy=False)

    out: Dict[int, tuple[int, int]] = {}
    n = len(minute_arr)
    if n > 0:
        start = 0
        cur = int(minute_arr[0])
        for i in range(1, n):
            m = int(minute_arr[i])
            if m != cur:
                out[cur] = (start, i)
                start = i
                cur = m
        out[cur] = (start, n)
    return prices, out


def build_candidate_features(df: pd.DataFrame, allow_absolute_price: bool, max_features: int) -> List[str]:
    exclude_exact = {"open", "high", "low", "close", "volume"}
    cand: List[str] = []

    for c in df.columns:
        if c in exclude_exact or c.startswith("fwd_ret_"):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue

        if allow_absolute_price:
            cand.append(c)
            continue

        if c.startswith(("open_tf", "high_tf", "low_tf", "close_tf", "volume_tf")):
            continue
        if c.startswith("ema") or c.startswith("vwap_tf"):
            continue

        if c.startswith("delta_") or c.startswith("dist_vwap") or c.startswith("dist_") or c.startswith(("rsi", "adx", "plus_di", "minus_di", "dx", "macd", "vol_z", "candle_", "break_", "fvg_", "session_")):
            cand.append(c)

    if max_features > 0:
        cand = cand[:max_features]
    return cand


def _parse_feature_meta(col: str) -> dict[str, object]:
    m = re.match(r"^delta_(\d+)_(.+)$", col)
    delta_w = None
    base = col
    if m:
        delta_w = int(m.group(1))
        base = m.group(2)

    family = None
    scale = None
    tf = None

    m = re.match(r"^dist_(support|resist)(\d+)_tf(\d+)$", base)
    if m:
        kind = m.group(1)
        lb = int(m.group(2))
        tf = int(m.group(3))
        family = f"dist_{kind}"
        scale = lb * tf
        return {"family": family, "scale": scale, "tf": tf, "delta_w": delta_w}

    m = re.match(r"^dist_ema(\d+)_tf(\d+)$", base)
    if m:
        scale = int(m.group(1)) * int(m.group(2))
        tf = int(m.group(2))
        family = "dist_ema"
        return {"family": family, "scale": scale, "tf": tf, "delta_w": delta_w}

    m = re.match(r"^(rsi|adx|plus_di|minus_di|dx)(\d+)_tf(\d+)$", base)
    if m:
        family = m.group(1)
        scale = int(m.group(2)) * int(m.group(3))
        tf = int(m.group(3))
        return {"family": family, "scale": scale, "tf": tf, "delta_w": delta_w}

    m = re.match(r"^(macd(?:_signal|_hist)?)_tf(\d+)$", base)
    if m:
        family = m.group(1)
        scale = int(m.group(2))
        tf = int(m.group(2))
        return {"family": family, "scale": scale, "tf": tf, "delta_w": delta_w}


    m = re.match(r"^candle_(.+)_tf(\d+)$", base)
    if m:
        family = f"candle_{m.group(1)}"
        tf = int(m.group(2))
        scale = tf
        return {"family": family, "scale": scale, "tf": tf, "delta_w": delta_w}

    m = re.match(r"^break_(up|dn)(?:_dist|_bars_since)?(\d+)_tf(\d+)$", base)
    if m:
        family = f"break_{m.group(1)}"
        scale = int(m.group(2)) * int(m.group(3))
        tf = int(m.group(3))
        return {"family": family, "scale": scale, "tf": tf, "delta_w": delta_w}

    m = re.match(r"^fvg_(bull|bear|any)_(?:flag|size|bars_since)_tf(\d+)$", base)
    if m:
        family = f"fvg_{m.group(1)}"
        tf = int(m.group(2))
        scale = tf
        return {"family": family, "scale": scale, "tf": tf, "delta_w": delta_w}

    if base.startswith("session_"):
        return {"family": base, "scale": None, "tf": None, "delta_w": delta_w}
    if base.startswith("regime_"):
        return {"family": base, "scale": None, "tf": None, "delta_w": delta_w}
    m = re.match(r"^(dist_vwap|vol_z)_tf(\d+)$", base)
    if m:
        family = m.group(1)
        scale = int(m.group(2))
        tf = int(m.group(2))
        return {"family": family, "scale": scale, "tf": tf, "delta_w": delta_w}

    return {"family": base.split("_")[0], "scale": None, "tf": None, "delta_w": delta_w}


def _build_same_reference_groups(df_train: pd.DataFrame, conds: List[Cond]) -> Dict[str, int]:
    """Find support/resist features that map to the same rolling reference point.

    Features are considered the same reference point if their full train-side series is
    identical (within rounding) for the same family and timeframe.
    """
    out: Dict[str, int] = {}
    cols = sorted({c for c, _, _ in conds})
    support_resist_cols = []
    for c in cols:
        meta = _parse_feature_meta(c)
        fam = str(meta.get("family"))
        tf = meta.get("tf")
        if fam in {"dist_support", "dist_resist"} and tf is not None and c in df_train.columns:
            support_resist_cols.append((c, fam, int(tf)))

    next_gid = 1
    by_sig: Dict[Tuple[str, int, bytes], List[str]] = {}
    for c, fam, tf in support_resist_cols:
        arr = df_train[c].to_numpy(dtype=np.float32, copy=False)
        arr = np.round(np.nan_to_num(arr, nan=np.float32(1e30), posinf=np.float32(1e31), neginf=np.float32(-1e31)), 6)
        sig = arr.tobytes()
        by_sig.setdefault((fam, tf, sig), []).append(c)

    for (_, _, _), same_cols in by_sig.items():
        if len(same_cols) < 2:
            continue
        gid = next_gid
        next_gid += 1
        for c in same_cols:
            out[c] = gid

    return out


def _rule_extension_allowed(
    current_idxs: List[int],
    cand_idx: int,
    candidates: list[dict[str, object]],
    same_reference_group: Optional[Dict[str, int]] = None,
) -> bool:
    cand = candidates[cand_idx]
    fam = str(cand.get("family"))
    c_scale = cand.get("scale")

    same_fam = []
    for i in current_idxs:
        meta = candidates[i]
        if str(meta.get("family")) == fam:
            same_fam.append(meta)

    if len(same_fam) >= 2:
        return False

    if c_scale is not None:
        for m in same_fam:
            s = m.get("scale")
            if s is None:
                continue
            lo = float(s) * 0.5
            hi = float(s) * 2.0
            if lo <= float(c_scale) <= hi:
                return False

    if same_reference_group and fam in {"dist_support", "dist_resist"}:
        cand_gid = int(same_reference_group.get(str(cand.get("primary_col", "")), 0))
        if cand_gid > 0:
            for meta in same_fam:
                if int(same_reference_group.get(str(meta.get("primary_col", "")), 0)) == cand_gid:
                    return False

    return True


def quantile_thresholds(train: pd.DataFrame, cols: List[str], qs: List[float]) -> Dict[str, Dict[float, float]]:
    out: Dict[str, Dict[float, float]] = {}
    for c in cols:
        s = train[c].dropna()
        if s.empty:
            continue
        out[c] = {q: float(s.quantile(q)) for q in qs}
    return out


def build_prefiltered_candidates(
    df: pd.DataFrame,
    binned_df: pd.DataFrame,
    cols: List[str],
    train_idx: int,
    y: np.ndarray,
    realistic_train_mask: np.ndarray,
    max_features: int,
    top_per_family: int,
    max_candidates: int,
    min_positive_hits: int,
    min_pos_rate: float,
    max_neg_rate: float,
    min_lift: float,
    min_coverage: float,
    max_coverage: float,
) -> list[dict[str, object]]:
    selected_train = np.asarray(realistic_train_mask[:train_idx], dtype=bool)
    positive_set = selected_train & (y[:train_idx] == 1)
    negative_set = selected_train & (y[:train_idx] == 0)
    pos_total = int(positive_set.sum())
    neg_total = int(negative_set.sum())
    sel_total = int(selected_train.sum())
    if pos_total == 0 or neg_total == 0 or sel_total == 0:
        return []

    candidates: list[dict[str, object]] = []
    eps = 1e-12

    for c in cols:
        if c not in binned_df.columns:
            continue
        meta = _parse_feature_meta(c)
        family = str(meta.get("family", c))
        bins_arr = pd.to_numeric(binned_df[c], errors="coerce").fillna(0).to_numpy(dtype=np.uint16, copy=False)
        train_bins = bins_arr[:train_idx]
        unique_bins = np.unique(train_bins[selected_train])
        unique_bins = unique_bins[unique_bins > 0]
        if unique_bins.size == 0:
            continue

        fam_candidates: list[dict[str, object]] = []
        for bin_id in unique_bins.tolist():
            mask = bins_arr == int(bin_id)
            mask_train = mask[:train_idx]
            coverage = float(mask_train[selected_train].mean()) if sel_total else 0.0
            if coverage < min_coverage or coverage > max_coverage:
                continue
            pos_hits = int(np.sum(mask_train & positive_set))
            neg_hits = int(np.sum(mask_train & negative_set))
            if pos_hits < min_positive_hits:
                continue
            pos_rate = pos_hits / max(1, pos_total)
            neg_rate = neg_hits / max(1, neg_total)
            lift = pos_rate / max(neg_rate, eps)
            if pos_rate < min_pos_rate or neg_rate > max_neg_rate or lift < min_lift:
                continue

            fam_candidates.append({
                "name": f"{c}_bin == {int(bin_id)}",
                "conds": [(c, "==", float(bin_id))],
                "mask": mask,
                "family": family,
                "scale": meta.get("scale"),
                "prefilter_score": float(pos_rate * np.log1p(lift)),
                "primary_col": c,
                "prefilter_pos_rate": float(pos_rate),
                "prefilter_neg_rate": float(neg_rate),
                "prefilter_lift": float(lift),
                "prefilter_coverage": float(coverage),
            })

        fam_candidates = sorted(fam_candidates, key=lambda x: float(x["prefilter_score"]), reverse=True)[:top_per_family]
        candidates.extend(fam_candidates)

    candidates = sorted(candidates, key=lambda x: float(x["prefilter_score"]), reverse=True)
    deduped: list[dict[str, object]] = []
    seen_names: set[str] = set()
    for cand in candidates:
        name = str(cand["name"])
        if name in seen_names:
            continue
        seen_names.add(name)
        deduped.append(cand)
        if len(deduped) >= max_candidates:
            break

    if max_features > 0:
        deduped = deduped[:max_features]
    return deduped


def parse_tp_weights(tps: List[float], w_str: str) -> np.ndarray:
    """Parse TP weights.

    Unlike older behavior, weights are NOT normalized automatically.
    Sum can be < 1.0 to leave a runner position managed by trailing stop.
    """
    if not w_str.strip():
        return np.full(len(tps), 1.0 / len(tps), dtype=np.float64)

    w = np.asarray([float(x) for x in w_str.split(",") if x.strip()], dtype=np.float64)
    if len(w) != len(tps):
        raise ValueError("tp-weights must have same length as tps")
    if np.any(~np.isfinite(w)) or np.any(w < 0):
        raise ValueError("tp-weights must be finite and >= 0")

    s = float(w.sum())
    if s <= 0:
        raise ValueError("tp-weights sum must be > 0")
    if s > 1.0 + 1e-12:
        raise ValueError("tp-weights sum must be <= 1.0 (runner uses the remainder)")
    return w


def simulate_multitp_trailing_pessimistic(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    tps: List[float],
    tp_w: np.ndarray,
    tp_enabled: bool,
    sl: float,
    hold: int,
    slippage_bps: float,
    spread_bps: float,
    trail: bool,
    trail_activate: float,
    trail_offset: float,
    trail_factor: float,
    trail_min_level: float,
    include_unrealized_at_test_end: bool,
    bar_time_ns: Optional[np.ndarray] = None,
    tick_prices_all: Optional[np.ndarray] = None,
    tick_minute_bounds: Optional[Dict[int, tuple[int, int]]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = close.shape[0]
    pnl = np.full(n, np.nan, dtype=np.float64)
    y = np.full(n, -1, dtype=np.int8)
    t_exit = np.full(n, -1, dtype=np.int32)
    t_qual = np.full(n, -1, dtype=np.int32)
    tp_hits = np.zeros(n, dtype=np.int8)

    slip = (slippage_bps + spread_bps) / 10000.0
    tps_arr = np.asarray(tps, dtype=np.float64)
    k_max = len(tps_arr)

    for i in range(max(0, n - 1)):
        entry = close[i] * (1.0 + slip)
        stop_ret = -sl
        stop_level = entry * (1.0 + stop_ret)

        tp_levels = entry * (1.0 + tps_arr)

        remaining = 1.0
        realized = 0.0
        hits = 0

        max_profit_ret = 0.0
        trailing_active = (not trail)

        max_k = n - 1 - i
        qualified = False
        for k in range(1, max_k + 1):
            j = i + k
            h = high[j]
            l = low[j]

            tick_prices = None
            if tick_minute_bounds is not None and tick_prices_all is not None and bar_time_ns is not None:
                minute_ns = int(bar_time_ns[j])
                bounds = tick_minute_bounds.get(minute_ns)
                if bounds is not None:
                    critical = (
                        (l <= stop_level)
                        or (tp_enabled and hits < k_max and h >= tp_levels[hits])
                        or (trail and (not trailing_active) and ((h / entry) - 1.0) >= trail_activate)
                    )
                    if critical:
                        s_idx, e_idx = bounds
                        tick_prices = tick_prices_all[s_idx:e_idx]

            if tick_prices is not None and tick_prices.size > 0:
                for px in tick_prices:
                    curr_profit_ret = (float(px) / entry) - 1.0
                    if curr_profit_ret > max_profit_ret:
                        max_profit_ret = curr_profit_ret

                    if trail and (not trailing_active) and max_profit_ret >= trail_activate:
                        trailing_active = True

                    if trail and trailing_active:
                        cand = (max_profit_ret - trail_offset) * trail_factor
                        if cand > stop_ret:
                            stop_ret = cand
                            stop_level = entry * (1.0 + stop_ret)

                    if (not qualified) and trail and k <= hold and max_profit_ret >= trail_activate:
                        qualified = True
                        t_qual[i] = k

                    if float(px) <= stop_level:
                        realized += remaining * stop_ret
                        pnl[i] = realized
                        y[i] = 1 if qualified else (1 if realized > 0 else 0)
                        t_exit[i] = k
                        tp_hits[i] = hits
                        break

                    if tp_enabled:
                        while hits < k_max and float(px) >= tp_levels[hits]:
                            w = min(float(tp_w[hits]), remaining)
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
                        if y[i] != -1:
                            break

                if y[i] != -1:
                    break
                continue

            # update max profit based on current bar high
            curr_profit_ret = (h / entry) - 1.0
            if curr_profit_ret > max_profit_ret:
                max_profit_ret = curr_profit_ret

            # activate trailing at x = trail_activate
            if trail and (not trailing_active) and max_profit_ret >= trail_activate:
                trailing_active = True

            # proportional trailing (independent of TP ladder steps)
            if trail and trailing_active:
                cand = (max_profit_ret - trail_offset) * trail_factor
                if cand > stop_ret:
                    stop_ret = cand
                    stop_level = entry * (1.0 + stop_ret)

            if (not qualified) and trail and k <= hold and max_profit_ret >= trail_activate:
                qualified = True
                t_qual[i] = k

            # pessimistic order: SL first in bar
            if l <= stop_level:
                realized += remaining * stop_ret
                pnl[i] = realized
                y[i] = 1 if qualified else (1 if realized > 0 else 0)
                t_exit[i] = k
                tp_hits[i] = hits
                break

            # then max one TP step per bar (optional TP-based exits)
            if tp_enabled and hits < k_max and h >= tp_levels[hits]:
                w = min(float(tp_w[hits]), remaining)
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

        if y[i] == -1:
            if include_unrealized_at_test_end:
                final_ret = (close[-1] / entry) - 1.0
                pnl[i] = final_ret
                y[i] = 1 if qualified else (1 if final_ret > 0 else 0)
                t_exit[i] = n - 1 - i
                tp_hits[i] = hits
            elif qualified:
                pnl[i] = trail_activate
                y[i] = 1
                t_exit[i] = n - 1 - i
                tp_hits[i] = hits

    return pnl, y, t_exit, t_qual, tp_hits


def simulate_selected_entries_with_ticks(
    entry_indices: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    tps: List[float],
    tp_w: np.ndarray,
    tp_enabled: bool,
    sl: float,
    hold: int,
    slippage_bps: float,
    spread_bps: float,
    trail: bool,
    trail_activate: float,
    trail_offset: float,
    trail_factor: float,
    include_unrealized_at_test_end: bool,
    bar_time_ns: np.ndarray,
    tick_prices_all: Optional[np.ndarray],
    tick_minute_bounds: Optional[Dict[int, tuple[int, int]]],
) -> dict[int, dict[str, float]]:
    if tick_prices_all is None or tick_minute_bounds is None or entry_indices.size == 0:
        return {}

    slip = (slippage_bps + spread_bps) / 10000.0
    tps_arr = np.asarray(tps, dtype=np.float64)
    k_max = len(tps_arr)
    out: dict[int, dict[str, float]] = {}

    for i in entry_indices.tolist():
        if i >= len(close) - 1:
            continue
        entry = float(close[i]) * (1.0 + slip)
        stop_ret = -sl
        stop_level = entry * (1.0 + stop_ret)
        tp_levels = entry * (1.0 + tps_arr)
        remaining = 1.0
        realized = 0.0
        hits = 0
        max_profit_ret = 0.0
        trailing_active = (not trail)
        qualified = False
        qual_k = -1
        exit_k = -1
        y_i = -1
        pnl_i = float("nan")

        for k in range(1, len(close) - i):
            j = i + k
            h = float(high[j])
            l = float(low[j])
            minute_ns = int(bar_time_ns[j])
            tick_prices = None
            bounds = tick_minute_bounds.get(minute_ns)
            if bounds is not None:
                critical = (
                    (l <= stop_level)
                    or (tp_enabled and hits < k_max and h >= tp_levels[hits])
                    or (trail and (not trailing_active) and ((h / entry) - 1.0) >= trail_activate)
                )
                if critical:
                    s_idx, e_idx = bounds
                    tick_prices = tick_prices_all[s_idx:e_idx]

            if tick_prices is not None and len(tick_prices) > 0:
                for px in tick_prices:
                    px = float(px)
                    curr_profit_ret = (px / entry) - 1.0
                    if curr_profit_ret > max_profit_ret:
                        max_profit_ret = curr_profit_ret
                    if trail and (not trailing_active) and max_profit_ret >= trail_activate:
                        trailing_active = True
                    if trail and trailing_active:
                        cand = (max_profit_ret - trail_offset) * trail_factor
                        if cand > stop_ret:
                            stop_ret = cand
                            stop_level = entry * (1.0 + stop_ret)
                    if (not qualified) and trail and k <= hold and max_profit_ret >= trail_activate:
                        qualified = True
                        qual_k = k
                    if px <= stop_level:
                        realized += remaining * stop_ret
                        pnl_i = realized
                        y_i = 1 if qualified else (1 if realized > 0 else 0)
                        exit_k = k
                        break
                    if tp_enabled:
                        while hits < k_max and px >= tp_levels[hits]:
                            w = min(float(tp_w[hits]), remaining)
                            if w > 0:
                                realized += w * float(tps_arr[hits])
                                remaining -= w
                            hits += 1
                            if remaining <= 1e-12:
                                pnl_i = realized
                                y_i = 1 if qualified else (1 if realized > 0 else 0)
                                exit_k = k
                                break
                        if y_i != -1:
                            break
                if y_i != -1:
                    out[i] = {"pnl": pnl_i, "y": y_i, "t_exit": exit_k, "t_qual": qual_k, "tp_hits": hits}
                    continue

            curr_profit_ret = (h / entry) - 1.0
            if curr_profit_ret > max_profit_ret:
                max_profit_ret = curr_profit_ret
            if trail and (not trailing_active) and max_profit_ret >= trail_activate:
                trailing_active = True
            if trail and trailing_active:
                cand = (max_profit_ret - trail_offset) * trail_factor
                if cand > stop_ret:
                    stop_ret = cand
                    stop_level = entry * (1.0 + stop_ret)
            if (not qualified) and trail and k <= hold and max_profit_ret >= trail_activate:
                qualified = True
                qual_k = k
            if l <= stop_level:
                realized += remaining * stop_ret
                pnl_i = realized
                y_i = 1 if qualified else (1 if realized > 0 else 0)
                exit_k = k
                out[i] = {"pnl": pnl_i, "y": y_i, "t_exit": exit_k, "t_qual": qual_k, "tp_hits": hits}
                break
            if tp_enabled and hits < k_max and h >= tp_levels[hits]:
                w = min(float(tp_w[hits]), remaining)
                if w > 0:
                    realized += w * float(tps_arr[hits])
                    remaining -= w
                hits += 1
                if remaining <= 1e-12:
                    pnl_i = realized
                    y_i = 1 if qualified else (1 if realized > 0 else 0)
                    exit_k = k
                    out[i] = {"pnl": pnl_i, "y": y_i, "t_exit": exit_k, "t_qual": qual_k, "tp_hits": hits}
                    break
        else:
            if include_unrealized_at_test_end:
                final_ret = (float(close[-1]) / entry) - 1.0
                out[i] = {"pnl": final_ret, "y": (1 if qualified else (1 if final_ret > 0 else 0)), "t_exit": len(close) - 1 - i, "t_qual": qual_k, "tp_hits": hits}
            elif qualified:
                out[i] = {"pnl": trail_activate, "y": 1, "t_exit": len(close) - 1 - i, "t_qual": qual_k, "tp_hits": hits}

    return out


def one_trade_at_a_time_from_masks(time_min: np.ndarray, candidate_mask: np.ndarray, minutes_to_exit: np.ndarray) -> np.ndarray:
    out = np.zeros_like(candidate_mask, dtype=bool)
    pos = np.flatnonzero(candidate_mask)
    if pos.size == 0:
        return out
    next_allowed = np.iinfo(np.int64).min

    for p in pos:
        t = int(time_min[p])
        if t < next_allowed:
            continue
        m = int(minutes_to_exit[p])
        if m < 1:
            continue
        out[p] = True
        next_allowed = t + m

    return out


def simplify_rule(rule: List[Cond]) -> List[Cond]:
    merged: Dict[str, Dict[str, float]] = {}
    for c, op, thr in rule:
        merged.setdefault(c, {})
        if op == "<=":
            merged[c]["<="] = thr if "<=" not in merged[c] else min(merged[c]["<="], thr)
        else:
            merged[c][">="] = thr if ">=" not in merged[c] else max(merged[c][">="], thr)
    out: List[Cond] = []
    for c in sorted(merged.keys()):
        for op in ("<=", ">="):
            if op in merged[c]:
                out.append((c, op, merged[c][op]))
    return out


def sharpe_proxy(x: np.ndarray) -> float:
    if x.size == 0:
        return float("nan")
    mu = float(np.mean(x))
    sd = float(np.std(x, ddof=0))
    return float("nan") if sd <= 0 else (mu / sd)


def sortino_proxy(x: np.ndarray, eps: float = 1e-12) -> float:
    if x.size == 0:
        return float("nan")
    mu = float(np.mean(x))
    dn = x[x < 0]
    if dn.size == 0:
        return float("inf") if mu > 0 else float("nan")
    dd = float(np.std(dn, ddof=0))
    return float("nan") if dd <= eps else (mu / dd)


def annualized_return_from_factor(ret_factor: float, years: float, eps: float = 1e-12) -> float:
    if not np.isfinite(ret_factor) or years <= eps:
        return float("nan")
    capital = 1.0 + float(ret_factor)
    if capital <= eps:
        return float("nan")
    return float(capital ** (1.0 / years) - 1.0)


def compounded_return_safe(x: np.ndarray, eps: float = 1e-12) -> float:
    """Stable compounded return for arrays of per-trade returns.

    Uses sum(log1p(r)) to avoid overflow in np.prod(1+r) for very long vectors.
    Returns NaN when any trade return is <= -100%.
    """
    if x.size == 0:
        return float("nan")
    x = np.asarray(x, dtype=np.float64)
    if np.any(~np.isfinite(x)):
        return float("nan")
    if np.any(x <= -1.0 + eps):
        return float("nan")

    s = float(np.sum(np.log1p(x)))
    # clamp to finite exp domain to avoid warnings and inf
    s = max(min(s, 700.0), -745.0)
    return float(np.expm1(s))


def linear_annual_return(sum_return: float, years: float, eps: float = 1e-12) -> float:
    if not np.isfinite(sum_return) or years <= eps:
        return float("nan")
    return float(sum_return / years)


def profit_factor(x: np.ndarray, eps: float = 1e-12) -> float:
    if x.size == 0:
        return float("nan")
    pos = float(np.sum(x[x > 0]))
    neg = float(-np.sum(x[x < 0]))
    if neg <= eps:
        return float("inf") if pos > eps else float("nan")
    return float(pos / neg)


def positive_trade_diagnostics(x: np.ndarray) -> dict[str, float]:
    pos = np.asarray(x[x > 0], dtype=np.float64)
    if pos.size == 0:
        return {
            "avg_positive_pnl": float("nan"),
            "median_positive_pnl": float("nan"),
            "p90_positive_pnl": float("nan"),
            "runner_outlier_share": float("nan"),
        }
    k = max(1, int(np.ceil(pos.size * 0.10)))
    top = np.sort(pos)[-k:]
    total = float(pos.sum())
    return {
        "avg_positive_pnl": float(np.mean(pos)),
        "median_positive_pnl": float(np.median(pos)),
        "p90_positive_pnl": float(np.quantile(pos, 0.90)),
        "runner_outlier_share": float(top.sum() / total) if total > 0 else float("nan"),
    }


def compute_balance_drawdowns(sum_returns: np.ndarray, start_balance: float = 1.0) -> dict[str, float]:
    if sum_returns.size == 0:
        return {
            "absolute_dd_pct": float("nan"),
            "max_dd_pct": float("nan"),
            "relative_dd_pct": float("nan"),
        }
    bal = start_balance + np.cumsum(np.asarray(sum_returns, dtype=np.float64))
    min_bal = float(np.min(bal))
    absolute_dd = max(0.0, (start_balance - min_bal) / start_balance)

    peak = start_balance
    max_dd_abs = 0.0
    max_dd_rel = 0.0
    for b in bal:
        if b > peak:
            peak = float(b)
        dd_abs = (peak - float(b)) / start_balance
        dd_rel = (peak - float(b)) / peak if peak > 0 else float("nan")
        if np.isfinite(dd_abs):
            max_dd_abs = max(max_dd_abs, dd_abs)
        if np.isfinite(dd_rel):
            max_dd_rel = max(max_dd_rel, dd_rel)
    return {
        "absolute_dd_pct": float(absolute_dd),
        "max_dd_pct": float(max_dd_abs),
        "relative_dd_pct": float(max_dd_rel),
    }


def estimate_lot_scenarios(
    close_test: pd.Series,
    account_margin_usd: float,
    broker_leverage: float,
    lot_step: float,
    contract_units_per_lot: float,
) -> dict[str, float]:
    px_max = float(close_test.max()) if len(close_test) else float("nan")
    if not np.isfinite(px_max) or px_max <= 0 or account_margin_usd <= 0 or broker_leverage <= 0:
        return {"lot_small": 0.01, "lot_max_single": float("nan")}

    max_lot = (account_margin_usd * broker_leverage) / (contract_units_per_lot * px_max)
    max_lot = max(0.0, np.floor(max_lot / lot_step) * lot_step)
    lot_small = max(lot_step, 0.01)
    return {
        "lot_small": float(lot_small),
        "lot_max_single": float(max_lot),
    }


def resolve_lot_run(
    lot_scenarios: dict[str, float],
    one_trade_at_a_time: bool,
    lot_step: float,
    seed: int,
    lot_run: float,
    lot_run_min: float,
    lot_run_choices: str,
) -> float:
    lot_max_single = float(lot_scenarios.get("lot_max_single", float("nan")))
    if not np.isfinite(lot_max_single) or lot_max_single <= 0:
        return float("nan")
    if one_trade_at_a_time:
        return lot_max_single

    parsed_choices = [float(x) for x in str(lot_run_choices).split(",") if str(x).strip()]
    if parsed_choices:
        choice = parsed_choices[abs(int(seed)) % len(parsed_choices)]
        chosen = min(choice, lot_max_single)
        return float(np.floor(chosen / lot_step) * lot_step)

    if np.isfinite(lot_run) and lot_run > 0:
        chosen = min(float(lot_run), lot_max_single)
        return float(np.floor(chosen / lot_step) * lot_step)

    lo = max(float(lot_run_min), lot_step, 0.01)
    hi = lot_max_single
    if hi < lo:
        return float(np.floor(max(0.0, hi) / lot_step) * lot_step)
    rng = np.random.default_rng(abs(int(seed)) + 17)
    chosen = float(rng.uniform(lo, hi))
    return float(np.floor(chosen / lot_step) * lot_step)


def max_open_trades_for_lot(
    lot: float,
    reference_price: float,
    account_margin_usd: float,
    broker_leverage: float,
    contract_units_per_lot: float,
) -> int:
    if lot <= 0 or reference_price <= 0 or account_margin_usd <= 0 or broker_leverage <= 0:
        return 0
    margin_per_trade = lot * contract_units_per_lot * reference_price / broker_leverage
    if margin_per_trade <= 0:
        return 0
    return max(0, int(np.floor(account_margin_usd / margin_per_trade)))


def select_trade_mask(
    time_min: np.ndarray,
    candidate_mask: np.ndarray,
    minutes_to_exit: np.ndarray,
    max_open_trades: int,
    cluster_gap_minutes: int,
    max_entries_per_cluster: int,
) -> np.ndarray:
    out = np.zeros_like(candidate_mask, dtype=bool)
    if max_open_trades <= 0 or max_entries_per_cluster <= 0:
        return out
    pos = np.flatnonzero(candidate_mask)
    if pos.size == 0:
        return out

    active_until: list[int] = []
    last_candidate_t: Optional[int] = None
    cluster_entries = 0

    for p in pos:
        t = int(time_min[p])
        m = int(minutes_to_exit[p])
        if m < 1:
            continue

        active_until = [x for x in active_until if x > t]

        if last_candidate_t is None or (t - last_candidate_t) > cluster_gap_minutes:
            cluster_entries = 0
        last_candidate_t = t

        if cluster_entries >= max_entries_per_cluster:
            continue
        if len(active_until) >= max_open_trades:
            continue

        out[p] = True
        active_until.append(t + m)
        cluster_entries += 1

    return out


def is_binary_feature(s: pd.Series) -> bool:
    vals = pd.unique(pd.to_numeric(s.dropna(), errors="coerce"))
    vals = vals[pd.notna(vals)]
    if vals.size == 0:
        return False
    vals = np.unique(np.asarray(vals, dtype=np.float64))
    return vals.size <= 2 and set(np.round(vals, 8).tolist()).issubset({0.0, 1.0})


def _normalize_higher_better(x: float, bad: float, mid: float, good: float, eps: float = 1e-12) -> float:
    if not np.isfinite(x):
        return 0.0
    bad = float(bad)
    mid = float(mid)
    good = float(good)
    if not (bad < mid < good):
        return 0.0
    if x <= bad:
        return 0.0
    if x >= good:
        return 1.0
    if x <= mid:
        return float(max(eps, 0.5 * ((x - bad) / (mid - bad))))
    return float(min(1.0, 0.5 + 0.5 * ((x - mid) / (good - mid))))


def _normalize_lower_better(x: float, good: float, mid: float, bad: float, eps: float = 1e-12) -> float:
    if not np.isfinite(x):
        return 0.0
    return _normalize_higher_better(-float(x), -float(bad), -float(mid), -float(good), eps=eps)


def compute_regime_breakdown(
    df: pd.DataFrame,
    selected_mask: np.ndarray,
    pnl_values: np.ndarray,
    score_cfg: dict,
    sl: float,
    trail_activate: float,
    trail_offset: float,
    trail_factor: float,
    prefix: str,
) -> dict[str, float]:
    out: dict[str, float] = {}
    regime_cols = [
        "regime_dir_bull",
        "regime_dir_bear",
        "regime_dir_sideways",
        "regime_vol_high",
        "regime_vol_normal",
        "regime_vol_low",
    ]
    total_hits = int(np.sum(selected_mask))
    if total_hits <= 0:
        return out
    trail_floor = max(0.0, (trail_activate - trail_offset) * trail_factor)

    for regime_col in regime_cols:
        if regime_col not in df.columns:
            continue
        regime_arr = pd.to_numeric(df[regime_col], errors="coerce").fillna(0).to_numpy(dtype=np.float32, copy=False)
        regime_sel = selected_mask & (regime_arr > 0.5)
        hits = int(np.sum(regime_sel))
        key = f"{prefix}_{regime_col}"
        out[f"{key}_trade_share"] = float(hits / total_hits) if total_hits else float("nan")
        out[f"{key}_activity_share"] = out[f"{key}_trade_share"]
        if hits <= 0:
            out[f"{key}_sum_return"] = float("nan")
            out[f"{key}_relative_dd_pct"] = float("nan")
            out[f"{key}_score"] = float("nan")
            continue

        pp = np.asarray(pnl_values[regime_sel], dtype=np.float64)
        sum_return = float(np.sum(pp))
        dd = compute_balance_drawdowns(pp)
        pf = profit_factor(pp)
        runner_eff = float(np.quantile(np.maximum(0.0, pp - trail_floor), 0.90) / max(sl, 1e-12))
        R = _normalize_higher_better(sum_return, score_cfg["ret_bad_test"], score_cfg["ret_mid_test"], score_cfg["ret_good_test"])
        DD = _normalize_lower_better(dd["relative_dd_pct"], score_cfg["dd_good"], score_cfg["dd_mid"], score_cfg["dd_bad"])
        PF = _normalize_higher_better(pf, score_cfg["pf_bad"], score_cfg["pf_mid"], score_cfg["pf_good"])
        RE = _normalize_higher_better(runner_eff, score_cfg["re_bad"], score_cfg["re_mid"], score_cfg["re_good"])
        score = 100.0 * (0.325 * R + 0.275 * DD + 0.225 * PF + 0.175 * RE) * np.sqrt(max(1e-9, min(R, DD, PF, RE)))
        out[f"{key}_sum_return"] = sum_return
        out[f"{key}_relative_dd_pct"] = dd["relative_dd_pct"]
        out[f"{key}_score"] = float(score)
    return out


def walk_forward_test_stats(
    time_min_test: np.ndarray,
    cand_test_mask: np.ndarray,
    t_exit_test: np.ndarray,
    pnl_test: np.ndarray,
    y_test: np.ndarray,
    wf_folds: int,
    one_trade_at_a_time: bool,
) -> Tuple[float, float, float, int]:
    n = len(y_test)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    edges = np.linspace(0, n, max(1, wf_folds) + 1, dtype=int)

    wins: List[float] = []
    evs: List[float] = []
    total = 0

    for i in range(len(edges) - 1):
        a, b = int(edges[i]), int(edges[i + 1])
        if b <= a:
            continue
        cand = cand_test_mask[a:b]
        if not cand.any():
            continue

        if one_trade_at_a_time:
            sel = one_trade_at_a_time_from_masks(time_min_test[a:b], cand, t_exit_test[a:b])
        else:
            sel = cand.copy()

        hits = int(sel.sum())
        if hits == 0:
            continue
        total += hits
        yy = y_test[a:b][sel]
        pp = pnl_test[a:b][sel]
        wins.append(float((yy == 1).mean()))
        evs.append(float(np.mean(pp)))

    if not wins:
        return float("nan"), float("nan"), float("nan"), 0
    return float(np.mean(wins)), float(np.min(wins)), float(np.mean(evs)), int(total)


def mine_best_rule(
    df: pd.DataFrame,
    binned_df: pd.DataFrame,
    cols: List[str],
    train_idx: int,
    y: np.ndarray,
    pnl: np.ndarray,
    t_exit: np.ndarray,
    t_qual: np.ndarray,
    tp_hits: np.ndarray,
    min_conds: int,
    max_conds: int,
    min_test_hits: int,
    min_test_hits_reduce_step: float,
    min_hits_return_override: float,
    wf_folds: int,
    objective: str,
    one_trade_at_a_time: bool,
    disable_same_reference_check: bool,
    two_starts: bool,
    two_starts_topk: int,
    two_starts_family_topn: int,
    score_cfg: dict,
    sl: float,
    trail_activate: float,
    trail_offset: float,
    trail_factor: float,
    cluster_gap_minutes: int,
    max_entries_per_cluster: int,
    account_margin_usd: float,
    broker_leverage: float,
    lot_step: float,
    contract_units_per_lot: float,
    lot_run: float,
    lot_run_min: float,
    lot_run_choices: str,
    prefilter_top_per_family: int,
    prefilter_max_candidates: int,
    prefilter_min_positive_hits: int,
    prefilter_min_pos_rate: float,
    prefilter_max_neg_rate: float,
    prefilter_min_lift: float,
    prefilter_min_coverage: float,
    prefilter_max_coverage: float,
    seed: int,
    final_tick_validation: bool,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    tps: List[float],
    tp_w: np.ndarray,
    tp_enabled: bool,
    trail: bool,
    hold: int,
    slippage_bps: float,
    spread_bps: float,
    include_unrealized_at_test_end: bool,
    bar_time_ns: np.ndarray,
    tick_prices_all: Optional[np.ndarray] = None,
    tick_minute_bounds: Optional[Dict[int, tuple[int, int]]] = None,
) -> Tuple[dict, pd.DataFrame]:
    n = len(df)
    y_test = y[train_idx:]
    pnl_test = pnl[train_idx:]
    t_exit_test = t_exit[train_idx:]
    t_qual_test = t_qual[train_idx:]
    tp_hits_test = tp_hits[train_idx:]
    tradable = (y == 0) | (y == 1)
    tradable_test = tradable[train_idx:]
    tradable_train = tradable[:train_idx]

    time_min = df.index.values.astype("datetime64[m]").astype("int64")
    time_min_test = time_min[train_idx:]

    day_id_test = df.index[train_idx:].normalize().astype("int64")
    day_id_train = df.index[:train_idx].normalize().astype("int64")

    def base_win(yy: np.ndarray) -> float:
        m = (yy == 0) | (yy == 1)
        return float((yy[m] == 1).mean()) if m.any() else float("nan")

    def base_ev(pp: np.ndarray, yy: np.ndarray) -> float:
        m = (yy == 0) | (yy == 1)
        return float(np.mean(pp[m])) if m.any() else float("nan")

    close_test_series = df["close"].iloc[train_idx:].astype(float)
    lot_scenarios = estimate_lot_scenarios(
        close_test=close_test_series,
        account_margin_usd=account_margin_usd,
        broker_leverage=broker_leverage,
        lot_step=lot_step,
        contract_units_per_lot=contract_units_per_lot,
    )
    reference_price_test = float(close_test_series.max()) if len(close_test_series) else float("nan")
    lot_run_resolved = resolve_lot_run(
        lot_scenarios=lot_scenarios,
        one_trade_at_a_time=one_trade_at_a_time,
        lot_step=lot_step,
        seed=seed,
        lot_run=lot_run,
        lot_run_min=lot_run_min,
        lot_run_choices=lot_run_choices,
    )
    max_open_run = 1 if one_trade_at_a_time else max_open_trades_for_lot(
        lot_run_resolved,
        reference_price_test,
        account_margin_usd,
        broker_leverage,
        contract_units_per_lot,
    )
    max_open_run = max(1, max_open_run) if one_trade_at_a_time else max_open_run

    realistic_train_mask = select_trade_mask(
        time_min=time_min[:train_idx],
        candidate_mask=tradable_train,
        minutes_to_exit=t_exit[:train_idx],
        max_open_trades=max_open_run,
        cluster_gap_minutes=cluster_gap_minutes,
        max_entries_per_cluster=max_entries_per_cluster,
    )

    candidates = build_prefiltered_candidates(
        df=df,
        binned_df=binned_df,
        cols=cols,
        train_idx=train_idx,
        y=y,
        realistic_train_mask=realistic_train_mask,
        max_features=max(len(cols), prefilter_max_candidates),
        top_per_family=prefilter_top_per_family,
        max_candidates=prefilter_max_candidates,
        min_positive_hits=prefilter_min_positive_hits,
        min_pos_rate=prefilter_min_pos_rate,
        max_neg_rate=prefilter_max_neg_rate,
        min_lift=prefilter_min_lift,
        min_coverage=prefilter_min_coverage,
        max_coverage=prefilter_max_coverage,
    )

    summary_empty = {
        "rule": "",
        "train_base_win": base_win(y[:train_idx]),
        "test_base_win": base_win(y_test),
        "train_base_ev": base_ev(pnl[:train_idx], y[:train_idx]),
        "test_base_ev": base_ev(pnl_test, y_test),
        "test_ev": float("nan"),
        "test_win": float("nan"),
        "test_hits": 0,
        "test_median_signals_day": 0.0,
        "test_median_minutes_to_exit": float("nan"),
        "test_median_minutes_to_qualify": float("nan"),
        "test_median_tp_hits": float("nan"),
        "conds": 0,
        "wf_mean_win": float("nan"),
        "wf_min_win": float("nan"),
        "wf_mean_ev": float("nan"),
        "wf_total_hits": 0,
        "min_test_hits_used": 0,
        "min_train_hits_used": 0,
        "min_hits_override_used": 0,
        "test_cumulative_return": float("nan"),
        "train_hits": 0,
    }
    if not candidates:
        return summary_empty, pd.DataFrame()

    two_starts_topk = max(2, int(two_starts_topk))
    two_starts_family_topn = max(1, int(two_starts_family_topn))
    fake_conds = [(str(c["primary_col"]), ">=", 0.0) for c in candidates]
    same_reference_group = {} if disable_same_reference_check else _build_same_reference_groups(df.iloc[:train_idx], fake_conds)
    test_weekdays = max(1, int((df.index[train_idx:].dayofweek < 5).sum()))
    train_weekdays = max(1, int((df.index[:train_idx].dayofweek < 5).sum()))
    test_years = test_weekdays / 252.0
    train_years = train_weekdays / 252.0

    def score(mask_full: np.ndarray, required_hits: int = min_test_hits) -> Optional[dict]:
        cand_train = mask_full[:train_idx] & tradable_train
        if not cand_train.any():
            return None
        max_open_train = max_open_run
        sel_train = select_trade_mask(
            time_min=time_min[:train_idx],
            candidate_mask=cand_train,
            minutes_to_exit=t_exit[:train_idx],
            max_open_trades=max_open_train,
            cluster_gap_minutes=cluster_gap_minutes,
            max_entries_per_cluster=max_entries_per_cluster,
        )
        train_hits = int(sel_train.sum())
        if train_hits == 0:
            return None
        train_pp = pnl[:train_idx][sel_train]
        train_sum_return = float(np.sum(train_pp)) if train_hits else float("nan")
        min_hits_ok = train_hits >= required_hits
        override_ok = (min_hits_return_override > 0) and (train_sum_return >= min_hits_return_override)
        if not (min_hits_ok or override_ok):
            return None

        cand_test = mask_full[train_idx:] & tradable_test
        if not cand_test.any():
            return None
        max_open_test = max_open_run
        sel = select_trade_mask(
            time_min=time_min_test,
            candidate_mask=cand_test,
            minutes_to_exit=t_exit_test,
            max_open_trades=max_open_test,
            cluster_gap_minutes=cluster_gap_minutes,
            max_entries_per_cluster=max_entries_per_cluster,
        )
        hits = int(sel.sum())
        if hits == 0:
            return None

        yy = y_test[sel]
        pp = pnl_test[sel]
        u, counts = np.unique(day_id_test[sel], return_counts=True)
        _ = u
        med_per_day = float(np.median(counts)) if counts.size else 0.0

        test_sum_return = float(np.sum(pp))
        test_ann = linear_annual_return(test_sum_return, test_years)
        train_ann = linear_annual_return(train_sum_return, train_years)
        dd_test = compute_balance_drawdowns(pp)
        dd_train = compute_balance_drawdowns(train_pp)
        pf_test = profit_factor(pp)
        pf_train = profit_factor(train_pp)
        trail_floor = max(0.0, (trail_activate - trail_offset) * trail_factor)
        runner_eff_test = float(np.quantile(np.maximum(0.0, pp - trail_floor), 0.90) / max(sl, 1e-12)) if hits else float("nan")
        runner_eff_train = float(np.quantile(np.maximum(0.0, train_pp - trail_floor), 0.90) / max(sl, 1e-12)) if train_hits else float("nan")
        rar_test = float(test_sum_return / max(dd_test["relative_dd_pct"], 1e-9)) if np.isfinite(dd_test["relative_dd_pct"]) else float("nan")
        rar_train = float(train_sum_return / max(dd_train["relative_dd_pct"], 1e-9)) if np.isfinite(dd_train["relative_dd_pct"]) else float("nan")

        ret_bad_train = float(score_cfg["ret_bad_train"])
        ret_mid_train = float(score_cfg["ret_mid_train"])
        ret_good_train = float(score_cfg["ret_good_train"])

        R_test = _normalize_higher_better(test_sum_return, score_cfg["ret_bad_test"], score_cfg["ret_mid_test"], score_cfg["ret_good_test"])
        DD_test = _normalize_lower_better(dd_test["relative_dd_pct"], score_cfg["dd_good"], score_cfg["dd_mid"], score_cfg["dd_bad"])
        PF_test = _normalize_higher_better(pf_test, score_cfg["pf_bad"], score_cfg["pf_mid"], score_cfg["pf_good"])
        RE_test = _normalize_higher_better(runner_eff_test, score_cfg["re_bad"], score_cfg["re_mid"], score_cfg["re_good"])

        R_train = _normalize_higher_better(train_sum_return, ret_bad_train, ret_mid_train, ret_good_train)
        DD_train = _normalize_lower_better(dd_train["relative_dd_pct"], score_cfg["dd_good"], score_cfg["dd_mid"], score_cfg["dd_bad"])
        PF_train = _normalize_higher_better(pf_train, score_cfg["pf_bad"], score_cfg["pf_mid"], score_cfg["pf_good"])
        RE_train = _normalize_higher_better(runner_eff_train, score_cfg["re_bad"], score_cfg["re_mid"], score_cfg["re_good"])

        score_test = 0.325 * R_test + 0.275 * DD_test + 0.225 * PF_test + 0.175 * RE_test
        score_train = 0.325 * R_train + 0.275 * DD_train + 0.225 * PF_train + 0.175 * RE_train
        base_score = 0.75 * score_test + 0.25 * score_train
        penalty = float(np.sqrt(max(1e-9, min(R_test, DD_test, PF_test, RE_test))))
        score_visible = float(100.0 * base_score * penalty)

        return {
            "ev": float(np.mean(pp)),
            "win": float((yy == 1).mean()),
            "hits": hits,
            "train_hits": train_hits,
            "median_per_day": med_per_day,
            "median_exit": float(np.median(t_exit_test[sel])) if hits else float("nan"),
            "median_qualify": float(np.median(t_qual_test[sel][t_qual_test[sel] >= 0])) if hits and np.any(t_qual_test[sel] >= 0) else float("nan"),
            "median_tp_hits": float(np.median(tp_hits_test[sel])) if hits else float("nan"),
            "test_sum_return": test_sum_return,
            "train_sum_return": train_sum_return,
            "min_hits_override_used": int((not min_hits_ok) and override_ok),
            "score": score_visible,
            "score_test": score_test,
            "score_train": score_train,
            "score_penalty": penalty,
            "test_ann_return": test_ann,
            "train_ann_return": train_ann,
            "test_profit_factor": pf_test,
            "train_profit_factor": pf_train,
            "test_runner_efficiency_p90": runner_eff_test,
            "train_runner_efficiency_p90": runner_eff_train,
            "test_profit_factor_norm": PF_test,
            "train_profit_factor_norm": PF_train,
            "test_runner_efficiency_norm": RE_test,
            "train_runner_efficiency_norm": RE_train,
            "test_drawdowns": dd_test,
            "train_drawdowns": dd_train,
            "test_risk_adjusted_return": rar_test,
            "train_risk_adjusted_return": rar_train,
            "mask": mask_full,
            "sel": sel,
        }

    def better(a: dict, b: Optional[dict]) -> bool:
        if b is None:
            return True
        if objective == "test_ev":
            a_main = a["ev"]
            b_main = b["ev"]
        elif objective == "test_win":
            a_main = a["win"]
            b_main = b["win"]
        else:
            a_main = a.get("score", float("nan"))
            b_main = b.get("score", float("nan"))
        if a_main != b_main:
            return a_main > b_main
        if a["hits"] != b["hits"]:
            return a["hits"] > b["hits"]
        return a["win"] > b["win"]

    def decay_hits_threshold(v: int) -> int:
        factor = max(0.0, min(0.99, float(min_test_hits_reduce_step)))
        nv = int(np.floor(v * (1.0 - factor)))
        if nv == v:
            nv = v - 1
        return max(1, nv)

    req_hits = max(1, int(min_test_hits))
    used_min_test_hits = req_hits
    current: Optional[dict] = None
    current_idxs: List[int] = []

    while True:
        best: Optional[dict] = None
        best_idxs: List[int] = []
        single_scores: List[Tuple[int, dict]] = []
        for i, cand in enumerate(candidates):
            m = np.asarray(cand["mask"], dtype=bool)
            sc = score(m, required_hits=req_hits)
            if sc is None:
                continue
            single_scores.append((i, sc))
            if better(sc, best):
                best = sc
                best_idxs = [i]

        if best is not None:
            current = best
            current_idxs = best_idxs[:]

            if two_starts and max_conds >= 2 and single_scores:
                base_i = current_idxs[0]

                best_ext = None
                best_ext_idxs: Optional[List[int]] = None
                used_feat = {str(candidates[base_i]["primary_col"])}
                for j, cand in enumerate(candidates):
                    if j == base_i or str(cand["primary_col"]) in used_feat:
                        continue
                    if not _rule_extension_allowed([base_i], j, candidates, same_reference_group):
                        continue
                    sc = score(np.asarray(candidates[base_i]["mask"], dtype=bool) & np.asarray(candidates[j]["mask"], dtype=bool), required_hits=req_hits)
                    if sc is not None and better(sc, best_ext):
                        best_ext = sc
                        best_ext_idxs = [base_i, j]

                key_main = (lambda d: d["ev"]) if objective == "test_ev" else (lambda d: d["win"])
                singles_sorted = sorted(single_scores, key=lambda x: key_main(x[1]), reverse=True)
                pair_seed_idxs = [i for i, _ in singles_sorted[:two_starts_topk]]
                fam_buckets: Dict[str, List[Tuple[int, dict]]] = {}
                for i0, sc0 in single_scores:
                    fam = str(candidates[i0].get("family", candidates[i0]["name"]))
                    fam_buckets.setdefault(fam, []).append((i0, sc0))
                fam_idxs: List[int] = []
                for _, arr in fam_buckets.items():
                    arr_sorted = sorted(arr, key=lambda x: key_main(x[1]), reverse=True)
                    fam_idxs.extend([i1 for i1, _ in arr_sorted[:two_starts_family_topn]])
                if fam_idxs:
                    pair_seed_idxs = list(dict.fromkeys(pair_seed_idxs + fam_idxs))

                best_pair = None
                best_pair_idxs: Optional[List[int]] = None
                for a in range(len(pair_seed_idxs)):
                    i = pair_seed_idxs[a]
                    for b in range(a + 1, len(pair_seed_idxs)):
                        j = pair_seed_idxs[b]
                        if candidates[i]["primary_col"] == candidates[j]["primary_col"]:
                            continue
                        if not _rule_extension_allowed([i], j, candidates, same_reference_group):
                            continue
                        sc = score(np.asarray(candidates[i]["mask"], dtype=bool) & np.asarray(candidates[j]["mask"], dtype=bool), required_hits=req_hits)
                        if sc is not None and better(sc, best_pair):
                            best_pair = sc
                            best_pair_idxs = [i, j]

                chosen = current
                chosen_idxs = current_idxs
                if best_ext is not None and better(best_ext, chosen):
                    chosen = best_ext
                    chosen_idxs = best_ext_idxs or chosen_idxs
                if best_pair is not None and better(best_pair, chosen):
                    chosen = best_pair
                    chosen_idxs = best_pair_idxs or chosen_idxs

                current = chosen
                current_idxs = chosen_idxs[:]

            while len(current_idxs) < max_conds:
                used_feat = {str(candidates[i]["primary_col"]) for i in current_idxs}
                best_cand = None
                best_idx = None
                cm = current["mask"]
                for i, cand in enumerate(candidates):
                    if i in current_idxs or str(cand["primary_col"]) in used_feat:
                        continue
                    if not _rule_extension_allowed(current_idxs, i, candidates, same_reference_group):
                        continue
                    sc = score(cm & np.asarray(candidates[i]["mask"], dtype=bool), required_hits=req_hits)
                    if sc is not None and better(sc, best_cand):
                        best_cand = sc
                        best_idx = i

                if best_cand is None:
                    break
                if objective == "test_ev":
                    improves = best_cand["ev"] > current["ev"]
                elif objective == "test_win":
                    improves = best_cand["win"] > current["win"]
                else:
                    improves = best_cand.get("score", float("nan")) > current.get("score", float("nan"))
                if not improves:
                    break

                current_idxs.append(int(best_idx))
                current = best_cand

            while len(current_idxs) < min_conds:
                used_feat = {str(candidates[i]["primary_col"]) for i in current_idxs}
                best_cand = None
                best_idx = None
                cm = current["mask"]
                for i, cand in enumerate(candidates):
                    if i in current_idxs or str(cand["primary_col"]) in used_feat:
                        continue
                    if not _rule_extension_allowed(current_idxs, i, candidates, same_reference_group):
                        continue
                    sc = score(cm & np.asarray(candidates[i]["mask"], dtype=bool), required_hits=req_hits)
                    if sc is not None and better(sc, best_cand):
                        best_cand = sc
                        best_idx = i
                if best_cand is None:
                    break
                current_idxs.append(int(best_idx))
                current = best_cand

            if len(current_idxs) >= min_conds:
                used_min_test_hits = req_hits
                break

        if req_hits <= 1:
            return summary_empty, pd.DataFrame()
        req_hits = decay_hits_threshold(req_hits)

    rule_parts: list[Cond] = []
    for i in current_idxs:
        rule_parts.extend(candidates[i]["conds"])
    rule = simplify_rule([(c, op, thr) for c, op, thr in rule_parts if op in {"<=", ">="}])
    binary_rule = [(c, op, thr) for c, op, thr in rule_parts if op == "=="]
    mask = np.ones(n, dtype=bool)
    for c, op, thr in rule:
        x = pd.to_numeric(df[c], errors="coerce").to_numpy(copy=False)
        mask &= (x <= thr) if op == "<=" else (x >= thr)
    for c, op, thr in binary_rule:
        x = pd.to_numeric(binned_df[c], errors="coerce").to_numpy(copy=False)
        mask &= np.isfinite(x) & (np.abs(x - thr) <= 1e-6)

    final = score(mask, required_hits=req_hits)
    if final is None:
        final = current
        mask = current["mask"]

    cand_test_final = mask[train_idx:] & tradable_test
    sel_test = select_trade_mask(
        time_min=time_min_test,
        candidate_mask=cand_test_final,
        minutes_to_exit=t_exit_test,
        max_open_trades=max_open_run,
        cluster_gap_minutes=cluster_gap_minutes,
        max_entries_per_cluster=max_entries_per_cluster,
    )

    dt_idx = df.index[train_idx:][sel_test]
    sig = pd.DataFrame(index=dt_idx)
    sig["close"] = df.loc[dt_idx, "close"].astype(float)
    sig["pnl"] = pnl_test[sel_test].astype(float)
    sig["outcome"] = np.where(y_test[sel_test] == 1, "TP", "SL")
    sig["minutes_to_hit"] = t_exit_test[sel_test].astype(int)
    sig["tp_hits"] = tp_hits_test[sel_test].astype(int)
    sig["row_close"] = sig["close"]
    sig["row_pnl"] = sig["pnl"]
    sig["row_outcome"] = sig["outcome"]
    sig["row_minutes_to_hit"] = sig["minutes_to_hit"]
    sig["row_tp_hits"] = sig["tp_hits"]

    wf_mean_win, wf_min_win, wf_mean_ev, wf_total_hits = walk_forward_test_stats(
        time_min_test=time_min_test,
        cand_test_mask=cand_test_final,
        t_exit_test=t_exit_test,
        pnl_test=pnl_test,
        y_test=y_test,
        wf_folds=wf_folds,
        one_trade_at_a_time=False,
    )

    yy = y_test[sel_test]
    pp = pnl_test[sel_test]
    hits = int(sel_test.sum())
    rule_str = " & ".join(
        [f"{c} {op} {thr:.6g}" for c, op, thr in rule] +
        [f"{c} == {int(thr)}" for c, _, thr in binary_rule]
    )

    # Train-side rule metrics (same gating logic as test)
    y_train = y[:train_idx]
    pnl_train = pnl[:train_idx]
    t_exit_train = t_exit[:train_idx]
    tp_hits_train = tp_hits[:train_idx]
    tradable_train = tradable[:train_idx]
    time_min_train = time_min[:train_idx]
    cand_train = mask[:train_idx] & tradable_train
    sel_train = select_trade_mask(
        time_min=time_min_train,
        candidate_mask=cand_train,
        minutes_to_exit=t_exit_train,
        max_open_trades=max_open_run,
        cluster_gap_minutes=cluster_gap_minutes,
        max_entries_per_cluster=max_entries_per_cluster,
    )
    late_tick_train_count = 0
    late_tick_test_count = 0
    if final_tick_validation and tick_prices_all is not None and tick_minute_bounds is not None:
        train_abs_idx = np.flatnonzero(sel_train)
        test_abs_idx = np.flatnonzero(sel_test) + train_idx
        refined_train = simulate_selected_entries_with_ticks(
            entry_indices=train_abs_idx.astype(np.int64, copy=False),
            high=high,
            low=low,
            close=close,
            tps=tps,
            tp_w=tp_w,
            tp_enabled=tp_enabled,
            sl=sl,
            hold=hold,
            slippage_bps=slippage_bps,
            spread_bps=spread_bps,
            trail=trail,
            trail_activate=trail_activate,
            trail_offset=trail_offset,
            trail_factor=trail_factor,
            include_unrealized_at_test_end=include_unrealized_at_test_end,
            bar_time_ns=bar_time_ns,
            tick_prices_all=tick_prices_all,
            tick_minute_bounds=tick_minute_bounds,
        )
        refined_test = simulate_selected_entries_with_ticks(
            entry_indices=test_abs_idx.astype(np.int64, copy=False),
            high=high,
            low=low,
            close=close,
            tps=tps,
            tp_w=tp_w,
            tp_enabled=tp_enabled,
            sl=sl,
            hold=hold,
            slippage_bps=slippage_bps,
            spread_bps=spread_bps,
            trail=trail,
            trail_activate=trail_activate,
            trail_offset=trail_offset,
            trail_factor=trail_factor,
            include_unrealized_at_test_end=include_unrealized_at_test_end,
            bar_time_ns=bar_time_ns,
            tick_prices_all=tick_prices_all,
            tick_minute_bounds=tick_minute_bounds,
        )
        for local_idx, abs_idx in enumerate(train_abs_idx.tolist()):
            upd = refined_train.get(abs_idx)
            if upd is None:
                continue
            train_yy_val = int(upd["y"])
            pnl_train[abs_idx] = float(upd["pnl"])
            y_train[abs_idx] = train_yy_val
            t_exit_train[abs_idx] = int(upd["t_exit"])
            tp_hits_train[abs_idx] = int(upd["tp_hits"])
            late_tick_train_count += 1
        for local_idx, abs_idx in enumerate(test_abs_idx.tolist()):
            upd = refined_test.get(abs_idx)
            if upd is None:
                continue
            rel = abs_idx - train_idx
            pnl_test[rel] = float(upd["pnl"])
            y_test[rel] = int(upd["y"])
            t_exit_test[rel] = int(upd["t_exit"])
            tp_hits_test[rel] = int(upd["tp_hits"])
            late_tick_test_count += 1
        # refresh signal-side views after late validation
        sig["pnl"] = pnl_test[sel_test].astype(float)
        sig["outcome"] = np.where(y_test[sel_test] == 1, "TP", "SL")
        sig["minutes_to_hit"] = t_exit_test[sel_test].astype(int)
        sig["tp_hits"] = tp_hits_test[sel_test].astype(int)
        sig["row_pnl"] = sig["pnl"]
        sig["row_outcome"] = sig["outcome"]
        sig["row_minutes_to_hit"] = sig["minutes_to_hit"]
        sig["row_tp_hits"] = sig["tp_hits"]
    train_hits = int(sel_train.sum())
    train_yy = y_train[sel_train] if train_hits else np.array([], dtype=np.int8)
    train_pp = pnl_train[sel_train] if train_hits else np.array([], dtype=np.float64)

    dd_test_all = compute_balance_drawdowns(pp) if hits else compute_balance_drawdowns(np.array([], dtype=np.float64))
    dd_train_all = compute_balance_drawdowns(train_pp) if train_hits else compute_balance_drawdowns(np.array([], dtype=np.float64))

    close_train = df["close"].iloc[:train_idx].astype(float)
    close_test = df["close"].iloc[train_idx:].astype(float)
    hodl_train_factor = float(close_train.iloc[-1] / close_train.iloc[0] - 1.0) if len(close_train) > 1 and close_train.iloc[0] != 0 else float("nan")
    hodl_test_factor = float(close_test.iloc[-1] / close_test.iloc[0] - 1.0) if len(close_test) > 1 and close_test.iloc[0] != 0 else float("nan")
    train_cal_days = max(1, int((df.index[train_idx - 1] - df.index[0]).total_seconds() / 86400.0)) if train_idx > 0 else 1
    test_cal_days = max(1, int((df.index[-1] - df.index[train_idx]).total_seconds() / 86400.0)) if train_idx < len(df.index) else 1
    hodl_train_ann = float((1.0 + hodl_train_factor) ** (365.0 / train_cal_days) - 1.0) if np.isfinite(hodl_train_factor) and hodl_train_factor > -1.0 else float("nan")
    hodl_test_ann = float((1.0 + hodl_test_factor) ** (365.0 / test_cal_days) - 1.0) if np.isfinite(hodl_test_factor) and hodl_test_factor > -1.0 else float("nan")

    pos_diag = positive_trade_diagnostics(pp) if hits else positive_trade_diagnostics(np.array([], dtype=np.float64))
    test_sum_return = float(np.sum(pp)) if hits else float("nan")
    train_sum_return = float(np.sum(train_pp)) if train_hits else float("nan")
    test_pf = profit_factor(pp) if hits else float("nan")
    train_pf = profit_factor(train_pp) if train_hits else float("nan")
    test_rar = float(test_sum_return / max(dd_test_all["relative_dd_pct"], 1e-9)) if hits else float("nan")
    train_rar = float(train_sum_return / max(dd_train_all["relative_dd_pct"], 1e-9)) if train_hits else float("nan")
    trail_floor = max(0.0, (trail_activate - trail_offset) * trail_factor)
    runner_eff = float(np.quantile(np.maximum(0.0, pp - trail_floor), 0.90) / max(1e-12, sl)) if hits else float("nan")
    dd_scaling_factor_to_10pct = float(0.10 / dd_test_all["relative_dd_pct"]) if hits and np.isfinite(dd_test_all["relative_dd_pct"]) and dd_test_all["relative_dd_pct"] > 0 else float("nan")
    projected_sum_pnl_usd_at_10pct_dd = (
        test_sum_return * reference_price_test * contract_units_per_lot * lot_run_resolved * dd_scaling_factor_to_10pct
        if hits and np.isfinite(dd_scaling_factor_to_10pct) and np.isfinite(reference_price_test)
        else float("nan")
    )

    summary = {
        "rule": rule_str,
        "train_base_win": base_win(y[:train_idx]),
        "test_base_win": base_win(y_test),
        "train_base_ev": base_ev(pnl[:train_idx], y[:train_idx]),
        "test_base_ev": base_ev(pnl_test, y_test),
        "ev_train": float(np.mean(train_pp)) if train_hits else float("nan"),
        "ev_test": float(np.mean(pp)) if hits else float("nan"),
        "train_win": float((train_yy == 1).mean()) if train_hits else float("nan"),
        "test_ev": float(np.mean(pp)) if hits else float("nan"),
        "test_win": float((yy == 1).mean()) if hits else float("nan"),
        "train_hits": train_hits,
        "test_hits": hits,
        "test_median_signals_day": final["median_per_day"] if hits else 0.0,
        "test_median_minutes_to_exit": final.get("median_exit", float("nan")) if hits else float("nan"),
        "test_median_minutes_to_qualify": final.get("median_qualify", float("nan")) if hits else float("nan"),
        "test_median_tp_hits": float(np.median(tp_hits_test[sel_test])) if hits else float("nan"),
        "test_return": test_sum_return,
        "train_return": train_sum_return,
        "test_max_drawdown_pct": dd_test_all["max_dd_pct"],
        "conds": len(rule) + len(binary_rule),
        "min_test_hits_used": used_min_test_hits,
        "min_train_hits_used": used_min_test_hits,
        "min_hits_override_used": int(final.get("min_hits_override_used", 0)) if hits else 0,
        "test_cumulative_return": test_sum_return,
        "train_cumulative_return": train_sum_return,
        "test_annualized_return": final.get("test_ann_return", float("nan")) if hits else float("nan"),
        "train_annualized_return": final.get("train_ann_return", float("nan")) if train_hits else float("nan"),
        "test_sum_return": test_sum_return,
        "train_sum_return": train_sum_return,
        "test_annual_return": final.get("test_ann_return", float("nan")) if hits else float("nan"),
        "train_annual_return": final.get("train_ann_return", float("nan")) if train_hits else float("nan"),
        "test_score": final.get("score", float("nan")) if hits else float("nan"),
        "score_test_component": final.get("score_test", float("nan")) if hits else float("nan"),
        "score_train_component": final.get("score_train", float("nan")) if hits else float("nan"),
        "score_penalty": final.get("score_penalty", float("nan")) if hits else float("nan"),
        "wf_hits": wf_total_hits,
        "wf_mean": wf_mean_ev,
        "wf_mean_win": wf_mean_win,
        "wf_min_win": wf_min_win,
        "wf_mean_ev": wf_mean_ev,
        "wf_total_hits": wf_total_hits,
        "hodl_train_return": hodl_train_factor,
        "hodl_test_return": hodl_test_factor,
        "hodl_train_annualized_return": hodl_train_ann,
        "hodl_test_annualized_return": hodl_test_ann,
        "profit_factor": test_pf,
        "train_profit_factor": train_pf,
        "test_risk_adjusted_return": test_rar,
        "train_risk_adjusted_return": train_rar,
        "avg_positive_pnl": pos_diag["avg_positive_pnl"],
        "median_positive_pnl": pos_diag["median_positive_pnl"],
        "p90_positive_pnl": pos_diag["p90_positive_pnl"],
        "runner_outlier_share": pos_diag["runner_outlier_share"],
        "runner_efficiency_p90": runner_eff,
        "runner_efficiency_norm": final.get("test_runner_efficiency_norm", float("nan")) if hits else float("nan"),
        "absolute_dd_pct": dd_test_all["absolute_dd_pct"],
        "max_dd_pct": dd_test_all["max_dd_pct"],
        "relative_dd_pct": dd_test_all["relative_dd_pct"],
        "dd_scaling_factor_to_10pct": dd_scaling_factor_to_10pct,
        "lot_small_at_10pct_dd": lot_scenarios["lot_small"] * dd_scaling_factor_to_10pct if np.isfinite(dd_scaling_factor_to_10pct) else float("nan"),
        "lot_run_at_10pct_dd": lot_run_resolved * dd_scaling_factor_to_10pct if np.isfinite(dd_scaling_factor_to_10pct) else float("nan"),
        "projected_sum_pnl_usd_at_10pct_dd": projected_sum_pnl_usd_at_10pct_dd,
        "lot_small": lot_scenarios["lot_small"],
        "lot_run": lot_run_resolved,
        "lot_max_single": lot_scenarios["lot_max_single"],
        "prefilter_candidates": len(candidates),
        "prefilter_realistic_train_rows": int(realistic_train_mask.sum()),
        "cluster_gap_minutes": cluster_gap_minutes,
        "max_entries_per_cluster": max_entries_per_cluster,
    }
    summary.update({
        "agg_train_hits": summary["train_hits"],
        "agg_test_hits": summary["test_hits"],
        "agg_train_sum_return": summary["train_sum_return"],
        "agg_test_sum_return": summary["test_sum_return"],
        "agg_test_profit_factor": summary["profit_factor"],
        "agg_test_runner_efficiency": summary["runner_efficiency_p90"],
        "risk_test_absolute_dd_pct": summary["absolute_dd_pct"],
        "risk_test_max_dd_pct": summary["max_dd_pct"],
        "risk_test_relative_dd_pct": summary["relative_dd_pct"],
        "risk_dd_scale_to_10pct": summary["dd_scaling_factor_to_10pct"],
        "row_mean_test_pnl": summary["test_ev"],
        "row_mean_train_pnl": summary["ev_train"],
    })
    summary["row_finalist_tick_validation"] = int(final_tick_validation)
    summary["row_finalist_tick_train_trades"] = late_tick_train_count
    summary["row_finalist_tick_test_trades"] = late_tick_test_count
    summary["row_tick_validation_scope"] = "late_finalists_exit_only"
    summary.update(
        compute_regime_breakdown(
            df=df.iloc[:train_idx],
            selected_mask=sel_train,
            pnl_values=pnl_train[:train_idx],
            score_cfg=score_cfg,
            sl=sl,
            trail_activate=trail_activate,
            trail_offset=trail_offset,
            trail_factor=trail_factor,
            prefix="regime_train",
        )
    )
    summary.update(
        compute_regime_breakdown(
            df=df.iloc[train_idx:],
            selected_mask=sel_test,
            pnl_values=pnl_test,
            score_cfg=score_cfg,
            sl=sl,
            trail_activate=trail_activate,
            trail_offset=trail_offset,
            trail_factor=trail_factor,
            prefix="regime_test",
        )
    )
    return summary, sig


def run_single_config(
    df: pd.DataFrame,
    binned_df: pd.DataFrame,
    train_idx: int,
    cols: List[str],
    tps: List[float],
    tp_w: np.ndarray,
    tp_enabled: bool,
    sl: float,
    hold: int,
    slippage_bps: float,
    spread_bps: float,
    trail: bool,
    trail_activate: float,
    trail_offset: float,
    trail_factor: float,
    trail_min_level: float,
    include_unrealized_at_test_end: bool,
    min_conds: int,
    max_conds: int,
    min_test_hits: int,
    min_test_hits_reduce_step: float,
    min_hits_return_override: float,
    wf_folds: int,
    objective: str,
    one_trade_at_a_time: bool,
    disable_same_reference_check: bool,
    two_starts: bool,
    two_starts_topk: int,
    two_starts_family_topn: int,
    score_cfg: dict,
    cluster_gap_minutes: int,
    max_entries_per_cluster: int,
    account_margin_usd: float,
    broker_leverage: float,
    lot_step: float,
    contract_units_per_lot: float,
    lot_run: float,
    lot_run_min: float,
    lot_run_choices: str,
    prefilter_top_per_family: int,
    prefilter_max_candidates: int,
    prefilter_min_positive_hits: int,
    prefilter_min_pos_rate: float,
    prefilter_max_neg_rate: float,
    prefilter_min_lift: float,
    prefilter_min_coverage: float,
    prefilter_max_coverage: float,
    finalist_tick_validation: bool,
    tp_summary_value: float,
    seed: int,
    tick_prices_all: Optional[np.ndarray] = None,
    tick_minute_bounds: Optional[Dict[int, tuple[int, int]]] = None,
) -> Tuple[dict, pd.DataFrame]:
    high = df["high"].to_numpy(dtype=np.float32, copy=False)
    low = df["low"].to_numpy(dtype=np.float32, copy=False)
    close = df["close"].to_numpy(dtype=np.float32, copy=False)
    bar_time_ns = np.asarray(pd.DatetimeIndex(df.index).floor("min").view("int64"), dtype=np.int64)

    pnl, y, t_exit, t_qual, tp_hits = simulate_multitp_trailing_pessimistic(
        high=high,
        low=low,
        close=close,
        tps=tps,
        tp_w=tp_w,
        tp_enabled=tp_enabled,
        sl=sl,
        hold=hold,
        slippage_bps=slippage_bps,
        spread_bps=spread_bps,
        trail=trail,
        trail_activate=trail_activate,
        trail_offset=trail_offset,
        trail_factor=trail_factor,
        trail_min_level=trail_min_level,
        include_unrealized_at_test_end=include_unrealized_at_test_end,
        bar_time_ns=bar_time_ns,
        tick_prices_all=tick_prices_all,
        tick_minute_bounds=tick_minute_bounds,
    )

    summary, sig = mine_best_rule(
        df=df,
        binned_df=binned_df,
        cols=cols,
        train_idx=train_idx,
        y=y,
        pnl=pnl,
        t_exit=t_exit,
        t_qual=t_qual,
        tp_hits=tp_hits,
        min_conds=min_conds,
        max_conds=max_conds,
        min_test_hits=min_test_hits,
        min_test_hits_reduce_step=min_test_hits_reduce_step,
        min_hits_return_override=min_hits_return_override,
        wf_folds=wf_folds,
        objective=objective,
        one_trade_at_a_time=one_trade_at_a_time,
        disable_same_reference_check=disable_same_reference_check,
        two_starts=two_starts,
        two_starts_topk=two_starts_topk,
        two_starts_family_topn=two_starts_family_topn,
        score_cfg=score_cfg,
        sl=sl,
        trail_activate=trail_activate,
        trail_offset=trail_offset,
        trail_factor=trail_factor,
        cluster_gap_minutes=cluster_gap_minutes,
        max_entries_per_cluster=max_entries_per_cluster,
        account_margin_usd=account_margin_usd,
        broker_leverage=broker_leverage,
        lot_step=lot_step,
        contract_units_per_lot=contract_units_per_lot,
        lot_run=lot_run,
        lot_run_min=lot_run_min,
        lot_run_choices=lot_run_choices,
        prefilter_top_per_family=prefilter_top_per_family,
        prefilter_max_candidates=prefilter_max_candidates,
        prefilter_min_positive_hits=prefilter_min_positive_hits,
        prefilter_min_pos_rate=prefilter_min_pos_rate,
        prefilter_max_neg_rate=prefilter_max_neg_rate,
        prefilter_min_lift=prefilter_min_lift,
        prefilter_min_coverage=prefilter_min_coverage,
        prefilter_max_coverage=prefilter_max_coverage,
        seed=seed,
        final_tick_validation=bool(finalist_tick_validation and tick_prices_all is not None and tick_minute_bounds is not None),
        high=high,
        low=low,
        close=close,
        tps=tps,
        tp_w=tp_w,
        tp_enabled=tp_enabled,
        trail=trail,
        hold=hold,
        slippage_bps=slippage_bps,
        spread_bps=spread_bps,
        include_unrealized_at_test_end=include_unrealized_at_test_end,
        bar_time_ns=bar_time_ns,
        tick_prices_all=tick_prices_all,
        tick_minute_bounds=tick_minute_bounds,
    )

    summary["tp"] = tp_summary_value
    summary["sl"] = sl
    summary["hold"] = hold
    summary["minutes"] = hold
    summary["objective"] = objective
    summary["one_trade_at_a_time"] = int(one_trade_at_a_time)
    summary["use_multi_tp"] = int(tp_enabled)
    summary["tps"] = ",".join(f"{x:.8g}" for x in tps)
    summary["tp_weights"] = ",".join(f"{x:.8g}" for x in tp_w.tolist())
    summary["trail"] = int(trail)
    summary["trail_activate"] = trail_activate
    summary["trail_offset"] = trail_offset
    summary["trail_factor"] = trail_factor
    summary["include_unrealized_at_test_end"] = int(include_unrealized_at_test_end)
    summary["slippage_bps"] = slippage_bps
    summary["spread_bps"] = spread_bps
    summary["seed"] = seed
    summary["cluster_gap_minutes"] = cluster_gap_minutes
    summary["max_entries_per_cluster"] = max_entries_per_cluster
    summary["account_margin_usd"] = account_margin_usd
    summary["broker_leverage"] = broker_leverage
    summary["lot_step"] = lot_step
    summary["contract_units_per_lot"] = contract_units_per_lot
    summary["test_start"] = str(df.index[train_idx]) if train_idx < len(df.index) else ""
    summary["test_end"] = str(df.index[-1]) if len(df.index) else ""

    if not sig.empty:
        sig = sig.copy()
        sig["tp"] = tp_summary_value
        sig["sl"] = sl
        sig["hold"] = hold
        sig["rule"] = summary["rule"]
        sig["tps"] = summary["tps"]
        sig["tp_weights"] = summary["tp_weights"]

    return summary, sig


def main() -> None:
    args = parse_args()

    if args.trail:
        if args.trail_activate <= 0:
            raise ValueError("--trail-activate must be > 0 when --trail is enabled")
        if args.trail_offset <= 0:
            raise ValueError("--trail-offset must be > 0 when --trail is enabled")
        if args.trail_factor <= 0:
            raise ValueError("--trail-factor must be > 0 when --trail is enabled")
        if args.trail_offset >= args.trail_activate:
            raise ValueError("--trail-offset must be < --trail-activate when --trail is enabled")

    if not (0.0 <= args.min_test_hits_reduce_step < 1.0):
        raise ValueError("--min-test-hits-reduce-step must be in [0,1)")
    if args.min_hits_return_override < 0:
        raise ValueError("--min-hits-return-override must be >= 0")
    if args.two_starts_topk < 2:
        raise ValueError("--two-starts-topk must be >= 2")
    if args.two_starts_family_topn < 1:
        raise ValueError("--two-starts-family-topn must be >= 1")
    if args.cluster_gap_minutes < 0:
        raise ValueError("--cluster-gap-minutes must be >= 0")
    if args.max_entries_per_cluster < 1:
        raise ValueError("--max-entries-per-cluster must be >= 1")
    if args.account_margin_usd <= 0 or args.broker_leverage <= 0:
        raise ValueError("--account-margin-usd and --broker-leverage must be > 0")
    if args.lot_run_min <= 0:
        raise ValueError("--lot-run-min must be > 0")
    if args.prefilter_bins < 2:
        raise ValueError("--prefilter-bins must be >= 2")

    df = load_features(args.features, args.tail_rows)
    if args.tail_rows > 0 and len(df) > args.tail_rows:
        df = df.tail(int(args.tail_rows))
        print(f"Applied --tail-rows={args.tail_rows}. Rows kept: {len(df)}")
    n = len(df)
    train_idx = int(n * args.train_frac)
    train_idx = max(1, min(train_idx, n - 1))

    tps_all = [float(x) for x in args.tps.split(",") if x.strip()]
    if not tps_all:
        raise ValueError("Need at least one TP value")
    cols = build_candidate_features(df, args.allow_absolute_price_features, args.max_features)
    print(f"Candidate features: {len(cols)}")
    if args.binned_features is not None:
        binned_df = load_binned_features(args.binned_features, args.tail_rows)
        if args.tail_rows > 0 and len(binned_df) > args.tail_rows:
            binned_df = binned_df.tail(int(args.tail_rows))
    else:
        print(f"Building fallback binned features in-process (bins={args.prefilter_bins})")
        binned_df = build_binned_feature_frame(df, cols, args.prefilter_bins)
    binned_df = binned_df.reindex(df.index)

    score_cfg = {
        "ret_bad_test": args.score_return_bad_test,
        "ret_mid_test": args.score_return_mid_test,
        "ret_good_test": args.score_return_good_test,
        "ret_bad_train": args.score_return_bad_train,
        "ret_mid_train": args.score_return_mid_test,
        "ret_good_train": args.score_return_good_test,
        "dd_good": args.score_dd_good,
        "dd_mid": args.score_dd_mid,
        "dd_bad": args.score_dd_bad,
        "pf_bad": args.score_profit_factor_bad,
        "pf_mid": args.score_profit_factor_mid,
        "pf_good": args.score_profit_factor_good,
        "re_bad": args.score_runner_efficiency_bad,
        "re_mid": args.score_runner_efficiency_mid,
        "re_good": args.score_runner_efficiency_good,
    }

    tick_prices_all: Optional[np.ndarray] = None
    tick_minute_bounds: Optional[Dict[int, tuple[int, int]]] = None
    if args.tick_data is not None:
        print(f"Loading tick data: {args.tick_data}")
        tick_prices_all, tick_minute_bounds = load_tick_minute_map(
            path=args.tick_data,
            datetime_col=args.tick_datetime_column,
            price_col=args.tick_price_column,
            sep=args.tick_sep,
            cache_parquet=args.tick_cache_parquet,
        )
        print(f"Tick minutes loaded: {len(tick_minute_bounds)} | ticks loaded: {len(tick_prices_all)}")

    summaries: List[dict] = []
    signals: List[pd.DataFrame] = []

    tp_runs = tps_all if args.sl_equals_tp else [tps_all[0]]

    for tp in tp_runs:
        tps = tps_all if args.use_multi_tp else [tp]
        tp_w = parse_tp_weights(tps, args.tp_weights)
        sl = tp if args.sl_equals_tp else float(args.sl)
        min_hits = args.min_test_hits

        print("\n===================================")
        print(f"run tp={tp:.6g} sl={sl:.6g} hold={args.hold} tp_exits={args.use_multi_tp} trail={args.trail} include_unrealized={int(args.include_unrealized_at_test_end)}")

        summary, sig = run_single_config(
            df=df,
            binned_df=binned_df,
            train_idx=train_idx,
            cols=cols,
            tps=tps,
            tp_w=tp_w,
            tp_enabled=args.use_multi_tp,
            sl=sl,
            hold=args.hold,
            slippage_bps=args.slippage_bps,
            spread_bps=args.spread_bps,
            trail=args.trail,
            trail_activate=args.trail_activate,
            trail_offset=args.trail_offset,
            trail_factor=args.trail_factor,
            trail_min_level=args.trail_min_level,
            include_unrealized_at_test_end=args.include_unrealized_at_test_end,
            min_conds=args.min_conds,
            max_conds=args.max_conds,
            min_test_hits=min_hits,
            min_test_hits_reduce_step=args.min_test_hits_reduce_step,
            min_hits_return_override=args.min_hits_return_override,
            wf_folds=args.wf_folds,
            objective=args.objective,
            one_trade_at_a_time=args.one_trade_at_a_time,
            disable_same_reference_check=args.disable_same_reference_check,
            two_starts=args.two_starts,
            two_starts_topk=args.two_starts_topk,
            two_starts_family_topn=args.two_starts_family_topn,
            score_cfg=score_cfg,
            cluster_gap_minutes=args.cluster_gap_minutes,
            max_entries_per_cluster=args.max_entries_per_cluster,
            account_margin_usd=args.account_margin_usd,
            broker_leverage=args.broker_leverage,
            lot_step=args.lot_step,
            contract_units_per_lot=args.contract_units_per_lot,
            lot_run=args.lot_run,
            lot_run_min=args.lot_run_min,
            lot_run_choices=args.lot_run_choices,
            prefilter_top_per_family=args.prefilter_top_per_family,
            prefilter_max_candidates=args.prefilter_max_candidates,
            prefilter_min_positive_hits=args.prefilter_min_positive_hits,
            prefilter_min_pos_rate=args.prefilter_min_pos_rate,
            prefilter_max_neg_rate=args.prefilter_max_neg_rate,
            prefilter_min_lift=args.prefilter_min_lift,
            prefilter_min_coverage=args.prefilter_min_coverage,
            prefilter_max_coverage=args.prefilter_max_coverage,
            finalist_tick_validation=args.finalist_tick_validation,
            tp_summary_value=tp,
            seed=args.seed,
            tick_prices_all=tick_prices_all,
            tick_minute_bounds=tick_minute_bounds,
        )

        print(f"Best rule: {summary['rule']}")
        print(
            f"TEST EV={summary['test_ev']:.8g} | win={summary['test_win']:.6f} | hits={summary['test_hits']} | "
            f"median/day={summary['test_median_signals_day']:.2f} | conds={summary['conds']}"
        )

        summaries.append(summary)
        if not sig.empty:
            signals.append(sig.reset_index().rename(columns={"index": "datetime"}))

    pd.DataFrame(summaries).to_csv(args.out_summary, index=False)
    if signals:
        pd.concat(signals, axis=0, ignore_index=True).to_csv(args.out_signals, index=False)
    else:
        pd.DataFrame(columns=["datetime", "close", "pnl", "outcome", "minutes_to_hit", "tp_hits", "tp", "sl", "hold", "rule", "tps", "tp_weights"]).to_csv(args.out_signals, index=False)

    print("\nSaved:", args.out_summary)
    print("Saved:", args.out_signals)


if __name__ == "__main__":
    main()
