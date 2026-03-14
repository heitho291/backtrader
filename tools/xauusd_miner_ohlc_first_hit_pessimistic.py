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
    p.add_argument("--trail-min-level", type=float, default=0.0,
                   help="Deprecated: ignored. Miner now qualifies on reaching --trail-activate within hold.")

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

    p.add_argument("--score-sortino-bad", type=float, default=0.5)
    p.add_argument("--score-sortino-mid", type=float, default=1.0)
    p.add_argument("--score-sortino-good", type=float, default=1.75)

    p.add_argument("--score-trades-bad-test", type=float, default=0.5)
    p.add_argument("--score-trades-mid-test", type=float, default=2.0)
    p.add_argument("--score-trades-good-test", type=float, default=4.5)
    p.add_argument("--score-trades-bad-train", type=float, default=0.5)
    p.add_argument("--score-trades-mid-train", type=float, default=1.75)
    p.add_argument("--score-trades-good-train", type=float, default=3.5)

    p.add_argument("--score-k-low", type=float, default=4.0, help="Lower-tail steepness for asymptotic score mapping")
    p.add_argument("--score-k-high", type=float, default=1.2, help="Upper-tail steepness for asymptotic score mapping")
    p.add_argument("--one-trade-at-a-time", action="store_true", default=True)
    p.add_argument("--no-one-trade-at-a-time", dest="one_trade_at_a_time", action="store_false")

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
    conds: List[Cond],
    same_reference_group: Optional[Dict[str, int]] = None,
) -> bool:
    c_col = conds[cand_idx][0]
    c_meta = _parse_feature_meta(c_col)
    fam = str(c_meta["family"])

    same_fam = []
    same_fam_cols = []
    for i in current_idxs:
        col = conds[i][0]
        meta = _parse_feature_meta(col)
        if str(meta["family"]) == fam:
            same_fam.append(meta)
            same_fam_cols.append(col)

    if len(same_fam) >= 2:
        return False

    c_scale = c_meta.get("scale")
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
        cand_gid = int(same_reference_group.get(c_col, 0))
        if cand_gid > 0:
            for col in same_fam_cols:
                if int(same_reference_group.get(col, 0)) == cand_gid:
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


def _score_higher_better(x: float, bad: float, mid: float, good: float, k_low: float, k_high: float, eps: float = 1e-12) -> float:
    if not np.isfinite(x):
        return eps
    bad = float(bad); mid = float(mid); good = float(good)
    if not (bad < mid < good):
        return eps
    if x <= bad:
        return float(max(eps, 0.25 * np.exp(-max(eps, k_low) * (bad - x))))
    if x < mid:
        u = (x - bad) / (mid - bad)
        return float(min(1.0 - eps, max(eps, 0.25 + 0.25 * u)))
    if x < good:
        u = (x - mid) / (good - mid)
        return float(min(1.0 - eps, max(eps, 0.50 + 0.25 * u)))
    v = 1.0 - np.exp(-max(eps, k_high) * (x - good))
    return float(min(1.0 - eps, max(eps, 0.75 + 0.25 * v)))


def _score_lower_better(x: float, good: float, mid: float, bad: float, k_low: float, k_high: float, eps: float = 1e-12) -> float:
    if not np.isfinite(x):
        return eps
    # invert by mapping -x as higher-better with mirrored anchors
    return _score_higher_better(-float(x), -float(bad), -float(mid), -float(good), k_low=k_low, k_high=k_high, eps=eps)


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
    cols: List[str],
    thresholds: Dict[str, Dict[float, float]],
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

    conds: List[Cond] = []
    for c in cols:
        if c not in thresholds:
            continue
        for q, thr in thresholds[c].items():
            conds.append((c, "<=" if q <= 0.5 else ">=", thr))

    summary_empty = {
        "rule": "",
        "train_base_win": base_win(y[:train_idx]),
        "test_base_win": base_win(y_test),
        "train_base_ev": base_ev(pnl[:train_idx], y[:train_idx]),
        "test_base_ev": base_ev(pnl_test, y_test),
        "test_ev": float("nan"),
        "test_win": float("nan"),
        "test_sharpe_proxy": float("nan"),
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
    if not conds:
        return summary_empty, pd.DataFrame()

    two_starts_topk = max(2, int(two_starts_topk))
    two_starts_family_topn = max(1, int(two_starts_family_topn))
    same_reference_group = {} if disable_same_reference_check else _build_same_reference_groups(df.iloc[:train_idx], conds)

    col_values = {c: df[c].to_numpy(copy=False) for c in cols}
    test_weekdays = max(1, int((df.index[train_idx:].dayofweek < 5).sum()))
    train_weekdays = max(1, int((df.index[:train_idx].dayofweek < 5).sum()))
    test_years = test_weekdays / 252.0
    train_years = train_weekdays / 252.0
    cond_masks = []
    for c, op, thr in conds:
        x = col_values[c]
        m = (x <= thr) if op == "<=" else (x >= thr)
        cond_masks.append(m)

    def score(mask_full: np.ndarray, required_hits: int = min_test_hits) -> Optional[dict]:
        cand_train = mask_full[:train_idx] & tradable_train
        if not cand_train.any():
            return None
        sel_train = one_trade_at_a_time_from_masks(time_min[:train_idx], cand_train, t_exit[:train_idx]) if one_trade_at_a_time else cand_train.copy()
        train_hits = int(sel_train.sum())
        if train_hits == 0:
            return None
        train_pp = pnl[:train_idx][sel_train]
        train_cum = float(np.prod(1.0 + train_pp) - 1.0) if train_hits else float("nan")
        min_hits_ok = train_hits >= required_hits
        override_ok = (min_hits_return_override > 0) and (train_cum >= min_hits_return_override)
        if not (min_hits_ok or override_ok):
            return None

        cand_test = mask_full[train_idx:] & tradable_test
        if not cand_test.any():
            return None
        sel = one_trade_at_a_time_from_masks(time_min_test, cand_test, t_exit_test) if one_trade_at_a_time else cand_test.copy()
        hits = int(sel.sum())
        if hits == 0:
            return None

        yy = y_test[sel]
        pp = pnl_test[sel]
        u, counts = np.unique(day_id_test[sel], return_counts=True)
        _ = u
        med_per_day = float(np.median(counts)) if counts.size else 0.0

        # score components (test side) using annualized return and trading-day activity
        test_ret_factor = float(np.prod(1.0 + pp) - 1.0)
        test_ann = annualized_return_from_factor(test_ret_factor, test_years)
        test_sortino = sortino_proxy(pp)
        test_trades_day = float(hits / test_weekdays)

        train_ann = annualized_return_from_factor(train_cum, train_years)

        ret_bad_train = float(score_cfg["ret_bad_train"])
        ret_mid_train = float(score_cfg["ret_mid_train"])
        ret_good_train = float(score_cfg["ret_good_train"])

        R_test = _score_higher_better(test_ann, score_cfg["ret_bad_test"], score_cfg["ret_mid_test"], score_cfg["ret_good_test"], score_cfg["k_low"], score_cfg["k_high"])
        eq_score = np.cumprod(1.0 + pp)
        dd_raw = abs(float(np.min((eq_score / np.maximum.accumulate(eq_score)) - 1.0))) if hits else float("nan")
        DD_test = _score_lower_better(dd_raw, score_cfg["dd_good"], score_cfg["dd_mid"], score_cfg["dd_bad"], score_cfg["k_low"], score_cfg["k_high"])
        So_test = _score_higher_better(test_sortino, score_cfg["so_bad"], score_cfg["so_mid"], score_cfg["so_good"], score_cfg["k_low"], score_cfg["k_high"])
        T_test = _score_higher_better(test_trades_day, score_cfg["tr_bad_test"], score_cfg["tr_mid_test"], score_cfg["tr_good_test"], score_cfg["k_low"], score_cfg["k_high"])

        R_train = _score_higher_better(train_ann, ret_bad_train, ret_mid_train, ret_good_train, score_cfg["k_low"], score_cfg["k_high"])
        So_train = So_test
        T_train = _score_higher_better(float(train_hits / train_weekdays), score_cfg["tr_bad_train"], score_cfg["tr_mid_train"], score_cfg["tr_good_train"], score_cfg["k_low"], score_cfg["k_high"])
        DD_train = DD_test

        score_test = 0.45 * R_test + 0.30 * DD_test + 0.10 * So_test + 0.15 * T_test
        score_train = 0.40 * R_train + 0.25 * DD_train + 0.10 * So_train + 0.25 * T_train
        base_score = 0.75 * score_test + 0.25 * score_train
        penalty = float(np.sqrt(max(1e-9, min(R_test, DD_test, So_test, T_test))))
        score_visible = float(100.0 * base_score * penalty)

        return {
            "ev": float(np.mean(pp)),
            "win": float((yy == 1).mean()),
            "sharpe": sharpe_proxy(pp),
            "hits": hits,
            "train_hits": train_hits,
            "median_per_day": med_per_day,
            "median_exit": float(np.median(t_exit_test[sel])) if hits else float("nan"),
            "median_qualify": float(np.median(t_qual_test[sel][t_qual_test[sel] >= 0])) if hits and np.any(t_qual_test[sel] >= 0) else float("nan"),
            "median_tp_hits": float(np.median(tp_hits_test[sel])) if hits else float("nan"),
            "cumulative_return": float(np.prod(1.0 + pp) - 1.0),
            "train_cumulative_return": train_cum,
            "min_hits_override_used": int((not min_hits_ok) and override_ok),
            "score": score_visible,
            "score_test": score_test,
            "score_train": score_train,
            "score_penalty": penalty,
            "test_ann_return": test_ann,
            "train_ann_return": train_ann,
            "test_sortino_proxy": test_sortino,
            "test_trades_per_day": test_trades_day,
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
        if np.isfinite(a["sharpe"]) and np.isfinite(b["sharpe"]) and a["sharpe"] != b["sharpe"]:
            return a["sharpe"] > b["sharpe"]
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
        for i, m in enumerate(cond_masks):
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
                used_feat = {conds[base_i][0]}
                for j, (c, _, _) in enumerate(conds):
                    if j == base_i or c in used_feat:
                        continue
                    if not _rule_extension_allowed([base_i], j, conds, same_reference_group):
                        continue
                    sc = score(cond_masks[base_i] & cond_masks[j], required_hits=req_hits)
                    if sc is not None and better(sc, best_ext):
                        best_ext = sc
                        best_ext_idxs = [base_i, j]

                key_main = (lambda d: d["ev"]) if objective == "test_ev" else (lambda d: d["win"])
                singles_sorted = sorted(single_scores, key=lambda x: key_main(x[1]), reverse=True)
                pair_seed_idxs = [i for i, _ in singles_sorted[:two_starts_topk]]
                fam_buckets: Dict[str, List[Tuple[int, dict]]] = {}
                for i0, sc0 in single_scores:
                    fam = str(_parse_feature_meta(conds[i0][0]).get("family", conds[i0][0]))
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
                        if conds[i][0] == conds[j][0]:
                            continue
                        if not _rule_extension_allowed([i], j, conds, same_reference_group):
                            continue
                        sc = score(cond_masks[i] & cond_masks[j], required_hits=req_hits)
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
                used_feat = {conds[i][0] for i in current_idxs}
                best_cand = None
                best_idx = None
                cm = current["mask"]
                for i, (c, _, _) in enumerate(conds):
                    if i in current_idxs or c in used_feat:
                        continue
                    if not _rule_extension_allowed(current_idxs, i, conds, same_reference_group):
                        continue
                    sc = score(cm & cond_masks[i], required_hits=req_hits)
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
                used_feat = {conds[i][0] for i in current_idxs}
                best_cand = None
                best_idx = None
                cm = current["mask"]
                for i, (c, _, _) in enumerate(conds):
                    if i in current_idxs or c in used_feat:
                        continue
                    if not _rule_extension_allowed(current_idxs, i, conds, same_reference_group):
                        continue
                    sc = score(cm & cond_masks[i], required_hits=req_hits)
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

    rule = simplify_rule([conds[i] for i in current_idxs])
    mask = np.ones(n, dtype=bool)
    for c, op, thr in rule:
        x = col_values[c]
        mask &= (x <= thr) if op == "<=" else (x >= thr)

    final = score(mask, required_hits=req_hits)
    if final is None:
        final = current
        mask = current["mask"]

    cand_test_final = mask[train_idx:] & tradable_test
    sel_test = one_trade_at_a_time_from_masks(time_min_test, cand_test_final, t_exit_test) if one_trade_at_a_time else cand_test_final.copy()

    dt_idx = df.index[train_idx:][sel_test]
    sig = pd.DataFrame(index=dt_idx)
    sig["close"] = df.loc[dt_idx, "close"].astype(float)
    sig["pnl"] = pnl_test[sel_test].astype(float)
    sig["outcome"] = np.where(y_test[sel_test] == 1, "TP", "SL")
    sig["minutes_to_hit"] = t_exit_test[sel_test].astype(int)
    sig["tp_hits"] = tp_hits_test[sel_test].astype(int)

    wf_mean_win, wf_min_win, wf_mean_ev, wf_total_hits = walk_forward_test_stats(
        time_min_test=time_min_test,
        cand_test_mask=cand_test_final,
        t_exit_test=t_exit_test,
        pnl_test=pnl_test,
        y_test=y_test,
        wf_folds=wf_folds,
        one_trade_at_a_time=one_trade_at_a_time,
    )

    yy = y_test[sel_test]
    pp = pnl_test[sel_test]
    hits = int(sel_test.sum())
    rule_str = " & ".join([f"{c} {op} {thr:.6g}" for c, op, thr in rule])

    # Train-side rule metrics (same gating logic as test)
    y_train = y[:train_idx]
    pnl_train = pnl[:train_idx]
    t_exit_train = t_exit[:train_idx]
    tp_hits_train = tp_hits[:train_idx]
    tradable_train = tradable[:train_idx]
    time_min_train = time_min[:train_idx]
    cand_train = mask[:train_idx] & tradable_train
    sel_train = one_trade_at_a_time_from_masks(time_min_train, cand_train, t_exit_train) if one_trade_at_a_time else cand_train.copy()
    train_hits = int(sel_train.sum())
    train_yy = y_train[sel_train] if train_hits else np.array([], dtype=np.int8)
    train_pp = pnl_train[sel_train] if train_hits else np.array([], dtype=np.float64)

    # Test equity drawdown from sequential selected trades
    if hits:
        eq = np.cumprod(1.0 + pp)
        peak = np.maximum.accumulate(eq)
        dd = (eq / peak) - 1.0
        test_max_drawdown_pct = float(np.min(dd))
    else:
        test_max_drawdown_pct = float("nan")

    # HODL annualized baselines
    close_train = df["close"].iloc[:train_idx].astype(float)
    close_test = df["close"].iloc[train_idx:].astype(float)
    hodl_train_factor = float(close_train.iloc[-1] / close_train.iloc[0] - 1.0) if len(close_train) > 1 and close_train.iloc[0] != 0 else float("nan")
    hodl_test_factor = float(close_test.iloc[-1] / close_test.iloc[0] - 1.0) if len(close_test) > 1 and close_test.iloc[0] != 0 else float("nan")
    train_weekdays_all = max(1, int((pd.Index(df.index[:train_idx]).dayofweek < 5).sum()))
    test_weekdays_all = max(1, int((pd.Index(df.index[train_idx:]).dayofweek < 5).sum()))
    hodl_train_ann = annualized_return_from_factor(hodl_train_factor, train_weekdays_all / 252.0)
    hodl_test_ann = annualized_return_from_factor(hodl_test_factor, test_weekdays_all / 252.0)

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
        "test_sharpe_proxy": sharpe_proxy(pp) if hits else float("nan"),
        "test_sortino_proxy": sortino_proxy(pp) if hits else float("nan"),
        "train_hits": train_hits,
        "test_hits": hits,
        "test_median_signals_day": final["median_per_day"] if hits else 0.0,
        "test_median_minutes_to_exit": final.get("median_exit", float("nan")) if hits else float("nan"),
        "test_median_minutes_to_qualify": final.get("median_qualify", float("nan")) if hits else float("nan"),
        "test_median_tp_hits": float(np.median(tp_hits_test[sel_test])) if hits else float("nan"),
        # Keep legacy column names, but compute realized compounded return for consistency
        # with the objective/override logic and sequential trade accounting.
        "test_return": float(np.prod(1.0 + pp) - 1.0) if hits else float("nan"),
        "train_return": float(np.prod(1.0 + train_pp) - 1.0) if train_hits else float("nan"),
        "test_max_drawdown_pct": test_max_drawdown_pct,
        "conds": len(rule),
        "min_test_hits_used": used_min_test_hits,
        "min_train_hits_used": used_min_test_hits,
        "min_hits_override_used": int(final.get("min_hits_override_used", 0)) if hits else 0,
        "test_cumulative_return": float(np.prod(1.0 + pp) - 1.0) if hits else float("nan"),
        "train_cumulative_return": float(np.prod(1.0 + train_pp) - 1.0) if train_hits else float("nan"),
        "test_trades_per_day": final.get("test_trades_per_day", float("nan")) if hits else float("nan"),
        "train_trades_per_day": float(train_hits / max(1, int((pd.Index(df.index[:train_idx]).dayofweek < 5).sum()))) if train_hits else float("nan"),
        "test_annualized_return": final.get("test_ann_return", float("nan")) if hits else float("nan"),
        "train_annualized_return": annualized_return_from_factor(float(np.prod(1.0 + train_pp) - 1.0), max(1e-9, max(1, int((pd.Index(df.index[:train_idx]).dayofweek < 5).sum()))/252.0)) if train_hits else float("nan"),
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
    }
    return summary, sig


def run_single_config(
    df: pd.DataFrame,
    train_idx: int,
    cols: List[str],
    thresholds: Dict[str, Dict[float, float]],
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
    tp_summary_value: float,
    seed: int,
) -> Tuple[dict, pd.DataFrame]:
    high = df["high"].to_numpy(dtype=np.float32, copy=False)
    low = df["low"].to_numpy(dtype=np.float32, copy=False)
    close = df["close"].to_numpy(dtype=np.float32, copy=False)

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
    )

    summary, sig = mine_best_rule(
        df=df,
        cols=cols,
        thresholds=thresholds,
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
    summary["trail_min_level"] = trail_min_level
    summary["trail_min_level_ignored"] = 1
    summary["include_unrealized_at_test_end"] = int(include_unrealized_at_test_end)
    summary["slippage_bps"] = slippage_bps
    summary["spread_bps"] = spread_bps
    summary["seed"] = seed
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

    if args.trail_min_level != 0:
        print("[WARN] --trail-min-level is deprecated/ignored; qualification uses --trail-activate.")

    if not (0.0 <= args.min_test_hits_reduce_step < 1.0):
        raise ValueError("--min-test-hits-reduce-step must be in [0,1)")
    if args.min_hits_return_override < 0:
        raise ValueError("--min-hits-return-override must be >= 0")
    if args.two_starts_topk < 2:
        raise ValueError("--two-starts-topk must be >= 2")
    if args.two_starts_family_topn < 1:
        raise ValueError("--two-starts-family-topn must be >= 1")

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
    qs = [float(x) for x in args.quantiles.split(",") if x.strip()]
    if not qs:
        raise ValueError("Need at least one quantile")

    cols = build_candidate_features(df, args.allow_absolute_price_features, args.max_features)
    print(f"Candidate features: {len(cols)}")
    thresholds = quantile_thresholds(df.iloc[:train_idx], cols, qs)

    summaries: List[dict] = []
    signals: List[pd.DataFrame] = []

    tp_runs = tps_all if args.sl_equals_tp else [tps_all[0]]

    for tp in tp_runs:
        tps = tps_all if args.use_multi_tp else [tp]
        tp_w = parse_tp_weights(tps, args.tp_weights)
        sl = tp if args.sl_equals_tp else float(args.sl)
        min_hits = args.min_test_hits

        print("\n===================================")
        print(f"run tp={tp:.6g} sl={sl:.6g} hold={args.hold} tp_exits={args.use_multi_tp} trail={args.trail} trail_min_level={args.trail_min_level:.6g} include_unrealized={int(args.include_unrealized_at_test_end)}")

        summary, sig = run_single_config(
            df=df,
            train_idx=train_idx,
            cols=cols,
            thresholds=thresholds,
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
            tp_summary_value=tp,
            seed=args.seed,
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
