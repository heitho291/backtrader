#!/usr/bin/env python3
"""
Rule mining on OHLC minute bars with pessimistic intra-bar ordering.

KEEP (from Script 1):
- Multi-TP partial exits (N steps, configurable weights)
- Optional trailing stop that "jumps" to the previous TP level once a TP is hit
- Optional trail activation threshold (profit level before trailing is allowed)
- Optional entry slippage (bps)
- Pessimistic tie-break per minute: if SL and next TP could both be hit in same minute -> SL wins
- Exclude absolute price-level features from rule search by default (reduce regime overfit)

ADD (requested):
- "One trade at a time" capital lock:
    A new signal can only occur AFTER the previous trade has exited (no fixed cooldown).
- Optimize rules primarily by mean PnL (EV) on TEST (i.e., average realized return per trade),
  not by winrate. Print winrate + sharpe proxy as info.
- Speed optimizations:
    * Precompute candidate condition masks once (numpy)
    * Greedy rule building uses cached masks and incremental AND
    * Use integer-minute timestamps for fast capital-lock gating
    * Avoid pandas ops inside scoring loops

Outputs:
- best_rules_summary.csv
- signals_best_rules.csv

Notes:
- This mines ONE event configuration (your multi-TP + trailing + slippage setup).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


# ----------------------------
# CLI
# ----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--features", type=Path, required=True, help="Path to feature_scan_features.csv or .csv.gz")
    p.add_argument("--hold", type=int, default=30, help="Max minutes to look ahead (trade max duration)")

    # Multi-TP settings
    p.add_argument("--tps", type=str, default="0.0015,0.0025,0.0035,0.0060", help="TP levels (fractions, long)")
    p.add_argument(
        "--tp-weights",
        type=str,
        default="",
        help="Optional weights, comma-separated, sum=1. Example: 0.25,0.25,0.25,0.25. If empty: equal.",
    )
    p.add_argument("--use-multi-tp", action="store_true", default=True, help="Enable multi-TP partial exits")
    p.add_argument("--no-multi-tp", dest="use_multi_tp", action="store_false", help="Disable multi-TP (single TP=tps[0])")

    # Stop settings
    p.add_argument("--sl", type=float, default=0.0015, help="Initial SL distance (fraction). Example 0.0015 = 0.15%%")
    p.add_argument("--slippage-bps", type=float, default=0.0, help="Entry slippage in bps. Long entry = close*(1+slip).")

    # Trailing stop settings
    p.add_argument("--trail", action="store_true", default=True, help="Enable trailing SL behavior (default True)")
    p.add_argument("--no-trail", dest="trail", action="store_false", help="Disable trailing SL")
    p.add_argument(
        "--trail-activate",
        type=float,
        default=0.0010,
        help="Trailing becomes active only after price reaches entry*(1+trail_activate). Example 0.001=0.1%%",
    )

    # Mining controls
    p.add_argument("--train-frac", type=float, default=0.7, help="Time split for train/test (by time order)")
    p.add_argument("--min-test-hits", type=int, default=150, help="Minimum TEST trades required for a rule")
    p.add_argument("--max-conds", type=int, default=5, help="Max conditions in rule (3-5 recommended)")
    p.add_argument("--min-conds", type=int, default=3, help="Min conditions in rule")

    # Threshold pool
    p.add_argument("--quantiles", type=str, default="0.05,0.10,0.90,0.95", help="Quantiles used for thresholds")

    # Walk-forward stability on test
    p.add_argument("--wf-folds", type=int, default=4, help="Walk-forward folds on the TEST range")

    # feature selection controls
    p.add_argument(
        "--allow-absolute-price-features",
        action="store_true",
        default=False,
        help="If set, allow open_tf*/close_tf*/ema*/vwap* as rule conditions (NOT recommended).",
    )

    # Outputs
    p.add_argument("--out-summary", type=Path, default=Path("best_rules_summary.csv"))
    p.add_argument("--out-signals", type=Path, default=Path("signals_best_rules.csv"))

    return p.parse_args()


# ----------------------------
# IO / features
# ----------------------------
def load_features(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, compression="infer")
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df = df.set_index("datetime")
    else:
        if df.columns[0].lower().startswith("unnamed"):
            df[df.columns[0]] = pd.to_datetime(df[df.columns[0]], errors="coerce")
            df = df.set_index(df.columns[0])
        else:
            raise ValueError("CSV must contain 'datetime' column or an unnamed index column.")

    df = df.sort_index()
    needed = {"open", "high", "low", "close"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing OHLC columns in features file: {missing}")
    return df


def build_candidate_features(df: pd.DataFrame, allow_absolute_price: bool) -> List[str]:
    """
    Exclude absolute price-level and OHLCV-like columns by default to avoid regime overfit.
    Keep:
      - dist_* (normalized)
      - dist_vwap_*
      - rsi*, adx*, plus_di*, minus_di*, dx*
      - macd* (indicator values)
      - vol_z* (normalized volume)
    """
    exclude_exact = {"open", "high", "low", "close", "volume"}
    cand: List[str] = []

    for c in df.columns:
        if c in exclude_exact:
            continue
        if c.startswith("fwd_ret_"):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue

        if allow_absolute_price:
            cand.append(c)
            continue

        # Exclude absolute price and level series
        if c.startswith(("open_tf", "high_tf", "low_tf", "close_tf", "volume_tf")):
            continue
        if c.startswith("ema"):       # ema{len}_tf{tf} is price-level
            continue
        if c.startswith("vwap_tf"):   # raw vwap is price-level
            continue

        # Allow normalized distance to vwap
        if c.startswith("dist_vwap"):
            cand.append(c)
            continue

        # Allow normalized distances
        if c.startswith("dist_"):
            cand.append(c)
            continue

        # Allow classic indicators (scale-stable)
        if c.startswith(("rsi", "adx", "plus_di", "minus_di", "dx", "macd", "vol_z")):
            cand.append(c)
            continue

    return cand


def quantile_thresholds(train: pd.DataFrame, cols: List[str], qs: List[float]) -> Dict[str, Dict[float, float]]:
    out: Dict[str, Dict[float, float]] = {}
    for c in cols:
        s = train[c].dropna()
        if s.empty:
            continue
        out[c] = {q: float(s.quantile(q)) for q in qs}
    return out


# ----------------------------
# Trade simulator (multi-TP + trailing + slippage) - pessimistic per-minute ordering
# ----------------------------
def _parse_tp_weights(tps: List[float], weights_str: str) -> np.ndarray:
    if not weights_str.strip():
        return np.full(len(tps), 1.0 / len(tps), dtype=np.float64)
    w = [float(x) for x in weights_str.split(",") if x.strip()]
    if len(w) != len(tps):
        raise ValueError("tp-weights must have same length as tps")
    w = np.asarray(w, dtype=np.float64)
    s = float(w.sum())
    if not np.isfinite(s) or s <= 0:
        raise ValueError("tp-weights must sum to > 0")
    return w / s


def simulate_multitp_trailing_pessimistic(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    tps: List[float],
    tp_w: np.ndarray,
    sl: float,
    hold: int,
    slippage_bps: float,
    trail: bool,
    trail_activate: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns arrays (len=n):
      pnl: realized return fraction (e.g. 0.001 = +0.1%%)
      y:   1 win (pnl>0), 0 loss (pnl<=0) for closed trades, -1 none (no exit within window)
      t_exit: minutes until trade closes, -1 if none
      tp_hits: number of TP steps hit (0..len(tps))

    Pessimistic order per minute:
      if low <= stop -> stop first (even if high >= next tp in same minute)
      else if high >= next tp -> take that tp (max one tp per minute)
    """
    n = close.shape[0]
    pnl = np.full(n, np.nan, dtype=np.float64)
    y = np.full(n, -1, dtype=np.int8)
    t_exit = np.full(n, -1, dtype=np.int16)
    tp_hits = np.full(n, 0, dtype=np.int8)

    slip = float(slippage_bps) / 10000.0
    tps_arr = np.asarray(tps, dtype=np.float64)
    k_max = tps_arr.shape[0]

    # Local bindings for speed
    high_a = high
    low_a = low
    close_a = close
    tpw = tp_w

    last_i = n - hold - 1
    for i in range(last_i):
        entry = close_a[i] * (1.0 + slip)

        # stop is tracked as a RETURN from entry (negative or later >=0 if trailed)
        stop_ret = -sl
        stop_level = entry * (1.0 + stop_ret)

        # TP price levels
        tp_levels = entry * (1.0 + tps_arr)

        remaining = 1.0
        realized = 0.0
        hits = 0

        # trailing activation uses HIGH reaching entry*(1+trail_activate)
        trailing_active = (not trail)
        trail_activation_level = entry * (1.0 + trail_activate)

        closed = False
        close_t = -1

        for k in range(1, hold + 1):
            j = i + k
            h = high_a[j]
            l = low_a[j]

            if trail and (not trailing_active) and (h >= trail_activation_level):
                trailing_active = True

            # pessimistic: SL first
            if l <= stop_level:
                realized += remaining * stop_ret
                remaining = 0.0
                closed = True
                close_t = k
                break

            # then 1 TP step max per minute
            if hits < k_max and h >= tp_levels[hits]:
                w = float(tpw[hits])
                if w > remaining:
                    w = remaining
                realized += w * float(tps_arr[hits])
                remaining -= w
                hits += 1

                # trailing "jumps" to previous TP (or entry after first TP)
                if trail and trailing_active:
                    if hits == 1:
                        stop_ret = 0.0
                    else:
                        stop_ret = float(tps_arr[hits - 2])
                    stop_level = entry * (1.0 + stop_ret)

                if remaining <= 1e-12:
                    closed = True
                    close_t = k
                    break

        if closed:
            pnl[i] = realized
            y[i] = 1 if realized > 0 else 0
            t_exit[i] = close_t
            tp_hits[i] = hits
        else:
            # no exit within hold
            # leave pnl nan, y=-1, t_exit=-1
            tp_hits[i] = hits

    return pnl, y, t_exit, tp_hits


# ----------------------------
# One-trade-at-a-time (capital lock)
# ----------------------------
def one_trade_at_a_time_from_masks(
    time_min: np.ndarray,          # int64 minutes since epoch
    candidate_mask: np.ndarray,    # bool
    minutes_to_exit: np.ndarray,   # int16 / int32 (minutes), same length
) -> np.ndarray:
    """
    Select signals in time order such that each new trade starts only after previous exits.
    Requires minutes_to_exit[pos] >= 1 for selected trades.
    """
    out = np.zeros_like(candidate_mask, dtype=bool)
    positions = np.flatnonzero(candidate_mask)
    if positions.size == 0:
        return out

    next_allowed_time = np.int64(-9223372036854775808)  # min int64

    for pos in positions:
        t = time_min[pos]
        if t < next_allowed_time:
            continue
        mte = int(minutes_to_exit[pos])
        if mte < 1:
            continue
        out[pos] = True
        next_allowed_time = t + np.int64(mte)

    return out


# ----------------------------
# Rule mining (fast: cache cond masks, optimize by EV/mean pnl)
# ----------------------------
Cond = Tuple[str, str, float]  # (col, op, thr)


def simplify_rule(rule: List[Cond]) -> List[Cond]:
    """
    Merge redundant thresholds per feature.
    - <= keeps strictest (smallest)
    - >= keeps strictest (largest)
    """
    merged: Dict[str, Dict[str, float]] = {}
    for col, op, thr in rule:
        merged.setdefault(col, {})
        if op == "<=":
            prev = merged[col].get("<=")
            merged[col]["<="] = thr if prev is None else min(prev, thr)
        else:
            prev = merged[col].get(">=")
            merged[col][">="] = thr if prev is None else max(prev, thr)

    out: List[Cond] = []
    for col in sorted(merged.keys()):
        for op in ("<=", ">="):
            if op in merged[col]:
                out.append((col, op, merged[col][op]))
    return out


def compute_sharpe_proxy(pnls: np.ndarray) -> float:
    """
    Sharpe proxy per-trade, no annualization:
      mean(pnl) / std(pnl)
    """
    if pnls.size == 0:
        return float("nan")
    mu = float(np.mean(pnls))
    sd = float(np.std(pnls, ddof=0))
    if sd <= 0:
        return float("nan")
    return mu / sd


def walk_forward_test_stats(
    time_min_test: np.ndarray,
    cand_test_mask: np.ndarray,     # bool on TEST range (rule AND tradable)
    t_exit_test: np.ndarray,        # minutes_to_exit on TEST range
    pnl_test: np.ndarray,           # pnl on TEST range (nan for non-tradable)
    y_test: np.ndarray,             # 1/0/-1 on TEST range
    wf_folds: int,
) -> Tuple[float, float, float, int]:
    """
    Evaluate fixed rule on test in sequential folds, applying one-trade-at-a-time inside each fold.
    Returns: (mean_win, min_win, mean_ev, total_hits)
    """
    n = y_test.shape[0]
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    wf_folds = max(1, int(wf_folds))
    edges = np.linspace(0, n, wf_folds + 1, dtype=int)

    wins = []
    evs = []
    total_hits = 0

    for i in range(wf_folds):
        a = int(edges[i])
        b = int(edges[i + 1])
        if b <= a:
            continue

        cand = cand_test_mask[a:b]
        if not cand.any():
            continue

        sel = one_trade_at_a_time_from_masks(time_min_test[a:b], cand, t_exit_test[a:b])
        hits = int(sel.sum())
        if hits <= 0:
            continue

        yy = y_test[a:b][sel]
        pp = pnl_test[a:b][sel]
        total_hits += hits

        wins.append(float((yy == 1).mean()))
        evs.append(float(np.mean(pp)))

    if not wins:
        return float("nan"), float("nan"), float("nan"), 0

    return float(np.mean(wins)), float(np.min(wins)), float(np.mean(evs)), int(total_hits)


def mine_best_rule(
    df: pd.DataFrame,
    cols: List[str],
    thresholds: Dict[str, Dict[float, float]],
    train_idx: int,
    y: np.ndarray,
    pnl: np.ndarray,
    t_exit: np.ndarray,
    tp_hits: np.ndarray,
    min_conds: int,
    max_conds: int,
    min_test_hits: int,
    wf_folds: int,
) -> Tuple[dict, pd.DataFrame]:
    """
    Optimize primarily by TEST EV (mean pnl), tie-break by SharpeProxy, then hits, then winrate.
    Uses one-trade-at-a-time gating (capital lock) on TEST selection.
    """
    n = len(df)
    test_n = n - train_idx

    # Time as integer minutes (fast gating)
    idx = df.index.values.astype("datetime64[m]")
    time_min = idx.astype("int64")  # minutes since epoch
    time_min_test = time_min[train_idx:]

    y_test = y[train_idx:]
    pnl_test = pnl[train_idx:]
    t_exit_test = t_exit[train_idx:]
    tp_hits_test = tp_hits[train_idx:]

    # Tradable masks: where trade closed within hold
    tradable = (y == 0) | (y == 1)
    tradable_test = tradable[train_idx:]

    # Baselines (no rule, only tradable times, still apply one-trade-at-a-time? -> for baseline, we do NOT,
    # because "no rule" doesn't define entries. We report raw baseline among tradable rows for context.)
    def base_winrate(yy: np.ndarray) -> float:
        m = (yy == 0) | (yy == 1)
        if not m.any():
            return 0.0
        return float((yy[m] == 1).mean())

    def base_ev(pp: np.ndarray, yy: np.ndarray) -> float:
        m = (yy == 0) | (yy == 1)
        if not m.any():
            return float("nan")
        return float(np.mean(pp[m]))

    base_train_win = base_winrate(y[:train_idx])
    base_test_win = base_winrate(y_test)
    base_train_ev = base_ev(pnl[:train_idx], y[:train_idx])
    base_test_ev = base_ev(pnl_test, y_test)

    # Build condition pool
    conds: List[Cond] = []
    for c in cols:
        if c not in thresholds:
            continue
        for q, thr in thresholds[c].items():
            if q <= 0.5:
                conds.append((c, "<=", thr))
            else:
                conds.append((c, ">=", thr))

    if not conds:
        summary = {
            "rule": "",
            "train_base_win": base_train_win,
            "test_base_win": base_test_win,
            "train_base_ev": base_train_ev,
            "test_base_ev": base_test_ev,
            "test_ev": float("nan"),
            "test_win": float("nan"),
            "test_sharpe_proxy": float("nan"),
            "test_hits": 0,
            "test_median_signals_day": 0.0,
            "test_median_minutes_to_exit": float("nan"),
            "test_median_tp_hits": float("nan"),
            "conds": 0,
            "wf_mean_win": float("nan"),
            "wf_min_win": float("nan"),
            "wf_mean_ev": float("nan"),
            "wf_total_hits": 0,
        }
        return summary, pd.DataFrame()

    # Cache numpy arrays for all candidate columns (avoid repeated df[col].to_numpy())
    col_values: Dict[str, np.ndarray] = {c: df[c].to_numpy(dtype=float, copy=False) for c in cols}

    # Precompute mask for each condition ON FULL RANGE (fast incremental AND)
    # Store as uint8 to reduce memory, convert to bool when needed.
    cond_masks: List[np.ndarray] = []
    for (c, op, thr) in conds:
        x = col_values[c]
        if op == "<=":
            m = (x <= thr)
        else:
            m = (x >= thr)
        # Also require feature not NaN, otherwise comparisons can be False already; keep as-is.
        cond_masks.append(m)

    cond_masks = [m.astype(bool, copy=False) for m in cond_masks]

    # Helper: score a rule given its mask (full-range mask)
    def score_from_full_mask(full_mask: np.ndarray) -> Optional[dict]:
        # Candidate entries in TEST: rule true AND tradable
        cand_test = full_mask[train_idx:] & tradable_test
        if not cand_test.any():
            return None

        sel_test = one_trade_at_a_time_from_masks(time_min_test, cand_test, t_exit_test)
        hits = int(sel_test.sum())
        if hits < min_test_hits:
            return None

        pnls = pnl_test[sel_test]
        yy = y_test[sel_test]

        ev = float(np.mean(pnls))
        win = float((yy == 1).mean())
        sharpe = compute_sharpe_proxy(pnls)

        # median signals/day
        # (pandas only once per scored candidate; still acceptable because candidates are limited by greedy search)
        dt_sel = df.index[train_idx:][sel_test]
        per_day = pd.Series(1, index=dt_sel).groupby(dt_sel.date).sum()
        med_per_day = float(per_day.median()) if len(per_day) else 0.0

        med_exit = float(np.median(t_exit_test[sel_test])) if hits else float("nan")
        med_tp_hits = float(np.median(tp_hits_test[sel_test])) if hits else float("nan")

        return {
            "ev": ev,
            "win": win,
            "sharpe": sharpe,
            "hits": hits,
            "median_per_day": med_per_day,
            "median_exit": med_exit,
            "median_tp_hits": med_tp_hits,
            "full_mask": full_mask,
            "cand_test": cand_test,
            "sel_test": sel_test,
        }

    def better(a: dict, b: Optional[dict]) -> bool:
        if b is None:
            return True
        if a["ev"] != b["ev"]:
            return a["ev"] > b["ev"]
        a_sh = a["sharpe"]
        b_sh = b["sharpe"]
        if np.isfinite(a_sh) and np.isfinite(b_sh) and a_sh != b_sh:
            return a_sh > b_sh
        if a["hits"] != b["hits"]:
            return a["hits"] > b["hits"]
        return a["win"] > b["win"]

    # ---------
    # Find best 1-condition rule
    # ---------
    best: Optional[dict] = None
    best_rule_idx: List[int] = []

    used_features: set[str] = set()

    for i, (cond, m) in enumerate(zip(conds, cond_masks)):
        sc = score_from_full_mask(m)
        if sc is None:
            continue
        if better(sc, best):
            best = sc
            best_rule_idx = [i]

    if best is None:
        summary = {
            "rule": "",
            "train_base_win": base_train_win,
            "test_base_win": base_test_win,
            "train_base_ev": base_train_ev,
            "test_base_ev": base_test_ev,
            "test_ev": float("nan"),
            "test_win": float("nan"),
            "test_sharpe_proxy": float("nan"),
            "test_hits": 0,
            "test_median_signals_day": 0.0,
            "test_median_minutes_to_exit": float("nan"),
            "test_median_tp_hits": float("nan"),
            "conds": 0,
            "wf_mean_win": float("nan"),
            "wf_min_win": float("nan"),
            "wf_mean_ev": float("nan"),
            "wf_total_hits": 0,
        }
        return summary, pd.DataFrame()

    # Greedy build
    current_idx = best_rule_idx[:]
    current_best = best

    def rule_features(idxs: List[int]) -> set[str]:
        return {conds[i][0] for i in idxs}

    # ---------
    # Greedy add conditions if EV improves
    # ---------
    while len(current_idx) < max_conds:
        current_mask = current_best["full_mask"]
        used = rule_features(current_idx)

        best_candidate: Optional[dict] = None
        best_candidate_idx: Optional[int] = None

        for i, cond in enumerate(conds):
            if i in current_idx:
                continue
            if cond[0] in used:
                continue

            combined = current_mask & cond_masks[i]
            sc = score_from_full_mask(combined)
            if sc is None:
                continue
            if better(sc, best_candidate):
                best_candidate = sc
                best_candidate_idx = i

        # accept only if EV improves
        if best_candidate is not None and best_candidate["ev"] > current_best["ev"]:
            current_idx.append(int(best_candidate_idx))  # type: ignore[arg-type]
            current_best = best_candidate
        else:
            break

    # Ensure min_conds (even if EV not improving)
    while len(current_idx) < min_conds:
        current_mask = current_best["full_mask"]
        used = rule_features(current_idx)

        best_candidate = None
        best_candidate_idx = None

        for i, cond in enumerate(conds):
            if i in current_idx:
                continue
            if cond[0] in used:
                continue
            combined = current_mask & cond_masks[i]
            sc = score_from_full_mask(combined)
            if sc is None:
                continue
            if better(sc, best_candidate):
                best_candidate = sc
                best_candidate_idx = i

        if best_candidate is None:
            break
        current_idx.append(int(best_candidate_idx))  # type: ignore[arg-type]
        current_best = best_candidate

    # Finalize: simplify thresholds (recompute masks from simplified rule)
    rule_conds = [conds[i] for i in current_idx]
    rule_conds = simplify_rule(rule_conds)

    # Rebuild final full mask from simplified rule (still using cached col arrays)
    final_mask = np.ones(n, dtype=bool)
    for c, op, thr in rule_conds:
        x = col_values[c]
        if op == "<=":
            final_mask &= (x <= thr)
        else:
            final_mask &= (x >= thr)

    final_sc = score_from_full_mask(final_mask)
    if final_sc is None:
        # fallback to current_best
        final_sc = current_best
        final_mask = current_best["full_mask"]
        rule_conds = [conds[i] for i in current_idx]
        rule_conds = simplify_rule(rule_conds)

    rule_str = " & ".join([f"{c} {op} {thr:.6g}" for c, op, thr in rule_conds])

    # Walk-forward stability on TEST range using candidate mask (rule AND tradable)
    cand_test_final = final_mask[train_idx:] & tradable_test
    wf_mean_win, wf_min_win, wf_mean_ev, wf_total_hits = walk_forward_test_stats(
        time_min_test=time_min_test,
        cand_test_mask=cand_test_final,
        t_exit_test=t_exit_test,
        pnl_test=pnl_test,
        y_test=y_test,
        wf_folds=wf_folds,
    )

    # Signals table on TEST
    sel_test = one_trade_at_a_time_from_masks(time_min_test, cand_test_final, t_exit_test)
    dt_idx = df.index[train_idx:][sel_test]
    sig = pd.DataFrame(index=dt_idx)
    sig["close"] = df.loc[dt_idx, "close"].astype(float)
    sig["pnl"] = pnl_test[sel_test].astype(float)
    sig["outcome"] = np.where(y_test[sel_test] == 1, "WIN", "LOSS")
    sig["minutes_to_exit"] = t_exit_test[sel_test].astype(int)
    sig["tp_hits"] = tp_hits_test[sel_test].astype(int)

    # Summary
    pnls = pnl_test[sel_test]
    yy = y_test[sel_test]
    hits = int(sel_test.sum())

    summary = {
        "rule": rule_str,
        "train_base_win": base_train_win,
        "test_base_win": base_test_win,
        "train_base_ev": base_train_ev,
        "test_base_ev": base_test_ev,
        "test_ev": float(np.mean(pnls)) if hits else float("nan"),
        "test_win": float((yy == 1).mean()) if hits else float("nan"),
        "test_sharpe_proxy": compute_sharpe_proxy(pnls) if hits else float("nan"),
        "test_hits": hits,
        "test_median_signals_day": float(
            pd.Series(1, index=dt_idx).groupby(dt_idx.date).sum().median()
        ) if hits else 0.0,
        "test_median_minutes_to_exit": float(np.median(t_exit_test[sel_test])) if hits else float("nan"),
        "test_median_tp_hits": float(np.median(tp_hits_test[sel_test])) if hits else float("nan"),
        "conds": len(rule_conds),
        "wf_mean_win": wf_mean_win,
        "wf_min_win": wf_min_win,
        "wf_mean_ev": wf_mean_ev,
        "wf_total_hits": wf_total_hits,
    }

    return summary, sig


# ----------------------------
# main
# ----------------------------
def main() -> None:
    args = parse_args()

    df = load_features(args.features)
    n = len(df)
    train_idx = int(n * float(args.train_frac))
    train_idx = max(1, min(train_idx, n - 1))

    tps = [float(x) for x in args.tps.split(",") if x.strip()]
    if len(tps) < 1:
        raise ValueError("Need at least one TP in --tps")
    if not args.use_multi_tp:
        tps = [tps[0]]

    tp_w = _parse_tp_weights(tps, args.tp_weights)

    qs = [float(x) for x in args.quantiles.split(",") if x.strip()]

    cols = build_candidate_features(df, allow_absolute_price=args.allow_absolute_price_features)
    print(f"Candidate features for rules: {len(cols)} (allow_absolute_price={args.allow_absolute_price_features})")

    thr = quantile_thresholds(df.iloc[:train_idx], cols, qs)

    # OHLC numpy
    high = df["high"].to_numpy(dtype=float, copy=False)
    low = df["low"].to_numpy(dtype=float, copy=False)
    close = df["close"].to_numpy(dtype=float, copy=False)

    print("\n===================================")
    print("EVENT CONFIG (multi-TP + trailing, pessimistic per-minute, one-trade-at-a-time in mining)")
    print(f"TPs: {tps}")
    print(f"TP weights: {tp_w.tolist()}")
    print(f"SL: {args.sl} | HOLD: {args.hold} | trail={args.trail} | trail_activate={args.trail_activate}")
    print(f"Entry slippage: {args.slippage_bps} bps")
    print(f"Train frac: {args.train_frac} | train rows: {train_idx} | test rows: {n - train_idx}")
    print(f"Rule conds: min={args.min_conds} max={args.max_conds} | min_test_hits={args.min_test_hits}")
    print("Optimize: TEST mean PnL (EV), tie-break SharpeProxy, hits, winrate")
    print("===================================\n")

    # 1) Simulate trade outcomes
    pnl, y, t_exit, tp_hits = simulate_multitp_trailing_pessimistic(
        high=high,
        low=low,
        close=close,
        tps=tps,
        tp_w=tp_w,
        sl=float(args.sl),
        hold=int(args.hold),
        slippage_bps=float(args.slippage_bps),
        trail=bool(args.trail),
        trail_activate=float(args.trail_activate),
    )

    # 2) Mine best rule
    summary, sig = mine_best_rule(
        df=df,
        cols=cols,
        thresholds=thr,
        train_idx=train_idx,
        y=y,
        pnl=pnl,
        t_exit=t_exit,
        tp_hits=tp_hits,
        min_conds=int(args.min_conds),
        max_conds=int(args.max_conds),
        min_test_hits=int(args.min_test_hits),
        wf_folds=int(args.wf_folds),
    )

    # Print result
    print("RESULT")
    print(f"Base winrate train: {summary['train_base_win']:.6f} | test: {summary['test_base_win']:.6f}")
    print(f"Base EV      train: {summary['train_base_ev']:.10g} | test: {summary['test_base_ev']:.10g}")
    print(f"Best rule: {summary['rule']}")
    print(
        f"TEST  EV(mean pnl): {summary['test_ev']:.10g} | Winrate: {summary['test_win']:.6f} | "
        f"SharpeProxy: {summary['test_sharpe_proxy']:.4f} | hits: {summary['test_hits']} | "
        f"median signals/day: {summary['test_median_signals_day']:.2f} | conds: {summary['conds']}"
    )
    print(
        f"TEST  median exit(min): {summary['test_median_minutes_to_exit']:.2f} | "
        f"median tp_hits: {summary['test_median_tp_hits']:.2f}"
    )
    print(
        f"WF mean_win: {summary['wf_mean_win']:.6f} | WF min_win: {summary['wf_min_win']:.6f} | "
        f"WF mean_EV: {summary['wf_mean_ev']:.10g} | WF hits: {summary['wf_total_hits']}"
    )

    # Save outputs
    out_summary = pd.DataFrame([summary])
    out_summary.to_csv(args.out_summary, index=False)

    if not sig.empty:
        out_sig = sig.reset_index().rename(columns={"index": "datetime"})
        out_sig["rule"] = summary["rule"]
        out_sig.to_csv(args.out_signals, index=False)
    else:
        pd.DataFrame(columns=["datetime", "close", "pnl", "outcome", "minutes_to_exit", "tp_hits", "rule"]).to_csv(
            args.out_signals, index=False
        )

    print("\nSaved:", args.out_summary)
    print("Saved:", args.out_signals)


if __name__ == "__main__":
    main()