#!/usr/bin/env python3
"""
Rule mining using OHLC-based first-hit events with pessimistic tie-break:
If in the same minute both TP and SL could be hit, SL wins (loss).

This v2 EXCLUDES absolute price-level columns from rule search by default,
to reduce regime/price-level overfitting (e.g., open_tf1 >= 2742).

Outputs:
- best_rules_summary.csv
- signals_best_rules.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--features", type=Path, required=True, help="Path to feature_scan_features.csv or .csv.gz")
    p.add_argument("--hold", type=int, default=30, help="Max minutes to look ahead")
    p.add_argument("--tps", type=str, default="0.0015,0.0025,0.0035,0.0045", help="TP levels")
    p.add_argument("--sl-equals-tp", action="store_true", default=True)
    p.add_argument("--train-frac", type=float, default=0.7, help="Time split for train/test")
    p.add_argument("--min-test-hits-015", type=int, default=150)
    p.add_argument("--min-test-hits-025plus", type=int, default=50)
    p.add_argument("--max-conds", type=int, default=5, help="Max conditions in rule (3-5 recommended)")
    p.add_argument("--min-conds", type=int, default=3, help="Min conditions in rule")
    p.add_argument("--out-summary", type=Path, default=Path("best_rules_summary.csv"))
    p.add_argument("--out-signals", type=Path, default=Path("signals_best_rules.csv"))
    p.add_argument("--quantiles", type=str, default="0.05,0.10,0.90,0.95", help="Quantiles used for thresholds")
    p.add_argument("--signal-cooldown-minutes", type=int, default=10, help="Cooldown for de-duplicating clustered minute-signals")
    p.add_argument("--wf-folds", type=int, default=4, help="Walk-forward coverage folds on test range")

    # feature selection controls
    p.add_argument(
        "--allow-absolute-price-features",
        action="store_true",
        default=False,
        help="If set, allow open_tf*/close_tf*/ema*/vwap* as rule conditions (NOT recommended).",
    )
    return p.parse_args()


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


def first_hit_labels_pessimistic(high: np.ndarray, low: np.ndarray, close: np.ndarray, tp: float, sl: float, hold: int):
    """
    outcome: 1=TP win, 0=SL loss, -1=none within window
    thit: minutes until hit, -1 if none
    """
    n = len(close)
    out = np.full(n, -1, dtype=np.int8)
    thit = np.full(n, -1, dtype=np.int16)

    for i in range(n - hold - 1):
        entry = close[i]
        tp_level = entry * (1.0 + tp)
        sl_level = entry * (1.0 - sl)

        hit = -1
        hit_t = -1

        for k in range(1, hold + 1):
            j = i + k
            h = high[j]
            l = low[j]

            tp_hit = h >= tp_level
            sl_hit = l <= sl_level

            if tp_hit and sl_hit:
                hit = 0  # pessimistic tie-break
                hit_t = k
                break
            if sl_hit:
                hit = 0
                hit_t = k
                break
            if tp_hit:
                hit = 1
                hit_t = k
                break

        out[i] = hit
        thit[i] = hit_t

    return out, thit


def build_candidate_features(df: pd.DataFrame, allow_absolute_price: bool) -> List[str]:
    """
    Exclude absolute price-level and OHLCV-like columns by default to avoid regime overfit.
    Keep:
      - dist_* (normalized)
      - rsi*, adx*, plus_di*, minus_di*, dx*
      - macd* (indicator values)
      - vol_z* (normalized volume)
      - vwap-derived distances (dist_vwap_*) but not raw vwap_tf*
    """
    exclude_exact = {"open", "high", "low", "close", "volume"}
    cand = []

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
        if c.startswith("ema"):  # ema{len}_tf{tf} is price-level
            continue
        if c.startswith("vwap_tf"):  # raw vwap is price-level
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


def eval_rule_mask(df: pd.DataFrame, rule: List[Tuple[str, str, float]]) -> np.ndarray:
    mask = np.ones(len(df), dtype=bool)
    for col, op, thr in rule:
        x = df[col].to_numpy()
        if op == "<=":
            mask &= x <= thr
        else:
            mask &= x >= thr
    return mask


def apply_cooldown(index: pd.Index, mask: np.ndarray, cooldown_minutes: int) -> np.ndarray:
    """Keep first hit, then suppress next hits for cooldown window."""
    if cooldown_minutes <= 0:
        return mask
    out = np.zeros_like(mask, dtype=bool)
    ts = pd.to_datetime(index)
    next_allowed = None
    hit_positions = np.flatnonzero(mask)
    for pos in hit_positions:
        t = ts[pos]
        if next_allowed is None or t >= next_allowed:
            out[pos] = True
            next_allowed = t + pd.Timedelta(minutes=cooldown_minutes)
    return out


def simplify_rule(rule: List[Tuple[str, str, float]]) -> List[Tuple[str, str, float]]:
    """Merge redundant thresholds per feature.

    - <= keeps strictest (smallest) threshold
    - >= keeps strictest (largest) threshold
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

    out: List[Tuple[str, str, float]] = []
    for col in sorted(merged.keys()):
        for op in ("<=", ">="):
            if op in merged[col]:
                out.append((col, op, merged[col][op]))
    return out


def walk_forward_coverage(
    index: pd.Index,
    y_test: np.ndarray,
    selected_test_mask: np.ndarray,
    cooldown_minutes: int,
    wf_folds: int,
) -> Tuple[float, float, int]:
    """Evaluate final fixed rule across sequential time chunks in test range."""
    if wf_folds <= 1:
        wf_folds = 1

    n = len(y_test)
    if n == 0:
        return float("nan"), float("nan"), 0

    fold_edges = np.linspace(0, n, wf_folds + 1, dtype=int)
    wins: List[float] = []
    total_hits = 0

    for i in range(wf_folds):
        a, b = int(fold_edges[i]), int(fold_edges[i + 1])
        if b <= a:
            continue
        mask_fold = selected_test_mask[a:b].copy()
        if mask_fold.sum() == 0:
            continue
        mask_fold = apply_cooldown(index[a:b], mask_fold, cooldown_minutes)
        sel = mask_fold & ((y_test[a:b] == 1) | (y_test[a:b] == 0))
        hits = int(sel.sum())
        if hits == 0:
            continue
        total_hits += hits
        wins.append(float((y_test[a:b][sel] == 1).mean()))

    if not wins:
        return float("nan"), float("nan"), 0
    return float(np.mean(wins)), float(np.min(wins)), int(total_hits)


def best_rules_for_tp(
    df: pd.DataFrame,
    y: np.ndarray,
    thit: np.ndarray,
    train_idx: int,
    cols: List[str],
    thresholds: Dict[str, Dict[float, float]],
    min_conds: int,
    max_conds: int,
    min_test_hits: int,
    signal_cooldown_minutes: int,
    wf_folds: int,
):
    test = df.iloc[train_idx:]
    y_train = y[:train_idx]
    y_test = y[train_idx:]
    th_test = thit[train_idx:]

    def base_winrate(yy: np.ndarray) -> float:
        hits = (yy == 1) | (yy == 0)
        if hits.sum() == 0:
            return 0.0
        return float((yy[hits] == 1).mean())

    base_train = base_winrate(y_train)
    base_test = base_winrate(y_test)

    # Build condition pool from quantiles
    conds: List[Tuple[str, str, float]] = []
    for c in cols:
        if c not in thresholds:
            continue
        for q, thr in thresholds[c].items():
            if q <= 0.5:
                conds.append((c, "<=", thr))
            else:
                conds.append((c, ">=", thr))

    def score_rule(rule: List[Tuple[str, str, float]]):
        m = eval_rule_mask(df, rule)
        m_test = m[train_idx:]
        hit_mask = (y_test == 1) | (y_test == 0)
        sel_pre = m_test & hit_mask
        sel = apply_cooldown(test.index, sel_pre, signal_cooldown_minutes)
        hits = int(sel.sum())
        if hits < min_test_hits:
            return None
        win = float((y_test[sel] == 1).mean())
        dt = test.index
        days = pd.Series(sel, index=dt).groupby(dt.date).sum()
        median_per_day = float(days.median()) if len(days) else 0.0
        return win, hits, median_per_day, m

    best = None

    # Start with best single condition
    for cond in conds:
        sc = score_rule([cond])
        if sc is None:
            continue
        win, hits, med, m = sc
        if best is None or win > best[0]:
            best = (win, hits, med, m, [cond])

    if best is None:
        return {
            "rule": "",
            "train_base_win": base_train,
            "test_base_win": base_test,
            "test_win": 0.0,
            "test_hits": 0,
            "test_median_signals_day": 0.0,
            "conds": 0,
        }, pd.DataFrame()

    current_rule = best[4]
    current_win = best[0]

    # Greedy add conditions if it improves winrate
    while len(current_rule) < max_conds:
        best_candidate = None
        for cond in conds:
            if cond in current_rule:
                continue
            if cond[0] in {c for c, _, _ in current_rule}:
                continue
            sc = score_rule(current_rule + [cond])
            if sc is None:
                continue
            win, hits, med, m = sc
            if best_candidate is None or win > best_candidate[0]:
                best_candidate = (win, hits, med, m, current_rule + [cond])
        if best_candidate and best_candidate[0] > current_win:
            best = best_candidate
            current_rule = best_candidate[4]
            current_win = best_candidate[0]
        else:
            break

    # Ensure min_conds by adding best possible even if not improving
    while len(current_rule) < min_conds:
        best_candidate = None
        for cond in conds:
            if cond in current_rule:
                continue
            if cond[0] in {c for c, _, _ in current_rule}:
                continue
            sc = score_rule(current_rule + [cond])
            if sc is None:
                continue
            win, hits, med, m = sc
            if best_candidate is None or win > best_candidate[0]:
                best_candidate = (win, hits, med, m, current_rule + [cond])
        if best_candidate is None:
            break
        best = best_candidate
        current_rule = best_candidate[4]

    win, hits, med, m_all, rule = best
    rule = simplify_rule(rule)
    m_all = eval_rule_mask(df, rule)
    m_test = m_all[train_idx:]
    hit_mask = (y_test == 1) | (y_test == 0)
    sel = apply_cooldown(test.index, m_test & hit_mask, signal_cooldown_minutes)

    sig = pd.DataFrame(index=test.index[sel])
    sig["close"] = test.loc[sig.index, "close"].astype(float)
    sig["outcome"] = np.where(y_test[sel] == 1, "TP", "SL")
    sig["minutes_to_hit"] = th_test[sel].astype(int)

    rule_str = " & ".join([f"{c} {op} {thr:.6g}" for c, op, thr in rule])

    full_hit_mask = (y == 1) | (y == 0)
    full_sel = apply_cooldown(df.index, (m_all & full_hit_mask), signal_cooldown_minutes)
    coverage_hits_full = int(full_sel.sum())
    coverage_win_full = float((y[full_sel] == 1).mean()) if coverage_hits_full else float("nan")

    wf_mean_win, wf_min_win, wf_total_hits = walk_forward_coverage(
        test.index,
        y_test,
        (m_test & hit_mask),
        cooldown_minutes=signal_cooldown_minutes,
        wf_folds=wf_folds,
    )

    summary = {
        "rule": rule_str,
        "train_base_win": base_train,
        "test_base_win": base_test,
        "test_win": win,
        "test_hits": hits,
        "test_median_signals_day": med,
        "conds": len(rule),
        "coverage_win_full": coverage_win_full,
        "coverage_hits_full": coverage_hits_full,
        "wf_mean_win": wf_mean_win,
        "wf_min_win": wf_min_win,
        "wf_total_hits": wf_total_hits,
    }
    return summary, sig


def main() -> None:
    args = parse_args()

    df = load_features(args.features)
    n = len(df)
    train_idx = int(n * args.train_frac)

    tps = [float(x) for x in args.tps.split(",") if x.strip()]
    qs = [float(x) for x in args.quantiles.split(",") if x.strip()]

    cols = build_candidate_features(df, allow_absolute_price=args.allow_absolute_price_features)
    print(f"Candidate features for rules: {len(cols)} (allow_absolute_price={args.allow_absolute_price_features})")

    thr = quantile_thresholds(df.iloc[:train_idx], cols, qs)

    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)

    summaries = []
    all_signals = []

    for tp in tps:
        sl = tp if args.sl_equals_tp else tp
        min_hits = args.min_test_hits_015 if tp <= 0.0015 else args.min_test_hits_025plus

        print("\n===================================")
        print(f"TP={tp:.4f} SL={sl:.4f} HOLD={args.hold} (pessimistic HL) | min_test_hits={min_hits}")

        y, thit = first_hit_labels_pessimistic(high, low, close, tp=tp, sl=sl, hold=args.hold)

        summary, sig = best_rules_for_tp(
            df=df,
            y=y,
            thit=thit,
            train_idx=train_idx,
            cols=cols,
            thresholds=thr,
            min_conds=args.min_conds,
            max_conds=args.max_conds,
            min_test_hits=min_hits,
            signal_cooldown_minutes=args.signal_cooldown_minutes,
            wf_folds=args.wf_folds,
        )

        summary["tp"] = tp
        summary["sl"] = sl
        summary["hold"] = args.hold

        print(f"Base winrate train: {summary['train_base_win']:.6f} | test: {summary['test_base_win']:.6f}")
        print(f"Best rule: {summary['rule']}")
        print(
            f"Test winrate: {summary['test_win']:.6f} | test_hits: {summary['test_hits']} | "
            f"median signals/day: {summary['test_median_signals_day']} | conds: {summary['conds']}"
        )

        summaries.append(summary)

        if not sig.empty:
            sig = sig.copy()
            sig["tp"] = tp
            sig["sl"] = sl
            sig["hold"] = args.hold
            sig["rule"] = summary["rule"]
            all_signals.append(sig.reset_index().rename(columns={"index": "datetime"}))

    out_summary = pd.DataFrame(summaries)
    out_summary.to_csv(args.out_summary, index=False)

    if all_signals:
        out_sig = pd.concat(all_signals, axis=0, ignore_index=True)
        out_sig.to_csv(args.out_signals, index=False)
    else:
        pd.DataFrame(
            columns=["datetime", "close", "outcome", "minutes_to_hit", "tp", "sl", "hold", "rule"]
        ).to_csv(args.out_signals, index=False)

    print("\nSaved:", args.out_summary)
    print("Saved:", args.out_signals)


if __name__ == "__main__":
    main()
