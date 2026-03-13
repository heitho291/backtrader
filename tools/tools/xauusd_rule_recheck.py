#!/usr/bin/env python3
"""Independent recheck for miner-style indicator rules with percent TP/SL first-hit labels.

Use this to validate mined rules (or custom rules) outside the miner search loop.
Supports:
- single rule via --rule (+ --tp/--sl/--hold)
- batch rules via --rules-csv (expects at least rule,tp,sl,hold by default)
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

RULE_PART_RE = re.compile(r"^\s*(?P<col>[A-Za-z0-9_]+)\s*(?P<op><=|>=)\s*(?P<thr>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Independent recheck of miner rules with percent TP/SL first-hit labels")
    p.add_argument("--features", type=Path, required=True, help="Path to feature_scan_features.csv or .csv.gz")

    p.add_argument("--rule", type=str, default="", help="Single rule string, e.g. 'adx14_tf15 <= 20 & dist_ema8_tf1 >= -0.0003'")
    p.add_argument("--tp", type=float, default=None, help="TP in fraction (e.g. 0.0015) for --rule mode")
    p.add_argument("--sl", type=float, default=None, help="SL in fraction (e.g. 0.0015) for --rule mode")
    p.add_argument("--hold", type=int, default=None, help="Hold in bars (minutes on tf1) for --rule mode")

    p.add_argument("--rules-csv", type=Path, default=None, help="CSV with rules to recheck (default key cols: rule,tp,sl,hold)")
    p.add_argument("--rule-column", type=str, default="rule")
    p.add_argument("--tp-column", type=str, default="tp")
    p.add_argument("--sl-column", type=str, default="sl")
    p.add_argument("--hold-column", type=str, default="hold")

    p.add_argument("--train-frac", type=float, default=0.7)
    p.add_argument("--signal-cooldown-minutes", type=int, default=10)
    p.add_argument("--wf-folds", type=int, default=4)

    p.add_argument("--expected-test-win-column", type=str, default="test_win", help="Optional column in --rules-csv to diff against recheck")
    p.add_argument("--expected-test-hits-column", type=str, default="test_hits", help="Optional column in --rules-csv to diff against recheck")

    p.add_argument("--out", type=Path, default=Path("rule_recheck_results.csv"))
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
        raise ValueError("CSV must contain 'datetime' column or an unnamed datetime index column")

    df = df.sort_index()
    needed = {"open", "high", "low", "close"}
    miss = needed - set(df.columns)
    if miss:
        raise ValueError(f"Missing OHLC columns in features file: {sorted(miss)}")
    return df


def parse_rule(rule_str: str) -> list[tuple[str, str, float]]:
    parts = [p.strip() for p in rule_str.split("&") if p.strip()]
    if not parts:
        raise ValueError("Rule string is empty")

    out: list[tuple[str, str, float]] = []
    for part in parts:
        m = RULE_PART_RE.match(part)
        if not m:
            raise ValueError(f"Could not parse rule condition: {part!r}")
        out.append((m.group("col"), m.group("op"), float(m.group("thr"))))
    return out


def first_hit_labels_pessimistic(high: np.ndarray, low: np.ndarray, close: np.ndarray, tp: float, sl: float, hold: int) -> tuple[np.ndarray, np.ndarray]:
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
                hit = 0
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


def eval_rule_mask(df: pd.DataFrame, rule: Iterable[tuple[str, str, float]]) -> np.ndarray:
    mask = np.ones(len(df), dtype=bool)
    for col, op, thr in rule:
        if col not in df.columns:
            raise ValueError(f"Rule references missing feature column: {col}")
        x = df[col].to_numpy()
        if op == "<=":
            mask &= x <= thr
        else:
            mask &= x >= thr
    return mask


def apply_cooldown(index: pd.Index, mask: np.ndarray, cooldown_minutes: int) -> np.ndarray:
    if cooldown_minutes <= 0:
        return mask
    out = np.zeros_like(mask, dtype=bool)
    ts = pd.to_datetime(index)
    next_allowed = None
    for pos in np.flatnonzero(mask):
        t = ts[pos]
        if next_allowed is None or t >= next_allowed:
            out[pos] = True
            next_allowed = t + pd.Timedelta(minutes=cooldown_minutes)
    return out


def walk_forward_coverage(index: pd.Index, y_test: np.ndarray, selected_test_mask: np.ndarray, cooldown_minutes: int, wf_folds: int) -> tuple[float, float, int]:
    if wf_folds <= 1:
        wf_folds = 1
    n = len(y_test)
    if n == 0:
        return float("nan"), float("nan"), 0

    fold_edges = np.linspace(0, n, wf_folds + 1, dtype=int)
    wins: list[float] = []
    total_hits = 0
    for i in range(wf_folds):
        a, b = int(fold_edges[i]), int(fold_edges[i + 1])
        if b <= a:
            continue
        m = selected_test_mask[a:b].copy()
        if m.sum() == 0:
            continue
        m = apply_cooldown(index[a:b], m, cooldown_minutes)
        sel = m & ((y_test[a:b] == 1) | (y_test[a:b] == 0))
        hits = int(sel.sum())
        if hits == 0:
            continue
        total_hits += hits
        wins.append(float((y_test[a:b][sel] == 1).mean()))

    if not wins:
        return float("nan"), float("nan"), 0
    return float(np.mean(wins)), float(np.min(wins)), int(total_hits)


def evaluate_one(df: pd.DataFrame, rule_str: str, tp: float, sl: float, hold: int, train_frac: float, cooldown_minutes: int, wf_folds: int) -> dict[str, object]:
    if hold <= 0:
        raise ValueError("hold must be > 0")
    if tp <= 0 or sl <= 0:
        raise ValueError("tp and sl must be > 0")

    rule = parse_rule(rule_str)

    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    y, _ = first_hit_labels_pessimistic(high, low, close, tp=tp, sl=sl, hold=hold)

    train_idx = int(len(df) * train_frac)
    if train_idx <= 0 or train_idx >= len(df):
        raise ValueError("train-frac must split inside data range")

    y_train = y[:train_idx]
    y_test = y[train_idx:]

    def base_winrate(yy: np.ndarray) -> float:
        hits = (yy == 1) | (yy == 0)
        if hits.sum() == 0:
            return float("nan")
        return float((yy[hits] == 1).mean())

    m_all = eval_rule_mask(df, rule)
    m_test = m_all[train_idx:]
    hit_mask_test = (y_test == 1) | (y_test == 0)
    sel_test = apply_cooldown(df.index[train_idx:], m_test & hit_mask_test, cooldown_minutes)

    test_hits = int(sel_test.sum())
    test_win = float((y_test[sel_test] == 1).mean()) if test_hits else float("nan")

    full_hit_mask = (y == 1) | (y == 0)
    sel_full = apply_cooldown(df.index, m_all & full_hit_mask, cooldown_minutes)
    full_hits = int(sel_full.sum())
    full_win = float((y[sel_full] == 1).mean()) if full_hits else float("nan")

    wf_mean, wf_min, wf_hits = walk_forward_coverage(
        df.index[train_idx:],
        y_test,
        m_test & hit_mask_test,
        cooldown_minutes=cooldown_minutes,
        wf_folds=wf_folds,
    )

    return {
        "rule": rule_str,
        "tp": tp,
        "sl": sl,
        "hold": hold,
        "train_base_win": base_winrate(y_train),
        "test_base_win": base_winrate(y_test),
        "test_win": test_win,
        "test_hits": test_hits,
        "coverage_win_full": full_win,
        "coverage_hits_full": full_hits,
        "wf_mean_win": wf_mean,
        "wf_min_win": wf_min,
        "wf_total_hits": wf_hits,
    }



def main() -> None:
    args = parse_args()
    df = load_features(args.features)

    # safer row loading for optional columns + bug-free expected value handling
    records: list[dict[str, object]] = []
    if args.rules_csv is None:
        if not args.rule:
            raise ValueError("Provide either --rule or --rules-csv")
        if args.tp is None or args.sl is None or args.hold is None:
            raise ValueError("--tp/--sl/--hold are required for single --rule mode")
        records = [{"rule": args.rule, "tp": float(args.tp), "sl": float(args.sl), "hold": int(args.hold)}]
    else:
        table = pd.read_csv(args.rules_csv, compression="infer")
        needed = {args.rule_column, args.tp_column, args.sl_column, args.hold_column}
        miss = needed - set(table.columns)
        if miss:
            raise ValueError(f"rules-csv missing columns: {sorted(miss)}")
        has_exp_win = args.expected_test_win_column in table.columns
        has_exp_hits = args.expected_test_hits_column in table.columns
        for _, r in table.iterrows():
            rec = {
                "rule": str(r[args.rule_column]),
                "tp": float(r[args.tp_column]),
                "sl": float(r[args.sl_column]),
                "hold": int(r[args.hold_column]),
            }
            if has_exp_win:
                rec["expected_test_win"] = float(r[args.expected_test_win_column])
            if has_exp_hits:
                rec["expected_test_hits"] = int(r[args.expected_test_hits_column])
            records.append(rec)

    rows: list[dict[str, object]] = []
    for i, rec in enumerate(records, start=1):
        out = evaluate_one(
            df=df,
            rule_str=str(rec["rule"]),
            tp=float(rec["tp"]),
            sl=float(rec["sl"]),
            hold=int(rec["hold"]),
            train_frac=args.train_frac,
            cooldown_minutes=args.signal_cooldown_minutes,
            wf_folds=args.wf_folds,
        )

        if "expected_test_win" in rec:
            out["expected_test_win"] = rec["expected_test_win"]
            out["delta_test_win"] = out["test_win"] - rec["expected_test_win"] if pd.notna(out["test_win"]) else float("nan")
        if "expected_test_hits" in rec:
            out["expected_test_hits"] = rec["expected_test_hits"]
            out["delta_test_hits"] = int(out["test_hits"]) - int(rec["expected_test_hits"])

        rows.append(out)
        print(f"[{i}/{len(records)}] done: tp={out['tp']:.6g} sl={out['sl']:.6g} hold={out['hold']} test_win={out['test_win']:.6f} hits={out['test_hits']}")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out, index=False)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
