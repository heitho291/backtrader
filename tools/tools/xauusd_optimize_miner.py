#!/usr/bin/env python3
"""Random-search optimizer for miner scripts.

Runs many miner configurations and ranks them by a chosen objective column from
miner summary CSV output (e.g. test_win, test_ev, wf_mean_win).
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import pandas as pd


def parse_int_range(v: str) -> tuple[int, int]:
    a, b = v.split(":", 1)
    return int(a), int(b)




def parse_float_range(v: str) -> tuple[float, float]:
    a, b = v.split(":", 1)
    return float(a), float(b)

def parse_str_choices(v: str) -> list[str]:
    return [x.strip() for x in v.split(";") if x.strip()]


def _validate_range(name: str, lo: float, hi: float) -> None:
    if lo > hi:
        raise ValueError(f"{name} range must satisfy min <= max")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Random optimize miner hyperparameters")
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--miner-script", type=Path, default=Path("tools/xauusd_miner_ohlc_first_hit_pessimistic.py"))
    p.add_argument("--runs", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=Path("miner_optimizer_results.csv"))
    p.add_argument("--checkpoint-every", type=int, default=5)
    p.add_argument("--max-attempts-per-run", type=int, default=5, help="Retry failed/empty runs up to N attempts per final run")
    p.add_argument("--keep-failed-runs", action="store_true", default=False, help="Keep failed attempts in output CSV")

    p.add_argument("--objective-column", type=str, default="test_score")
    p.add_argument(
        "--objective-agg",
        choices=["mean", "max", "min", "weighted_by_hits"],
        default="weighted_by_hits",
        help="How to aggregate objective over multiple summary rows (e.g. multiple TP rows)",
    )
    p.add_argument("--objective-hit-column", type=str, default="test_hits", help="Hit column used for weighted objective")
    p.add_argument("--min-objective-hits", type=int, default=0, help="Drop rows below this hit count before aggregation")

    p.add_argument("--train-frac", type=float, default=0.7)
    p.add_argument("--hold-range", type=parse_int_range, default=(60, 120))
    p.add_argument("--min-conds-range", type=parse_int_range, default=(2, 4))
    p.add_argument("--max-conds-range", type=parse_int_range, default=(4, 6))
    p.add_argument("--wf-folds-range", type=parse_int_range, default=(2, 6))
    p.add_argument("--min-test-hits-range", type=parse_int_range, default=(80, 250))
    p.add_argument("--max-features-range", type=parse_int_range, default=(120, 400),
                   help="Sample miner --max-features cap (smaller is faster; 0 allows all)")
    p.add_argument("--tail-rows-range", type=parse_int_range, default=(250000, 700000),
                   help="Sample miner --tail-rows cap to control memory/runtime")
    p.add_argument("--p-one-trade-at-a-time", type=float, default=1.0,
                   help="Probability to keep one-trade-at-a-time enabled")
    p.add_argument("--p-disable-same-reference-check", type=float, default=0.0,
                   help="Probability to disable miner support/resist same-reference dedup check")
    p.add_argument("--timeout-sec", type=int, default=1800,
                   help="Hard timeout per miner subprocess run in seconds (0=disabled)")
    p.add_argument("--capture-stdout", action="store_true", default=False,
                   help="Capture miner stdout (off by default to reduce memory usage)")
    p.add_argument("--sl-range", type=parse_float_range, default=(0.0010, 0.0035))
    p.add_argument("--slippage-bps-range", type=parse_float_range, default=(0.0, 4.0))
    p.add_argument("--spread-bps-range", type=parse_float_range, default=(0.0, 4.0))
    p.add_argument("--trail-activate-range", type=parse_float_range, default=(0.0008, 0.0020))
    p.add_argument("--trail-offset-range", type=parse_float_range, default=(0.0002, 0.0012))
    p.add_argument("--trail-factor-range", type=parse_float_range, default=(0.2, 0.8))
    p.add_argument("--p-trail", type=float, default=1.0, help="Probability to enable trailing per run")
    p.add_argument("--p-use-multi-tp", type=float, default=1.0, help="Probability to enable TP-based exits per run")
    p.add_argument("--trail-min-level-range", type=parse_float_range, default=(0.0, 0.0040))
    p.add_argument("--min-test-hits-reduce-step-range", type=parse_float_range, default=(0.05, 0.20))
    p.add_argument("--min-hits-return-override-range", type=parse_float_range, default=(2.0, 4.0))
    p.add_argument("--p-include-unrealized-at-test-end", type=float, default=1.0,
                   help="Probability to include unrealized pnl at test_end in miner scoring")
    p.add_argument("--objective-choice", type=str, choices=["test_ev", "test_win", "test_score"], default="test_score")
    p.add_argument("--two-starts-family-topn", type=int, default=3)
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
    p.add_argument("--score-k-low", type=float, default=4.0)
    p.add_argument("--score-k-high", type=float, default=1.2)
    p.add_argument(
        "--quantiles-choices",
        type=parse_str_choices,
        default=["0.05,0.10,0.90,0.95", "0.10,0.20,0.80,0.90", "0.05,0.15,0.85,0.95"],
        help="Semicolon-separated choices, each value is full --quantiles string",
    )
    p.add_argument(
        "--tps-choices",
        type=parse_str_choices,
        default=["0.0015,0.0025,0.0035,0.0045", "0.001,0.0015,0.002,0.0025,0.003", "0.002,0.003,0.004,0.005"],
        help="Semicolon-separated choices, each value is full --tps string",
    )
    p.add_argument("--tp-weight-total-pct-min", type=float, default=35.0, help="Minimum total TP close percent")
    p.add_argument("--tp-weight-total-pct-max", type=float, default=90.0, help="Maximum total TP close percent")
    p.add_argument("--p-allow-absolute-price-features", type=float, default=0.0)

    return p.parse_args()


def sample_cfg(rng: random.Random, args: argparse.Namespace) -> dict[str, object]:
    min_conds = rng.randint(*args.min_conds_range)
    max_conds = rng.randint(*args.max_conds_range)
    if max_conds < min_conds:
        max_conds = min_conds

    tps = rng.choice(args.tps_choices)
    n_tps = len([x for x in tps.split(",") if x.strip()])

    # Equal TP chunks with constrained total close percent:
    # tp_weight_each_pct = y and total = n_tps * y in [min,max]
    lo_total = float(args.tp_weight_total_pct_min)
    hi_total = float(args.tp_weight_total_pct_max)
    if lo_total <= 0 or hi_total <= 0 or hi_total < lo_total:
        raise ValueError("tp-weight-total-pct min/max must satisfy 0 < min <= max")

    lo_each = max(lo_total / n_tps, 0.01)
    hi_each = min(hi_total / n_tps, 100.0 / n_tps)
    if hi_each < lo_each:
        # fallback safe value: keep runner with 50% total close
        each_pct = min(max(50.0 / n_tps, 0.01), 100.0 / n_tps)
    else:
        each_pct = rng.uniform(lo_each, hi_each)

    each_frac = each_pct / 100.0
    tp_weights = ",".join([f"{each_frac:.6g}"] * n_tps)

    trail_activate = rng.uniform(*args.trail_activate_range)
    max_offset = min(args.trail_offset_range[1], trail_activate - 1e-12)
    if max_offset < args.trail_offset_range[0]:
        raise ValueError(
            "No valid trailing sample possible: trail-offset-range min must be < sampled trail_activate. "
            "Adjust --trail-activate-range/--trail-offset-range."
        )
    trail_offset = rng.uniform(args.trail_offset_range[0], max_offset)
    trail_activate = round(trail_activate, 6)
    trail_offset = round(trail_offset, 6)
    if trail_offset >= trail_activate:
        trail_offset = max(trail_activate - 1e-6, 1e-6)

    return {
        "hold": rng.randint(*args.hold_range),
        "min_conds": min_conds,
        "max_conds": max_conds,
        "wf_folds": rng.randint(*args.wf_folds_range),
        "min_test_hits": rng.randint(*args.min_test_hits_range),
        "max_features": rng.randint(*args.max_features_range),
        "tail_rows": rng.randint(*args.tail_rows_range),
        "sl": round(rng.uniform(*args.sl_range), 6),
        "slippage_bps": round(rng.uniform(*args.slippage_bps_range), 4),
        "spread_bps": round(rng.uniform(*args.spread_bps_range), 4),
        "trail_activate": trail_activate,
        "trail_offset": trail_offset,
        "trail_factor": round(rng.uniform(*args.trail_factor_range), 6),
        "trail_min_level": round(rng.uniform(*args.trail_min_level_range), 6),
        "min_test_hits_reduce_step": round(rng.uniform(*args.min_test_hits_reduce_step_range), 4),
        "min_hits_return_override": round(rng.uniform(*args.min_hits_return_override_range), 6),
        "include_unrealized_at_test_end": rng.random() < args.p_include_unrealized_at_test_end,
        "trail": rng.random() < args.p_trail,
        "use_multi_tp": rng.random() < args.p_use_multi_tp,
        "objective": args.objective_choice,
        "miner_seed": rng.randint(1, 10**9),
        "quantiles": rng.choice(args.quantiles_choices),
        "tps": tps,
        "tp_weights": tp_weights,
        "tp_weight_each_pct": round(each_pct, 4),
        "tp_weight_total_pct": round(each_pct * n_tps, 4),
        "allow_absolute_price_features": rng.random() < args.p_allow_absolute_price_features,
        "one_trade_at_a_time": rng.random() < args.p_one_trade_at_a_time,
        "disable_same_reference_check": rng.random() < args.p_disable_same_reference_check,
    }


def build_cmd(args: argparse.Namespace, cfg: dict[str, object], out_summary: Path, out_signals: Path) -> list[str]:
    raw_tps = str(cfg["tps"])
    tps_list = [x.strip() for x in raw_tps.split(",") if x.strip()]
    use_multi_tp = bool(cfg["use_multi_tp"])
    cmd_tps = raw_tps if use_multi_tp else (tps_list[0] if tps_list else "0.002")
    cmd_tp_weights = str(cfg["tp_weights"]) if use_multi_tp else "1.0"

    cmd = [
        sys.executable,
        str(args.miner_script),
        "--features",
        str(args.features),
        "--hold",
        str(cfg["hold"]),
        "--train-frac",
        str(args.train_frac),
        "--min-conds",
        str(cfg["min_conds"]),
        "--max-conds",
        str(cfg["max_conds"]),
        "--wf-folds",
        str(cfg["wf_folds"]),
        "--min-test-hits",
        str(cfg["min_test_hits"]),
        "--max-features",
        str(cfg["max_features"]),
        "--tail-rows",
        str(cfg["tail_rows"]),
        "--sl",
        str(cfg["sl"]),
        "--slippage-bps",
        str(cfg["slippage_bps"]),
        "--spread-bps",
        str(cfg["spread_bps"]),
        "--trail-activate",
        str(cfg["trail_activate"]),
        "--trail-offset",
        str(cfg["trail_offset"]),
        "--trail-factor",
        str(cfg["trail_factor"]),
        "--trail-min-level",
        str(cfg["trail_min_level"]),
        "--min-test-hits-reduce-step",
        str(cfg["min_test_hits_reduce_step"]),
        "--min-hits-return-override",
        str(cfg["min_hits_return_override"]),
        "--objective",
        str(cfg["objective"]),
        "--seed",
        str(cfg["miner_seed"]),
        "--quantiles",
        str(cfg["quantiles"]),
        "--two-starts-family-topn",
        str(args.two_starts_family_topn),
        "--score-return-bad-test",
        str(args.score_return_bad_test),
        "--score-return-mid-test",
        str(args.score_return_mid_test),
        "--score-return-good-test",
        str(args.score_return_good_test),
        "--score-return-bad-train",
        str(args.score_return_bad_train),
        "--score-dd-good",
        str(args.score_dd_good),
        "--score-dd-mid",
        str(args.score_dd_mid),
        "--score-dd-bad",
        str(args.score_dd_bad),
        "--score-sortino-bad",
        str(args.score_sortino_bad),
        "--score-sortino-mid",
        str(args.score_sortino_mid),
        "--score-sortino-good",
        str(args.score_sortino_good),
        "--score-trades-bad-test",
        str(args.score_trades_bad_test),
        "--score-trades-mid-test",
        str(args.score_trades_mid_test),
        "--score-trades-good-test",
        str(args.score_trades_good_test),
        "--score-trades-bad-train",
        str(args.score_trades_bad_train),
        "--score-trades-mid-train",
        str(args.score_trades_mid_train),
        "--score-trades-good-train",
        str(args.score_trades_good_train),
        "--score-k-low",
        str(args.score_k_low),
        "--score-k-high",
        str(args.score_k_high),
        "--tps",
        cmd_tps,
        "--tp-weights",
        cmd_tp_weights,
        "--out-summary",
        str(out_summary),
        "--out-signals",
        str(out_signals),
    ]
    if bool(cfg["allow_absolute_price_features"]):
        cmd.append("--allow-absolute-price-features")
    if bool(cfg["one_trade_at_a_time"]):
        cmd.append("--one-trade-at-a-time")
    else:
        cmd.append("--no-one-trade-at-a-time")
    if bool(cfg["disable_same_reference_check"]):
        cmd.append("--disable-same-reference-check")
    if bool(cfg["trail"]):
        cmd.append("--trail")
    else:
        cmd.append("--no-trail")
    if use_multi_tp:
        cmd.append("--use-multi-tp")
    else:
        cmd.append("--no-multi-tp")
    if bool(cfg["include_unrealized_at_test_end"]):
        cmd.append("--include-unrealized-at-test-end")
    else:
        cmd.append("--no-include-unrealized-at-test-end")
    return cmd


def aggregate_objective(df: pd.DataFrame, objective_col: str, mode: str, hit_col: str, min_hits: int) -> tuple[float, int]:
    if objective_col not in df.columns:
        return float("nan"), 0

    x = df.copy()
    x[objective_col] = pd.to_numeric(x[objective_col], errors="coerce")
    if hit_col in x.columns:
        x[hit_col] = pd.to_numeric(x[hit_col], errors="coerce")
        if min_hits > 0:
            x = x[x[hit_col] >= min_hits]

    x = x[x[objective_col].notna()]
    if x.empty:
        return float("nan"), 0

    if mode == "mean":
        return float(x[objective_col].mean()), int(len(x))
    if mode == "max":
        return float(x[objective_col].max()), int(len(x))
    if mode == "min":
        return float(x[objective_col].min()), int(len(x))

    if hit_col in x.columns:
        w = x[hit_col].to_numpy(dtype=float)
        v = x[objective_col].to_numpy(dtype=float)
        s = float(w.sum())
        if s > 0:
            return float((v * w).sum() / s), int(len(x))

    return float(x[objective_col].mean()), int(len(x))





def compute_loss_streak_risk_metrics(signals_df: pd.DataFrame) -> dict[str, object]:
    out: dict[str, object] = {}
    if signals_df.empty or "pnl" not in signals_df.columns:
        return out

    pnl = pd.to_numeric(signals_df["pnl"], errors="coerce").dropna().to_numpy(dtype=float)
    if pnl.size == 0:
        return out

    losses = pnl < 0
    loss_rate = float(losses.mean())
    out["loss_rate"] = loss_rate

    max_streak = 0
    cur = 0
    for is_loss in losses:
        if is_loss:
            cur += 1
            if cur > max_streak:
                max_streak = cur
        else:
            cur = 0
    out["max_consecutive_losses_observed"] = int(max_streak)

    if not losses.any():
        return out

    worst_loss_ret = float(max(-p for p in pnl if p < 0))
    out["worst_loss_return"] = worst_loss_ret

    def losses_needed_for_drawdown(L: int, dd: float) -> float:
        one_step = 1.0 - (L * worst_loss_ret)
        if one_step <= 0:
            return 1.0
        target = 1.0 - dd
        if target <= 0:
            return float("inf")
        return float(math.ceil(math.log(target) / math.log(one_step)))

    for L in (5, 10, 20, 30):
        k75 = losses_needed_for_drawdown(L, 0.75)
        k99 = losses_needed_for_drawdown(L, 0.99)
        out[f"L{L}_losses_to_m75"] = k75
        out[f"L{L}_losses_to_m99"] = k99
        out[f"L{L}_p_streak_m75"] = float(loss_rate ** k75) if math.isfinite(k75) else float("nan")
        out[f"L{L}_p_streak_m99"] = float(loss_rate ** k99) if math.isfinite(k99) else float("nan")

    return out


def extract_best_row_metrics(summary_df: pd.DataFrame, objective_col: str) -> dict[str, object]:
    if summary_df.empty:
        return {}

    # choose row by objective column if available, else first row
    if objective_col in summary_df.columns:
        c = pd.to_numeric(summary_df[objective_col], errors="coerce")
        if c.notna().any():
            idx = int(c.idxmax())
            row = summary_df.loc[idx]
        else:
            row = summary_df.iloc[0]
    else:
        row = summary_df.iloc[0]

    wanted = [
        "rule",
        "ev_train", "ev_test", "train_base_ev", "test_base_ev",
        "train_win", "test_win", "train_base_win", "test_base_win",
        "train_hits", "test_hits",
        "test_median_minutes_to_exit", "test_median_minutes_to_qualify", "test_median_signals_day", "test_median_tp_hits",
        "test_sharpe_proxy", "test_sortino_proxy", "test_return", "train_return", "test_max_drawdown_pct",
        "test_score", "score_test_component", "score_train_component", "score_penalty",
        "test_annualized_return", "train_annualized_return",
        "hodl_test_annualized_return", "hodl_train_annualized_return",
        "wf_hits", "wf_mean", "wf_mean_win", "wf_min_win",
        "test_start", "test_end",
        "conds", "min_test_hits_used", "min_train_hits_used", "min_hits_override_used",
        "test_cumulative_return", "train_cumulative_return", "include_unrealized_at_test_end",
    ]

    out: dict[str, object] = {}
    for k in wanted:
        if k in row.index:
            out[f"best_{k}"] = row[k]
    return out



def _tail_line(text: str) -> str:
    lines = text.strip().splitlines()
    return lines[-1] if lines else ""

def save_results(path: Path, rows: list[dict[str, object]]) -> None:
    target = path

    if not rows:
        headers = ["run", "attempt", "exit_code", "objective", "failed_reason"]
        try:
            with target.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
                w.writeheader()
        except PermissionError:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            alt = target.with_name(f"{target.stem}_{ts}{target.suffix}")
            with alt.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
                w.writeheader()
            print(f"[WARN] Could not write {target} (locked?). Wrote {alt} instead.")
        return

    sorted_rows = sorted(rows, key=lambda r: (float(r.get("objective", float("-inf"))) if r.get("exit_code") == 0 else float("-inf")), reverse=True)

    headers: list[str] = []
    seen: set[str] = set()
    for r in sorted_rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                headers.append(k)

    try:
        with target.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
            w.writeheader()
            w.writerows(sorted_rows)
    except PermissionError:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        alt = target.with_name(f"{target.stem}_{ts}{target.suffix}")
        with alt.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
            w.writeheader()
            w.writerows(sorted_rows)
        print(f"[WARN] Could not write {target} (locked?). Wrote {alt} instead.")


def main() -> None:
    args = parse_args()

    if not (0.0 <= args.p_allow_absolute_price_features <= 1.0):
        raise ValueError("p-allow-absolute-price-features must be in [0,1]")
    if not (0.0 <= args.p_trail <= 1.0):
        raise ValueError("p-trail must be in [0,1]")
    if not (0.0 <= args.p_use_multi_tp <= 1.0):
        raise ValueError("p-use-multi-tp must be in [0,1]")
    if args.max_attempts_per_run < 1:
        raise ValueError("max-attempts-per-run must be >= 1")
    if not (0.0 <= args.p_include_unrealized_at_test_end <= 1.0):
        raise ValueError("p-include-unrealized-at-test-end must be in [0,1]")
    if not (0.0 <= args.p_one_trade_at_a_time <= 1.0):
        raise ValueError("p-one-trade-at-a-time must be in [0,1]")
    if not (0.0 <= args.p_disable_same_reference_check <= 1.0):
        raise ValueError("p-disable-same-reference-check must be in [0,1]")
    if args.timeout_sec < 0:
        raise ValueError("timeout-sec must be >= 0")
    if args.two_starts_family_topn < 1:
        raise ValueError("two-starts-family-topn must be >= 1")

    _validate_range("trail-activate", *args.trail_activate_range)
    _validate_range("trail-offset", *args.trail_offset_range)
    _validate_range("trail-factor", *args.trail_factor_range)
    _validate_range("trail-min-level", *args.trail_min_level_range)
    _validate_range("min-test-hits-reduce-step", *args.min_test_hits_reduce_step_range)
    _validate_range("min-hits-return-override", *args.min_hits_return_override_range)
    _validate_range("max-features", *args.max_features_range)
    _validate_range("tail-rows", *args.tail_rows_range)

    if args.trail_activate_range[1] <= 0:
        raise ValueError("trail-activate-range max must be > 0")
    if args.trail_offset_range[1] <= 0:
        raise ValueError("trail-offset-range max must be > 0")
    if args.trail_factor_range[1] <= 0:
        raise ValueError("trail-factor-range max must be > 0")
    if args.trail_min_level_range[0] < 0:
        raise ValueError("trail-min-level-range min must be >= 0")
    if args.min_test_hits_reduce_step_range[0] < 0 or args.min_test_hits_reduce_step_range[1] >= 1:
        raise ValueError("min-test-hits-reduce-step-range must be within [0,1)")
    if args.min_hits_return_override_range[0] < 0:
        raise ValueError("min-hits-return-override-range min must be >= 0")
    if args.max_features_range[0] < 0:
        raise ValueError("max-features-range min must be >= 0")
    if args.tail_rows_range[0] < 1:
        raise ValueError("tail-rows-range min must be >= 1")

    if args.p_trail > 0:
        if args.trail_activate_range[0] <= 0:
            raise ValueError("trail-activate-range min must be > 0 when trailing can be enabled")
        if args.trail_offset_range[0] <= 0:
            raise ValueError("trail-offset-range min must be > 0 when trailing can be enabled")
        if args.trail_factor_range[0] <= 0:
            raise ValueError("trail-factor-range min must be > 0 when trailing can be enabled")
        if args.trail_offset_range[0] >= args.trail_activate_range[1]:
            raise ValueError("trail-offset-range min must be < trail-activate-range max when trailing can be enabled")

    rng = random.Random(args.seed)
    rows: list[dict[str, object]] = []

    try:
        for run in range(1, args.runs + 1):
            success = False
            attempt = 0
            last_row: dict[str, object] | None = None
            while attempt < args.max_attempts_per_run and not success:
                attempt += 1
                cfg = sample_cfg(rng, args)

                with tempfile.TemporaryDirectory(prefix="miner_opt_") as td:
                    out_summary = Path(td) / "summary.csv"
                    out_signals = Path(td) / "signals.csv"
                    cmd = build_cmd(args, cfg, out_summary, out_signals)
                    t0 = time.time()
                    timed_out = False
                    try:
                        proc = subprocess.run(
                            cmd,
                            stdout=(subprocess.PIPE if args.capture_stdout else subprocess.DEVNULL),
                            stderr=subprocess.PIPE,
                            text=True,
                            timeout=(args.timeout_sec if args.timeout_sec > 0 else None),
                        )
                    except subprocess.TimeoutExpired as e:
                        timed_out = True
                        proc = subprocess.CompletedProcess(
                            cmd,
                            returncode=124,
                            stdout=(e.stdout or "") if args.capture_stdout else "",
                            stderr=(e.stderr or "") + "\n[timeout] miner subprocess exceeded timeout",
                        )
                    sec = time.time() - t0

                    objective = float("nan")
                    rows_used = 0
                    best_rule = ""
                    best_metrics: dict[str, object] = {}
                    risk_metrics: dict[str, object] = {}
                    if proc.returncode == 0 and out_summary.exists():
                        try:
                            s = pd.read_csv(out_summary)
                            objective, rows_used = aggregate_objective(
                                s,
                                objective_col=args.objective_column,
                                mode=args.objective_agg,
                                hit_col=args.objective_hit_column,
                                min_hits=args.min_objective_hits,
                            )
                            best_metrics = extract_best_row_metrics(s, args.objective_column)
                            if "best_rule" in best_metrics:
                                best_rule = str(best_metrics["best_rule"])
                            if out_signals.exists():
                                sig = pd.read_csv(out_signals)
                                risk_metrics = compute_loss_streak_risk_metrics(sig)
                        except Exception:
                            pass

                    valid_result = proc.returncode == 0 and math.isfinite(objective) and rows_used > 0 and bool(best_rule)
                    err_tail = _tail_line(proc.stderr or "") if proc.returncode != 0 else ""
                    out_tail = _tail_line(proc.stdout or "") if proc.returncode != 0 else ""

                    row = {
                        "run": run,
                        "attempt": attempt,
                        **cfg,
                        "exit_code": proc.returncode,
                        "runtime_sec": round(sec, 3),
                        "objective": objective,
                        "objective_rows_used": rows_used,
                        "best_rule": best_rule,
                        **best_metrics,
                        **risk_metrics,
                        "error_tail": err_tail,
                        "stdout_tail": out_tail,
                    }

                    last_row = row

                    if valid_result:
                        rows.append(row)
                        success = True
                    elif args.keep_failed_runs:
                        if timed_out:
                            row["failed_reason"] = "timeout"
                        elif proc.returncode != 0:
                            row["failed_reason"] = f"miner_exit_{proc.returncode}"
                        elif not math.isfinite(objective):
                            row["failed_reason"] = "non_finite_objective"
                        elif rows_used <= 0:
                            row["failed_reason"] = "no_objective_rows"
                        elif not best_rule:
                            row["failed_reason"] = "empty_best_rule"
                        else:
                            row["failed_reason"] = "non_finite_objective_or_no_rule"
                        rows.append(row)

                    print(
                        f"[{run}/{args.runs} attempt {attempt}/{args.max_attempts_per_run}] "
                        f"code={proc.returncode} objective={objective:.8g} rows={rows_used} "
                        f"valid={int(valid_result)} sec={sec:.2f} tail_rows={cfg.get('tail_rows')} max_feat={cfg.get('max_features')}"
                        + (f" err={err_tail[:160]}" if err_tail else "")
                    )

            if not success:
                print(f"[WARN] run {run}: no valid result after {args.max_attempts_per_run} attempts")
                if (not args.keep_failed_runs) and (last_row is not None):
                    fail_row = dict(last_row)
                    fail_row["failed_reason"] = "no_valid_result_after_attempts"
                    rows.append(fail_row)

            if args.checkpoint_every > 0 and run % args.checkpoint_every == 0:
                save_results(args.out, rows)
                print(f"checkpoint saved: {args.out} ({run} runs)")

    except KeyboardInterrupt:
        print("\n[WARN] Interrupted by user, saving checkpoint before exit...")
        save_results(args.out, rows)
        print(f"Saved (partial): {args.out}")
        return

    save_results(args.out, rows)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
