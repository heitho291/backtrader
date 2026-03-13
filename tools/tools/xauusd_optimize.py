#!/usr/bin/env python3
"""Random-search optimizer for tools/xauusd_step1.py.

Runs multiple backtests with sampled EMA/RSI/Volume filter settings and records
ROI metrics into CSV. Intended for local execution with large datasets.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path


ANNUALIZED_RE = re.compile(r"Annualized ROI:\s*([-+]?\d+(?:\.\d+)?)%")
ROI_PERIOD_RE = re.compile(r"ROI period:\s*([-+]?\d+(?:\.\d+)?)%")
WINRATE_RE = re.compile(r"Win rate:\s*([-+]?\d+(?:\.\d+)?)%")
END_VALUE_RE = re.compile(r"End portfolio value:\s*([-+]?\d+(?:\.\d+)?)")
CLOSED_TRADES_RE = re.compile(r"Closed trades:\s*(\d+)")


def parse_metric(pattern: re.Pattern[str], output: str, default: float = float("nan")) -> float:
    m = pattern.search(output)
    return float(m.group(1)) if m else default


def build_command(args: argparse.Namespace, cfg: dict[str, object]) -> list[str]:
    entry_cash = round(args.cash / cfg["max_entries"], 8) if args.auto_entry_cash_from_max_entries else cfg["entry_cash"]
    cfg["effective_entry_cash"] = entry_cash
    cmd = [
        sys.executable,
        str(args.step1_script),
        "--data",
        str(args.data),
        "--tail",
        str(args.tail),
        "--cash",
        str(args.cash),
        "--entry-cash",
        str(entry_cash),
        "--max-entries",
        str(cfg["max_entries"]),
        "--trail-activation-usdt",
        str(cfg["trail_activation_usdt"]),
        "--initial-stoploss-usdt",
        str(cfg["initial_stoploss_usdt"]),
        "--short-initial-stoploss-usdt",
        str(cfg["short_initial_stoploss_usdt"]),
        "--trail-keep-pct",
        str(cfg["trail_keep_pct"]),
        "--trail-activation-offset-usdt",
        str(cfg["trail_activation_offset_usdt"]),
        "--min-add-minutes",
        str(args.min_add_minutes),
        "--min-add-drop-pct",
        str(cfg["min_add_drop_pct"]),
        "--rsi-add-drop-pct",
        str(cfg["rsi_add_drop_pct"]),
    ]

    if cfg["use_tp_ladder"]:
        cmd.extend([
            "--use-tp-ladder",
            "--tp-levels",
            str(cfg["tp_levels"]),
            "--tp-step-pct",
            str(cfg["tp_step_pct"]),
            "--tp-close-pct",
            str(cfg["tp_close_pct"]),
        ])

    if args.enable_longs:
        cmd.append("--enable-longs")
    else:
        cmd.append("--disable-longs")

    if args.enable_shorts:
        cmd.append("--enable-shorts")
    else:
        cmd.append("--disable-shorts")

    if args.use_short_tp_ladder:
        cmd.append("--use-short-tp-ladder")
    else:
        cmd.append("--no-use-short-tp-ladder")

    if args.add_only_lower_price:
        cmd.append("--add-only-lower-price")
    else:
        cmd.append("--no-add-only-lower-price")

    if cfg["use_ema"]:
        cmd.extend([
            "--use-ema",
            "--ema-period",
            str(cfg["ema_period"]),
            "--ema-tf",
            str(cfg["ema_tf"]),
            "--ema-condition",
            str(cfg["ema_condition"]),
            "--ema-min-above-hours",
            str(cfg["ema_min_above_hours"]),
        ])
    else:
        cmd.append("--no-use-ema")

    if cfg["use_rsi"]:
        cmd.extend([
            "--use-rsi",
            "--rsi-len",
            str(cfg["rsi_len"]),
            "--rsi-min",
            str(cfg["rsi_min"]),
            "--rsi-max",
            str(cfg["rsi_max"]),
            "--rsi-tf",
            str(cfg["rsi_tf"]),
        ])

    if args.short_rsi_filters:
        cmd.extend(["--short-rsi-filters", args.short_rsi_filters])
    elif args.use_short_rsi:
        cmd.extend([
            "--use-short-rsi",
            "--short-rsi-len",
            str(args.short_rsi_len),
            "--short-rsi-min",
            str(args.short_rsi_min),
            "--short-rsi-max",
            str(args.short_rsi_max),
            "--short-rsi-tf",
            str(args.short_rsi_tf),
        ])

    if args.use_short_ema:
        cmd.extend([
            "--use-short-ema",
            "--short-ema-period",
            str(args.short_ema_period),
            "--short-ema-condition",
            str(args.short_ema_condition),
            "--short-ema-min-below-hours",
            str(args.short_ema_min_below_hours),
            "--short-ema-tf",
            str(args.short_ema_tf),
        ])
    else:
        cmd.append("--no-use-short-ema")

    if cfg["use_volume"]:
        cmd.extend([
            "--use-volume",
            "--vol-len",
            str(cfg["vol_len"]),
            "--vol-multiplier",
            str(cfg["vol_multiplier"]),
            "--vol-tf",
            str(cfg["vol_tf"]),
        ])

    return cmd


def sample_config(rng: random.Random, args: argparse.Namespace) -> dict[str, object]:
    use_ema = True
    use_rsi = rng.random() < args.p_use_rsi
    use_volume = rng.random() < args.p_use_volume

    rsi_min_low, rsi_min_high = args.rsi_min_range
    rsi_max_low, rsi_max_high = args.rsi_max_range

    use_zero_rsi_min = rng.random() < args.p_rsi_min_zero
    if use_zero_rsi_min:
        rsi_min = 0.0
    else:
        rsi_min = rng.uniform(rsi_min_low, rsi_min_high)
    rsi_max_floor = max(rsi_min + 5.0, rsi_max_low)
    if rsi_max_floor > rsi_max_high:
        rsi_max_floor = rsi_max_high
        rsi_min = min(rsi_min, rsi_max_high - 5.0)
    rsi_max = rng.uniform(rsi_max_floor, rsi_max_high)

    ema_min_above_hours = 0.0
    use_ema_min_above = (
        args.ema_min_above_hours_range is not None
        and (rng.random() < args.p_use_ema_min_above_hours)
    )
    if use_ema_min_above:
        ema_min_above_hours = round(rng.uniform(*args.ema_min_above_hours_range), 2)

    min_activation = max(args.trail_activation_usdt_range[0], args.trail_activation_offset_usdt_range[0] + 0.01)
    trail_activation_usdt = round(rng.uniform(min_activation, args.trail_activation_usdt_range[1]), 2)
    max_offset_for_activation = min(args.trail_activation_offset_usdt_range[1], trail_activation_usdt - 0.01)
    trail_activation_offset_usdt = round(rng.uniform(args.trail_activation_offset_usdt_range[0], max_offset_for_activation), 2)

    use_tp_ladder = rng.random() < args.p_use_tp_ladder
    tp_levels = rng.randint(*args.tp_levels_range)
    tp_close_pct = round(rng.uniform(*args.tp_close_pct_range), 2)
    if use_tp_ladder:
        tp_levels = rng.randint(max(1, args.tp_levels_range[0]), args.tp_levels_range[1])
        min_close_from_total = 50.0 / tp_levels
        max_close_from_total = 100.0 / tp_levels
        lo = max(args.tp_close_pct_range[0], min_close_from_total)
        hi = min(args.tp_close_pct_range[1], max_close_from_total)
        tp_close_pct = round(rng.uniform(lo, hi), 2)

    return {
        "entry_cash": args.entry_cash,
        "max_entries": rng.randint(*args.max_entries_range),
        "trail_activation_usdt": trail_activation_usdt,
        "initial_stoploss_usdt": round(rng.uniform(*args.initial_stoploss_usdt_range), 2),
        "short_initial_stoploss_usdt": round(rng.uniform(*args.short_initial_stoploss_usdt_range), 2),
        "trail_keep_pct": round(rng.uniform(*args.trail_keep_pct_range), 4),
        "trail_activation_offset_usdt": trail_activation_offset_usdt,
        "min_add_drop_pct": round(rng.uniform(*args.min_add_drop_pct_range), 2),
        "rsi_add_drop_pct": round(rng.uniform(*args.rsi_add_drop_pct_range), 2),
        "use_tp_ladder": use_tp_ladder,
        "tp_levels": tp_levels,
        "tp_step_pct": round(rng.uniform(*args.tp_step_pct_range), 2),
        "tp_close_pct": tp_close_pct,
        "use_ema": use_ema,
        "ema_period": rng.randint(*args.ema_period_range),
        "ema_tf": rng.choice(args.ema_tf_choices),
        "ema_condition": rng.choice(args.ema_condition_choices),
        "ema_min_above_hours": ema_min_above_hours,
        "use_ema_min_above_hours": use_ema_min_above,
        "use_rsi": use_rsi,
        "rsi_len": rng.randint(*args.rsi_len_range),
        "rsi_min": round(rsi_min, 2),
        "rsi_max": round(rsi_max, 2),
        "use_zero_rsi_min": use_zero_rsi_min,
        "rsi_tf": rng.choice(args.rsi_tf_choices),
        "use_volume": use_volume,
        "vol_len": rng.randint(*args.vol_len_range),
        "vol_multiplier": round(rng.uniform(*args.vol_multiplier_range), 2),
        "vol_tf": rng.choice(args.vol_tf_choices),
    }


def parse_range(value: str) -> tuple[int, int]:
    a, b = value.split(":", 1)
    return int(a), int(b)


def parse_float_range(value: str) -> tuple[float, float]:
    a, b = value.split(":", 1)
    return float(a), float(b)


def parse_float_range_or_disabled(value: str) -> tuple[float, float] | None:
    value = value.strip()
    if value == "--":
        return None
    return parse_float_range(value)


def parse_int_choices(value: str) -> list[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def parse_str_choices(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def parse_run_list(value: str) -> list[int]:
    runs: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        runs.append(int(token))
    return runs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Random optimize xauusd_step1 settings")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--step1-script", type=Path, default=Path("tools/xauusd_step1.py"))
    p.add_argument("--tail", type=int, default=350000)
    p.add_argument("--cash", type=float, default=1500.0)
    p.add_argument("--runs", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--out", type=Path, default=Path("optimizer_results.csv"))
    p.add_argument("--checkpoint-every", type=int, default=5, help="Write partial results every N runs")
    p.add_argument("--replay-runs", type=parse_run_list, default=None, help="Only execute specified run indices, e.g. 17,18,51")

    p.add_argument("--entry-cash", type=float, default=500.0)
    p.add_argument("--enable-longs", dest="enable_longs", action="store_true", default=True)
    p.add_argument("--disable-longs", dest="enable_longs", action="store_false")
    p.add_argument("--enable-shorts", dest="enable_shorts", action="store_true", default=False)
    p.add_argument("--disable-shorts", dest="enable_shorts", action="store_false")
    p.add_argument("--use-short-tp-ladder", dest="use_short_tp_ladder", action="store_true", default=False)
    p.add_argument("--no-use-short-tp-ladder", dest="use_short_tp_ladder", action="store_false")
    p.add_argument("--max-entries", type=int, default=3)
    p.add_argument("--max-entries-range", type=parse_range, default=(3, 3))
    p.add_argument("--auto-entry-cash-from-max-entries", action="store_true", default=True)
    p.add_argument("--no-auto-entry-cash-from-max-entries", dest="auto_entry_cash_from_max_entries", action="store_false")
    p.add_argument("--min-add-minutes", type=int, default=5)
    p.add_argument("--add-only-lower-price", action="store_true", default=True)
    p.add_argument("--trail-activation-usdt", type=float, default=4.0)
    p.add_argument("--initial-stoploss-usdt", type=float, default=-20.0)
    p.add_argument("--short-initial-stoploss-usdt", type=float, default=-20.0)
    p.add_argument("--trail-keep-pct", type=float, default=0.5)
    p.add_argument("--trail-activation-usdt-range", type=parse_float_range, default=(4.0, 4.0))
    p.add_argument("--initial-stoploss-usdt-range", type=parse_float_range, default=(-20.0, -20.0))
    p.add_argument("--short-initial-stoploss-usdt-range", type=parse_float_range, default=(-20.0, -20.0))
    p.add_argument("--trail-keep-pct-range", type=parse_float_range, default=(0.5, 0.5))
    p.add_argument("--trail-activation-offset-usdt-range", type=parse_float_range, default=(0.0, 0.0))
    p.add_argument("--min-add-drop-pct-range", type=parse_float_range, default=(0.0, 0.0))
    p.add_argument("--rsi-add-drop-pct-range", type=parse_float_range, default=(0.0, 0.0))
    p.add_argument("--p-use-tp-ladder", type=float, default=0.0)
    tp_mode = p.add_mutually_exclusive_group()
    tp_mode.add_argument("--force-use-tp-ladder", action="store_true", help="Force TP ladder on for every sampled run")
    tp_mode.add_argument("--force-no-tp-ladder", action="store_true", help="Force TP ladder off for every sampled run")
    p.add_argument("--tp-levels-range", type=parse_range, default=(0, 0))
    p.add_argument("--tp-step-pct-range", type=parse_float_range, default=(10.0, 10.0))
    p.add_argument("--tp-close-pct-range", type=parse_float_range, default=(10.0, 10.0))

    p.add_argument("--p-use-rsi", type=float, default=0.8)
    p.add_argument("--p-use-volume", type=float, default=0.8)

    p.add_argument("--ema-period-range", type=parse_range, default=(20, 120))
    p.add_argument("--ema-tf-choices", type=parse_int_choices, default=[1, 5, 15, 30, 60])
    p.add_argument("--ema-condition-choices", type=parse_str_choices, default=["below_after_above", "below", "above"])
    p.add_argument(
        "--ema-min-above-hours-range",
        type=parse_float_range_or_disabled,
        default=(0.5, 6.0),
        help="low:high range in hours or '--' to disable the feature globally",
    )
    p.add_argument(
        "--p-use-ema-min-above-hours",
        type=float,
        default=1.0,
        help="Probability to apply ema-min-above-hours per run (e.g. 0.5 ~= every second run)",
    )

    p.add_argument("--rsi-len-range", type=parse_range, default=(7, 80))
    p.add_argument("--rsi-min-range", type=parse_float_range, default=(0.0, 40.0))
    p.add_argument("--rsi-max-range", type=parse_float_range, default=(50.0, 95.0))
    p.add_argument(
        "--p-rsi-min-zero",
        type=float,
        default=0.0,
        help="Probability to force rsi_min=0 per run (e.g. 0.5 ~= every second run)",
    )
    p.add_argument("--rsi-tf-choices", type=parse_int_choices, default=[1, 5, 15, 30])
    p.add_argument(
        "--short-rsi-filters",
        type=str,
        default="",
        help="Pass-through multi short RSI filters to step1: 'tf:len:min:max;tf:len:min:max'",
    )
    p.add_argument("--use-short-rsi", action="store_true", default=False)
    p.add_argument("--short-rsi-len", type=int, default=50)
    p.add_argument("--short-rsi-min", type=float, default=20.0)
    p.add_argument("--short-rsi-max", type=float, default=80.0)
    p.add_argument("--short-rsi-tf", type=int, default=1)
    p.add_argument("--use-short-ema", action="store_true", default=True)
    p.add_argument("--no-use-short-ema", dest="use_short_ema", action="store_false")
    p.add_argument("--short-ema-period", type=int, default=50)
    p.add_argument(
        "--short-ema-condition",
        type=str,
        choices=["above", "below", "above_after_below"],
        default="above_after_below",
    )
    p.add_argument("--short-ema-min-below-hours", type=float, default=2.0)
    p.add_argument("--short-ema-tf", type=int, default=15)

    p.add_argument("--vol-len-range", type=parse_range, default=(10, 120))
    p.add_argument("--vol-multiplier-range", type=parse_float_range, default=(1.0, 5.0))
    p.add_argument("--vol-tf-choices", type=parse_int_choices, default=[1, 5, 15, 30])

    return p.parse_args()




def sort_results(results: list[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(
        results,
        key=lambda r: (r["annualized_roi_pct"] if r["exit_code"] == 0 else -1e18),
        reverse=True,
    )


def save_results(path: Path, results: list[dict[str, object]]) -> Path | None:
    if not results:
        return None

    sorted_results = sort_results(results)
    headers = list(sorted_results[0].keys())
    for _ in range(3):
        try:
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                writer.writerows(sorted_results)
            return path
        except PermissionError:
            time.sleep(0.2)

    fallback = path.with_name(f"{path.stem}.pid{os.getpid()}{path.suffix}")
    with fallback.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(sorted_results)
    return fallback


def print_top(results: list[dict[str, object]], top: int) -> None:
    sorted_results = sort_results(results)
    print("\nTop results by annualized ROI:")
    for r in sorted_results[:top]:
        print(
            f"run={r['run']} annualized={r['annualized_roi_pct']:.2f}% roi={r['roi_period_pct']:.2f}% "
            f"win={r['win_rate_pct']:.2f}% ema_tf={r['ema_tf']} ema_period={r['ema_period']} "
            f"rsi={r['use_rsi']} vol={r['use_volume']}"
        )

def main() -> None:
    args = parse_args()
    if args.force_use_tp_ladder:
        args.p_use_tp_ladder = 1.0
    elif args.force_no_tp_ladder:
        args.p_use_tp_ladder = 0.0
    if args.replay_runs and min(args.replay_runs) < 1:
        raise ValueError("replay-runs values must be >= 1")
    if args.rsi_min_range[0] > args.rsi_min_range[1]:
        raise ValueError("rsi-min-range must be low:high")
    if args.rsi_max_range[0] > args.rsi_max_range[1]:
        raise ValueError("rsi-max-range must be low:high")
    if args.max_entries_range[0] < 1 or args.max_entries_range[0] > args.max_entries_range[1]:
        raise ValueError("max-entries-range must be low:high and >= 1")
    if args.trail_activation_usdt_range[0] > args.trail_activation_usdt_range[1]:
        raise ValueError("trail-activation-usdt-range must be low:high")
    if args.initial_stoploss_usdt_range[0] > args.initial_stoploss_usdt_range[1]:
        raise ValueError("initial-stoploss-usdt-range must be low:high")
    if args.short_initial_stoploss_usdt_range[0] > args.short_initial_stoploss_usdt_range[1]:
        raise ValueError("short-initial-stoploss-usdt-range must be low:high")
    if args.trail_keep_pct_range[0] > args.trail_keep_pct_range[1] or args.trail_keep_pct_range[0] < 0:
        raise ValueError("trail-keep-pct-range must be low:high and >= 0")
    if args.trail_activation_offset_usdt_range[0] > args.trail_activation_offset_usdt_range[1] or args.trail_activation_offset_usdt_range[0] < 0:
        raise ValueError("trail-activation-offset-usdt-range must be low:high and >= 0")
    if args.trail_activation_usdt_range[1] <= args.trail_activation_offset_usdt_range[0]:
        raise ValueError("trail-activation-usdt-range upper bound must be greater than trail-activation-offset-usdt-range lower bound")
    if args.min_add_drop_pct_range[0] > args.min_add_drop_pct_range[1] or args.min_add_drop_pct_range[0] < 0:
        raise ValueError("min-add-drop-pct-range must be low:high and >= 0")
    if args.rsi_add_drop_pct_range[0] > args.rsi_add_drop_pct_range[1] or args.rsi_add_drop_pct_range[0] < 0:
        raise ValueError("rsi-add-drop-pct-range must be low:high and >= 0")
    if not (0.0 <= args.p_use_tp_ladder <= 1.0):
        raise ValueError("p-use-tp-ladder must be in [0, 1]")
    if args.tp_levels_range[0] < 0 or args.tp_levels_range[0] > args.tp_levels_range[1]:
        raise ValueError("tp-levels-range must be low:high and >= 0")
    if args.tp_step_pct_range[0] <= 0 or args.tp_step_pct_range[0] > args.tp_step_pct_range[1]:
        raise ValueError("tp-step-pct-range must be low:high and > 0")
    if args.tp_close_pct_range[0] <= 0 or args.tp_close_pct_range[0] > args.tp_close_pct_range[1]:
        raise ValueError("tp-close-pct-range must be low:high and > 0")
    if args.p_use_tp_ladder > 0:
        if args.tp_levels_range[1] < 1:
            raise ValueError("tp-levels-range must allow at least 1 when p-use-tp-ladder > 0")
        feasible = False
        for levels in range(max(1, args.tp_levels_range[0]), args.tp_levels_range[1] + 1):
            lo = max(args.tp_close_pct_range[0], 50.0 / levels)
            hi = min(args.tp_close_pct_range[1], 100.0 / levels)
            if lo <= hi:
                feasible = True
                break
        if not feasible:
            raise ValueError("No feasible TP ladder combo for constraints: need some levels where 50/levels <= tp-close-pct <= 100/levels")
    if not (0.0 <= args.p_rsi_min_zero <= 1.0):
        raise ValueError("p-rsi-min-zero must be in [0, 1]")
    if not (0.0 <= args.p_use_ema_min_above_hours <= 1.0):
        raise ValueError("p-use-ema-min-above-hours must be in [0, 1]")
    if args.ema_min_above_hours_range is not None and args.ema_min_above_hours_range[0] > args.ema_min_above_hours_range[1]:
        raise ValueError("ema-min-above-hours-range must be low:high or '--'")

    results: list[dict[str, object]] = []

    run_indices = list(range(1, args.runs + 1)) if not args.replay_runs else sorted(set(args.replay_runs))

    try:
        for run_idx in run_indices:
            # Rebuild the exact config for run_idx from the same seed.
            # This guarantees replay parity independent of replay order.
            run_rng = random.Random(args.seed)
            cfg = None
            for _ in range(run_idx):
                cfg = sample_config(run_rng, args)
            cmd = build_command(args, cfg)
            proc = subprocess.run(cmd, capture_output=True, text=True)
            out = proc.stdout + "\n" + proc.stderr

            row = {
                "run": run_idx,
                **cfg,
                "exit_code": proc.returncode,
                "annualized_roi_pct": parse_metric(ANNUALIZED_RE, out),
                "roi_period_pct": parse_metric(ROI_PERIOD_RE, out),
                "win_rate_pct": parse_metric(WINRATE_RE, out),
                "end_portfolio": parse_metric(END_VALUE_RE, out),
                "closed_trades": int(parse_metric(CLOSED_TRADES_RE, out, default=0.0)),
            }

            results.append(row)

            print(
                f"[{len(results)}/{len(run_indices)} | run={run_idx}] code={row['exit_code']} annualized={row['annualized_roi_pct']:.2f}% "
                f"roi={row['roi_period_pct']:.2f}% win={row['win_rate_pct']:.2f}%"
            )

            if args.checkpoint_every > 0 and len(results) % args.checkpoint_every == 0:
                saved_to = save_results(args.out, results)
                print(f"checkpoint saved: {saved_to} ({len(results)} runs)")

    except KeyboardInterrupt:
        print("\nInterrupted by user (Ctrl+C). Saving partial results ...")

    saved_to = save_results(args.out, results)
    if results:
        print_top(results, args.top)
    print(f"\nSaved: {saved_to or args.out}")


if __name__ == "__main__":
    main()
