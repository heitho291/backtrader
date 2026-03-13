#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import random
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional


# -----------------------------
# helpers: spec parsing + sampling
# -----------------------------
INT_RE = re.compile(r"^-?\d+$")


def is_int_token(s: str) -> bool:
    return bool(INT_RE.match(s.strip()))


def decimals_in_token(s: str) -> int:
    s = s.strip()
    if "." in s:
        return len(s.split(".", 1)[1])
    return 0


def quantized_float_sample(rng: random.Random, lo_s: str, hi_s: str) -> Tuple[float, int]:
    """
    Sample from [lo, hi] in steps of 10^-d, where d = max(decimals(lo), decimals(hi)).
    Returns (value, d).
    """
    d = max(decimals_in_token(lo_s), decimals_in_token(hi_s))
    lo = float(lo_s)
    hi = float(hi_s)
    if hi < lo:
        raise ValueError(f"Invalid float range {lo_s}:{hi_s} (high < low)")

    scale = 10 ** d
    lo_i = int(round(lo * scale))
    hi_i = int(round(hi * scale))
    if hi_i < lo_i:
        raise ValueError(f"Invalid quantized float range after scaling: {lo_s}:{hi_s}")

    v_i = rng.randint(lo_i, hi_i)  # inclusive
    v = v_i / scale
    return v, d


def sample_from_spec(rng: random.Random, spec: str) -> Tuple[str, str]:
    """
    Returns (sampled_value_as_string, kind)
      kind in {"fixed","choice","int_range","float_range","int_range_step"}
    Rules:
      - choice: a|b|c
      - int range step: a:b:step  (always int)
      - 2-part range:
          if both tokens are ints => int_range
          else => float_range quantized by decimals in tokens
      - otherwise fixed
    """
    spec = str(spec)

    # choice
    if "|" in spec:
        parts = [p for p in spec.split("|") if p != ""]
        v = rng.choice(parts)
        return str(v), "choice"

    # range
    if ":" in spec:
        parts = [p.strip() for p in spec.split(":")]
        if len(parts) == 3:
            lo_s, hi_s, step_s = parts
            lo = int(lo_s)
            hi = int(hi_s)
            step = int(step_s)
            if step <= 0:
                raise ValueError(f"Invalid step in range {spec}")
            if hi < lo:
                raise ValueError(f"Invalid int range {spec} (high < low)")
            n = ((hi - lo) // step) + 1
            v = lo + step * rng.randrange(n)
            return str(v), "int_range_step"

        if len(parts) == 2:
            lo_s, hi_s = parts
            if is_int_token(lo_s) and is_int_token(hi_s):
                lo = int(lo_s)
                hi = int(hi_s)
                if hi < lo:
                    raise ValueError(f"Invalid int range {spec} (high < low)")
                v = rng.randint(lo, hi)
                return str(v), "int_range"

            # float quantized
            v, d = quantized_float_sample(rng, lo_s, hi_s)
            v_str = f"{v:.{d}f}"
            return v_str, "float_range"

    # fixed
    return spec, "fixed"


# -----------------------------
# parse miner stdout (RESULT block)
# -----------------------------
RESULT_PATTERNS = {
    "base_winrate_train": re.compile(r"Base winrate train:\s*([0-9.]+)"),
    "base_winrate_test":  re.compile(r"Base winrate train:\s*[0-9.]+\s*\|\s*test:\s*([0-9.]+)"),
    "base_ev_train":      re.compile(r"Base EV\s+train:\s*([-\d.]+)"),
    "base_ev_test":       re.compile(r"Base EV\s+train:\s*[-\d.]+\s*\|\s*test:\s*([-\d.]+)"),
    "best_rule":          re.compile(r"Best rule:\s*(.+)"),
    "test_ev":            re.compile(r"TEST\s+EV\(mean pnl\):\s*([-\d.]+)"),
    "test_winrate":       re.compile(r"Winrate:\s*([0-9.]+)"),
    "test_sharpe_proxy":  re.compile(r"SharpeProxy:\s*([-\d.]+)"),
    "test_hits":          re.compile(r"hits:\s*(\d+)"),
    "test_median_signals_day": re.compile(r"median signals/day:\s*([0-9.]+)"),
    "conds":              re.compile(r"conds:\s*(\d+)"),
    "test_median_exit_min": re.compile(r"TEST\s+median exit\(min\):\s*([0-9.]+)"),
    "test_median_tp_hits": re.compile(r"median tp_hits:\s*([0-9.]+)"),
    "wf_mean_win":        re.compile(r"WF mean_win:\s*([0-9.]+)"),
    "wf_min_win":         re.compile(r"WF min_win:\s*([0-9.]+)"),
    "wf_mean_ev":         re.compile(r"WF mean_EV:\s*([-\d.]+)"),
    "wf_hits":            re.compile(r"WF hits:\s*(\d+)"),
}


def parse_result(stdout_text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k, pat in RESULT_PATTERNS.items():
        m = pat.search(stdout_text)
        if m:
            out[k] = m.group(1).strip()
    return out


# -----------------------------
# argument handling for optimizer
# -----------------------------
def strip_arg_with_value(args: List[str], flag: str) -> List[str]:
    """
    Remove occurrences of: flag <value>
    """
    out = []
    i = 0
    while i < len(args):
        if args[i] == flag:
            i += 2
            continue
        out.append(args[i])
        i += 1
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-csv", default="best_rules_optimizer.csv")
    p.add_argument("--workdir", default="", help="Optional base dir for temp run folders. Default: ./optimizer_runs_<timestamp>")
    p.add_argument("--keep-artifacts", action="store_true", help="Keep per-run temp folders (outputs + stdout). Default: delete.")
    p.add_argument("miner_script", type=str, help="Path to miner .py")
    p.add_argument("miner_args", nargs=argparse.REMAINDER, help="Everything after miner_script is passed to miner (ranges/choices allowed).")
    return p


def main() -> None:
    args = build_parser().parse_args()

    if args.runs < 1:
        raise ValueError("--runs must be >= 1")

    miner_args = list(args.miner_args)

    # Prevent user-provided out files from overwriting (we manage them)
    miner_args = strip_arg_with_value(miner_args, "--out-signals")
    miner_args = strip_arg_with_value(miner_args, "--out-summary")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = Path(args.workdir) if args.workdir else Path(f"optimizer_runs_{ts}")
    base.mkdir(parents=True, exist_ok=True)

    rng_master = random.Random(args.seed)

    rows: List[Dict[str, str]] = []

    for run_idx in range(1, args.runs + 1):
        run_dir = base / f"run_{run_idx:05d}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # per-run rng derived deterministically
        run_seed = rng_master.randint(0, 2**31 - 1)
        rng = random.Random(run_seed)

        # sample args
        sampled_params: Dict[str, str] = {}
        final_miner_args: List[str] = []

        i = 0
        while i < len(miner_args):
            a = miner_args[i]

            # pass through non-flag tokens (rare)
            if not a.startswith("--"):
                final_miner_args.append(a)
                i += 1
                continue

            # boolean flag with no value
            if i + 1 >= len(miner_args) or miner_args[i + 1].startswith("--"):
                final_miner_args.append(a)
                i += 1
                continue

            v_spec = miner_args[i + 1]
            v_s, _kind = sample_from_spec(rng, v_spec)

            # Special: allow "--trail true|false" and "--use-multi-tp true|false"
            if a == "--trail":
                t = v_s.strip().lower()
                if t in ("true", "1", "yes", "y", "on"):
                    final_miner_args.append("--trail")
                    sampled_params["trail"] = "true"
                elif t in ("false", "0", "no", "n", "off"):
                    final_miner_args.append("--no-trail")
                    sampled_params["trail"] = "false"
                else:
                    raise ValueError(f"Invalid bool spec for --trail: {v_spec} (sampled {v_s})")
                i += 2
                continue

            if a == "--use-multi-tp":
                t = v_s.strip().lower()
                if t in ("true", "1", "yes", "y", "on"):
                    final_miner_args.append("--use-multi-tp")
                    sampled_params["use-multi-tp"] = "true"
                elif t in ("false", "0", "no", "n", "off"):
                    final_miner_args.append("--no-multi-tp")
                    sampled_params["use-multi-tp"] = "false"
                else:
                    raise ValueError(f"Invalid bool spec for --use-multi-tp: {v_spec} (sampled {v_s})")
                i += 2
                continue

            # Normal key/value
            final_miner_args.append(a)
            final_miner_args.append(v_s)
            sampled_params[a.lstrip("-")] = v_s
            i += 2

        # Ensure no overwrites: we set unique out files ourselves (still per-run, but in temp folder)
        out_summary = run_dir / "best_rules_summary.csv"
        out_signals = run_dir / "signals_best_rules.csv"
        final_miner_args += ["--out-summary", str(out_summary), "--out-signals", str(out_signals)]

        cmd = [sys.executable, args.miner_script] + final_miner_args

        print(f"\nRUN {run_idx}/{args.runs}")
        print("CMD:", " ".join([f"\"{c}\"" if (" " in c) else c for c in cmd]))

        proc = subprocess.run(cmd, capture_output=True, text=True)
        stdout_text = proc.stdout or ""
        stderr_text = proc.stderr or ""

        # write stdout/stderr for debugging (in run_dir). Will be deleted unless keep_artifacts.
        (run_dir / "stdout.txt").write_text(stdout_text, encoding="utf-8")
        (run_dir / "stderr.txt").write_text(stderr_text, encoding="utf-8")

        if proc.returncode != 0:
            # Record failure row and stop (safer for long runs)
            row = {
                "run": str(run_idx),
                "returncode": str(proc.returncode),
                "run_seed": str(run_seed),
                "error": "miner_failed",
            }
            # include sampled params
            for k, v in sampled_params.items():
                row[f"param_{k}"] = v
            rows.append(row)
            print("Run failed. Stopping.")
            break

        metrics = parse_result(stdout_text)

        # one row per run
        row = {
            "run": str(run_idx),
            "returncode": "0",
            "run_seed": str(run_seed),
        }
        for k, v in sampled_params.items():
            row[f"param_{k}"] = v
        for k, v in metrics.items():
            row[k] = v

        rows.append(row)

        # cleanup run artifacts unless requested
        if not args.keep_artifacts:
            shutil.rmtree(run_dir, ignore_errors=True)

    # write final CSV (end only)
    out_csv = Path(args.out_csv)
    if not out_csv.is_absolute():
        out_csv = base / out_csv

    # collect all columns
    cols = sorted({k for r in rows for k in r.keys()})
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    print("\nDONE")
    print("Saved:", str(out_csv))


if __name__ == "__main__":
    main()