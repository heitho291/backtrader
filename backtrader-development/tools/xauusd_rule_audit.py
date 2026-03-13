#!/usr/bin/env python3
"""Audit mined XAUUSD rules for overlap bias, summary consistency and suspicious win-rates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit rule-miner CSV exports")
    p.add_argument("--best-rules", type=Path, required=True, help="Path to best_rules_summary.csv or .csv.gz")
    p.add_argument("--signals", type=Path, required=True, help="Path to signals_best_rules.csv or .csv.gz")
    p.add_argument("--cooldown-minutes", type=int, default=None, help="Cooldown used to cluster sequential hits into one event")
    p.add_argument("--out-csv", type=Path, default=Path("rule_audit.csv"), help="Per-rule audit CSV")
    p.add_argument("--out-json", type=Path, default=Path("rule_audit_summary.json"), help="Global audit summary JSON")
    return p.parse_args()


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, compression="infer")


def _event_ids(ts: pd.Series, cooldown_minutes: int) -> pd.Series:
    ts = pd.to_datetime(ts, errors="coerce")
    deltas = ts.diff().dt.total_seconds().div(60.0)
    starts = (deltas.isna()) | (deltas > cooldown_minutes)
    return starts.cumsum()


def _with_rule_key(df: pd.DataFrame, required: set[str]) -> pd.DataFrame:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing key columns: {sorted(missing)}")
    x = df.copy()
    x["rule"] = x["rule"].astype(str)
    x["tp"] = pd.to_numeric(x["tp"], errors="coerce")
    x["sl"] = pd.to_numeric(x["sl"], errors="coerce")
    x["hold"] = pd.to_numeric(x["hold"], errors="coerce").astype("Int64")
    x["rule_key"] = (
        "rule=" + x["rule"].astype(str)
        + "|tp=" + x["tp"].map(lambda v: "nan" if pd.isna(v) else f"{float(v):.8g}")
        + "|sl=" + x["sl"].map(lambda v: "nan" if pd.isna(v) else f"{float(v):.8g}")
        + "|hold=" + x["hold"].astype(str)
    )
    return x


def audit_rule(rule_signals: pd.DataFrame, cooldown_minutes: int) -> dict[str, float | int]:
    x = rule_signals.copy()
    x["datetime"] = pd.to_datetime(x["datetime"], errors="coerce")
    x = x.dropna(subset=["datetime", "outcome"]).sort_values("datetime")

    if x.empty:
        return {
            "signal_hits": 0,
            "signal_winrate": float("nan"),
            "event_hits": 0,
            "event_winrate": float("nan"),
            "tp_hits": 0,
            "sl_hits": 0,
            "other_hits": 0,
        }

    x["outcome"] = x["outcome"].astype(str).str.upper()
    x["event_id"] = _event_ids(x["datetime"], cooldown_minutes)

    tp_hits = int((x["outcome"] == "TP").sum())
    sl_hits = int((x["outcome"] == "SL").sum())
    other_hits = int(len(x) - tp_hits - sl_hits)

    by_event = x.groupby("event_id")["outcome"].agg(lambda s: "TP" if (s == "TP").any() else ("SL" if (s == "SL").any() else "OTHER"))
    event_tp = int((by_event == "TP").sum())
    event_count = int(len(by_event))

    return {
        "signal_hits": int(len(x)),
        "signal_winrate": tp_hits / len(x) if len(x) else float("nan"),
        "event_hits": event_count,
        "event_winrate": event_tp / event_count if event_count else float("nan"),
        "tp_hits": tp_hits,
        "sl_hits": sl_hits,
        "other_hits": other_hits,
        "event_tp_hits": event_tp,
    }


def main() -> None:
    args = parse_args()
    best = _read_csv(args.best_rules)
    sig = _read_csv(args.signals)

    required_best = {"rule", "test_win", "test_hits", "tp", "sl", "hold"}
    required_sig = {"rule", "datetime", "outcome", "tp", "sl", "hold"}
    if not required_best.issubset(best.columns):
        raise ValueError(f"best-rules missing columns: {sorted(required_best - set(best.columns))}")
    if not required_sig.issubset(sig.columns):
        raise ValueError(f"signals missing columns: {sorted(required_sig - set(sig.columns))}")

    best = _with_rule_key(best, required_best)
    sig = _with_rule_key(sig, required_sig)

    rows: list[dict[str, object]] = []
    suspicious: list[dict[str, object]] = []

    for _, row in best.iterrows():
        rule = row["rule"]
        key = row["rule_key"]
        hold = int(row["hold"])
        cooldown = args.cooldown_minutes if args.cooldown_minutes is not None else hold
        rsig = sig[sig["rule_key"] == key]
        stats = audit_rule(rsig, cooldown)

        summary_hits = int(row["test_hits"])
        summary_wr = float(row["test_win"])
        signal_hits = int(stats["signal_hits"])
        signal_wr = float(stats["signal_winrate"])

        hit_delta = signal_hits - summary_hits
        wr_delta = signal_wr - summary_wr if pd.notna(signal_wr) else float("nan")
        overlap_factor = signal_hits / max(1, int(stats["event_hits"]))

        out = {
            "rule_key": key,
            "rule": rule,
            "tp": float(row["tp"]),
            "sl": float(row["sl"]),
            "hold": hold,
            "cooldown_minutes": cooldown,
            "summary_test_hits": summary_hits,
            "summary_test_win": summary_wr,
            "signal_hits": signal_hits,
            "signal_winrate": signal_wr,
            "event_hits": stats["event_hits"],
            "event_winrate": stats["event_winrate"],
            "overlap_factor": overlap_factor,
            "tp_hits": stats["tp_hits"],
            "sl_hits": stats["sl_hits"],
            "other_hits": stats["other_hits"],
            "delta_hits_signal_minus_summary": hit_delta,
            "delta_winrate_signal_minus_summary": wr_delta,
        }
        rows.append(out)

        if (
            signal_hits == 0
            or stats["sl_hits"] == 0
            or overlap_factor > 1.5
            or abs(hit_delta) > 0
            or (pd.notna(signal_wr) and signal_wr >= 0.99 and signal_hits >= 30)
        ):
            suspicious.append(out)

    audit_df = pd.DataFrame(rows).sort_values(["summary_test_win", "summary_test_hits"], ascending=[False, False])
    audit_df.to_csv(args.out_csv, index=False)

    summary = {
        "rules": int(len(audit_df)),
        "suspicious_rules": int(len(suspicious)),
        "max_signal_winrate": float(audit_df["signal_winrate"].max()) if not audit_df.empty else None,
        "max_event_winrate": float(audit_df["event_winrate"].max()) if not audit_df.empty else None,
        "max_overlap_factor": float(audit_df["overlap_factor"].max()) if not audit_df.empty else None,
        "notes": [
            "Rule matching uses compound key rule+tp+sl+hold to prevent cross-TP contamination.",
            "Use event_winrate over signal_winrate when many adjacent bars trigger same move.",
            "If suspicious_rules > 0, inspect train/test split and target labeling for leakage.",
            "CSV.GZ is supported directly: pandas compression='infer'.",
        ],
        "suspicious_examples": suspicious[:10],
    }
    args.out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved audit CSV: {args.out_csv}")
    print(f"Saved audit JSON: {args.out_json}")
    print(f"Rules audited: {len(audit_df)} | Suspicious: {len(suspicious)}")


if __name__ == "__main__":
    main()

