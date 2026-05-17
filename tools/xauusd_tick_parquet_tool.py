#!/usr/bin/env python3
"""Build, append and validate normalized XAUUSD tick Parquet files."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

NS_PER_MINUTE = 60_000_000_000
NORMALIZED_COLUMNS = ["datetime_ns", "minute_ns", "price", "bid", "ask", "volume", "source_order"]
DEDUP_COLUMNS = ["datetime_ns", "price", "bid", "ask", "volume"]


def _import_deps():
    import numpy as np  # type: ignore
    import pandas as pd  # type: ignore

    return np, pd


def _lower_col_map(cols) -> dict[str, str]:
    return {str(c).strip().lower(): str(c) for c in cols}


def _first_col(cols_lower: dict[str, str], names: list[str]) -> str | None:
    for name in names:
        key = str(name).strip().lower()
        if key in cols_lower:
            return cols_lower[key]
    return None


def _read_csv_input(path: Path, args: argparse.Namespace):
    np, pd = _import_deps()
    sep = None if str(args.sep).lower() == "auto" else str(args.sep)
    has_header = str(args.has_header).lower()
    if has_header == "auto":
        sample = path.read_text(encoding="utf-8", errors="ignore").splitlines()[:1]
        first = sample[0] if sample else ""
        header = any(ch.isalpha() for ch in first)
    else:
        header = has_header == "yes"
    read_kwargs: dict[str, Any] = {"sep": sep, "engine": "python"} if sep is None else {"sep": sep}
    if header:
        return pd.read_csv(path, **read_kwargs)
    df = pd.read_csv(path, header=None, **read_kwargs)
    default_cols = ["date", "time", "ask", "bid", "volume"] if df.shape[1] >= 5 else ["datetime", "price", "bid", "ask", "volume"][: df.shape[1]]
    df.columns = default_cols + [f"col_{i}" for i in range(len(default_cols), df.shape[1])]
    return df


def _read_input_frame(path: Path, args: argparse.Namespace):
    _, pd = _import_deps()
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return _read_csv_input(path, args)


def _normalize_frame(df, args: argparse.Namespace, source_offset: int = 0):
    np, pd = _import_deps()
    rows_total_in = int(len(df))
    cols_lower = _lower_col_map(df.columns)

    if set(NORMALIZED_COLUMNS).issubset(set(str(c) for c in df.columns)):
        out = df[NORMALIZED_COLUMNS].copy()
        for c in NORMALIZED_COLUMNS:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    else:
        dt_col = str(args.datetime_column).strip() if args.datetime_column else ""
        date_col = str(args.date_column).strip() if args.date_column else ""
        time_col = str(args.time_column).strip() if args.time_column else ""
        if not dt_col:
            dt_col = _first_col(cols_lower, ["datetime", "timestamp", "time", "date"] ) or ""
        if (not date_col) and "date" in cols_lower and (not dt_col or dt_col == cols_lower.get("time")):
            date_col = cols_lower["date"]
        if not time_col:
            time_col = _first_col(cols_lower, ["time"] ) or ""

        if date_col and time_col and date_col in df.columns and time_col in df.columns:
            dt_raw = df[date_col].astype(str).str.strip() + " " + df[time_col].astype(str).str.strip()
        elif dt_col and dt_col in df.columns:
            dt_raw = df[dt_col]
        else:
            raise ValueError("No datetime input found; use --datetime-column or --date-column/--time-column")

        dt = pd.to_datetime(dt_raw, errors="coerce", utc=False)
        dt_ns = dt.astype("int64")
        minute_ns = (dt_ns // NS_PER_MINUTE) * NS_PER_MINUTE

        bid_col = str(args.bid_column).strip() if args.bid_column else (_first_col(cols_lower, ["bid", "Bid", "BID"]) or "")
        ask_col = str(args.ask_column).strip() if args.ask_column else (_first_col(cols_lower, ["ask", "Ask", "ASK"]) or "")
        price_col = str(args.price_column).strip() if args.price_column else (_first_col(cols_lower, ["price", "last", "close", "mid"]) or "")
        vol_col = str(args.volume_column).strip() if args.volume_column else (_first_col(cols_lower, ["volume", "vol", "Volume"]) or "")

        bid = pd.to_numeric(df[bid_col], errors="coerce") if bid_col in df.columns else pd.Series(np.nan, index=df.index, dtype="float64")
        ask = pd.to_numeric(df[ask_col], errors="coerce") if ask_col in df.columns else pd.Series(np.nan, index=df.index, dtype="float64")
        if bid_col in df.columns and ask_col in df.columns:
            price = (bid + ask) / 2.0
        elif price_col in df.columns:
            price = pd.to_numeric(df[price_col], errors="coerce")
        elif bid_col in df.columns:
            price = bid
        elif ask_col in df.columns:
            price = ask
        else:
            raise ValueError("No usable price source found; provide bid/ask or --price-column")
        volume = pd.to_numeric(df[vol_col], errors="coerce") if vol_col in df.columns else pd.Series(np.nan, index=df.index, dtype="float64")
        out = pd.DataFrame({
            "datetime_ns": dt_ns,
            "minute_ns": minute_ns,
            "price": price.astype("float64"),
            "bid": bid.astype("float64"),
            "ask": ask.astype("float64"),
            "volume": volume.astype("float64"),
            "source_order": np.arange(source_offset, source_offset + len(df), dtype=np.int64),
        })

    valid_dt = out["datetime_ns"].notna() & (out["datetime_ns"].astype("int64") != np.iinfo(np.int64).min)
    valid_price = pd.to_numeric(out["price"], errors="coerce") > 0
    valid_bid = out["bid"].isna() | (pd.to_numeric(out["bid"], errors="coerce") > 0)
    valid_ask = out["ask"].isna() | (pd.to_numeric(out["ask"], errors="coerce") > 0)
    valid = valid_dt & valid_price & valid_bid & valid_ask
    dropped = int(rows_total_in - int(valid.sum()))
    out = out.loc[valid, NORMALIZED_COLUMNS].copy()
    out["datetime_ns"] = out["datetime_ns"].astype("int64")
    out["minute_ns"] = ((out["datetime_ns"] // NS_PER_MINUTE) * NS_PER_MINUTE).astype("int64")
    out["source_order"] = pd.to_numeric(out["source_order"], errors="coerce").fillna(0).astype("int64")
    for c in ["price", "bid", "ask", "volume"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
    return out, {"rows_total_in": rows_total_in, "rows_dropped_invalid": dropped}


def _sort_and_dedupe(df):
    _, pd = _import_deps()
    before = int(len(df))
    out = df.sort_values(["datetime_ns", "source_order"], kind="mergesort").drop_duplicates(subset=DEDUP_COLUMNS, keep="first").copy()
    out = out.sort_values(["datetime_ns", "source_order"], kind="mergesort").reset_index(drop=True)
    out["source_order"] = range(len(out))
    return out, before - int(len(out))


def _ns_to_iso(ns: int) -> str:
    _, pd = _import_deps()
    return pd.Timestamp(int(ns), unit="ns").isoformat()


def _parse_gap_ts(value: str) -> int:
    _, pd = _import_deps()
    return int(pd.Timestamp(value).value)


def _load_allowed_gaps(path: Path | None) -> list[tuple[int, int, str]]:
    if path is None:
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for item in raw.get("allowed_gaps", []):
        out.append((_parse_gap_ts(str(item["start"])), _parse_gap_ts(str(item["end"])), str(item.get("reason", "allowed"))))
    return out


def _is_xauusd_offmarket_minute(minute_ns: int) -> bool:
    _, pd = _import_deps()
    ts = pd.Timestamp(int(minute_ns), unit="ns")
    wd = int(ts.weekday())
    if wd == 5:
        return True
    if wd == 6 and ts.hour < 22:
        return True
    if wd == 4 and ts.hour >= 22:
        return True
    return False


def _gap_allowed(start_ns: int, end_ns: int, allowed_gaps: list[tuple[int, int, str]], market_schedule: str) -> bool:
    for a, b, _reason in allowed_gaps:
        if start_ns >= a and end_ns <= b:
            return True
    if market_schedule == "xauusd_auto":
        # Regular weekend/off-market blocks are allowed; holidays remain warnings unless supplied via JSON.
        for minute_ns in range(int(start_ns), int(end_ns) + 1, NS_PER_MINUTE):
            if not _is_xauusd_offmarket_minute(minute_ns):
                return False
        return True
    return False


def _gap_summary(df, args: argparse.Namespace) -> dict[str, Any]:
    np, pd = _import_deps()
    if bool(args.no_gap_check) or df.empty:
        return {"missing_minutes_total": 0, "allowed_gap_count": 0, "unexpected_gap_count": 0, "unexpected_gaps": []}
    minutes = np.asarray(sorted(pd.Series(df["minute_ns"].unique()).astype("int64").tolist()), dtype=np.int64)
    if minutes.size <= 1:
        return {"missing_minutes_total": 0, "allowed_gap_count": 0, "unexpected_gap_count": 0, "unexpected_gaps": []}
    allowed_gaps = _load_allowed_gaps(args.allowed_gaps_json)
    missing_total = 0
    allowed_count = 0
    unexpected = []
    diffs = np.diff(minutes)
    gap_positions = np.flatnonzero(diffs > NS_PER_MINUTE)
    for pos in gap_positions.tolist():
        start = int(minutes[pos] + NS_PER_MINUTE)
        end = int(minutes[pos + 1] - NS_PER_MINUTE)
        missing = int((end - start) // NS_PER_MINUTE) + 1
        missing_total += missing
        if _gap_allowed(start, end, allowed_gaps, str(args.market_schedule)):
            allowed_count += 1
        else:
            unexpected.append({"start": _ns_to_iso(start), "end": _ns_to_iso(end), "duration_minutes": missing})
    unexpected = sorted(unexpected, key=lambda x: int(x["duration_minutes"]), reverse=True)
    return {
        "missing_minutes_total": int(missing_total),
        "allowed_gap_count": int(allowed_count),
        "unexpected_gap_count": int(len(unexpected)),
        "unexpected_gaps": unexpected[:10],
    }


def _overlap_conflicts(base, new) -> dict[str, Any]:
    _, pd = _import_deps()
    if base.empty or new.empty:
        return {"overlap_conflict_count": 0, "overlap_conflict_first": "", "overlap_conflict_last": ""}
    joined = pd.concat([base.assign(_src="base"), new.assign(_src="new")], ignore_index=True)
    dup_dt = joined[joined["datetime_ns"].duplicated(keep=False)]
    conflict_dts = []
    for dt, grp in dup_dt.groupby("datetime_ns", sort=False):
        if grp["_src"].nunique() < 2:
            continue
        distinct = grp[DEDUP_COLUMNS].drop_duplicates()
        if len(distinct) > 1:
            conflict_dts.append(int(dt))
    if not conflict_dts:
        return {"overlap_conflict_count": 0, "overlap_conflict_first": "", "overlap_conflict_last": ""}
    return {
        "overlap_conflict_count": int(len(conflict_dts)),
        "overlap_conflict_first": _ns_to_iso(min(conflict_dts)),
        "overlap_conflict_last": _ns_to_iso(max(conflict_dts)),
    }


def _atomic_write_parquet(path: Path, df, row_group_size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", delete=False, dir=str(path.parent), suffix=".parquet") as tf:
        tmp = Path(tf.name)
    try:
        kwargs = {"index": False}
        if int(row_group_size) > 0:
            kwargs["row_group_size"] = int(row_group_size)
        try:
            df.to_parquet(tmp, **kwargs)
        except TypeError:
            kwargs.pop("row_group_size", None)
            df.to_parquet(tmp, **kwargs)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except Exception:
                pass


def _summary(df, args: argparse.Namespace, extra: dict[str, Any]) -> dict[str, Any]:
    out = dict(extra)
    out["rows_total_out"] = int(len(df))
    out["first_datetime"] = _ns_to_iso(int(df["datetime_ns"].min())) if len(df) else ""
    out["last_datetime"] = _ns_to_iso(int(df["datetime_ns"].max())) if len(df) else ""
    out["unique_minutes"] = int(df["minute_ns"].nunique()) if len(df) else 0
    out.update(_gap_summary(df, args))
    return out


def _print_summary(summary: dict[str, Any]) -> None:
    for key in [
        "rows_total_in", "rows_total_out", "rows_dropped_invalid", "duplicate_rows_removed",
        "overlap_conflict_count", "overlap_conflict_first", "overlap_conflict_last",
        "first_datetime", "last_datetime", "unique_minutes", "missing_minutes_total",
        "allowed_gap_count", "unexpected_gap_count",
    ]:
        if key in summary and summary[key] != "":
            print(f"{key}={summary[key]}")
    for i, gap in enumerate(summary.get("unexpected_gaps", [])[:10], start=1):
        print(f"unexpected_gap_{i}=start={gap['start']} end={gap['end']} duration_minutes={gap['duration_minutes']}")


def _common_parser(p: argparse.ArgumentParser) -> None:
    p.add_argument("--datetime-column", default="")
    p.add_argument("--date-column", default="")
    p.add_argument("--time-column", default="")
    p.add_argument("--bid-column", default="")
    p.add_argument("--ask-column", default="")
    p.add_argument("--price-column", default="")
    p.add_argument("--volume-column", default="")
    p.add_argument("--sep", default="auto")
    p.add_argument("--has-header", choices=["auto", "yes", "no"], default="auto")
    p.add_argument("--no-gap-check", action="store_true", default=False)
    p.add_argument("--allowed-gaps-json", type=Path, default=None)
    p.add_argument("--market-schedule", choices=["xauusd_auto", "none"], default="xauusd_auto")
    p.add_argument("--strict-gaps", action="store_true", default=False)
    p.add_argument("--row-group-size", type=int, default=0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build, append and validate normalized tick Parquet files")
    sub = p.add_subparsers(dest="mode", required=True)
    b = sub.add_parser("build", help="Build normalized Parquet from CSV or Parquet input")
    b.add_argument("--input", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    _common_parser(b)
    a = sub.add_parser("append", help="Append/merge new CSV or Parquet ticks into a normalized Parquet")
    a.add_argument("--base", type=Path, required=True)
    a.add_argument("--input", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    _common_parser(a)
    v = sub.add_parser("validate", help="Validate a normalized tick Parquet")
    v.add_argument("--input", type=Path, required=True)
    _common_parser(v)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "build":
        df_raw = _read_input_frame(args.input, args)
        df, stats = _normalize_frame(df_raw, args)
        df, dup_removed = _sort_and_dedupe(df)
        summary = _summary(df, args, {**stats, "duplicate_rows_removed": int(dup_removed), "overlap_conflict_count": 0})
        _atomic_write_parquet(args.out, df, int(args.row_group_size))
        _print_summary(summary)
        if bool(args.strict_gaps) and int(summary.get("unexpected_gap_count", 0)) > 0:
            raise SystemExit(2)
        return

    if args.mode == "append":
        base_raw = _read_input_frame(args.base, args)
        new_raw = _read_input_frame(args.input, args)
        base, base_stats = _normalize_frame(base_raw, args)
        new, new_stats = _normalize_frame(new_raw, args, source_offset=len(base))
        _, pd = _import_deps()
        conflicts = _overlap_conflicts(base, new)
        merged, dup_removed = _sort_and_dedupe(pd.concat([base, new], ignore_index=True))
        stats = {
            "rows_total_in": int(base_stats["rows_total_in"] + new_stats["rows_total_in"]),
            "rows_dropped_invalid": int(base_stats["rows_dropped_invalid"] + new_stats["rows_dropped_invalid"]),
            "duplicate_rows_removed": int(dup_removed),
            **conflicts,
        }
        summary = _summary(merged, args, stats)
        _atomic_write_parquet(args.out, merged, int(args.row_group_size))
        _print_summary(summary)
        if int(conflicts.get("overlap_conflict_count", 0)) > 0:
            print("warning=overlap timestamp conflicts were kept; no silent overwrite was performed")
        if bool(args.strict_gaps) and int(summary.get("unexpected_gap_count", 0)) > 0:
            raise SystemExit(2)
        return

    if args.mode == "validate":
        df_raw = _read_input_frame(args.input, args)
        df, stats = _normalize_frame(df_raw, args)
        sorted_ok = bool(df["datetime_ns"].is_monotonic_increasing) if len(df) else True
        dup_count = int(df.duplicated(subset=DEDUP_COLUMNS).sum()) if len(df) else 0
        summary = _summary(df, args, {**stats, "duplicate_rows_removed": dup_count, "overlap_conflict_count": 0})
        summary["schema_ok"] = int(set(NORMALIZED_COLUMNS).issubset(set(str(c) for c in df_raw.columns)))
        summary["sorted_ok"] = int(sorted_ok)
        _print_summary(summary)
        print(f"schema_ok={summary['schema_ok']}")
        print(f"sorted_ok={summary['sorted_ok']}")
        if dup_count > 0:
            print(f"warning=duplicate full rows found count={dup_count}")
        if bool(args.strict_gaps) and int(summary.get("unexpected_gap_count", 0)) > 0:
            raise SystemExit(2)
        return


if __name__ == "__main__":
    main()
