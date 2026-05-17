#!/usr/bin/env python3
"""Build, append and validate XAUUSD tick Parquet files with DateTime/Bid/Ask/Volume schema."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterator

NS_PER_MINUTE = 60_000_000_000
OUTPUT_COLUMNS = ["DateTime", "Bid", "Ask", "Volume"]
DEDUP_COLUMNS = ["DateTime", "Bid", "Ask", "Volume"]


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


def _csv_read_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    sep = None if str(args.sep).lower() == "auto" else str(args.sep)
    return {"sep": sep, "engine": "python"} if sep is None else {"sep": sep}


def _csv_has_header(path: Path, args: argparse.Namespace) -> bool:
    mode = str(args.has_header).lower()
    if mode != "auto":
        return mode == "yes"
    sample = path.read_text(encoding="utf-8", errors="ignore").splitlines()[:1]
    first = sample[0] if sample else ""
    return any(ch.isalpha() for ch in first)


def _iter_csv_chunks(path: Path, args: argparse.Namespace) -> Iterator[Any]:
    _, pd = _import_deps()
    kwargs = _csv_read_kwargs(args)
    chunksize = max(10_000, int(args.chunk_size))
    if _csv_has_header(path, args):
        yield from pd.read_csv(path, chunksize=chunksize, **kwargs)
        return
    for ch in pd.read_csv(path, header=None, chunksize=chunksize, **kwargs):
        if ch.shape[1] >= 5:
            ch = ch.iloc[:, :5].copy()
            ch.columns = ["Date", "Time", "Ask", "Bid", "Volume"]
        elif ch.shape[1] >= 4:
            ch = ch.iloc[:, :4].copy()
            ch.columns = ["DateTime", "Bid", "Ask", "Volume"]
        else:
            raise ValueError("Headerless CSV needs at least DateTime,Bid,Ask,Volume columns")
        yield ch


def _iter_input_chunks(path: Path, args: argparse.Namespace) -> Iterator[Any]:
    _, pd = _import_deps()
    if path.suffix.lower() in {".parquet", ".pq"}:
        yield pd.read_parquet(path)
    else:
        yield from _iter_csv_chunks(path, args)


def _normalize_chunk(df, args: argparse.Namespace, source_offset: int = 0):
    np, pd = _import_deps()
    rows_total_in = int(len(df))
    cols_lower = _lower_col_map(df.columns)

    dt_col = str(args.datetime_column).strip() if args.datetime_column else ""
    date_col = str(args.date_column).strip() if args.date_column else ""
    time_col = str(args.time_column).strip() if args.time_column else ""
    bid_col = str(args.bid_column).strip() if args.bid_column else ""
    ask_col = str(args.ask_column).strip() if args.ask_column else ""
    vol_col = str(args.volume_column).strip() if args.volume_column else ""

    if not dt_col:
        dt_col = _first_col(cols_lower, ["datetime", "date_time", "timestamp", "time", "date"]) or ""
    if (not date_col) and "date" in cols_lower and (not dt_col or dt_col == cols_lower.get("time")):
        date_col = cols_lower["date"]
    if not time_col:
        time_col = _first_col(cols_lower, ["time"] ) or ""
    if not bid_col:
        bid_col = _first_col(cols_lower, ["bid", "Bid", "BID"]) or ""
    if not ask_col:
        ask_col = _first_col(cols_lower, ["ask", "Ask", "ASK"]) or ""
    if not vol_col:
        vol_col = _first_col(cols_lower, ["volume", "vol", "Volume"]) or ""

    if date_col and time_col and date_col in df.columns and time_col in df.columns:
        dt_raw = df[date_col].astype(str).str.strip() + " " + df[time_col].astype(str).str.strip()
    elif dt_col and dt_col in df.columns:
        dt_raw = df[dt_col]
    else:
        raise ValueError("Input requires DateTime or date/time columns")
    if bid_col not in df.columns or ask_col not in df.columns:
        raise ValueError("Input requires Bid and Ask columns")
    if vol_col not in df.columns:
        raise ValueError("Input requires Volume column")

    dt = pd.to_datetime(dt_raw, errors="coerce", utc=False)
    bid = pd.to_numeric(df[bid_col], errors="coerce")
    ask = pd.to_numeric(df[ask_col], errors="coerce")
    volume = pd.to_numeric(df[vol_col], errors="coerce")
    valid = dt.notna() & np.isfinite(bid.to_numpy(dtype=np.float64)) & np.isfinite(ask.to_numpy(dtype=np.float64)) & (bid > 0) & (ask > 0)
    dropped = int(rows_total_in - int(valid.sum()))
    out = pd.DataFrame({
        "DateTime": dt.loc[valid].astype("datetime64[ns]"),
        "Bid": bid.loc[valid].astype("float64"),
        "Ask": ask.loc[valid].astype("float64"),
        "Volume": volume.loc[valid],
    })
    out["_row_order"] = np.arange(source_offset, source_offset + len(out), dtype=np.int64)
    out["_datetime_ns"] = out["DateTime"].astype("int64")
    out["_minute_ns"] = (out["_datetime_ns"] // NS_PER_MINUTE) * NS_PER_MINUTE
    return out, {"rows_total_in": rows_total_in, "rows_dropped_invalid": dropped}


def _write_temp_chunk(df, temp_dir: Path, idx: int) -> Path:
    path = temp_dir / f"chunk_{idx:06d}.parquet"
    df.to_parquet(path, index=False)
    return path


def _load_normalized_via_chunks(path: Path, args: argparse.Namespace, temp_dir: Path, start_order: int = 0):
    _, pd = _import_deps()
    paths: list[Path] = []
    stats = {"rows_total_in": 0, "rows_dropped_invalid": 0}
    row_order = int(start_order)
    for idx, chunk in enumerate(_iter_input_chunks(path, args)):
        norm, st = _normalize_chunk(chunk, args, source_offset=row_order)
        row_order += len(norm)
        stats["rows_total_in"] += int(st["rows_total_in"])
        stats["rows_dropped_invalid"] += int(st["rows_dropped_invalid"])
        if len(norm):
            paths.append(_write_temp_chunk(norm, temp_dir, len(paths)))
    if not paths:
        empty = pd.DataFrame(columns=OUTPUT_COLUMNS + ["_row_order", "_datetime_ns", "_minute_ns"])
        return empty, stats
    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    return df, stats


def _sort_dedupe_and_conflicts(df):
    _, pd = _import_deps()
    if df.empty:
        return df[OUTPUT_COLUMNS + ["_row_order", "_datetime_ns", "_minute_ns"]].copy(), 0, {"overlap_conflict_count": 0, "overlap_conflict_first": "", "overlap_conflict_last": ""}
    conflict_dts = []
    for dt, grp in df.groupby("DateTime", sort=False):
        if len(grp[DEDUP_COLUMNS].drop_duplicates()) > 1:
            conflict_dts.append(dt)
    before = int(len(df))
    out = df.sort_values(["_datetime_ns", "_row_order"], kind="mergesort").drop_duplicates(subset=DEDUP_COLUMNS, keep="first").copy()
    out = out.sort_values(["_datetime_ns", "_row_order"], kind="mergesort").reset_index(drop=True)
    out["_row_order"] = range(len(out))
    conflicts = {"overlap_conflict_count": int(len(conflict_dts)), "overlap_conflict_first": "", "overlap_conflict_last": ""}
    if conflict_dts:
        conflicts["overlap_conflict_first"] = pd.Timestamp(min(conflict_dts)).isoformat()
        conflicts["overlap_conflict_last"] = pd.Timestamp(max(conflict_dts)).isoformat()
    return out, before - int(len(out)), conflicts


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
    return [(_parse_gap_ts(str(x["start"])), _parse_gap_ts(str(x["end"])), str(x.get("reason", "allowed"))) for x in raw.get("allowed_gaps", [])]


def _is_xauusd_offmarket_minute(minute_ns: int) -> bool:
    _, pd = _import_deps()
    ts = pd.Timestamp(int(minute_ns), unit="ns")
    wd = int(ts.weekday())
    if wd == 5:
        return True
    if wd == 6:
        return True
    if wd == 4 and ts.hour >= 20:
        return True
    if wd == 0 and ts.hour < 1:
        return True
    return False


def _gap_allowed(start_ns: int, end_ns: int, allowed_gaps: list[tuple[int, int, str]], market_schedule: str) -> bool:
    for a, b, _reason in allowed_gaps:
        if start_ns >= a and end_ns <= b:
            return True
    if market_schedule == "xauusd_auto":
        # Lenient weekend/off-market allowance; holidays must be supplied via allowed_gaps_json.
        for minute_ns in range(int(start_ns), int(end_ns) + 1, NS_PER_MINUTE):
            if not _is_xauusd_offmarket_minute(minute_ns):
                return False
        return True
    return False


def _gap_summary(df, args: argparse.Namespace) -> dict[str, Any]:
    np, pd = _import_deps()
    if bool(args.no_gap_check) or df.empty:
        return {"missing_minutes_total": 0, "allowed_gap_count": 0, "unexpected_gap_count": 0, "unexpected_gaps": []}
    minutes = np.asarray(sorted(pd.Series(df["_minute_ns"].unique()).astype("int64").tolist()), dtype=np.int64)
    if minutes.size <= 1:
        return {"missing_minutes_total": 0, "allowed_gap_count": 0, "unexpected_gap_count": 0, "unexpected_gaps": []}
    allowed_gaps = _load_allowed_gaps(args.allowed_gaps_json)
    missing_total = 0
    allowed_count = 0
    unexpected = []
    for pos in np.flatnonzero(np.diff(minutes) > NS_PER_MINUTE).tolist():
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


def _write_parquet(path: Path, df, row_group_size: int, atomic: bool = True) -> None:
    out_df = df[OUTPUT_COLUMNS].copy()
    path.parent.mkdir(parents=True, exist_ok=True)
    dest = path
    tmp = None
    if atomic:
        with tempfile.NamedTemporaryFile("wb", delete=False, dir=str(path.parent), suffix=".parquet") as tf:
            tmp = Path(tf.name)
        dest = tmp
    try:
        kwargs: dict[str, Any] = {"index": False, "compression": "snappy"}
        if int(row_group_size) > 0:
            kwargs["row_group_size"] = int(row_group_size)
        try:
            out_df.to_parquet(dest, **kwargs)
        except TypeError:
            kwargs.pop("row_group_size", None)
            out_df.to_parquet(dest, **kwargs)
        except Exception:
            kwargs["compression"] = None
            out_df.to_parquet(dest, **kwargs)
        if atomic and tmp is not None:
            os.replace(tmp, path)
    finally:
        if tmp is not None and tmp.exists():
            try:
                tmp.unlink()
            except Exception:
                pass


def _summary(df, args: argparse.Namespace, extra: dict[str, Any]) -> dict[str, Any]:
    out = dict(extra)
    out["rows_total_out"] = int(len(df))
    out["first_datetime"] = df["DateTime"].min().isoformat() if len(df) else ""
    out["last_datetime"] = df["DateTime"].max().isoformat() if len(df) else ""
    out["unique_minutes"] = int(df["_minute_ns"].nunique()) if len(df) else 0
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
    p.add_argument("--volume-column", default="")
    p.add_argument("--sep", default="auto")
    p.add_argument("--has-header", choices=["auto", "yes", "no"], default="auto")
    p.add_argument("--no-gap-check", action="store_true", default=False)
    p.add_argument("--allowed-gaps-json", type=Path, default=None)
    p.add_argument("--market-schedule", choices=["xauusd_auto", "none"], default="xauusd_auto")
    p.add_argument("--strict-gaps", action="store_true", default=False)
    p.add_argument("--row-group-size", type=int, default=0)
    p.add_argument("--chunk-size", type=int, default=500_000)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build, append and validate DateTime/Bid/Ask/Volume tick Parquet files")
    sub = p.add_subparsers(dest="mode", required=True)
    b = sub.add_parser("build", help="Build tick Parquet from CSV or Parquet input")
    b.add_argument("--input", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    _common_parser(b)
    a = sub.add_parser("append", help="Append/merge new CSV or Parquet ticks into a tick Parquet")
    a.add_argument("--base", type=Path, required=True)
    a.add_argument("--input", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    _common_parser(a)
    v = sub.add_parser("validate", help="Validate a tick Parquet")
    v.add_argument("--input", type=Path, required=True)
    _common_parser(v)
    return p.parse_args()


def _load_for_mode(path: Path, args: argparse.Namespace, temp_dir: Path, start_order: int = 0):
    return _load_normalized_via_chunks(path, args, temp_dir=temp_dir, start_order=start_order)


def main() -> None:
    args = parse_args()
    temp_dir = Path(tempfile.mkdtemp(prefix="tick_parquet_tool_"))
    try:
        if args.mode == "build":
            df, stats = _load_for_mode(args.input, args, temp_dir)
            df, dup_removed, conflicts = _sort_dedupe_and_conflicts(df)
            summary = _summary(df, args, {**stats, "duplicate_rows_removed": int(dup_removed), **conflicts})
            _write_parquet(args.out, df, int(args.row_group_size), atomic=True)
            _print_summary(summary)
            if int(conflicts.get("overlap_conflict_count", 0)) > 0:
                print("warning=same DateTime with differing Bid/Ask/Volume was kept; no silent overwrite was performed")
            if bool(args.strict_gaps) and int(summary.get("unexpected_gap_count", 0)) > 0:
                raise SystemExit(2)
            return

        if args.mode == "append":
            base, base_stats = _load_for_mode(args.base, args, temp_dir)
            new, new_stats = _load_for_mode(args.input, args, temp_dir, start_order=len(base))
            _, pd = _import_deps()
            merged = pd.concat([base, new], ignore_index=True)
            merged, dup_removed, conflicts = _sort_dedupe_and_conflicts(merged)
            stats = {
                "rows_total_in": int(base_stats["rows_total_in"] + new_stats["rows_total_in"]),
                "rows_dropped_invalid": int(base_stats["rows_dropped_invalid"] + new_stats["rows_dropped_invalid"]),
                "duplicate_rows_removed": int(dup_removed),
                **conflicts,
            }
            summary = _summary(merged, args, stats)
            _write_parquet(args.out, merged, int(args.row_group_size), atomic=True)
            _print_summary(summary)
            if int(conflicts.get("overlap_conflict_count", 0)) > 0:
                print("warning=same DateTime with differing Bid/Ask/Volume was kept; no silent overwrite was performed")
            if bool(args.strict_gaps) and int(summary.get("unexpected_gap_count", 0)) > 0:
                raise SystemExit(2)
            return

        if args.mode == "validate":
            _, pd = _import_deps()
            raw = pd.read_parquet(args.input)
            schema_ok = list(raw.columns) == OUTPUT_COLUMNS
            if not schema_ok:
                print(f"schema_ok=0 expected_columns={','.join(OUTPUT_COLUMNS)} actual_columns={','.join(map(str, raw.columns))}")
            df, stats = _normalize_chunk(raw, args)
            sorted_ok = bool(df["_datetime_ns"].is_monotonic_increasing) if len(df) else True
            duplicate_rows = int(df.duplicated(subset=DEDUP_COLUMNS).sum()) if len(df) else 0
            _sorted, _dup_removed, conflicts = _sort_dedupe_and_conflicts(df)
            summary = _summary(df, args, {**stats, "duplicate_rows_removed": duplicate_rows, **conflicts})
            _print_summary(summary)
            print(f"schema_ok={int(schema_ok)}")
            print(f"sorted_ok={int(sorted_ok)}")
            if duplicate_rows > 0:
                print(f"warning=duplicate full rows found count={duplicate_rows}")
            if int(conflicts.get("overlap_conflict_count", 0)) > 0:
                print("warning=same DateTime with differing Bid/Ask/Volume found")
            if bool(args.strict_gaps) and int(summary.get("unexpected_gap_count", 0)) > 0:
                raise SystemExit(2)
            return
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
