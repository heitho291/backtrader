#!/usr/bin/env python3
"""Build, append and validate XAUUSD tick Parquet files with DateTime/Bid/Ask/Volume schema."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterator

NS_PER_MINUTE = 60_000_000_000
OUTPUT_COLUMNS = ["DateTime", "Bid", "Ask", "Volume"]
DEDUP_COLUMNS = ["DateTime", "Bid", "Ask", "Volume"]
HELPER_COLUMNS = ["_row_order", "_datetime_ns", "_minute_ns"]


def _import_deps():
    import numpy as np  # type: ignore
    import pandas as pd  # type: ignore

    return np, pd


def _require_pyarrow():
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on local optional dependency
        raise RuntimeError(
            "pyarrow is required for bucketed build/append parquet writing. "
            "Install pyarrow; the tool no longer falls back to RAM-heavy full-data concat."
        ) from exc
    return pa, pq


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


def _iter_parquet_chunks(path: Path, args: argparse.Namespace) -> Iterator[Any]:
    _pa, pq = _require_pyarrow()
    pf = pq.ParquetFile(path)
    batch_size = max(10_000, int(args.chunk_size))
    for batch in pf.iter_batches(batch_size=batch_size):
        yield batch.to_pandas()


def _iter_input_chunks(path: Path, args: argparse.Namespace) -> Iterator[Any]:
    if path.suffix.lower() in {".parquet", ".pq"}:
        yield from _iter_parquet_chunks(path, args)
    else:
        yield from _iter_csv_chunks(path, args)


def _resolve_columns(df, args: argparse.Namespace) -> tuple[Any, str, str, str]:
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
        time_col = _first_col(cols_lower, ["time"]) or ""
    if not bid_col:
        bid_col = _first_col(cols_lower, ["bid"]) or ""
    if not ask_col:
        ask_col = _first_col(cols_lower, ["ask"]) or ""
    if not vol_col:
        vol_col = _first_col(cols_lower, ["volume", "vol"]) or ""

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
    return dt_raw, bid_col, ask_col, vol_col


def _datetime_ns_for_boundaries(df, args: argparse.Namespace):
    np, pd = _import_deps()
    dt_raw, _bid_col, _ask_col, _vol_col = _resolve_columns(df, args)
    dt = pd.to_datetime(dt_raw, errors="coerce", utc=False)
    valid = dt.notna()
    if not bool(valid.any()):
        return np.asarray([], dtype=np.int64)
    return dt.loc[valid].astype("datetime64[ns]").astype("int64").to_numpy(dtype=np.int64)


def _normalize_chunk(df, args: argparse.Namespace, source_offset: int = 0):
    np, pd = _import_deps()
    rows_total_in = int(len(df))
    dt_raw, bid_col, ask_col, vol_col = _resolve_columns(df, args)

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
        "Volume": volume.loc[valid].astype("float64"),
    })
    out["_row_order"] = np.arange(source_offset, source_offset + len(out), dtype=np.int64)
    out["_datetime_ns"] = out["DateTime"].astype("int64")
    out["_minute_ns"] = (out["_datetime_ns"] // NS_PER_MINUTE) * NS_PER_MINUTE
    return out, {"rows_total_in": rows_total_in, "rows_dropped_invalid": dropped}


def _estimate_boundaries(inputs: list[Path], args: argparse.Namespace) -> list[int]:
    np, _pd = _import_deps()
    buckets = max(1, int(args.merge_buckets))
    sample_count = 0
    if buckets == 1:
        print("[tick-parquet] boundary pass done samples=0 boundaries=0", flush=True)
        return []
    samples: list[np.ndarray] = []
    max_per_chunk = max(100, int(args.boundary_sample_per_chunk))
    for path in inputs:
        for chunk in _iter_input_chunks(path, args):
            vals = _datetime_ns_for_boundaries(chunk, args)
            if vals.size == 0:
                continue
            if vals.size > max_per_chunk:
                take = np.linspace(0, vals.size - 1, max_per_chunk, dtype=np.int64)
                vals = np.sort(vals)[take]
            sample_count += int(vals.size)
            samples.append(vals.astype(np.int64, copy=False))
    if not samples:
        print("[tick-parquet] boundary pass done samples=0 boundaries=0", flush=True)
        return []
    all_samples = np.sort(np.concatenate(samples).astype(np.int64, copy=False))
    boundaries: list[int] = []
    for k in range(1, buckets):
        pos = min(len(all_samples) - 1, max(0, int(round(k * len(all_samples) / buckets))))
        boundaries.append(int(all_samples[pos]))
    # Strictly increasing boundaries keep searchsorted bucket assignment stable.
    out: list[int] = []
    last: int | None = None
    for b in boundaries:
        if last is None or b > last:
            out.append(b)
            last = b
    print(f"[tick-parquet] boundary pass done samples={sample_count} boundaries={len(out)}", flush=True)
    return out


def _write_temp_part(df, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df[OUTPUT_COLUMNS + HELPER_COLUMNS].to_parquet(path, index=False)


def _bucketize_inputs(inputs: list[Path], args: argparse.Namespace, temp_dir: Path, boundaries: list[int]):
    np, _pd = _import_deps()
    print("[tick-parquet] bucketize start", flush=True)
    bucket_count = max(1, int(args.merge_buckets))
    bucket_parts: list[list[Path]] = [[] for _ in range(bucket_count)]
    stats = {"rows_total_in": 0, "rows_dropped_invalid": 0}
    row_order = 0
    part_id = 0
    chunk_no = 0
    for path in inputs:
        for chunk in _iter_input_chunks(path, args):
            chunk_no += 1
            norm, st = _normalize_chunk(chunk, args, source_offset=row_order)
            row_order += len(norm)
            rows_in = int(st["rows_total_in"])
            rows_dropped = int(st["rows_dropped_invalid"])
            rows_valid = int(len(norm))
            stats["rows_total_in"] += rows_in
            stats["rows_dropped_invalid"] += rows_dropped
            print(
                f"[tick-parquet] bucketize chunk input={path} chunk_no={chunk_no} "
                f"rows_in={rows_in} rows_valid={rows_valid} rows_dropped={rows_dropped}",
                flush=True,
            )
            if norm.empty:
                continue
            bucket_ids = np.searchsorted(np.asarray(boundaries, dtype=np.int64), norm["_datetime_ns"].to_numpy(dtype=np.int64), side="right")
            norm["_bucket"] = bucket_ids.astype(np.int16, copy=False)
            for bid in sorted(set(int(x) for x in bucket_ids.tolist())):
                part = norm.loc[norm["_bucket"] == bid, OUTPUT_COLUMNS + HELPER_COLUMNS]
                if part.empty:
                    continue
                part_path = temp_dir / "buckets" / f"bucket_{bid:03d}" / f"part_{part_id:08d}.parquet"
                part_id += 1
                _write_temp_part(part, part_path)
                bucket_parts[bid].append(part_path)
    parts_total = sum(len(x) for x in bucket_parts)
    print(
        f"[tick-parquet] bucketize done parts={parts_total} rows_total_in={stats['rows_total_in']} "
        f"rows_dropped_invalid={stats['rows_dropped_invalid']}",
        flush=True,
    )
    return bucket_parts, stats


def _sort_dedupe_and_conflicts(df):
    _, pd = _import_deps()
    if df.empty:
        return df[OUTPUT_COLUMNS + HELPER_COLUMNS].copy(), 0, {"overlap_conflict_count": 0, "overlap_conflict_first": "", "overlap_conflict_last": ""}
    before = int(len(df))
    out = df.sort_values(["_datetime_ns", "_row_order"], kind="mergesort").drop_duplicates(subset=DEDUP_COLUMNS, keep="first").copy()

    conflict_count = 0
    conflict_first = ""
    conflict_last = ""
    duplicated_dt_mask = out["DateTime"].duplicated(keep=False)
    if bool(duplicated_dt_mask.any()):
        duplicated = out.loc[duplicated_dt_mask, DEDUP_COLUMNS]
        combo_counts = duplicated.groupby("DateTime", sort=False)[["Bid", "Ask", "Volume"]].nunique(dropna=False)
        conflict_mask = (combo_counts > 1).any(axis=1)
        if bool(conflict_mask.any()):
            conflict_index = combo_counts.index[conflict_mask]
            conflict_count = int(len(conflict_index))
            conflict_first = pd.Timestamp(conflict_index.min()).isoformat()
            conflict_last = pd.Timestamp(conflict_index.max()).isoformat()

    out = out.sort_values(["_datetime_ns", "_row_order"], kind="mergesort").reset_index(drop=True)
    out["_row_order"] = range(len(out))
    conflicts = {
        "overlap_conflict_count": int(conflict_count),
        "overlap_conflict_first": conflict_first,
        "overlap_conflict_last": conflict_last,
    }
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


class GapAggregator:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.allowed_gaps = _load_allowed_gaps(args.allowed_gaps_json)
        self.unique_minutes = 0
        self.missing_minutes_total = 0
        self.allowed_gap_count = 0
        self.unexpected: list[dict[str, Any]] = []
        self._prev_minute: int | None = None

    def update(self, minutes) -> None:
        np, _pd = _import_deps()
        if len(minutes) == 0:
            return
        arr = np.asarray(sorted(set(int(x) for x in minutes)), dtype=np.int64)
        if self._prev_minute is not None:
            arr = arr[arr > int(self._prev_minute)]
        if arr.size == 0:
            return
        self.unique_minutes += int(arr.size)
        if bool(self.args.no_gap_check):
            self._prev_minute = int(arr[-1])
            return
        if self._prev_minute is not None:
            self._record_gap_if_any(int(self._prev_minute), int(arr[0]))
        for pos in np.flatnonzero(np.diff(arr) > NS_PER_MINUTE).tolist():
            self._record_gap_if_any(int(arr[pos]), int(arr[pos + 1]))
        self._prev_minute = int(arr[-1])

    def _record_gap_if_any(self, left_minute: int, right_minute: int) -> None:
        if right_minute - left_minute <= NS_PER_MINUTE:
            return
        start = int(left_minute + NS_PER_MINUTE)
        end = int(right_minute - NS_PER_MINUTE)
        missing = int((end - start) // NS_PER_MINUTE) + 1
        self.missing_minutes_total += missing
        if _gap_allowed(start, end, self.allowed_gaps, str(self.args.market_schedule)):
            self.allowed_gap_count += 1
        else:
            self.unexpected.append({"start": _ns_to_iso(start), "end": _ns_to_iso(end), "duration_minutes": missing})

    def summary(self) -> dict[str, Any]:
        unexpected = sorted(self.unexpected, key=lambda x: int(x["duration_minutes"]), reverse=True)
        return {
            "unique_minutes": int(self.unique_minutes),
            "missing_minutes_total": int(self.missing_minutes_total),
            "allowed_gap_count": int(self.allowed_gap_count),
            "unexpected_gap_count": int(len(unexpected)),
            "unexpected_gaps": unexpected[:10],
        }


def _make_output_schema():
    pa, _pq = _require_pyarrow()
    return pa.schema([
        pa.field("DateTime", pa.timestamp("ns")),
        pa.field("Bid", pa.float64()),
        pa.field("Ask", pa.float64()),
        pa.field("Volume", pa.float64()),
    ])


def _frame_to_table(df):
    pa, _pq = _require_pyarrow()
    out_df = df[OUTPUT_COLUMNS].copy()
    out_df["Volume"] = out_df["Volume"].astype("float64")
    return pa.Table.from_pandas(out_df, schema=_make_output_schema(), preserve_index=False)


def _open_writer(path: Path, schema, compression: str = "snappy"):
    _pa, pq = _require_pyarrow()
    try:
        return pq.ParquetWriter(path, schema=schema, compression=compression)
    except Exception:
        return pq.ParquetWriter(path, schema=schema, compression=None)


def _write_table_with_row_groups(writer, table, row_group_size: int) -> None:
    if row_group_size > 0:
        for start in range(0, table.num_rows, row_group_size):
            writer.write_table(table.slice(start, row_group_size))
    else:
        writer.write_table(table)


def _process_one_bucket(bucket_index: int, bucket_count: int, parts: list[Path], args: argparse.Namespace, processed_dir: Path) -> dict[str, Any]:
    np, pd = _import_deps()
    out_path = processed_dir / f"processed_bucket_{bucket_index:03d}.parquet"
    print(f"[tick-parquet] bucket {bucket_index + 1}/{bucket_count} start parts={len(parts)}", flush=True)
    if parts:
        bucket_df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    else:
        bucket_df = pd.DataFrame(columns=OUTPUT_COLUMNS + HELPER_COLUMNS)
    rows_before = int(len(bucket_df))
    print(f"[tick-parquet] bucket {bucket_index + 1}/{bucket_count} loaded rows={rows_before}", flush=True)
    bucket_df, dups, conflicts = _sort_dedupe_and_conflicts(bucket_df)
    c_count = int(conflicts.get("overlap_conflict_count", 0))
    print(
        f"[tick-parquet] bucket {bucket_index + 1}/{bucket_count} dedupe "
        f"duplicate_rows_removed={int(dups)} conflicts={c_count}",
        flush=True,
    )
    minutes = bucket_df["_minute_ns"].drop_duplicates().to_numpy(dtype=np.int64) if len(bucket_df) else np.asarray([], dtype=np.int64)
    first_dt = bucket_df["DateTime"].min().isoformat() if len(bucket_df) else ""
    last_dt = bucket_df["DateTime"].max().isoformat() if len(bucket_df) else ""
    rows_after = int(len(bucket_df))
    if rows_after:
        _pa, pq = _require_pyarrow()
        pq.write_table(_frame_to_table(bucket_df), out_path, compression="snappy")
    print(f"[tick-parquet] bucket {bucket_index + 1}/{bucket_count} written rows={rows_after}", flush=True)
    return {
        "bucket_index": int(bucket_index),
        "processed_path": str(out_path) if rows_after else "",
        "rows_before": rows_before,
        "rows_after": rows_after,
        "duplicate_rows_removed": int(dups),
        "overlap_conflict_count": c_count,
        "overlap_conflict_first": str(conflicts.get("overlap_conflict_first", "")),
        "overlap_conflict_last": str(conflicts.get("overlap_conflict_last", "")),
        "first_datetime": first_dt,
        "last_datetime": last_dt,
        "minutes": minutes.tolist(),
    }


def _merge_bucket_result(summary_state: dict[str, Any], result: dict[str, Any], gapper: GapAggregator) -> None:
    summary_state["rows_out"] += int(result.get("rows_after", 0))
    summary_state["dup_removed"] += int(result.get("duplicate_rows_removed", 0))
    c_count = int(result.get("overlap_conflict_count", 0))
    if c_count:
        summary_state["conflict_count"] += c_count
        c_first = str(result.get("overlap_conflict_first", ""))
        c_last = str(result.get("overlap_conflict_last", ""))
        if c_first:
            summary_state["conflict_first"] = c_first if not summary_state["conflict_first"] else min(summary_state["conflict_first"], c_first)
        if c_last:
            summary_state["conflict_last"] = c_last if not summary_state["conflict_last"] else max(summary_state["conflict_last"], c_last)
    first_dt = str(result.get("first_datetime", ""))
    last_dt = str(result.get("last_datetime", ""))
    if first_dt and not summary_state["first_dt"]:
        summary_state["first_dt"] = first_dt
    if last_dt:
        summary_state["last_dt"] = last_dt
    gapper.update(result.get("minutes", []))


def _process_buckets_to_output(bucket_parts: list[list[Path]], args: argparse.Namespace, out_path: Path, stats: dict[str, int], temp_dir: Path) -> dict[str, Any]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", delete=False, dir=str(out_path.parent), suffix=".parquet") as tf:
        tmp_out = Path(tf.name)
    writer = None
    gapper = GapAggregator(args)
    state: dict[str, Any] = {
        "rows_out": 0,
        "dup_removed": 0,
        "conflict_count": 0,
        "conflict_first": "",
        "conflict_last": "",
        "first_dt": "",
        "last_dt": "",
    }
    bucket_count = len(bucket_parts)
    processed_dir = temp_dir / "processed_buckets"
    processed_dir.mkdir(parents=True, exist_ok=True)
    workers = max(1, int(args.workers))
    print(f"[tick-parquet] process buckets start buckets={bucket_count}", flush=True)
    try:
        schema = _make_output_schema()
        writer = _open_writer(tmp_out, schema=schema, compression="snappy")
        if workers == 1:
            for bucket_index, parts in enumerate(bucket_parts):
                result = _process_one_bucket(bucket_index, bucket_count, parts, args, processed_dir)
                _merge_bucket_result(state, result, gapper)
                processed_path = str(result.get("processed_path", ""))
                if processed_path:
                    table = _require_pyarrow()[1].read_table(processed_path)
                    _write_table_with_row_groups(writer, table, int(args.row_group_size))
        else:
            results: dict[int, dict[str, Any]] = {}
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
                future_map = {
                    executor.submit(_process_one_bucket, bucket_index, bucket_count, parts, args, processed_dir): bucket_index
                    for bucket_index, parts in enumerate(bucket_parts)
                }
                for future in concurrent.futures.as_completed(future_map):
                    result = future.result()
                    results[int(result["bucket_index"])] = result
            for bucket_index in range(bucket_count):
                result = results[bucket_index]
                _merge_bucket_result(state, result, gapper)
                processed_path = str(result.get("processed_path", ""))
                if processed_path:
                    table = _require_pyarrow()[1].read_table(processed_path)
                    _write_table_with_row_groups(writer, table, int(args.row_group_size))
        if writer is not None:
            writer.close()
            writer = None
        os.replace(tmp_out, out_path)
    finally:
        if writer is not None:
            writer.close()
        if tmp_out.exists():
            try:
                tmp_out.unlink()
            except Exception:
                pass
    print(f"[tick-parquet] process buckets done rows_out={state['rows_out']}", flush=True)
    summary = {
        **stats,
        "rows_total_out": int(state["rows_out"]),
        "duplicate_rows_removed": int(state["dup_removed"]),
        "overlap_conflict_count": int(state["conflict_count"]),
        "overlap_conflict_first": state["conflict_first"],
        "overlap_conflict_last": state["conflict_last"],
        "first_datetime": state["first_dt"],
        "last_datetime": state["last_dt"],
    }
    summary.update(gapper.summary())
    return summary


def _validate_summary(df, args: argparse.Namespace, schema_ok: bool) -> dict[str, Any]:
    sorted_ok = bool(df["_datetime_ns"].is_monotonic_increasing) if len(df) else True
    first_datetime = df["DateTime"].min().isoformat() if len(df) else ""
    last_datetime = df["DateTime"].max().isoformat() if len(df) else ""
    deduped, dup_removed, conflicts = _sort_dedupe_and_conflicts(df)
    duplicate_rows = int(dup_removed)
    gapper = GapAggregator(args)
    if len(deduped):
        gapper.update(deduped["_minute_ns"].drop_duplicates().to_numpy())
    summary = {
        "rows_total_in": int(len(df)),
        "rows_total_out": int(len(deduped)),
        "rows_dropped_invalid": 0,
        "duplicate_rows_removed": duplicate_rows,
        **conflicts,
        "first_datetime": first_datetime,
        "last_datetime": last_datetime,
        "schema_ok": int(schema_ok),
        "sorted_ok": int(sorted_ok),
    }
    summary.update(gapper.summary())
    return summary


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
    p.add_argument("--merge-buckets", type=int, default=20)
    p.add_argument("--boundary-sample-per-chunk", type=int, default=10_000)
    p.add_argument("--workers", type=int, default=1)


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


def _run_bucketed(inputs: list[Path], out: Path, args: argparse.Namespace) -> dict[str, Any]:
    if int(args.merge_buckets) < 1:
        raise ValueError("--merge-buckets must be >= 1")
    if int(args.workers) < 1:
        raise ValueError("--workers must be >= 1")
    _require_pyarrow()
    temp_dir = Path(tempfile.mkdtemp(prefix="tick_parquet_tool_"))
    try:
        print("[tick-parquet] boundary pass start", flush=True)
        boundaries = _estimate_boundaries(inputs, args)
        bucket_parts, stats = _bucketize_inputs(inputs, args, temp_dir, boundaries)
        return _process_buckets_to_output(bucket_parts, args, out, stats, temp_dir)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main() -> None:
    args = parse_args()
    if args.mode == "build":
        summary = _run_bucketed([args.input], args.out, args)
        _print_summary(summary)
        if int(summary.get("overlap_conflict_count", 0)) > 0:
            print("warning=same DateTime with differing Bid/Ask/Volume was kept; no silent overwrite was performed")
        if bool(args.strict_gaps) and int(summary.get("unexpected_gap_count", 0)) > 0:
            raise SystemExit(2)
        return

    if args.mode == "append":
        summary = _run_bucketed([args.base, args.input], args.out, args)
        _print_summary(summary)
        if int(summary.get("overlap_conflict_count", 0)) > 0:
            print("warning=same DateTime with differing Bid/Ask/Volume was kept; no silent overwrite was performed")
        if bool(args.strict_gaps) and int(summary.get("unexpected_gap_count", 0)) > 0:
            raise SystemExit(2)
        return

    if args.mode == "validate":
        _np, pd = _import_deps()
        raw = pd.read_parquet(args.input)
        schema_ok = list(raw.columns) == OUTPUT_COLUMNS
        if not schema_ok:
            print(f"schema_ok=0 expected_columns={','.join(OUTPUT_COLUMNS)} actual_columns={','.join(map(str, raw.columns))}")
        df, stats = _normalize_chunk(raw, args)
        summary = _validate_summary(df, args, schema_ok=schema_ok)
        summary["rows_total_in"] = int(stats["rows_total_in"])
        summary["rows_dropped_invalid"] = int(stats["rows_dropped_invalid"])
        _print_summary(summary)
        print(f"schema_ok={int(schema_ok)}")
        print(f"sorted_ok={int(summary.get('sorted_ok', 0))}")
        if int(summary.get("duplicate_rows_removed", 0)) > 0:
            print(f"warning=duplicate full rows found count={summary['duplicate_rows_removed']}")
        if int(summary.get("overlap_conflict_count", 0)) > 0:
            print("warning=same DateTime with differing Bid/Ask/Volume found")
        if bool(args.strict_gaps) and int(summary.get("unexpected_gap_count", 0)) > 0:
            raise SystemExit(2)
        return


if __name__ == "__main__":
    main()
