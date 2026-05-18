#!/usr/bin/env python3
"""Standalone in-detail prefilter combinatorial search.

Builds labels via TP/SL/Hold/Trail simulation directly from features OHLC,
then searches complete rule combinations (no greedy extension) over unlocked pools.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import collections
import hashlib
import importlib.util
import itertools
import json
import math
import os
import random
import tempfile
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


def _load_miner_module(path: Path):
    spec = importlib.util.spec_from_file_location("xau_miner_module_for_prefilter", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load miner script: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Standalone in-detail co-occurrence combinatorial search")
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--binned-features", type=Path, required=True)
    p.add_argument("--binned-metadata", type=Path, required=True)
    p.add_argument("--miner-script", type=Path, default=Path("tools/xauusd_miner_ohlc_first_hit_pessimistic.py"))

    p.add_argument("--train-frac", type=float, default=0.7)
    p.add_argument("--wf-folds", type=int, default=4)
    p.add_argument("--quantiles", type=str, default="0.05,0.10,0.90,0.95")

    p.add_argument("--tps", type=str, default="0.0015,0.0025,0.0035,0.0045")
    p.add_argument("--tp-weights", type=str, default="")
    p.add_argument("--use-multi-tp", action="store_true", default=True)
    p.add_argument("--no-use-multi-tp", dest="use_multi_tp", action="store_false")
    p.add_argument("--no-tp", action="store_true", default=False, help="Disable TP exits (allowed only with --trail)")
    p.add_argument("--sl", type=float, default=0.0015)
    p.add_argument("--hold", type=int, default=90)
    p.add_argument("--slippage-bps", type=float, default=0.0)
    p.add_argument("--spread-bps", type=float, default=0.0)
    p.add_argument("--trail", action="store_true", default=True)
    p.add_argument("--no-trail", dest="trail", action="store_false")
    p.add_argument("--trail-activate", type=float, default=0.0010)
    p.add_argument("--trail-offset", type=float, default=0.0006)
    p.add_argument("--trail-factor", type=float, default=0.5)
    p.add_argument("--trail-min-level", type=float, default=0.0)
    p.add_argument("--include-unrealized-at-test-end", action="store_true", default=True)
    p.add_argument("--no-include-unrealized-at-test-end", dest="include_unrealized_at_test_end", action="store_false")
    p.add_argument("--cluster-gap-minutes", type=int, default=1)
    p.add_argument("--max-entries-per-cluster", type=int, default=15)
    p.add_argument("--max-open-trades", type=int, default=15)
    p.add_argument("--cluster-only-lower-entry", dest="cluster_only_lower_entry", action="store_true", default=True)
    p.add_argument("--no-cluster-only-lower-entry", dest="cluster_only_lower_entry", action="store_false")

    p.add_argument("--tick-data", type=Path, default=None)
    p.add_argument("--tick-cache-parquet", type=Path, default=None)
    p.add_argument("--tick-entry-cache-npz", type=Path, default=None)
    p.add_argument("--tick-datetime-column", type=str, default="datetime")
    p.add_argument("--tick-price-column", type=str, default="auto")
    p.add_argument("--tick-sep", type=str, default=",")
    p.add_argument("--tick-refine-scope", type=str, default="fam_top", choices=["fam_top", "filtered", "all_coarse"])
    p.add_argument("--tick-chunk-size", type=str, default="auto")
    p.add_argument("--debug-tick-cache-scope-counts", action="store_true", default=False)
    p.add_argument("--tick-minute-load-mode", type=str, default="critical", choices=["critical", "full_window"])

    p.add_argument("--step-size", type=int, default=5)
    p.add_argument("--top-paths", type=int, default=24)
    p.add_argument("--max-path-conds", type=int, default=8)
    p.add_argument("--workers", type=str, default="auto")
    p.add_argument("--batch-size", type=int, default=50000)
    p.add_argument("--memory-soft-limit-gb", type=float, default=20.0)
    p.add_argument("--max-valids", type=int, default=1000)
    p.add_argument("--early-stop-top-k", type=int, default=100)
    p.add_argument("--early-stop-window-combos", type=int, default=150000)
    p.add_argument("--early-stop-avg-improve-pct", type=float, default=0.25)
    p.add_argument("--batch-random-seed", type=int, default=42)
    p.add_argument("--debug-reject-stats", action="store_true", default=False)
    p.add_argument("--debug-timing-breakdown", action="store_true", default=False)
    p.add_argument("--debug-atr-candidates", action="store_true", default=False)
    p.add_argument("--label-cache-npz", type=Path, default=None)
    p.add_argument("--min-single-pos-hits", type=int, default=2)
    p.add_argument("--min-single-lift", type=float, default=1.01)
    p.add_argument("--max-single-mask-count", type=int, default=0)
    p.add_argument("--family-top-n", type=int, default=40)
    p.add_argument("--family-split-delta-window", action="store_true", default=False)
    p.add_argument("--include-candidates-file", type=Path, default=None)
    p.add_argument("--out-candidates-coarse-csv", type=Path, default=None)
    p.add_argument("--out-candidates-refined-csv", type=Path, default=None)
    p.add_argument("--out-candidates-csv", type=Path, default=None, help="Deprecated (forbidden): use coarse/refined candidate CSV paths")

    p.add_argument("--min-pos-per-week", type=float, default=1.0)
    p.add_argument("--min-main-score", type=float, default=1.0)
    p.add_argument("--binary-cap-per-block", type=int, default=6)
    p.add_argument("--binary-cap-per-list-block", type=int, default=2)
    p.add_argument("--binary-anchor-lookahead-blocks", type=int, default=2)

    p.add_argument("--out-rules-json", type=Path, default=Path("prefilter_in_detail_rules.json"))
    p.add_argument("--out-rules-csv", type=Path, default=Path("prefilter_in_detail_rules.csv"))
    p.add_argument("--control-file", type=Path, default=None)
    p.add_argument("--phase-c-beam-width", type=int, default=64)
    p.add_argument("--phase-c-add-min", type=int, default=1)
    p.add_argument("--phase-c-add-max", type=int, default=2)
    p.add_argument("--phase-c-max-conds", type=int, default=9)
    p.add_argument("--phase-c-max-generated-per-level", type=int, default=0)
    p.add_argument("--phase-d-beam-width", type=int, default=64)
    p.add_argument("--phase-d-start-conds", type=int, default=2)
    p.add_argument("--phase-d-add-min", type=int, default=1)
    p.add_argument("--phase-d-add-max", type=int, default=2)
    p.add_argument("--phase-d-max-conds", type=int, default=9)
    p.add_argument("--phase-d-max-generated-per-level", type=int, default=0)
    return p.parse_args()


def _batched(iterable: Iterable[Tuple[int, ...]], n: int):
    buf = []
    for x in iterable:
        buf.append(x)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf


def _chunk_list(xs: list, n: int) -> list[list]:
    return [xs[i:i + n] for i in range(0, len(xs), n)]


def _next_batch(it: Iterable[Tuple[int, ...]], n: int) -> list[Tuple[int, ...]]:
    return list(itertools.islice(it, max(1, int(n))))


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(path.parent), encoding="utf-8") as tf:
        tf.write(text)
        tmp = Path(tf.name)
    os.replace(tmp, path)


def _atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(path.parent), encoding="utf-8", newline="") as tf:
        frame.to_csv(tf.name, index=False)
        tmp = Path(tf.name)
    os.replace(tmp, path)




def _ordered_frame(rows: list[dict], preferred_cols: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    extra = [c for c in frame.columns if c not in preferred_cols]
    return frame[[c for c in preferred_cols if c in frame.columns] + extra]


def _atomic_write_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", delete=False, dir=str(path.parent), suffix=".npz") as tf:
        tmp = Path(tf.name)
    try:
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except Exception:
                pass


def _load_tick_entry_cache(path: Path | None, cache_key: str) -> dict[int, dict] | None:
    if path is None:
        return None
    if not path.exists():
        return {}
    try:
        with np.load(path, allow_pickle=False) as z:
            stored = str(np.asarray(z["cache_key"]).item()) if "cache_key" in z.files else ""
            if stored != str(cache_key):
                print("[prefilter-tick-cache] cache context mismatch; ignoring old cache.")
                return {}
            entries = np.asarray(z["entry_idx"], dtype=np.int64)
            tick_y = np.asarray(z["tick_y"])
            tick_pnl = np.asarray(z["tick_pnl"])
            tick_t_exit = np.asarray(z["tick_t_exit"])
            tick_t_qual = np.asarray(z["tick_t_qual"])
            tick_tp_hits = np.asarray(z["tick_tp_hits"])
        out: dict[int, dict] = {}
        for pos, idx in enumerate(entries.tolist()):
            out[int(idx)] = {
                "y": int(tick_y[pos]),
                "pnl": float(tick_pnl[pos]),
                "t_exit": int(tick_t_exit[pos]),
                "t_qual": int(tick_t_qual[pos]),
                "tp_hits": int(tick_tp_hits[pos]),
            }
        print(f"[prefilter-tick-cache] loaded entries={len(out)} from {path}")
        return out
    except Exception as e:
        print(f"[prefilter-tick-cache] failed to load cache; ignoring old cache: {e}")
        return {}


def _write_tick_entry_cache(path: Path | None, cache_key: str, cache: dict[int, dict]) -> None:
    if path is None:
        return
    idxs = np.asarray(sorted(int(k) for k in cache.keys()), dtype=np.int64)
    def arr(name: str, dtype, default):
        return np.asarray([cache[int(i)].get(name, default) for i in idxs.tolist()], dtype=dtype)
    _atomic_write_npz(
        path,
        cache_key=np.asarray(str(cache_key)),
        entry_idx=idxs,
        tick_y=arr("y", np.int8, -1),
        tick_pnl=arr("pnl", np.float32, np.nan),
        tick_t_exit=arr("t_exit", np.int32, -1),
        tick_t_qual=arr("t_qual", np.int32, -1),
        tick_tp_hits=arr("tp_hits", np.int8, 0),
    )
    print(f"[prefilter-tick-cache] wrote entries={len(idxs)} to {path}")

def _write_control_none(control_file: Path | None) -> None:
    if control_file is None:
        return
    _atomic_write_text(control_file, "none\n")


def _read_control_command(control_file: Path | None) -> str:
    if control_file is None:
        return "none"
    try:
        txt = control_file.read_text(encoding="utf-8").strip().lower()
    except Exception:
        return "none"
    return txt or "none"


def _iter_combinations_with_new(idxs: list[int], r: int, new_start: int):
    if r < 1 or r > len(idxs):
        return
    old = [x for x in idxs if x < new_start]
    new = [x for x in idxs if x >= new_start]
    if not new:
        return
    for k_new in range(1, min(r, len(new)) + 1):
        k_old = r - k_new
        if k_old > len(old):
            continue
        for n_part in itertools.combinations(new, k_new):
            if k_old == 0:
                yield tuple(sorted(n_part))
            else:
                for o_part in itertools.combinations(old, k_old):
                    yield tuple(sorted(o_part + n_part))


def _iter_parent_extensions(parent: dict, idxs: list[int], max_path_conds: int):
    cset = set(int(x) for x in parent.get("_combo", ()))
    remaining = int(max_path_conds) - len(cset)
    if remaining <= 0:
        return
    add_candidates = [x for x in idxs if x not in cset]
    for add_k in range(1, remaining + 1):
        for adds in itertools.combinations(add_candidates, add_k):
            combo = tuple(sorted(cset | set(adds)))
            if 2 <= len(combo) <= int(max_path_conds):
                yield combo, parent


def _iter_all_parent_extensions(seeds: list[dict], idxs: list[int], max_path_conds: int):
    for p in seeds:
        yield from _iter_parent_extensions(p, idxs, max_path_conds)



def _is_atr_candidate_col(col: str) -> bool:
    c = str(col).lower()
    if c.startswith("delta_"):
        parts = c.split("_", 2)
        c = parts[2] if len(parts) == 3 and parts[1].isdigit() else c[6:]
    return bool(c.startswith("atr") or "_atr" in c)


def _stable_candidate_key(col: str, op: str, value: float) -> str:
    return f"{str(col)}|{str(op)}|{float(value):.17g}"

def _file_sig(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        st = path.stat()
        return f"{path.resolve()}|{int(st.st_mtime_ns)}|{int(st.st_size)}"
    except Exception:
        return str(path)


def _ctx_sig(obj: dict) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True).encode("utf-8")).hexdigest()


def _load_stage_csv_if_match(path: Path | None, expected_stage: str, expected_ctx_sig: str) -> pd.DataFrame | None:
    if path is None or (not path.exists()):
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    if "__stage" not in df.columns or "__ctx_sig" not in df.columns:
        return None
    stage_vals = {str(x) for x in df["__stage"].dropna().unique().tolist()}
    sig_vals = {str(x) for x in df["__ctx_sig"].dropna().unique().tolist()}
    if stage_vals != {expected_stage}:
        return None
    if sig_vals != {expected_ctx_sig}:
        got_sig = next(iter(sig_vals), "")
        print(f"[prefilter-resume] {expected_stage} CSV context mismatch")
        print(f"[prefilter-resume] csv_ctx_sig={got_sig}")
        print(f"[prefilter-resume] expected_ctx_sig={expected_ctx_sig}")
        return None
    return df


def _load_allowlist(path: Path | None) -> tuple[set[str] | None, bool]:
    if path is None:
        return None, False
    txt = path.read_text(encoding="utf-8").strip()
    if not txt:
        return set(), False
    if path.suffix.lower() != ".txt":
        raise ValueError(f"{path}: include-candidates-file must be a .txt file")
    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    if not lines:
        return set(), False
    header = lines[0]
    if header not in {"candidate_key", "candidate_key_refined"}:
        raise ValueError(f"{path}: first line must be exactly candidate_key or candidate_key_refined")
    out = []
    for ln in lines[1:]:
        s = ln.strip()
        if not s:
            continue
        out.append(s)
    return set(out), (header == "candidate_key_refined")


def _candidate_family(col: str, split_delta_window: bool) -> str:
    c = str(col)
    if c.startswith("delta_"):
        parts = c.split("_")
        if split_delta_window and len(parts) > 2 and parts[1].isdigit():
            return f"delta_{parts[1]}_{parts[2]}"
        if "rsi" in c:
            return "delta_rsi"
        if "macd" in c:
            return "delta_macd"
        if "adx" in c:
            return "delta_adx"
        if "plus_di" in c:
            return "delta_plus_di"
        if "minus_di" in c:
            return "delta_minus_di"
        if "dx" in c:
            return "delta_dx"
        if "mfi" in c:
            return "delta_mfi"
        if "kdj" in c:
            return "delta_kdj"
        if "vol_z" in c:
            return "delta_vol_z"
        if "atr" in c:
            return "delta_atr"
    if "dist_ema" in c:
        return "dist_ema"
    if c.startswith("ema"):
        return "ema"
    if c.startswith("rsi"):
        return "rsi"
    if c.startswith("macd"):
        return "macd"
    if c.startswith("adx"):
        return "adx"
    if c.startswith("plus_di"):
        return "plus_di"
    if c.startswith("minus_di"):
        return "minus_di"
    if c.startswith("dx"):
        return "dx"
    if c.startswith("mfi"):
        return "mfi"
    if c.startswith("kdj"):
        return "kdj"
    if "orderblock" in c:
        return "orderblock"
    if "support" in c:
        return "support"
    if "resist" in c:
        return "resistance"
    if c.startswith("break_up"):
        return "break_up"
    if c.startswith("break_dn"):
        return "break_dn"
    if c.startswith("candle_"):
        return "candle"
    if "vol_z" in c:
        return "vol_z"
    if c.startswith("fvg_"):
        return "fvg"
    if c.startswith("liq_sweep"):
        return "liq_sweep"
    if c.startswith("bos_") or c.startswith("choch_") or c.startswith("ms_"):
        return "market_structure"
    if c.startswith("atr"):
        return "atr"
    return c.split("_")[0]


def _critical_minutes_for_entries(
    entry_indices: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    bar_time_ns: np.ndarray,
    hold: int,
    tp_mode: str,
    trail: bool,
    trail_activate: float,
    tps: np.ndarray,
    sl: float,
    slippage_bps: float,
    spread_bps: float,
) -> set[int]:
    crit: set[int] = set()
    if entry_indices.size == 0:
        return crit
    slip = (slippage_bps + spread_bps) / 10000.0
    n = len(close)
    for i in entry_indices.tolist():
        if i >= n - 1:
            continue
        entry = float(close[i]) * (1.0 + slip)
        stop_level = entry * (1.0 - float(sl))
        end = min(n - 1, i + max(1, int(hold)))
        for j in range(i + 1, end + 1):
            h = float(high[j]); l = float(low[j])
            stop_hit = l <= stop_level
            tp_hit = bool(tp_mode in {"single", "multi"} and np.any(h >= entry * (1.0 + tps)))
            trail_can_activate = bool(trail and ((h / entry) - 1.0) >= float(trail_activate))
            is_critical = False
            if (not trail) and tp_mode == "single":
                is_critical = stop_hit and tp_hit
            elif (not trail) and tp_mode == "multi":
                is_critical = stop_hit and tp_hit
            elif trail and tp_mode == "none":
                is_critical = trail_can_activate
            else:
                event_count = int(stop_hit) + int(tp_hit) + int(trail_can_activate)
                is_critical = trail_can_activate or (event_count >= 2)
            if is_critical:
                crit.add(int(bar_time_ns[j]))
    return crit


def _load_tick_minute_map_partial(
    path: Path,
    datetime_col: str,
    price_col: str,
    sep: str,
    minute_filter: set[int],
    tick_chunk_size: int | str = "auto",
) -> tuple[np.ndarray, dict[int, tuple[int, int]], np.ndarray | None, np.ndarray | None, int]:
    if not minute_filter:
        return np.asarray([], dtype=np.float64), {}, None, None, 0
    if path.suffix.lower() in {".parquet", ".pq"}:
        try:
            import pyarrow.parquet as pa_parquet  # type: ignore
        except Exception as exc:
            raise RuntimeError("pyarrow is required for memory-safe tick parquet loading") from exc

        cols = ["DateTime", "Bid", "Ask", "Volume"]
        pf = pa_parquet.ParquetFile(path)
        schema_cols = set(str(c) for c in pf.schema_arrow.names)
        missing = set(cols) - schema_cols
        if missing:
            raise ValueError(f"tick parquet missing columns: {sorted(missing)}")

        chunk_size_eff = 150_000 if str(tick_chunk_size).lower() == "auto" else max(10_000, int(tick_chunk_size))
        minute_filter_arr = np.asarray(list(minute_filter), dtype=np.int64)
        filter_min = int(minute_filter_arr.min())
        filter_max = int(minute_filter_arr.max())
        nat_ns = np.iinfo(np.int64).min
        minute_parts: list[np.ndarray] = []
        time_parts: list[np.ndarray] = []
        bid_parts: list[np.ndarray] = []
        ask_parts: list[np.ndarray] = []
        row_order_parts: list[np.ndarray] = []
        row_groups_total = int(pf.num_row_groups)
        row_groups_read = 0
        row_groups_skipped = 0
        rows_matched = 0
        source_row_offset = 0

        def _row_group_overlaps_filter(row_group_index: int) -> bool:
            rg = pf.metadata.row_group(row_group_index)
            dt_col_idx = pf.schema_arrow.names.index("DateTime")
            stats = rg.column(dt_col_idx).statistics
            if stats is None or stats.min is None or stats.max is None:
                return True
            try:
                rg_min = int(pd.Timestamp(stats.min).tz_convert(None).value) if getattr(stats.min, "tzinfo", None) is not None else int(pd.Timestamp(stats.min).value)
                rg_max = int(pd.Timestamp(stats.max).tz_convert(None).value) if getattr(stats.max, "tzinfo", None) is not None else int(pd.Timestamp(stats.max).value)
            except Exception:
                return True
            rg_minute_min = (rg_min // NS_PER_MINUTE) * NS_PER_MINUTE
            rg_minute_max = (rg_max // NS_PER_MINUTE) * NS_PER_MINUTE
            return not (rg_minute_max < filter_min or rg_minute_min > filter_max)

        for row_group_index in range(row_groups_total):
            row_group_rows = int(pf.metadata.row_group(row_group_index).num_rows)
            if not _row_group_overlaps_filter(row_group_index):
                row_groups_skipped += 1
                source_row_offset += row_group_rows
                continue
            row_groups_read += 1
            batch_offset = 0
            for batch in pf.iter_batches(row_groups=[row_group_index], columns=cols, batch_size=chunk_size_eff):
                batch_rows = int(batch.num_rows)
                ch = batch.to_pandas()
                dt_full = pd.to_datetime(ch["DateTime"], errors="coerce", utc=True)
                bid = pd.to_numeric(ch["Bid"], errors="coerce").to_numpy(dtype=np.float64)
                ask = pd.to_numeric(ch["Ask"], errors="coerce").to_numpy(dtype=np.float64)
                dt_valid = dt_full.notna().to_numpy()
                dt_ns = dt_full.astype("int64").to_numpy(dtype=np.int64)
                minute_ns = ((dt_ns // NS_PER_MINUTE) * NS_PER_MINUTE).astype(np.int64, copy=False)
                valid = dt_valid & (dt_ns != nat_ns) & np.isfinite(bid) & np.isfinite(ask) & (bid > 0) & (ask > 0)
                if np.any(valid):
                    keep = valid & np.isin(minute_ns, minute_filter_arr)
                    if np.any(keep):
                        keep_indices = np.flatnonzero(keep)
                        minute_parts.append(minute_ns[keep_indices].astype(np.int64, copy=False))
                        time_parts.append(dt_ns[keep_indices].astype(np.int64, copy=False))
                        bid_parts.append(bid[keep_indices].astype(np.float64, copy=False))
                        ask_parts.append(ask[keep_indices].astype(np.float64, copy=False))
                        row_order_parts.append((source_row_offset + batch_offset + keep_indices).astype(np.int64, copy=False))
                        rows_matched += int(len(keep_indices))
                batch_offset += batch_rows
            source_row_offset += row_group_rows

        if not minute_parts:
            print(
                f"[prefilter-tick-parquet-load] row_groups_total={row_groups_total} row_groups_read={row_groups_read} "
                f"row_groups_skipped={row_groups_skipped} rows_matched=0 matched_minutes=0",
                flush=True,
            )
            return np.asarray([], dtype=np.float64), {}, None, None, 0

        mins = np.concatenate(minute_parts).astype(np.int64, copy=False)
        times = np.concatenate(time_parts).astype(np.int64, copy=False)
        bids = np.concatenate(bid_parts).astype(np.float64, copy=False)
        asks = np.concatenate(ask_parts).astype(np.float64, copy=False)
        row_order = np.concatenate(row_order_parts).astype(np.int64, copy=False)
        order = np.lexsort((row_order, times, mins))
        mins = mins[order]
        bids = bids[order]
        asks = asks[order]
        prices = bids.copy()
        bounds: dict[int, tuple[int, int]] = {}
        i = 0
        while i < len(mins):
            m = int(mins[i]); j = i + 1
            while j < len(mins) and int(mins[j]) == m:
                j += 1
            bounds[m] = (i, j)
            i = j
        print(
            f"[prefilter-tick-parquet-load] row_groups_total={row_groups_total} row_groups_read={row_groups_read} "
            f"row_groups_skipped={row_groups_skipped} rows_matched={rows_matched} matched_minutes={len(bounds)}",
            flush=True,
        )
        return prices, bounds, bids, asks, int(len(bounds))
    use_price_col = None if str(price_col).lower() == "auto" else str(price_col)
    chunk_size_eff = 150_000 if str(tick_chunk_size).lower() == "auto" else max(10_000, int(tick_chunk_size))
    minute_filter_arr = np.asarray(list(minute_filter), dtype=np.int64)
    minute_parts: list[np.ndarray] = []
    time_parts: list[np.ndarray] = []
    price_parts: list[np.ndarray] = []
    bid_parts: list[np.ndarray] = []
    ask_parts: list[np.ndarray] = []
    saw_bid = False
    saw_ask = False
    matched_minute_set: set[int] = set()
    for ch in pd.read_csv(path, sep=sep, chunksize=chunk_size_eff):
        pcol = use_price_col
        cols_lower = {str(c).lower(): str(c) for c in ch.columns}
        dt_col = cols_lower.get(str(datetime_col).strip().lower())
        if dt_col is None:
            for cand in ("datetime", "time", "timestamp", "date"):
                if cand in cols_lower:
                    dt_col = cols_lower[cand]
                    break
        if dt_col is None:
            raise ValueError(f"tick-data missing datetime column: {datetime_col}")
        bid_col = cols_lower.get("bid")
        ask_col = cols_lower.get("ask")
        if pcol is None:
            for cand in ("price", "last", "close", "mid", "bid", "ask"):
                if cand in cols_lower:
                    pcol = cols_lower[cand]
                    break
            if pcol is None:
                raise ValueError("tick-data price column not found")
        dt_full = pd.to_datetime(ch[dt_col], errors="coerce", utc=True)
        dt_ns = dt_full.astype("int64").to_numpy()
        minute_ns = dt_full.dt.floor("min").astype("int64").to_numpy()
        keep = np.isin(minute_ns, minute_filter_arr)
        if not np.any(keep):
            continue
        price_arr = pd.to_numeric(ch.loc[keep, pcol], errors="coerce").to_numpy(dtype=np.float64)
        kept_minutes = minute_ns[keep].astype(np.int64, copy=False)
        kept_times = dt_ns[keep].astype(np.int64, copy=False)
        valid = np.isfinite(price_arr) & (kept_times != np.iinfo(np.int64).min)
        bid_arr = None
        ask_arr = None
        if bid_col is not None:
            bid_arr = pd.to_numeric(ch.loc[keep, bid_col], errors="coerce").to_numpy(dtype=np.float64)
            saw_bid = True
        if ask_col is not None:
            ask_arr = pd.to_numeric(ch.loc[keep, ask_col], errors="coerce").to_numpy(dtype=np.float64)
            saw_ask = True
        if not np.any(valid):
            continue
        kept_minutes = kept_minutes[valid]
        kept_times = kept_times[valid]
        price_arr = price_arr[valid]
        minute_parts.append(kept_minutes)
        time_parts.append(kept_times)
        price_parts.append(price_arr)
        if bid_arr is not None:
            bid_parts.append(bid_arr[valid])
        elif saw_bid:
            bid_parts.append(np.full(price_arr.shape, np.nan, dtype=np.float64))
        if ask_arr is not None:
            ask_parts.append(ask_arr[valid])
        elif saw_ask:
            ask_parts.append(np.full(price_arr.shape, np.nan, dtype=np.float64))
        matched_minute_set.update({int(x) for x in np.unique(kept_minutes).tolist()})
    if not price_parts:
        return np.asarray([], dtype=np.float64), {}, None, None, 0
    mins = np.concatenate(minute_parts).astype(np.int64, copy=False)
    times = np.concatenate(time_parts).astype(np.int64, copy=False)
    prices = np.concatenate(price_parts).astype(np.float64, copy=False)
    bids = np.concatenate(bid_parts).astype(np.float64, copy=False) if saw_bid and bid_parts else None
    asks = np.concatenate(ask_parts).astype(np.float64, copy=False) if saw_ask and ask_parts else None
    order = np.lexsort((times, mins))
    mins = mins[order]
    prices = prices[order]
    if bids is not None:
        bids = bids[order]
    if asks is not None:
        asks = asks[order]
    bounds: dict[int, tuple[int, int]] = {}
    i = 0
    while i < len(mins):
        m = int(mins[i]); j = i + 1
        while j < len(mins) and int(mins[j]) == m:
            j += 1
        bounds[m] = (i, j)
        i = j
    return prices, bounds, bids, asks, int(len(matched_minute_set))


def _mask_hash_arr(mask: np.ndarray) -> str:
    packed = np.packbits(mask.astype(np.uint8, copy=False))
    return hashlib.sha1(packed.tobytes()).hexdigest()


def _with_rank_context(ranked: list[dict], rank_name: str) -> list[dict]:
    out: list[dict] = []
    for i, it in enumerate(ranked):
        z = dict(it)
        z["_rank_name"] = rank_name
        z["_rank_pos"] = int(i)
        out.append(z)
    return out


def _build_unlocked_pool(
    miner_mod,
    rank_lists: list[list[dict]],
    all_candidate_cols: list[str],
    unlocked_next: int,
    step_size: int,
    binary_cap_per_list_block: int,
    binary_anchor_lookahead_blocks: int,
    binary_cap_per_block: int,
) -> list[dict]:
    if unlocked_next <= 0:
        return []
    block_size = max(1, int(step_size))
    unlocked_blocks = int(math.ceil(float(unlocked_next) / float(block_size)))
    lookahead_blocks = max(0, int(binary_anchor_lookahead_blocks))
    merged: dict[tuple[str, str, float], dict] = {}

    blocks_by_rank = [_chunk_list(rank, block_size) for rank in rank_lists]

    def _has_anchor_within_lookahead(binary_col: str, block_idx: int) -> bool:
        anchors = miner_mod.candidate_anchor_columns(binary_col, all_candidate_cols)
        if not anchors:
            return True
        cols_window: set[str] = set()
        b0 = max(0, int(block_idx))
        b1 = b0 + lookahead_blocks
        for blocks in blocks_by_rank:
            for bi in range(b0, min(len(blocks), b1 + 1)):
                cols_window.update(str(x["col"]) for x in blocks[bi])
        return any(a in cols_window for a in anchors)

    def _add_item(it: dict, source_block: int) -> None:
        key = (str(it["col"]), str(it["op"]), float(it["value"]))
        cur = merged.get(key)
        cand = dict(it)
        cand["_source_block"] = int(source_block)
        if cur is None:
            merged[key] = cand
            return
        if int(cand.get("_rank_pos", 10 ** 9)) < int(cur.get("_rank_pos", 10 ** 9)):
            merged[key] = cand

    for blocks in blocks_by_rank:
        for bi in range(min(len(blocks), unlocked_blocks)):
            bin_count = 0
            for it in blocks[bi]:
                if bool(it.get("binary", False)):
                    if not _has_anchor_within_lookahead(str(it["col"]), bi):
                        continue
                    if bin_count >= int(binary_cap_per_list_block):
                        continue
                    bin_count += 1
                _add_item(it, source_block=bi)

    if int(binary_cap_per_block) > 0:
        per_step_cap = int(binary_cap_per_block)
        by_block: dict[int, list[int]] = {}
        pool = list(merged.values())
        for i, it in enumerate(pool):
            bi = int(it.get("_source_block", 0))
            by_block.setdefault(bi, []).append(i)
        keep_idx: set[int] = set()
        for bi, idxs in by_block.items():
            bidx = [i for i in idxs if bool(pool[i].get("binary", False))]
            nidx = [i for i in idxs if not bool(pool[i].get("binary", False))]
            keep_idx.update(nidx)
            keep_idx.update(bidx[:per_step_cap])
        merged = {
            (str(pool[i]["col"]), str(pool[i]["op"]), float(pool[i]["value"])): pool[i]
            for i in sorted(keep_idx)
        }

    return list(merged.values())


def _calc_wf(mask: np.ndarray, y_test: np.ndarray, wf_folds: int) -> tuple[float, float, int]:
    n = len(y_test)
    if n == 0:
        return float("nan"), float("nan"), 0
    edges = np.linspace(0, n, max(1, wf_folds) + 1, dtype=int)
    ratios = []
    total = 0
    for i in range(len(edges) - 1):
        a, b = int(edges[i]), int(edges[i + 1])
        m = mask[a:b]
        if not m.any():
            continue
        pos = int(np.sum(m & (y_test[a:b] == 1)))
        neg = int(np.sum(m & (y_test[a:b] == 0)))
        total += (pos + neg)
        ratios.append(pos / max(1, neg))
    if not ratios:
        return float("nan"), float("nan"), 0
    return float(np.mean(ratios)), float(np.min(ratios)), int(total)


def main() -> None:
    args = parse_args()
    if int(args.cluster_gap_minutes) < 0:
        raise ValueError("--cluster-gap-minutes must be >= 0")
    if int(args.max_entries_per_cluster) < 1:
        raise ValueError("--max-entries-per-cluster must be >= 1")
    if int(args.max_open_trades) < 1:
        raise ValueError("--max-open-trades must be >= 1")
    if int(args.phase_c_max_generated_per_level) < 0:
        raise ValueError("--phase-c-max-generated-per-level must be >= 0")
    if int(args.phase_d_max_generated_per_level) < 0:
        raise ValueError("--phase-d-max-generated-per-level must be >= 0")
    if args.out_candidates_csv is not None:
        raise ValueError("--out-candidates-csv is deprecated. Use --out-candidates-coarse-csv and --out-candidates-refined-csv.")
    miner = _load_miner_module(args.miner_script)
    timing: dict[str, float] = {}

    t0 = time.perf_counter()
    df = miner.load_features(args.features)
    timing["load_features_sec"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    bdf = miner.load_binned_features(args.binned_features, tail_rows=0).reindex(df.index)
    timing["load_binned_features_sec"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    meta = miner.load_binned_metadata(args.binned_metadata)
    timing["load_binned_metadata_sec"] = time.perf_counter() - t0

    n = len(df)
    train_idx = max(1, min(int(n * float(args.train_frac)), n - 1))

    high = df["high"].to_numpy(dtype=np.float32, copy=False)
    low = df["low"].to_numpy(dtype=np.float32, copy=False)
    close = df["close"].to_numpy(dtype=np.float32, copy=False)
    bar_time_ns = np.asarray(pd.DatetimeIndex(df.index).floor("min").view("int64"), dtype=np.int64)

    tick_prices_all = None
    tick_minute_bounds = None

    tps_all = [float(x) for x in str(args.tps).split(",") if x.strip()]
    if bool(args.no_tp):
        tp_mode = "none"
    else:
        tp_mode = "multi" if bool(args.use_multi_tp) else "single"
    if (not bool(args.trail)) and tp_mode == "none":
        raise ValueError("Invalid TP mode: no TP is not allowed when --trail is disabled.")
    if tp_mode == "multi":
        tps = tps_all
        tp_w = miner.parse_tp_weights(tps, str(args.tp_weights))
        tp_enabled = True
    elif tp_mode == "single":
        tps = [tps_all[0]]
        tp_w = np.asarray([1.0], dtype=np.float64)
        tp_enabled = True
    else:
        tps = [tps_all[0]]
        tp_w = np.asarray([1.0], dtype=np.float64)
        tp_enabled = False

    cache_key_obj = {
        "features_sig": _file_sig(args.features),
        "tps": [float(x) for x in tps],
        "tp_weights": [float(x) for x in tp_w.tolist()],
        "tp_mode": str(tp_mode),
        "sl": float(args.sl),
        "hold": int(args.hold),
        "slippage_bps": float(args.slippage_bps),
        "spread_bps": float(args.spread_bps),
        "trail": bool(args.trail),
        "trail_activate": float(args.trail_activate),
        "trail_offset": float(args.trail_offset),
        "trail_factor": float(args.trail_factor),
        "trail_min_level": float(args.trail_min_level),
        "include_unrealized_at_test_end": bool(args.include_unrealized_at_test_end),
        "label_semantics_version": "hold_endpoint_v2",
        "train_frac": float(args.train_frac),
        "train_idx": int(train_idx),
        "n_rows": int(n),
        "full_end_idx": int(n - 1),
        "period_end_semantics": "train_test_split_endpoints_v1",
    }
    cache_key = hashlib.sha1(json.dumps(cache_key_obj, sort_keys=True).encode("utf-8")).hexdigest()
    loaded_cache = False
    if args.label_cache_npz is not None and args.label_cache_npz.exists():
        try:
            z = np.load(args.label_cache_npz, allow_pickle=False)
            if str(z["cache_key"].item()) == cache_key:
                pnl = z["pnl"]
                y = z["y"]
                t_exit = z["t_exit"]
                t_qual = z["t_qual"]
                tp_hits = z["tp_hits"]
                loaded_cache = True
                print(f"[prefilter-cache] loaded labels from {args.label_cache_npz}")
        except Exception as e:
            print(f"[prefilter-cache] failed to load cache: {e}")
    t0 = time.perf_counter()
    if not loaded_cache:
        pnl, y, t_exit, t_qual, tp_hits = miner.simulate_multitp_trailing_pessimistic(
            high=high,
            low=low,
            close=close,
            tps=tps,
            tp_w=tp_w,
            tp_enabled=bool(tp_enabled),
            sl=float(args.sl),
            hold=int(args.hold),
            slippage_bps=float(args.slippage_bps),
            spread_bps=float(args.spread_bps),
            trail=bool(args.trail),
            trail_activate=float(args.trail_activate),
            trail_offset=float(args.trail_offset),
            trail_factor=float(args.trail_factor),
            trail_min_level=float(args.trail_min_level),
            include_unrealized_at_test_end=bool(args.include_unrealized_at_test_end),
            bar_time_ns=bar_time_ns,
            tick_prices_all=None,
            tick_minute_bounds=None,
            period_end_indices=np.where(np.arange(n) < train_idx, train_idx - 1, n - 1).astype(np.int64),
        )
        if args.label_cache_npz is not None:
            args.label_cache_npz.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                args.label_cache_npz,
                cache_key=np.asarray(cache_key),
                pnl=pnl,
                y=y,
                t_exit=t_exit,
                t_qual=t_qual,
                tp_hits=tp_hits,
            )
            print(f"[prefilter-cache] wrote labels cache to {args.label_cache_npz}")
    timing["simulate_labels_sec"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    cols = miner.build_candidate_features(df, allow_absolute_price=False, max_features=0)
    timing["build_candidate_features_sec"] = time.perf_counter() - t0
    atr_cols_available = sum(1 for c in cols if _is_atr_candidate_col(c))
    atr_cols_sample = [str(c) for c in cols if _is_atr_candidate_col(c)][:30]
    y_train = y[:train_idx]
    y_test = y[train_idx:]
    tradable_train = ((y_train == 0) | (y_train == 1))
    tradable_test = ((y_test == 0) | (y_test == 1))
    period_end_indices = np.where(np.arange(n) < train_idx, train_idx - 1, n - 1).astype(np.int64)

    def _exit_abs_idx(entry_idx: int, exit_rel: int) -> int:
        i = int(entry_idx)
        if i < 0 or i >= n:
            return i
        if int(exit_rel) > 0:
            return min(n - 1, i + int(exit_rel))
        period_end = int(period_end_indices[i]) if i < len(period_end_indices) else n - 1
        period_end = max(i, min(period_end, n - 1))
        return min(period_end, i + max(1, int(args.hold)))

    def _select_entries_for_mask(
        raw_mask: np.ndarray,
        y_eval: np.ndarray,
        t_exit_eval: np.ndarray,
        only_lower_entry: bool | None = None,
    ) -> tuple[np.ndarray, int, int]:
        use_lower = bool(args.cluster_only_lower_entry) if only_lower_entry is None else bool(only_lower_entry)
        eval_y = np.asarray(y_eval)
        eval_exit = np.asarray(t_exit_eval)
        raw = np.asarray(raw_mask, dtype=bool)
        evaluable = raw & ((eval_y == 0) | (eval_y == 1))
        raw_idxs = np.flatnonzero(evaluable)
        selected = np.zeros(n, dtype=bool)
        if raw_idxs.size == 0:
            return selected, 0, 0

        gap_ns = int(args.cluster_gap_minutes) * 60 * 1_000_000_000
        open_trades: list[int] = []
        prev_idx: int | None = None
        entries_in_cluster = 0
        cluster_min_close = float("nan")
        selected_cluster_ids: set[int] = set()
        cluster_id = -1

        for idx_i in raw_idxs.tolist():
            i = int(idx_i)
            new_cluster = prev_idx is None
            if prev_idx is not None:
                if bar_time_ns is not None and len(bar_time_ns) > i and len(bar_time_ns) > prev_idx:
                    new_cluster = (int(bar_time_ns[i]) - int(bar_time_ns[prev_idx])) > gap_ns
                else:
                    new_cluster = (i - int(prev_idx)) > int(args.cluster_gap_minutes)
            if new_cluster:
                cluster_id += 1
                entries_in_cluster = 0
                cluster_min_close = float("nan")
            prev_idx = i

            open_trades = [x for x in open_trades if int(x) > i]
            if entries_in_cluster >= int(args.max_entries_per_cluster):
                continue
            if use_lower and entries_in_cluster > 0:
                c = float(close[i])
                if (not np.isfinite(c)) or (not np.isfinite(cluster_min_close)) or not (c < cluster_min_close):
                    continue
            if len(open_trades) >= int(args.max_open_trades):
                continue

            exit_idx = _exit_abs_idx(i, int(eval_exit[i]) if i < len(eval_exit) else -1)
            selected[i] = True
            entries_in_cluster += 1
            selected_cluster_ids.add(int(cluster_id))
            c = float(close[i])
            if np.isfinite(c):
                cluster_min_close = c if not np.isfinite(cluster_min_close) else min(cluster_min_close, c)
            open_trades.append(int(exit_idx))
        return selected, int(raw_idxs.size), int(len(selected_cluster_ids))

    t0 = time.perf_counter()
    allow_keys, allow_keys_refined = _load_allowlist(args.include_candidates_file)
    coarse_ctx = {
        "features_sig": _file_sig(args.features),
        "binned_sig": _file_sig(args.binned_features),
        "meta_sig": _file_sig(args.binned_metadata),
        "label_cache_key": cache_key,
        "train_frac": float(args.train_frac),
        "quantiles": str(args.quantiles),
        "family_split_delta_window": int(bool(args.family_split_delta_window)),
        "entry_selection_semantics_version": "cluster_gap_lower_maxopen_v1",
        "cluster_gap_minutes": int(args.cluster_gap_minutes),
        "max_entries_per_cluster": int(args.max_entries_per_cluster),
        "max_open_trades": int(args.max_open_trades),
        "cluster_only_lower_entry": int(bool(args.cluster_only_lower_entry)),
    }
    coarse_ctx_sig = _ctx_sig(coarse_ctx)
    coarse_out = args.out_candidates_coarse_csv
    refined_out = args.out_candidates_refined_csv
    coarse_resume = _load_stage_csv_if_match(coarse_out, "coarse", coarse_ctx_sig)
    coarse_existing_rows: dict[str, dict] = {}
    coarse_build_min_pos = int(args.min_single_pos_hits)
    coarse_build_max_mask = int(args.max_single_mask_count)
    coarse_build_min_lift = float(args.min_single_lift)
    if coarse_resume is not None and len(coarse_resume) > 0:
        b_pos = int(coarse_resume.get("build_min_single_pos_hits", pd.Series([args.min_single_pos_hits])).iloc[0])
        b_mask = int(coarse_resume.get("build_max_single_mask_count", pd.Series([args.max_single_mask_count])).iloc[0])
        b_lift = float(coarse_resume.get("build_min_single_lift", pd.Series([args.min_single_lift])).iloc[0])
        c_pos = int(args.min_single_pos_hits)
        c_mask = int(args.max_single_mask_count)
        c_lift = float(args.min_single_lift)
        coarse_build_min_pos = min(b_pos, c_pos)
        coarse_build_min_lift = min(b_lift, c_lift)
        coarse_build_max_mask = 0 if (b_mask == 0 or c_mask == 0) else max(b_mask, c_mask)
        too_narrow_reasons = []
        if b_mask == 0:
            pass
        elif c_mask == 0:
            too_narrow_reasons.append("max_single_mask_count")
        elif c_mask > b_mask:
            too_narrow_reasons.append("max_single_mask_count")
        if c_pos < b_pos:
            too_narrow_reasons.append("min_single_pos_hits")
        if c_lift < b_lift:
            too_narrow_reasons.append("min_single_lift")
        for _, r in coarse_resume.iterrows():
            try:
                stable_k = str(r.get("stable_candidate_key", "")).strip() or _stable_candidate_key(str(r.get("col", "")), str(r.get("op", "")), float(r.get("value", np.nan)))
            except Exception:
                stable_k = str(r.get("candidate_key", "")).strip()
            if stable_k:
                row = {k: r.get(k) for k in r.index}
                row["stable_candidate_key"] = stable_k
                coarse_existing_rows[stable_k] = row
        print(f"[prefilter-resume] coarse_build_min_single_pos_hits={b_pos} current_min_single_pos_hits={c_pos}")
        print(f"[prefilter-resume] coarse_build_max_single_mask_count={b_mask} current_max_single_mask_count={c_mask}")
        print(f"[prefilter-resume] coarse_build_min_single_lift={b_lift} current_min_single_lift={c_lift}")
        if too_narrow_reasons:
            print(f"[prefilter-resume] coarse CSV will be extended for wider runtime filters: {','.join(too_narrow_reasons)}")
            coarse_resume = None
        else:
            print("[prefilter-resume] coarse CSV reused (build width is sufficient)")

    items = []
    filtered_items: list[dict] = []
    single_rejects = {"min_pos": 0, "min_lift": 0, "mask_count": 0, "allowlist": 0}
    if coarse_resume is not None:
        for _, r in coarse_resume.iterrows():
            key = str(r.get("candidate_key", "")).strip()
            stable_k = str(r.get("stable_candidate_key", "")).strip()
            col = str(r["col"]); op = str(r["op"]); val = float(r["value"])
            if not stable_k:
                stable_k = _stable_candidate_key(col, op, val)
            if not key:
                key = stable_k
            xvec = pd.to_numeric(bdf[col] if op == "==" else df[col], errors="coerce").to_numpy(copy=False)
            if op == "==":
                m = np.isfinite(xvec) & (np.abs(xvec - val) <= 1e-6)
            elif op == ">=":
                m = np.isfinite(xvec) & (xvec >= val)
            else:
                m = np.isfinite(xvec) & (xvec <= val)
            m_sel, _, _ = _select_entries_for_mask(m, y, t_exit)
            pos_hits = int(np.sum(m_sel[:train_idx] & (y_train == 1)))
            neg_hits = int(np.sum(m_sel[:train_idx] & (y_train == 0)))
            mask_count = int(pos_hits + neg_hits)
            ratio = pos_hits / max(1, neg_hits)
            lift = (pos_hits / max(1, int(np.sum(y_train == 1)))) / max(1e-12, (neg_hits / max(1, int(np.sum(y_train == 0)))))
            filtered_items.append({
                "candidate_key": key, "stable_candidate_key": stable_k, "col": col, "op": op, "value": val, "mask": m,
                "binary": bool(int(r.get("binary", 0))), "_family": str(r.get("family", _candidate_family(col, bool(args.family_split_delta_window)))),
                "_single_pos_hits": pos_hits,
                "_single_neg_hits": neg_hits,
                "_single_mask_count": mask_count,
                "_single_ratio": float(ratio),
                "coarse_single_pos_hits": pos_hits,
                "coarse_single_neg_hits": neg_hits,
                "coarse_single_mask_count": mask_count,
                "coarse_single_ratio": float(ratio),
                "coarse_lift": float(lift),
                "lift": float(lift),
                "ratio": float(ratio),
            })
        print(f"[prefilter-resume] loaded coarse candidates from {coarse_out} rows={len(filtered_items)}")
    else:
        qs = [float(x) for x in str(args.quantiles).split(",") if x.strip()]
        t0 = time.perf_counter()
        qmap = miner.quantile_thresholds(df.iloc[:train_idx], cols, qs)
        timing["quantile_thresholds_sec"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        for c in cols:
            if c.startswith("dist_"):
                x = pd.to_numeric(df[c], errors="coerce").to_numpy(copy=False)
                for _, thr in qmap.get(c, {}).items():
                    for op in (">=", "<="):
                        m = np.isfinite(x) & ((x >= thr) if op == ">=" else (x <= thr))
                        m_sel, _, _ = _select_entries_for_mask(m, y, t_exit)
                        pos = int(np.sum(m_sel[:train_idx] & (y_train == 1)))
                        neg = int(np.sum(m_sel[:train_idx] & (y_train == 0)))
                        if pos <= 0:
                            continue
                        items.append({"col": c, "op": op, "value": float(thr), "mask": m, "binary": False,
                                      "freq": pos / max(1, int(np.sum(y_train == 1))),
                                      "lift": (pos / max(1, int(np.sum(y_train == 1)))) / max(1e-12, (neg / max(1, int(np.sum(y_train == 0))))),
                                      "ratio": pos / max(1, neg)})
                continue
            miss = int(meta.get(c, {}).get("missing_code", 0))
            b = pd.to_numeric(bdf[c], errors="coerce").fillna(miss).to_numpy()
            vals = np.unique(b[:train_idx]); vals = vals[vals != miss]
            is_binary = str(meta.get(c, {}).get("feature_type", "")) == "binary"
            for v in vals.tolist():
                m = np.isfinite(b) & (np.abs(b - float(v)) <= 1e-6)
                m_sel, _, _ = _select_entries_for_mask(m, y, t_exit)
                pos = int(np.sum(m_sel[:train_idx] & (y_train == 1))); neg = int(np.sum(m_sel[:train_idx] & (y_train == 0)))
                if pos <= 0:
                    continue
                items.append({"col": c, "op": "==", "value": float(v), "mask": m, "binary": bool(is_binary),
                              "freq": pos / max(1, int(np.sum(y_train == 1))),
                              "lift": (pos / max(1, int(np.sum(y_train == 1)))) / max(1e-12, (neg / max(1, int(np.sum(y_train == 0))))),
                              "ratio": pos / max(1, neg)})
        timing["build_items_sec"] = time.perf_counter() - t0
        for i, it in enumerate(items, start=1):
            x = dict(it); x["candidate_key"] = f"cand_{i:06d}"
            x["stable_candidate_key"] = _stable_candidate_key(str(x["col"]), str(x["op"]), float(x["value"]))
            m = np.asarray(x["mask"], dtype=bool)
            m_sel, _, _ = _select_entries_for_mask(m, y, t_exit)
            pos_hits = int(np.sum(m_sel[:train_idx] & (y_train == 1))); neg_hits = int(np.sum(m_sel[:train_idx] & (y_train == 0)))
            mask_count = int(pos_hits + neg_hits)
            if float(x["lift"]) < float(args.min_single_lift):
                single_rejects["min_lift"] += 1; continue
            x["_single_pos_hits"] = pos_hits; x["_single_neg_hits"] = neg_hits; x["_single_mask_count"] = mask_count
            x["_single_ratio"] = pos_hits / max(1, neg_hits)
            x["coarse_single_pos_hits"] = int(pos_hits); x["coarse_single_neg_hits"] = int(neg_hits)
            x["coarse_single_mask_count"] = int(mask_count); x["coarse_single_ratio"] = float(x["_single_ratio"]); x["coarse_lift"] = float(x["lift"])
            x["_family"] = _candidate_family(str(x["col"]), bool(args.family_split_delta_window))
            filtered_items.append(x)
        by_mask: dict[str, dict] = {}
        for it in filtered_items:
            mh = _mask_hash_arr(np.asarray(it["mask"][:train_idx], dtype=bool))
            cur = by_mask.get(mh)
            if cur is None:
                by_mask[mh] = it; continue
            tf_cur = int(miner._parse_feature_meta(str(cur["col"])).get("tf") or 10**9)
            tf_new = int(miner._parse_feature_meta(str(it["col"])).get("tf") or 10**9)
            if tf_new < tf_cur:
                by_mask[mh] = it
        filtered_items = list(by_mask.values())
    atr_debug_source = list(items) if items else list(filtered_items)
    def _atr_debug_pass_counts(source_items: list[dict]) -> tuple[int, int, int, int]:
        atr_items = [it for it in source_items if _is_atr_candidate_col(str(it.get("col", "")))]
        after_pos = []
        after_lift = []
        after_mask = []
        for it in atr_items:
            m = np.asarray(it.get("mask"), dtype=bool)
            pos_hits = int(it.get("coarse_single_pos_hits", it.get("_single_pos_hits", np.sum(m[:train_idx] & (y_train == 1)))))
            mask_count = int(it.get("coarse_single_mask_count", it.get("_single_mask_count", np.sum(m[:train_idx]))))
            lift_v = float(it.get("coarse_lift", it.get("lift", 0.0)))
            if pos_hits >= int(args.min_single_pos_hits):
                after_pos.append(it)
                if lift_v >= float(args.min_single_lift):
                    after_lift.append(it)
                    if int(args.max_single_mask_count) <= 0 or mask_count <= int(args.max_single_mask_count):
                        after_mask.append(it)
        return len(atr_items), len(after_pos), len(after_lift), len(after_mask)
    atr_candidates_built, atr_candidates_after_min_pos, atr_candidates_after_min_lift, atr_candidates_after_mask_count = _atr_debug_pass_counts(atr_debug_source)
    full_filtered_items = list(filtered_items)
    filtered_items = list(full_filtered_items)
    if int(args.min_single_pos_hits) > 0:
        before = len(filtered_items)
        filtered_items = [it for it in filtered_items if int(it.get("coarse_single_pos_hits", it.get("_single_pos_hits", 0))) >= int(args.min_single_pos_hits)]
        single_rejects["min_pos"] = max(0, before - len(filtered_items))
    if float(args.min_single_lift) > 0:
        before = len(filtered_items)
        filtered_items = [it for it in filtered_items if float(it.get("coarse_lift", it.get("lift", 0.0))) >= float(args.min_single_lift)]
        single_rejects["min_lift"] = max(single_rejects.get("min_lift", 0), max(0, before - len(filtered_items)))
    if int(args.max_single_mask_count) > 0:
        before = len(filtered_items)
        filtered_items = [it for it in filtered_items if int(it.get("coarse_single_mask_count", it.get("_single_mask_count", 0))) <= int(args.max_single_mask_count)]
        single_rejects["mask_count"] = max(0, before - len(filtered_items))
    if allow_keys is not None:
        before = len(filtered_items)
        filtered_items = [it for it in filtered_items if str(it.get("candidate_key")) in allow_keys]
        single_rejects["allowlist"] = max(0, before - len(filtered_items))
    fam_top: list[dict] = []
    fam_groups: dict[str, list[dict]] = {}
    for it in filtered_items:
        fam_groups.setdefault(str(it["_family"]), []).append(it)
    for _, arr in fam_groups.items():
        arr = sorted(arr, key=lambda z: (-float(z["lift"]), -int(z["_single_pos_hits"]), int(z["_single_mask_count"])))
        fam_top.extend(arr[: int(args.family_top_n)])

    tick_refined_mode = False
    refined_ctx = {
        **coarse_ctx,
        "tick_data_sig": _file_sig(args.tick_data),
        "tick_cache_sig": _file_sig(args.tick_cache_parquet),
        "tp_mode": str(tp_mode),
        "trail": int(bool(args.trail)),
        "trail_activate": float(args.trail_activate),
        "hold": int(args.hold),
        "tick_datetime_column": str(args.tick_datetime_column),
        "tick_price_column": str(args.tick_price_column),
        "tick_sep": str(args.tick_sep),
    }
    refined_ctx_sig = _ctx_sig(refined_ctx)
    tick_entry_cache_ctx = {
        **refined_ctx,
        "features_sig": _file_sig(args.features),
        "tps": [float(x) for x in tps],
        "tp_weights": [float(x) for x in np.asarray(tp_w, dtype=np.float64).tolist()],
        "use_multi_tp": int(bool(args.use_multi_tp)),
        "no_tp": int(bool(args.no_tp)),
        "sl": float(args.sl),
        "slippage_bps": float(args.slippage_bps),
        "spread_bps": float(args.spread_bps),
        "trail_offset": float(args.trail_offset),
        "trail_factor": float(args.trail_factor),
        "trail_min_level": float(args.trail_min_level),
        "include_unrealized_at_test_end": int(bool(args.include_unrealized_at_test_end)),
        "train_idx": int(train_idx),
        "n_rows": int(n),
        "period_end_semantics": "train_test_split_period_end_v1",
        "entry_semantics_version": "next_bar_first_ask_v1",
        "bid_ask_semantics_version": "long_entry_ask_exit_bid_v1",
        "label_outcome_semantics_version": "hold_or_period_end_endpoint_pnl_v1",
    }
    tick_entry_cache_sig = _ctx_sig(tick_entry_cache_ctx)
    tick_refine_t0 = time.perf_counter() if (args.tick_data is not None or (refined_out is not None and refined_out.exists())) else None
    refined_resume = _load_stage_csv_if_match(refined_out, "refined", refined_ctx_sig)
    if str(args.tick_refine_scope) == "fam_top":
        tick_scope_items = fam_top
    elif str(args.tick_refine_scope) == "filtered":
        tick_scope_items = filtered_items
    else:
        tick_scope_items = full_filtered_items
    all_by_key = {str(it["candidate_key"]): it for it in full_filtered_items}
    req_tick_cols = ["tick_single_pos_hits", "tick_single_neg_hits", "tick_single_mask_count", "tick_single_ratio", "tick_lift"]
    def _has_full_tick_metrics(obj: dict) -> bool:
        try:
            return all(np.isfinite(float(obj.get(k, np.nan))) for k in req_tick_cols)
        except Exception:
            return False

    if refined_resume is not None:
        by_key = {str(it["candidate_key"]): it for it in tick_scope_items}
        refined_rows_total = int(len(refined_resume))
        refined_rows_with_tick = 0
        refined_rows_missing_tick = 0
        for _, r in refined_resume.iterrows():
            k = str(r.get("candidate_key_refined", "")).strip()
            stable_k = str(r.get("stable_candidate_key", "")).strip()
            col_r = str(r.get("col", ""))
            op_r = str(r.get("op", ""))
            val_r = float(r.get("value", np.nan)) if pd.notna(r.get("value", np.nan)) else np.nan
            if not stable_k and col_r and op_r and np.isfinite(val_r):
                stable_k = _stable_candidate_key(col_r, op_r, val_r)
            it = all_by_key.get(k)
            if it is None and stable_k:
                it = next((x for x in full_filtered_items if _stable_candidate_key(str(x["col"]), str(x["op"]), float(x["value"])) == stable_k), None)
            if it is None:
                continue
            req_vals = [r.get(c, np.nan) for c in req_tick_cols]
            if any(pd.isna(v) for v in req_vals):
                refined_rows_missing_tick += 1
                continue
            refined_rows_with_tick += 1
            it["tick_single_pos_hits"] = int(r.get("tick_single_pos_hits", 0))
            it["tick_single_neg_hits"] = int(r.get("tick_single_neg_hits", 0))
            it["tick_single_mask_count"] = int(r.get("tick_single_mask_count", 0))
            it["tick_single_ratio"] = float(r.get("tick_single_ratio", 0.0))
            it["tick_lift"] = float(r.get("tick_lift", 0.0))
            if str(it.get("candidate_key")) in by_key:
                it["_single_pos_hits"] = int(it["tick_single_pos_hits"])
                it["_single_neg_hits"] = int(it["tick_single_neg_hits"])
                it["_single_mask_count"] = int(it["tick_single_mask_count"])
                it["_single_ratio"] = float(it["tick_single_ratio"])
                it["ratio"] = float(it["tick_single_ratio"])
                it["lift"] = float(it["tick_lift"])
        tick_refined_mode = any(_has_full_tick_metrics(x) for x in by_key.values())
        print(f"[prefilter-resume] loaded refined candidates from {refined_out}")
        print(f"[prefilter-resume] refined_rows_total={refined_rows_total}")
        print(f"[prefilter-resume] refined_rows_with_tick_metrics={refined_rows_with_tick}")
        print(f"[prefilter-resume] refined_rows_missing_tick_metrics={refined_rows_missing_tick}")
        print(f"[prefilter-resume] requested_allowlist_keys={len(allow_keys) if allow_keys is not None else 0}")
        print(f"[prefilter-resume] usable_refined_keys_for_fam_top={sum(1 for it in fam_top if 'tick_single_ratio' in it)}")
        print(f"[prefilter-resume] usable_refined_keys_for_tick_scope={sum(1 for it in tick_scope_items if 'tick_single_ratio' in it)}")
    elif bool(allow_keys_refined):
        if refined_out is None or not refined_out.exists():
            raise ValueError("candidate_key_refined input requires --out-candidates-refined-csv file to reload refined metrics")
        raise ValueError("Refined TXT header provided, but refined candidate CSV context did not match current run.")
    tick_missing_items = [it for it in tick_scope_items if not _has_full_tick_metrics(it)]
    if args.tick_entry_cache_npz is None and refined_resume is not None and not tick_missing_items and any(_has_full_tick_metrics(it) for it in tick_scope_items):
        print("[prefilter-tick] warning: using tick_lift from refined CSV; without --tick-entry-cache-npz it cannot be exactly rebaselined to the current tick_refine_scope.")
    if (args.tick_data is not None or args.tick_entry_cache_npz is not None) and len(tick_scope_items) > 0 and (tick_missing_items or args.tick_entry_cache_npz is not None):
        print(f"[prefilter-tick] tick_refine_scope={args.tick_refine_scope}")
        print(f"[prefilter-tick] tick_refine_candidates_count={len(tick_scope_items)}")
        print(f"[prefilter-tick] tick_refine_missing_to_compute={len(tick_missing_items)}")
        fam_union = np.zeros(n, dtype=bool)
        for it in tick_scope_items:
            fam_union |= np.asarray(it["mask"], dtype=bool)
        entry_indices = np.flatnonzero(fam_union).astype(np.int64, copy=False)
        entry_index_set = {int(i) for i in entry_indices.tolist()}
        tick_period_end_indices = np.where(np.arange(n) < train_idx, train_idx - 1, n - 1).astype(np.int64)

        def _scope_tick_minute_counts(scope_entries: np.ndarray) -> tuple[int, int, int]:
            scope_entry_minutes = {int(bar_time_ns[int(i) + 1]) for i in scope_entries.tolist() if int(i) + 1 < n}
            scope_critical_minutes = _critical_minutes_for_entries(
                entry_indices=scope_entries,
                high=high,
                low=low,
                close=close,
                bar_time_ns=bar_time_ns,
                hold=int(args.hold),
                tp_mode=str(tp_mode),
                trail=bool(args.trail),
                trail_activate=float(args.trail_activate),
                tps=np.asarray(tps, dtype=np.float64),
                sl=float(args.sl),
                slippage_bps=float(args.slippage_bps),
                spread_bps=float(args.spread_bps),
            )
            return len(scope_entry_minutes), len(scope_critical_minutes), len(scope_entry_minutes | scope_critical_minutes)

        def _full_window_minutes(scope_entries: np.ndarray) -> set[int]:
            out: set[int] = set()
            for idx_i in scope_entries.tolist():
                i = int(idx_i)
                if i + 1 >= n:
                    continue
                period_end = max(i, min(int(tick_period_end_indices[i]), n - 1))
                end = min(period_end, i + max(1, int(args.hold)))
                for j in range(i + 1, end + 1):
                    out.add(int(bar_time_ns[j]))
            return out

        cached_entries = _load_tick_entry_cache(args.tick_entry_cache_npz, tick_entry_cache_sig) if args.tick_entry_cache_npz is not None else {}
        cached_entries = cached_entries if cached_entries is not None else {}
        cached_present = {int(k) for k in cached_entries.keys() if int(k) in entry_index_set}
        tick_map: dict[int, dict] = {int(i): cached_entries[int(i)] for i in cached_present if int(i) in cached_entries}

        def _is_structurally_invalid_entry(ii: int) -> bool:
            if ii < 0 or ii >= n:
                return True
            entry_i = ii + 1
            if entry_i >= n:
                return True
            if ii >= len(tick_period_end_indices) or entry_i > int(tick_period_end_indices[ii]):
                return True
            try:
                _ = int(bar_time_ns[entry_i])
            except Exception:
                return True
            return False

        structurally_invalid = [int(i) for i in entry_indices.tolist() if int(i) not in cached_present and _is_structurally_invalid_entry(int(i))]
        if structurally_invalid:
            invalid_rec = {"y": -1, "pnl": float("nan"), "t_exit": -1, "t_qual": -1, "tp_hits": 0}
            for idx_i in structurally_invalid:
                tick_map[int(idx_i)] = dict(invalid_rec)
                if args.tick_entry_cache_npz is not None:
                    cached_entries[int(idx_i)] = dict(invalid_rec)
            cached_present.update(structurally_invalid)
            if args.tick_entry_cache_npz is not None:
                _write_tick_entry_cache(args.tick_entry_cache_npz, tick_entry_cache_sig, cached_entries)
            print(f"[prefilter-tick-cache] cached_structurally_invalid_entries={len(structurally_invalid)}")
        cached_valid = {int(k) for k in cached_present if int(k) in cached_entries and int(cached_entries[int(k)].get("y", -1)) in (0, 1)}
        missing_entry_indices = np.asarray([int(i) for i in entry_indices.tolist() if int(i) not in cached_present], dtype=np.int64)
        used_real_ticks = bool(args.tick_entry_cache_npz is not None and len(missing_entry_indices) == 0 and len(entry_indices) > 0)
        if missing_entry_indices.size > 0:
            print(f"[prefilter-tick-cache] requested_entries={len(entry_indices)} cached_entries_used={len(cached_present)} missing_entries={len(missing_entry_indices)}")
            if args.tick_data is None:
                print("[prefilter-tick-cache] warning: missing entries require --tick-data; current-scope missing entries remain unresolved.")
                critical_minutes: set[int] = set()
                entry_minutes: set[int] = set()
                minutes_to_load: set[int] = set()
                matched_critical_minutes_count = 0
                matched_total_minutes_count = 0
                used_real_ticks = bool(len(tick_map) > 0)
            else:
                critical_minutes = _critical_minutes_for_entries(
                    entry_indices=missing_entry_indices,
                    high=high,
                    low=low,
                    close=close,
                    bar_time_ns=bar_time_ns,
                    hold=int(args.hold),
                    tp_mode=str(tp_mode),
                    trail=bool(args.trail),
                    trail_activate=float(args.trail_activate),
                    tps=np.asarray(tps, dtype=np.float64),
                    sl=float(args.sl),
                    slippage_bps=float(args.slippage_bps),
                    spread_bps=float(args.spread_bps),
                )
                entry_minutes = {int(bar_time_ns[int(i) + 1]) for i in missing_entry_indices.tolist() if int(i) + 1 < n}
                if str(args.tick_minute_load_mode) == "full_window":
                    window_minutes = _full_window_minutes(missing_entry_indices)
                    minutes_to_load = set(critical_minutes) | entry_minutes | window_minutes
                else:
                    minutes_to_load = set(critical_minutes) | entry_minutes
                if bool(args.debug_tick_cache_scope_counts):
                    scope_entry_count, scope_critical_count, scope_would_load_count = _scope_tick_minute_counts(entry_indices)
                    print(f"[prefilter-tick-cache] scope_entry_minutes_count={scope_entry_count}")
                    print(f"[prefilter-tick-cache] scope_critical_minutes_count={scope_critical_count}")
                    print(f"[prefilter-tick-cache] scope_minutes_would_load_count={scope_would_load_count}")
                    print("[prefilter-tick-cache] scope_counts_source=ohlc_diagnostic_no_raw_tick_load")
                if minutes_to_load:
                    tick_prices_all, tick_minute_bounds, tick_bids_all, tick_asks_all, matched_total_minutes_count = _load_tick_minute_map_partial(
                        path=args.tick_data,
                        datetime_col=args.tick_datetime_column,
                        price_col=args.tick_price_column,
                        sep=args.tick_sep,
                        minute_filter=minutes_to_load,
                        tick_chunk_size=args.tick_chunk_size,
                    )
                    matched_critical_minutes_count = sum(1 for m in critical_minutes if int(m) in tick_minute_bounds)
                else:
                    tick_prices_all, tick_minute_bounds = np.asarray([], dtype=np.float64), {}
                    tick_bids_all, tick_asks_all, matched_critical_minutes_count, matched_total_minutes_count = None, None, 0, 0
                used_real_ticks_partial = bool(matched_critical_minutes_count > 0 or (len(entry_minutes) > 0 and matched_total_minutes_count >= len(entry_minutes)))
                used_real_ticks_full = bool(len(minutes_to_load) > 0 and matched_total_minutes_count >= len(minutes_to_load) and matched_critical_minutes_count >= len(critical_minutes))
                print(f"[prefilter-tick-raw-load] tick_minute_load_mode={args.tick_minute_load_mode}")
                if str(args.tick_minute_load_mode) == "full_window":
                    print("[prefilter-tick-raw-load] warning=full_window_may_be_expensive")
                print(f"[prefilter-tick-raw-load] critical_minutes_count={len(critical_minutes)}")
                print(f"[prefilter-tick-raw-load] entry_minutes_count={len(entry_minutes)}")
                print(f"[prefilter-tick-raw-load] minutes_to_load_count={len(minutes_to_load)}")
                print(f"[prefilter-tick-raw-load] matched_critical_minutes_count={int(matched_critical_minutes_count)}")
                print(f"[prefilter-tick-raw-load] matched_total_minutes_count={int(matched_total_minutes_count)}")
                print(f"[prefilter-tick-raw-load] raw_tick_coverage_full={bool(used_real_ticks_full)}")
                print(f"[prefilter-tick-raw-load] used_real_ticks_source={'parquet_ticks' if args.tick_data is not None and args.tick_data.suffix.lower() in {'.parquet', '.pq'} else 'raw_ticks'}")
                if used_real_ticks_partial:
                    entry_tick_stats: dict[str, int] = {}
                    tick_map_new = miner.simulate_selected_entries_with_ticks(
                        entry_indices=missing_entry_indices,
                        high=high,
                        low=low,
                        close=close,
                        tps=tps,
                        tp_w=tp_w,
                        tp_enabled=bool(tp_enabled),
                        sl=float(args.sl),
                        hold=int(args.hold),
                        slippage_bps=float(args.slippage_bps),
                        spread_bps=float(args.spread_bps),
                        trail=bool(args.trail),
                        trail_activate=float(args.trail_activate),
                        trail_offset=float(args.trail_offset),
                        trail_factor=float(args.trail_factor),
                        include_unrealized_at_test_end=bool(args.include_unrealized_at_test_end),
                        bar_time_ns=bar_time_ns,
                        tick_prices_all=tick_prices_all,
                        tick_minute_bounds=tick_minute_bounds,
                        tick_bids_all=tick_bids_all,
                        tick_asks_all=tick_asks_all,
                        use_tick_bid_ask=bool(tick_bids_all is not None and tick_asks_all is not None),
                        period_end_indices=tick_period_end_indices,
                        entry_tick_stats=entry_tick_stats,
                    )
                    print(f"[prefilter-tick] entry_tick_missing_count={int(entry_tick_stats.get('entry_tick_missing_count', 0))}")
                    invalid_rec = {"y": -1, "pnl": float("nan"), "t_exit": -1, "t_qual": -1, "tp_hits": 0}
                    for idx_i, rec in tick_map_new.items():
                        rec_to_store = dict(rec)
                        ii = int(idx_i)
                        if bool(tick_bids_all is not None and tick_asks_all is not None) and ii + 1 < n:
                            b_entry = tick_minute_bounds.get(int(bar_time_ns[ii + 1]))
                            if b_entry is None or b_entry[1] <= b_entry[0] or not np.isfinite(tick_asks_all[b_entry[0]:b_entry[1]]).any():
                                rec_to_store = dict(invalid_rec)
                        tick_map[int(idx_i)] = rec_to_store
                        if args.tick_entry_cache_npz is not None:
                            cached_entries[int(idx_i)] = rec_to_store
                    unresolved_invalid = []
                    for idx_i in missing_entry_indices.tolist():
                        ii = int(idx_i)
                        if ii in tick_map_new:
                            continue
                        if _is_structurally_invalid_entry(ii):
                            unresolved_invalid.append(ii)
                            continue
                        if int(bar_time_ns[ii + 1]) not in tick_minute_bounds:
                            unresolved_invalid.append(ii)
                    if unresolved_invalid:
                        for idx_i in unresolved_invalid:
                            tick_map[int(idx_i)] = dict(invalid_rec)
                            if args.tick_entry_cache_npz is not None:
                                cached_entries[int(idx_i)] = dict(invalid_rec)
                        print(f"[prefilter-tick] entry_tick_invalid_unresolved_count={len(unresolved_invalid)}")
                    if args.tick_entry_cache_npz is not None:
                        _write_tick_entry_cache(args.tick_entry_cache_npz, tick_entry_cache_sig, cached_entries)
                    used_real_ticks = True
        else:
            print("[prefilter-tick-cache] tick_data_source=entry_cache_only")
            print("[prefilter-tick-cache] raw_tick_load_skipped=True reason=entry_cache_complete")
            print(f"[prefilter-tick-cache] requested_entries={len(entry_indices)}")
            print(f"[prefilter-tick-cache] cached_entries_used={len(cached_present)}")
            print("[prefilter-tick-cache] missing_entries=0")
            print("[prefilter-tick-cache] used_real_ticks_source=entry_cache")
            print("[prefilter-tick-raw-load] skipped=True reason=entry_cache_complete")
            if bool(args.debug_tick_cache_scope_counts):
                scope_entry_count, scope_critical_count, scope_would_load_count = _scope_tick_minute_counts(entry_indices)
                print(f"[prefilter-tick-cache] scope_entry_minutes_count={scope_entry_count}")
                print(f"[prefilter-tick-cache] scope_critical_minutes_count={scope_critical_count}")
                print(f"[prefilter-tick-cache] scope_minutes_would_load_count={scope_would_load_count}")
                print("[prefilter-tick-cache] scope_counts_source=ohlc_diagnostic_no_raw_tick_load")
            else:
                print("[prefilter-tick-cache] scope_critical_minutes_count=not_computed")
                print("[prefilter-tick-cache] scope_counts_hint=use --debug-tick-cache-scope-counts")
        if tick_map:
            y_ref = np.full(n, -1, dtype=np.int8)
            t_exit_ref = np.full(n, -1, dtype=np.int32)
            for idx_i, rec in tick_map.items():
                if int(idx_i) in entry_index_set:
                    y_ref[int(idx_i)] = np.int8(int(rec.get("y", -1)))
                    t_exit_ref[int(idx_i)] = np.int32(int(rec.get("t_exit", -1)))
            y_ref_train = y_ref[:train_idx]
            union_sel, _, _ = _select_entries_for_mask(fam_union, y_ref, t_exit_ref)
            union_train_mask = union_sel[:train_idx] & tradable_train & ((y_ref_train == 0) | (y_ref_train == 1))
            union_pos = int(np.sum(union_train_mask & (y_ref_train == 1)))
            union_neg = int(np.sum(union_train_mask & (y_ref_train == 0)))
            union_ratio = union_pos / max(1, union_neg)
            metrics_complete = True
            tick_missing_ids = {id(x) for x in tick_missing_items}
            for it in tick_scope_items:
                raw_m = np.asarray(it["mask"], dtype=bool)
                selected_m, raw_evaluable, _ = _select_entries_for_mask(raw_m, y_ref, t_exit_ref)
                valid_m = selected_m[:train_idx] & tradable_train & ((y_ref_train == 0) | (y_ref_train == 1))
                requested_entries = int(np.sum(raw_m & fam_union))
                cached_for_item = sum(1 for idx_i in np.flatnonzero(raw_m).tolist() if int(idx_i) in tick_map)
                if cached_for_item < requested_entries:
                    metrics_complete = False
                    if id(it) in tick_missing_ids:
                        continue
                tick_pos = int(np.sum(valid_m & (y_ref_train == 1)))
                tick_neg = int(np.sum(valid_m & (y_ref_train == 0)))
                tick_ratio = tick_pos / max(1, tick_neg)
                tick_lift = tick_ratio / max(1e-12, union_ratio)
                it["tick_single_pos_hits"] = tick_pos
                it["tick_single_neg_hits"] = tick_neg
                it["tick_single_mask_count"] = int(tick_pos + tick_neg)
                it["tick_single_ratio"] = float(tick_ratio)
                it["tick_lift"] = float(tick_lift)
                it["_single_pos_hits"] = tick_pos
                it["_single_neg_hits"] = tick_neg
                it["_single_mask_count"] = int(tick_pos + tick_neg)
                it["_single_ratio"] = float(tick_ratio)
                it["ratio"] = float(tick_ratio)
                it["lift"] = float(tick_lift)
            tick_refined_mode = any(_has_full_tick_metrics(it) for it in tick_scope_items)
            if not metrics_complete:
                print("[prefilter-tick-cache] warning: some current-scope entries are missing tick cache results; affected candidates remain missing.")
            print(f"[prefilter] tick-refined {args.tick_refine_scope} scope on {len(entry_indices)} union entry rows.")
    elif args.tick_data is not None and len(tick_scope_items) > 0:
        print(f"[prefilter-tick] tick_refine_scope={args.tick_refine_scope}")
        print(f"[prefilter-tick] tick_refine_candidates_count={len(tick_scope_items)}")
        print("[prefilter-tick] all requested candidates already have full tick metrics")
    elif args.tick_data is not None:
        print(f"[prefilter-tick] skipped: tick_refine_scope={args.tick_refine_scope} produced empty candidate scope")
    if coarse_out is not None:
        fam_top_keys = {str(z.get("candidate_key")) for z in fam_top}
        existing_rows = dict(coarse_existing_rows)
        for it in full_filtered_items:
            stable_k = str(it.get("stable_candidate_key", "")).strip() or _stable_candidate_key(str(it["col"]), str(it["op"]), float(it["value"]))
            row = existing_rows.get(stable_k, {})
            row.update({
                "candidate_key": str(it["candidate_key"]),
                "stable_candidate_key": stable_k,
                "col": str(it["col"]),
                "op": str(it["op"]),
                "value": float(it["value"]),
                "family": str(it["_family"]),
                "coarse_single_pos_hits": int(it.get("coarse_single_pos_hits", it.get("_single_pos_hits", 0))),
                "coarse_single_neg_hits": int(it.get("coarse_single_neg_hits", it.get("_single_neg_hits", 0))),
                "coarse_single_mask_count": int(it.get("coarse_single_mask_count", it.get("_single_mask_count", 0))),
                "coarse_single_ratio": float(it.get("coarse_single_ratio", it.get("_single_ratio", it.get("ratio", 0.0)))),
                "coarse_lift": float(it.get("coarse_lift", it.get("lift", 0.0))),
                "binary": int(bool(it.get("binary", False))),
                "kept_after_family_topn": int(str(it["candidate_key"]) in fam_top_keys),
                "build_min_single_pos_hits": int(coarse_build_min_pos),
                "build_max_single_mask_count": int(coarse_build_max_mask),
                "build_min_single_lift": float(coarse_build_min_lift),
                "__stage": "coarse",
                "__ctx_sig": coarse_ctx_sig,
            })
            existing_rows[stable_k] = row
        for row in existing_rows.values():
            row["build_min_single_pos_hits"] = int(coarse_build_min_pos)
            row["build_max_single_mask_count"] = int(coarse_build_max_mask)
            row["build_min_single_lift"] = float(coarse_build_min_lift)
            row["__stage"] = "coarse"
            row["__ctx_sig"] = coarse_ctx_sig
        coarse_rows = list(existing_rows.values())
        coarse_cols = ["candidate_key", "stable_candidate_key", "col", "op", "value", "family", "coarse_single_pos_hits", "coarse_single_neg_hits", "coarse_single_mask_count", "coarse_single_ratio", "coarse_lift", "binary", "kept_after_family_topn", "build_min_single_pos_hits", "build_max_single_mask_count", "build_min_single_lift", "__stage", "__ctx_sig"]
        _atomic_write_csv(coarse_out, _ordered_frame(coarse_rows, coarse_cols))
        print(f"[prefilter-candidates] wrote coarse CSV: {coarse_out} rows={len(coarse_rows)}")

    if bool(args.debug_atr_candidates):
        atr_candidates_written_to_csv = sum(1 for it in full_filtered_items if _is_atr_candidate_col(str(it.get("col", ""))))
        print(
            "[prefilter-atr-debug] "
            f"atr_cols_available={atr_cols_available} "
            f"atr_candidates_built={atr_candidates_built} "
            f"atr_candidates_after_min_pos={atr_candidates_after_min_pos} "
            f"atr_candidates_after_min_lift={atr_candidates_after_min_lift} "
            f"atr_candidates_after_mask_count={atr_candidates_after_mask_count} "
            f"atr_candidates_written_to_csv={atr_candidates_written_to_csv}"
        )
        if atr_cols_sample:
            print("[prefilter-atr-debug] atr_cols_sample=" + ",".join(atr_cols_sample))

    phase_d_pool_base: list[dict] = []
    tick_scope_keys = {str(it.get("candidate_key")) for it in tick_scope_items}
    for it in tick_scope_items:
        if _has_full_tick_metrics(it):
            phase_d_pool_base.append(it)

    if refined_out is not None:
        fam_top_keys = {str(z.get("candidate_key")) for z in fam_top}
        status_rank = {"full": 3, "missing": 2, "out_of_scope": 1}

        def _refined_row_key(row: dict) -> str:
            stable_k = str(row.get("stable_candidate_key", "")).strip()
            if stable_k:
                return stable_k
            try:
                return _stable_candidate_key(str(row.get("col", "")), str(row.get("op", "")), float(row.get("value", np.nan)))
            except Exception:
                return str(row.get("candidate_key_refined", row.get("candidate_key", ""))).strip()

        def _prefer_refined_row(old: dict | None, new: dict) -> dict:
            if old is None:
                return new
            old_rank = int(status_rank.get(str(old.get("tick_metric_status", "out_of_scope")), 0))
            new_rank = int(status_rank.get(str(new.get("tick_metric_status", "out_of_scope")), 0))
            if new_rank >= old_rank:
                return new
            old = dict(old)
            old["__ctx_sig"] = refined_ctx_sig
            old["__stage"] = "refined"
            return old

        existing_rows: dict[str, dict] = {}
        if refined_resume is not None:
            for _, r in refined_resume.iterrows():
                stable_k = _refined_row_key({k: r.get(k) for k in r.index})
                if not stable_k:
                    continue
                row = {k: r.get(k) for k in r.index}
                row["stable_candidate_key"] = stable_k
                row["tick_metric_status"] = str(row.get("tick_metric_status", "out_of_scope") or "out_of_scope")
                row["__ctx_sig"] = refined_ctx_sig
                row["__stage"] = "refined"
                existing_rows[stable_k] = _prefer_refined_row(existing_rows.get(stable_k), row)
        for it in full_filtered_items:
            stable_k = _stable_candidate_key(str(it["col"]), str(it["op"]), float(it["value"]))
            row = existing_rows.get(stable_k, {})
            row.update({
                "candidate_key_refined": str(it["candidate_key"]),
                "stable_candidate_key": stable_k,
                "col": str(it["col"]),
                "op": str(it["op"]),
                "value": float(it["value"]),
                "family": str(it["_family"]),
                "coarse_single_pos_hits": int(it.get("coarse_single_pos_hits", it.get("_single_pos_hits", 0))),
                "coarse_single_neg_hits": int(it.get("coarse_single_neg_hits", it.get("_single_neg_hits", 0))),
                "coarse_single_mask_count": int(it.get("coarse_single_mask_count", it.get("_single_mask_count", 0))),
                "coarse_single_ratio": float(it.get("coarse_single_ratio", it.get("_single_ratio", it.get("ratio", 0.0)))),
                "coarse_lift": float(it.get("coarse_lift", it.get("lift", 0.0))),
                "binary": int(bool(it.get("binary", False))),
                "kept_after_family_topn": int(str(it["candidate_key"]) in fam_top_keys),
                "__stage": "refined",
                "__ctx_sig": refined_ctx_sig,
            })
            for col_tick in req_tick_cols:
                row[col_tick] = float(it.get(col_tick, row.get(col_tick, np.nan)))
            if str(it.get("candidate_key")) not in tick_scope_keys:
                row["tick_metric_status"] = "out_of_scope"
            elif _has_full_tick_metrics(row):
                row["tick_metric_status"] = "full"
            else:
                row["tick_metric_status"] = "missing"
            existing_rows[stable_k] = _prefer_refined_row(existing_rows.get(stable_k), row)
        deduped_rows: dict[str, dict] = {}
        for row in existing_rows.values():
            k = _refined_row_key(row)
            if not k:
                continue
            deduped_rows[k] = _prefer_refined_row(deduped_rows.get(k), row)
        cand_rows = list(deduped_rows.values())
        refined_cols = ["candidate_key_refined", "stable_candidate_key", "col", "op", "value", "family", "coarse_single_pos_hits", "coarse_single_neg_hits", "coarse_single_mask_count", "coarse_single_ratio", "coarse_lift", "binary", "kept_after_family_topn", "tick_single_pos_hits", "tick_single_neg_hits", "tick_single_mask_count", "tick_single_ratio", "tick_lift", "tick_metric_status", "__stage", "__ctx_sig", "build_min_single_pos_hits", "build_max_single_mask_count", "build_min_single_lift"]
        _atomic_write_csv(refined_out, _ordered_frame(cand_rows, refined_cols))
        print(f"[prefilter-candidates] wrote refined CSV: {refined_out} rows={len(cand_rows)}")

    tick_candidates_requested = len(tick_scope_items)
    tick_candidates_with_metrics = sum(1 for it in tick_scope_items if _has_full_tick_metrics(it))
    tick_candidates_missing_metrics = max(0, tick_candidates_requested - tick_candidates_with_metrics)
    tick_refined_mode = bool(tick_refined_mode or len(phase_d_pool_base) > 0)
    print(f"[prefilter-tick] tick_candidates_requested={tick_candidates_requested}")
    print(f"[prefilter-tick] tick_candidates_with_metrics={tick_candidates_with_metrics}")
    print(f"[prefilter-tick] tick_candidates_missing_metrics={tick_candidates_missing_metrics}")
    print(f"[prefilter-tick] tick_candidates_used_for_phase_d={len(phase_d_pool_base)}")
    if tick_refine_t0 is not None:
        timing["tick_refine_sec"] = time.perf_counter() - tick_refine_t0


    if tick_refined_mode:
        rank_lift = _with_rank_context(sorted(fam_top, key=lambda z: (-float(z.get("ratio", 0.0)), -int(z.get("_single_pos_hits", 0)), int(z.get("_single_mask_count", 0)))), "ratio")
    else:
        rank_lift = _with_rank_context(sorted(fam_top, key=lambda z: z["lift"], reverse=True), "lift")
    rank_freq: list[dict] = []
    rank_ratio: list[dict] = []
    timing["ranking_prep_sec"] = time.perf_counter() - t0

    if bool(args.debug_timing_breakdown):
        print("[prefilter-timing] " + " | ".join([f"{k}={v:.3f}s" for k, v in timing.items()]))
    if bool(args.debug_timing_breakdown) or bool(args.debug_reject_stats):
        dist_items = sum(1 for x in items if str(x["col"]).startswith("dist_"))
        non_dist_items = len(items) - dist_items
        bin_items = sum(1 for x in items if bool(x.get("binary", False)))
        non_bin_items = len(items) - bin_items
        print(
            f"[prefilter-items] cols={len(cols)} items={len(items)} "
            f"filtered_items={len(filtered_items)} family_top_pool={len(rank_lift)} "
            f"dist_items={dist_items} non_dist_items={non_dist_items} "
            f"binary_items={bin_items} non_binary_items={non_bin_items}"
        )
        if bool(args.debug_reject_stats):
            print("[prefilter-single-rejects] " + " ".join([f"{k}={v}" for k, v in single_rejects.items()]))

    workers = max(1, int(os.cpu_count() or 1)) if str(args.workers).lower() == "auto" else max(1, int(args.workers))

    best_paths: list[dict] = []
    unlocked = 0
    prev_best = -np.inf
    same_ref = miner._build_same_reference_groups(df.iloc[:train_idx], [(str(c), ">=", 0.0) for c in cols])
    reject_stats: dict[str, int] = {
        "rejected_min_pos_per_week": 0,
        "rejected_min_main_score": 0,
        "rejected_same_parent_mask": 0,
        "rejected_not_strictly_better_than_parent": 0,
        "rejected_same_reference": 0,
        "rejected_bundle_anchor": 0,
        "rejected_binary_cap": 0,
        "rejected_duplicate_mask": 0,
    }

    def _build_mask_from_conds(conds: list[dict]) -> np.ndarray:
        mask = np.ones(n, dtype=bool)
        for c in conds:
            col = str(c["col"])
            op = str(c["op"])
            val = float(c["value"])
            domain = str(c.get("domain", "auto"))
            ftype = str(meta.get(col, {}).get("feature_type", "unknown"))
            use_binned = (domain == "binned") or (domain == "auto" and ftype in {"binary", "discrete", "continuous"})
            if use_binned and op in {"==", ">=", "<="}:
                x = pd.to_numeric(bdf[col], errors="coerce").to_numpy(copy=False)
            else:
                x = pd.to_numeric(df[col], errors="coerce").to_numpy(copy=False)
            if op == "==":
                if "lo_bin" in c and "hi_bin" in c:
                    lo_bin = float(c["lo_bin"])
                    hi_bin = float(c["hi_bin"])
                    mask &= np.isfinite(x) & (x >= lo_bin - 1e-6) & (x <= hi_bin + 1e-6)
                else:
                    mask &= np.isfinite(x) & (np.abs(x - val) <= 1e-6)
            elif op == ">=":
                mask &= np.isfinite(x) & (x >= val)
            elif op == "<=":
                mask &= np.isfinite(x) & (x <= val)
            else:
                return np.zeros(n, dtype=bool)
        return mask

    def _score_from_mask(mask: np.ndarray, only_lower_entry: bool | None = None, enforce_filters: bool = True) -> dict | None:
        selected_mask, raw_evaluable, clusters_count = _select_entries_for_mask(mask, y, t_exit, only_lower_entry=only_lower_entry)
        mt = selected_mask[:train_idx] & tradable_train
        pos_hits = int(np.sum(mt & (y_train == 1)))
        neg_hits = int(np.sum(mt & (y_train == 0)))
        days = max(1.0, float((df.index[train_idx - 1] - df.index[0]).total_seconds() / 86400.0))
        weeks = days / 7.0
        if enforce_filters and pos_hits < float(args.min_pos_per_week) * weeks:
            if bool(args.debug_reject_stats):
                reject_stats["rejected_min_pos_per_week"] += 1
            return None
        ratio = pos_hits / max(1, neg_hits)
        if enforce_filters and ratio < float(args.min_main_score):
            if bool(args.debug_reject_stats):
                reject_stats["rejected_min_main_score"] += 1
            return None
        mt_test = selected_mask[train_idx:] & tradable_test
        pos_test = int(np.sum(mt_test & (y_test == 1)))
        neg_test = int(np.sum(mt_test & (y_test == 0)))
        ratio_test = pos_test / max(1, neg_test)
        wf_mean, wf_min, wf_hits = _calc_wf(mt_test, y_test, int(args.wf_folds))
        precision = pos_hits / max(1, (pos_hits + neg_hits))
        selected_evaluable = int(pos_hits + neg_hits + pos_test + neg_test)
        return {
            "pos_hits": pos_hits,
            "neg_hits": neg_hits,
            "ratio": float(ratio),
            "precision": float(precision),
            "test_pos_hits": pos_test,
            "test_neg_hits": neg_test,
            "test_ratio": float(ratio_test),
            "wf_mean_ratio": wf_mean,
            "wf_min_ratio": wf_min,
            "wf_hits": wf_hits,
            "selection_ratio": float(selected_evaluable / raw_evaluable) if raw_evaluable > 0 else 0.0,
            "clusters_count": int(clusters_count),
        }

    def _train_key(r: dict) -> tuple[float, int, int]:
        return (float(r["ratio"]), int(r["pos_hits"]), -int(r["neg_hits"]))

    def _test_key(r: dict) -> tuple[float, int, int]:
        return (float(r["test_ratio"]), int(r["test_pos_hits"]), -int(r["test_neg_hits"]))

    def _mask_hash(mask: np.ndarray) -> str:
        return _mask_hash_arr(mask)

    def evaluate_combo(combo: Tuple[int, ...], pool_items: List[dict], parent: dict | None = None, source: str = "A") -> dict | None:
        conds = [pool_items[i] for i in combo]
        dedup = []
        seen_gid = set()
        for c in conds:
            fam = str(miner._parse_feature_meta(str(c["col"])).get("family", ""))
            if fam in {"dist_support", "dist_resist"}:
                gid = int(same_ref.get(str(c["col"]), 0))
                if gid > 0 and gid in seen_gid:
                    if bool(args.debug_reject_stats):
                        reject_stats["rejected_same_reference"] += 1
                    continue
                if gid > 0:
                    seen_gid.add(gid)
            dedup.append(c)
        conds = dedup
        # Binary flood cap per unlocked block
        if sum(1 for c in conds if c["binary"]) > int(args.binary_cap_per_block):
            if bool(args.debug_reject_stats):
                reject_stats["rejected_binary_cap"] += 1
            return None
        # bundle/anchor validity
        cond_triplets = [(str(c["col"]), str(c["op"]), float(c["value"])) for c in conds]
        ok_bundle, _ = miner.validate_binary_anchor_invariant(cond_triplets, cols)
        if not ok_bundle:
            if bool(args.debug_reject_stats):
                reject_stats["rejected_bundle_anchor"] += 1
            return None

        mask = _build_mask_from_conds(
            [{"col": c["col"], "op": c["op"], "value": c["value"], "domain": "auto"} for c in conds]
        )
        sc = _score_from_mask(mask)
        if sc is None:
            return None
        mh = _mask_hash(mask)
        if parent is not None:
            if str(parent.get("_mask_hash", "")) == mh:
                if bool(args.debug_reject_stats):
                    reject_stats["rejected_same_parent_mask"] += 1
                return None
            if _train_key(sc) <= _train_key(parent):
                if bool(args.debug_reject_stats):
                    reject_stats["rejected_not_strictly_better_than_parent"] += 1
                return None
        return {
            "conds": [{"col": str(c["col"]), "op": str(c["op"]), "value": float(c["value"]), "domain": "auto"} for c in conds],
            "_combo": tuple(sorted(int(x) for x in combo)),
            "_mask_hash": mh,
            "search_source": str(source),
            **sc,
        }

    csv_columns = [
        "path_index", "rule_human", "rule_json_id", "decode_type_info", "decode_bin_info", "pos_hits", "neg_hits",
        "remaining_hit_ratio", "precision_info", "test_pos_hits", "test_neg_hits", "test_ratio",
        "wf_mean_ratio", "wf_min_ratio", "wf_hits", "selection_ratio", "clusters_count",
        *( ["le_off_train_ratio", "le_off_test_ratio", "le_off_total_hits"] if bool(args.cluster_only_lower_entry) else [] ),
        "tp", "sl", "hold", "trail",
        "trail_activate", "trail_offset", "trail_factor", "search_source", "is_fallback_export",
    ]

    def _decode_cond_struct(c: dict) -> dict:
        col = str(c.get("col", ""))
        op = str(c.get("op", ""))
        m = meta.get(col, {}) if isinstance(meta, dict) else {}
        ftype = str(m.get("feature_type", "unknown"))
        out = {"feature_type": ftype, "lo_bin": None, "hi_bin": None, "raw_lo": None, "raw_hi": None, "raw_values": None, "bin_range_text": ""}
        if op != "==":
            return out
        try:
            lo = int(c.get("lo_bin", round(float(c.get("value", 0.0)))))
            hi = int(c.get("hi_bin", lo))
        except Exception:
            return out
        missing = int(m.get("missing_code", 0) or 0)
        eff = int(m.get("effective_bin_count", 0) or 0)
        if lo == missing or hi == missing or lo < 1 or hi < lo or (eff > 0 and hi > eff):
            return out
        out["lo_bin"] = int(lo)
        out["hi_bin"] = int(hi)
        out["bin_range_text"] = f"bin[{lo}]" if lo == hi else f"bin[{lo}..{hi}]"
        bin_to_raw = m.get("bin_to_raw", {})
        if ftype in {"binary", "discrete"} and isinstance(bin_to_raw, dict) and bin_to_raw:
            def _raw_for_bin(b: int):
                v = bin_to_raw.get(str(b), bin_to_raw.get(int(b)))
                if v is None:
                    return None
                try:
                    return float(v)
                except Exception:
                    return v

            raw_values = [_raw_for_bin(b) for b in range(lo, hi + 1)]
            raw_values = [v for v in raw_values if v is not None]
            out["raw_values"] = raw_values if raw_values else None
            out["raw_lo"] = _raw_for_bin(lo)
            out["raw_hi"] = _raw_for_bin(hi)
            return out

        edges = m.get("bin_edges", [])
        if isinstance(edges, list) and len(edges) >= hi:
            lo_pair = edges[lo - 1]
            hi_pair = edges[hi - 1]
            if isinstance(lo_pair, list) and isinstance(hi_pair, list) and len(lo_pair) == 2 and len(hi_pair) == 2:
                out["raw_lo"] = float(lo_pair[0])
                out["raw_hi"] = float(hi_pair[1])
        return out

    def _attach_decode_info(rule: dict) -> dict:
        rr = dict(rule)
        conds = []
        decodes = []
        for c in rr.get("conds", []):
            cc = dict(c)
            dec = _decode_cond_struct(cc)
            cc["decode"] = dec
            conds.append(cc)
            decodes.append(dec)
        rr["conds"] = conds
        rr["decode_type_info_structured"] = decodes
        return rr

    def _rules_to_rows(rules: list[dict], is_fallback_export: bool = False) -> list[dict]:
        def _decode_interval(col: str, lo: int, hi: int) -> str:
            m = meta.get(col, {}) if isinstance(meta, dict) else {}
            edges = m.get("bin_edges", [])
            eff = int(m.get("effective_bin_count", 0) or 0)
            if not isinstance(edges, list) or eff <= 0 or lo < 1 or hi > eff or lo > hi:
                raise ValueError(f"Undecodable interval for {col}: lo={lo} hi={hi} eff={eff}")
            lo_pair = edges[lo - 1]
            hi_pair = edges[hi - 1]
            if not (isinstance(lo_pair, list) and isinstance(hi_pair, list) and len(lo_pair) == 2 and len(hi_pair) == 2):
                raise ValueError(f"Invalid bin_edges for {col}")
            lo_raw = float(lo_pair[0])
            hi_raw = float(hi_pair[1])
            return f"{col} in [{lo_raw:.6g}, {hi_raw:.6g}] (bin[{lo}..{hi}])"

        rows: list[dict] = []
        for i, r in enumerate(rules, start=1):
            decoded_parts = []
            type_parts = []
            bin_parts = []
            for c in r["conds"]:
                col = str(c["col"])
                op = str(c["op"])
                val = float(c["value"])
                ftype = str(meta.get(col, {}).get("feature_type", "unknown"))
                if op == "==":
                    if "lo_bin" in c and "hi_bin" in c and int(c["lo_bin"]) != int(c["hi_bin"]):
                        lo_bin = int(c["lo_bin"])
                        hi_bin = int(c["hi_bin"])
                        decoded_parts.append(_decode_interval(col, lo_bin, hi_bin))
                    else:
                        decoded_parts.append(miner.decode_bin_condition(col, val, meta))
                else:
                    decoded_parts.append(f"{col} {op} {val:.6g}")
                dec = c.get("decode") if isinstance(c.get("decode"), dict) else _decode_cond_struct(c)
                if dec.get("bin_range_text"):
                    raw_lo = dec.get("raw_lo")
                    raw_hi = dec.get("raw_hi")
                    raw_txt = "" if raw_lo is None or raw_hi is None else f" raw[{float(raw_lo):.6g}..{float(raw_hi):.6g}]"
                    type_parts.append(f"{col}:{ftype} {dec.get('bin_range_text')}{raw_txt}")
                    bin_parts.append(json.dumps({"col": col, **dec}, ensure_ascii=False, sort_keys=True))
                else:
                    type_parts.append(f"{col}:{ftype}")
            row = {
                "path_index": i,
                "rule_human": " & ".join(decoded_parts),
                "rule_json_id": f"rule_{i}",
                "decode_type_info": " | ".join(type_parts),
                "decode_bin_info": " | ".join(bin_parts),
                "pos_hits": r["pos_hits"],
                "neg_hits": r["neg_hits"],
                "remaining_hit_ratio": r["ratio"],
                "precision_info": r["precision"],
                "test_pos_hits": r["test_pos_hits"],
                "test_neg_hits": r["test_neg_hits"],
                "test_ratio": r["test_ratio"],
                "wf_mean_ratio": r["wf_mean_ratio"],
                "wf_min_ratio": r["wf_min_ratio"],
                "wf_hits": r["wf_hits"],
                "selection_ratio": float(r.get("selection_ratio", 0.0)),
                "clusters_count": int(r.get("clusters_count", 0)),
                "tp": ",".join([f"{x:.8g}" for x in tps]),
                "sl": float(args.sl),
                "hold": int(args.hold),
                "trail": int(args.trail),
                "trail_activate": float(args.trail_activate),
                "trail_offset": float(args.trail_offset),
                "trail_factor": float(args.trail_factor),
                "search_source": str(r.get("search_source", "")),
                "is_fallback_export": int(bool(is_fallback_export)),
            }
            if bool(args.cluster_only_lower_entry):
                row["le_off_train_ratio"] = float(r.get("le_off_train_ratio", np.nan))
                row["le_off_test_ratio"] = float(r.get("le_off_test_ratio", np.nan))
                row["le_off_total_hits"] = int(r.get("le_off_total_hits", 0))
            rows.append(row)
        return rows

    def _validate_rows(rows: list[dict]) -> None:
        if not rows:
            return
        keys0 = set(rows[0].keys())
        for i, r in enumerate(rows, start=1):
            if set(r.keys()) != keys0:
                raise ValueError(f"CSV row key mismatch at row {i}")
            txt = str(r.get("rule_human", ""))
            if txt.count("(") != txt.count(")"):
                raise ValueError(f"rule_human parentheses mismatch at row {i}: {txt!r}")

    def _dedupe_mask(rows: list[dict]) -> list[dict]:
        buckets: dict[tuple[int, int, int, int], dict[str, dict]] = {}
        for r in rows:
            key = (int(r["pos_hits"]), int(r["neg_hits"]), int(r["test_pos_hits"]), int(r["test_neg_hits"]))
            buckets.setdefault(key, {})
            mh = str(r.get("_mask_hash", ""))
            cur = buckets[key].get(mh)
            if cur is None:
                buckets[key][mh] = r
                continue
            if bool(args.debug_reject_stats):
                reject_stats["rejected_duplicate_mask"] += 1
            if len(r.get("conds", [])) < len(cur.get("conds", [])):
                buckets[key][mh] = r
            elif len(r.get("conds", [])) == len(cur.get("conds", [])):
                if _test_key(r) > _test_key(cur) or (_test_key(r) == _test_key(cur) and _train_key(r) > _train_key(cur)):
                    buckets[key][mh] = r
        out: list[dict] = []
        for m in buckets.values():
            out.extend(m.values())
        return out

    def _save_progress(valid_pool: list[dict], state: dict) -> None:
        non_d_rows = [r for r in valid_pool if str(r.get("search_source", "")) != "D"]
        d_rows = [r for r in valid_pool if str(r.get("search_source", "")) == "D"]
        non_d_rows = sorted(_dedupe_mask(non_d_rows), key=lambda z: (-float(z["ratio"]), -int(z["pos_hits"]), int(z["neg_hits"])))[: int(args.top_paths)]
        d_rows = sorted(_dedupe_mask(d_rows), key=lambda z: (-float(z["ratio"]), -int(z["pos_hits"]), int(z["neg_hits"])))[: int(args.top_paths)]
        export_subset = [_attach_decode_info(r) for r in (non_d_rows + d_rows)]
        rules_only = {"version": 2, "rules": export_subset}
        _atomic_write_text(args.out_rules_json, json.dumps(rules_only, ensure_ascii=False, indent=2))
        rows = _rules_to_rows(export_subset, is_fallback_export=False)
        _validate_rows(rows)
        if rows:
            _atomic_write_csv(args.out_rules_csv, pd.DataFrame(rows))
        else:
            _atomic_write_csv(args.out_rules_csv, pd.DataFrame(columns=csv_columns))

    rng = random.Random(int(args.batch_random_seed))
    valid_pool: list[dict] = []
    progress_state: dict = {}
    phase_c_was_active = False
    original_early_stop_window_combos = int(args.early_stop_window_combos)
    stop_after_batch_requested = False
    force_phase_b_requested = False
    completed_phase_b_pool_keys: set[tuple[str, ...]] = set()

    def _phase_b_pool_key(pool_items: list[dict]) -> tuple[str, ...]:
        keys: list[str] = []
        for it in pool_items:
            stable_k = str(it.get("stable_candidate_key", "")).strip()
            if not stable_k:
                try:
                    stable_k = _stable_candidate_key(str(it.get("col", "")), str(it.get("op", "")), float(it.get("value", np.nan)))
                except Exception:
                    stable_k = ""
            keys.append(stable_k or str(it.get("candidate_key", "")))
        return tuple(keys)
    if args.control_file is not None:
        _write_control_none(args.control_file)
    try:
        while True:
            force_phase = "A"
            unlocked_next = unlocked + int(args.step_size)
            pool = _build_unlocked_pool(
                miner_mod=miner,
                rank_lists=[rank_lift],
                all_candidate_cols=cols,
                unlocked_next=unlocked_next,
                step_size=int(args.step_size),
                binary_cap_per_list_block=10 ** 9,
                binary_anchor_lookahead_blocks=int(args.binary_anchor_lookahead_blocks),
                binary_cap_per_block=10 ** 9,
            )
            if not pool:
                break
            idxs = list(range(len(pool)))
            old_pool_size = max(0, len(pool) - int(args.step_size))
            phase_a = (len(valid_pool) < int(args.max_valids)) and (not force_phase_b_requested)
            phase_b_pool_key = _phase_b_pool_key(pool)
            if (not phase_a) and phase_b_pool_key in completed_phase_b_pool_keys:
                print(f"[prefilter-phase-b] skipped: completed pool key already processed. pool_size={len(pool)}")
                force_phase = "C"
                break
            if (not phase_a) and not valid_pool:
                print("[prefilter-phase-b] skipped: no parent seeds in valid_pool after phase A.")
                force_phase = "C"
                break
            shard_specs: list[tuple[int, int, int, int]] = []
            batch_eff = max(256, int(args.batch_size))
            sid = 0
            if phase_a:
                for r in range(2, int(args.max_path_conds) + 1):
                    if r > len(idxs):
                        continue
                    total_r = sum(1 for _ in _iter_combinations_with_new(idxs, r, old_pool_size))
                    for st in range(0, total_r, batch_eff):
                        shard_specs.append((sid, r, st, min(batch_eff, total_r - st)))
                        sid += 1
            else:
                # Phase B streams directly from parent_ext_iter until it is exhausted.
                pass
            rng.shuffle(shard_specs)
            tested = 0
            valid_round = 0
            mut_i = 0
            seeds = sorted(valid_pool, key=lambda z: (-float(z["ratio"]), -int(z["pos_hits"]), int(z["neg_hits"])))[: int(args.max_valids)]
            mut_combos: collections.deque[tuple[tuple[int, ...], dict]] = collections.deque()
            parent_ext_iter = None
            if (not phase_a) or (phase_a and len(seeds) > 0 and unlocked_next > int(args.step_size)):
                parent_ext_iter = _iter_all_parent_extensions(seeds, idxs, int(args.max_path_conds))
            combos_total_free = 0
            rmax = min(int(args.max_path_conds), len(idxs))
            if phase_a:
                for rr in range(2, rmax + 1):
                    combos_total_free += sum(1 for _ in _iter_combinations_with_new(idxs, rr, old_pool_size))
                for _ in range(max(1, batch_eff // 2)):
                    nxt = next(parent_ext_iter, None) if parent_ext_iter is not None else None
                    if nxt is None:
                        break
                    mut_combos.append(nxt)
                if mut_combos:
                    tmp = list(mut_combos)
                    rng.shuffle(tmp)
                    mut_combos = collections.deque(tmp)
            else:
                combos_total_free = -1
            combos_total_parent_extensions = int(len(mut_combos)) if phase_a else -1
            combos_total = int(combos_total_free + combos_total_parent_extensions) if phase_a else -1

            hist: list[tuple[int, float, tuple[float, int, int]]] = []
            phase_b_exhausted = False
            executor_cache: dict[int, concurrent.futures.ThreadPoolExecutor] = {}
            for spec in (shard_specs if phase_a else itertools.repeat((-1, -1, 0, 0))):
                est_mem_gb = (batch_eff * max(1, len(pool)) * 8.0) / (1024 ** 3)
                target_mem_gb = max(0.5, float(args.memory_soft_limit_gb) * 0.75)
                while est_mem_gb > target_mem_gb and batch_eff > 256:
                    batch_eff = max(256, batch_eff // 2)
                    est_mem_gb = (batch_eff * max(1, len(pool)) * 8.0) / (1024 ** 3)
                workers_eff = workers if est_mem_gb <= target_mem_gb else max(1, workers // 2)
                if phase_a:
                    _sid, rr, st, cnt = spec
                    new_combos = list(itertools.islice(_iter_combinations_with_new(idxs, rr, old_pool_size), st, st + cnt))
                    combos = list(new_combos)
                    parents = [None] * len(new_combos)
                    # from pool 2 onward in phase A: also expand existing valid rules in parallel
                    if mut_combos:
                        fill_n = max(1, batch_eff // 2)
                        exp_part = []
                        for _ in range(fill_n):
                            if not mut_combos:
                                break
                            exp_part.append(mut_combos.popleft())
                        combos.extend([x[0] for x in exp_part])
                        parents.extend([x[1] for x in exp_part])
                    if parent_ext_iter is not None and len(mut_combos) < max(1, batch_eff // 2):
                        refill = []
                        for _ in range(max(1, batch_eff // 2)):
                            nxt = next(parent_ext_iter, None)
                            if nxt is None:
                                break
                            refill.append(nxt)
                        if refill:
                            rng.shuffle(refill)
                            mut_combos.extend(refill)
                else:
                    combos_par = list(itertools.islice(parent_ext_iter, batch_eff)) if parent_ext_iter is not None else []
                    combos = [x[0] for x in combos_par]
                    parents = [x[1] for x in combos_par]
                    if not combos:
                        phase_b_exhausted = True
                        break
                out_batch: list[dict] = []
                if workers_eff <= 1:
                    for i, cb in enumerate(combos):
                        rr = evaluate_combo(cb, pool, parents[i], source=("A" if phase_a else "B"))
                        if rr is not None:
                            out_batch.append(rr)
                else:
                    ex = executor_cache.get(int(workers_eff))
                    if ex is None:
                        ex = concurrent.futures.ThreadPoolExecutor(max_workers=int(workers_eff))
                        executor_cache[int(workers_eff)] = ex
                    futs = [ex.submit(evaluate_combo, cb, pool, parents[i], ("A" if phase_a else "B")) for i, cb in enumerate(combos)]
                    for f in futs:
                        rr = f.result()
                        if rr is not None:
                            out_batch.append(rr)
                tested += len(combos)
                valid_round += len(out_batch)
                valid_pool.extend(out_batch)
                valid_pool = _dedupe_mask(valid_pool)
                valid_pool = sorted(valid_pool, key=lambda z: (-float(z["ratio"]), -int(z["pos_hits"]), int(z["neg_hits"])))[: int(args.max_valids)]
                topk = valid_pool[: int(args.early_stop_top_k)]
                a1 = _train_key(topk[0]) if topk else (-np.inf, -1, 1)
                avg_ratio = float(np.mean([float(x["ratio"]) for x in topk])) if topk else float("nan")
                hist.append((len(combos), avg_ratio, a1))
                progress_state = {
                    "round": int(unlocked_next),
                    "pool_size": int(len(pool)),
                    "combos_total_free": int(combos_total_free),
                    "combos_total_parent_extensions": int(combos_total_parent_extensions),
                    "combos_total": int(combos_total) if combos_total_parent_extensions >= 0 else -1,
                    "tested": int(tested),
                    "valid_kept": int(len(valid_pool)),
                    "valid_new_round": int(valid_round),
                    "batch_size": int(batch_eff),
                    "workers": int(workers_eff),
                    "best_a1": list(a1),
                    "topk_summary": [{"ratio": float(x["ratio"]), "pos_hits": int(x["pos_hits"]), "neg_hits": int(x["neg_hits"])} for x in topk[:10]],
                    "phase": "A" if phase_a else "B",
                }
                _save_progress(valid_pool, progress_state)
                if phase_a:
                    print(
                        f"[prefilter-progress] round={unlocked_next} phase=A "
                        f"pool_size={len(pool)} combos_total_free={progress_state['combos_total_free']} "
                        f"combos_total_parent_extensions={progress_state['combos_total_parent_extensions']} "
                        f"combos_total={progress_state['combos_total']} "
                        f"tested={tested} valid_kept={len(valid_pool)} valid_new_round={valid_round} batch_size={batch_eff} workers={workers_eff}"
                    )
                else:
                    print(
                        f"[prefilter-progress] round={unlocked_next} phase=B "
                        f"pool_size={len(pool)} combos_total=unknown/streaming parent_extension_mode=streaming "
                        f"tested={tested} valid_kept={len(valid_pool)} valid_new_round={valid_round} batch_size={batch_eff} workers={workers_eff}"
                    )
                cmd = _read_control_command(args.control_file)
                if cmd == "export_now":
                    print("[prefilter-control] export_now")
                    _save_progress(valid_pool, progress_state)
                    _write_control_none(args.control_file)
                elif cmd == "disable_early_stop":
                    print("[prefilter-control] disable_early_stop")
                    args.early_stop_window_combos = 10 ** 18
                    _write_control_none(args.control_file)
                elif cmd == "enable_early_stop":
                    print("[prefilter-control] enable_early_stop")
                    args.early_stop_window_combos = int(original_early_stop_window_combos)
                    _write_control_none(args.control_file)
                elif cmd == "stop_after_batch":
                    print("[prefilter-control] stop_after_batch requested; will stop cleanly after batch")
                    _save_progress(valid_pool, progress_state)
                    _write_control_none(args.control_file)
                    stop_after_batch_requested = True
                    break
                elif cmd == "force_phase_b":
                    print("[prefilter-control] force_phase_b requested; will end phase A and continue with phase B")
                    force_phase_b_requested = True
                    _write_control_none(args.control_file)
                    if phase_a:
                        break
                elif cmd == "force_phase_c":
                    print("[prefilter-control] force_phase_c")
                    force_phase = "C"
                    phase_c_was_active = True
                    _write_control_none(args.control_file)
                    break
                elif cmd == "force_phase_d":
                    if phase_c_was_active:
                        print("[prefilter-control] force_phase_d")
                        force_phase = "D"
                    else:
                        print("[prefilter-control] force_phase_d ignored (phase C not active yet)")
                    _write_control_none(args.control_file)
                if bool(args.debug_reject_stats):
                    print("[prefilter-reject-stats] " + " ".join([f"{k}={v}" for k, v in reject_stats.items()]))
                covered = 0
                win = []
                for h in reversed(hist):
                    win.append(h)
                    covered += int(h[0])
                    if covered >= int(args.early_stop_window_combos):
                        break
                if covered >= int(args.early_stop_window_combos) and len(win) >= 2:
                    old = win[-1][1]
                    new = win[0][1]
                    improve_pct = 0.0 if (not np.isfinite(old) or old == 0.0) else ((new - old) / abs(old) * 100.0)
                    a1_improved = any(h[2] > win[-1][2] for h in win[:-1])
                    if improve_pct < float(args.early_stop_avg_improve_pct) and not a1_improved:
                        if phase_a:
                            print("[prefilter-progress] early stop in phase A; continuing with phase B next.")
                            force_phase_b_requested = True
                        else:
                            print("[prefilter-progress] early stop in phase B; continuing with phase C/D next.")
                            force_phase = "C"
                        break

            for ex in executor_cache.values():
                ex.shutdown(wait=True)

            if (not phase_a) and (phase_b_exhausted or force_phase == "C"):
                completed_phase_b_pool_keys.add(phase_b_pool_key)
            if stop_after_batch_requested:
                print("[prefilter-progress] stopped after batch as requested.")
                break
            if (not phase_a) and phase_b_exhausted:
                print("[prefilter-phase-b] completed: parent_ext_iter exhausted")
                force_phase = "C"
            if force_phase in {"C", "D"}:
                phase_c_was_active = phase_c_was_active or (force_phase == "C")
                print("[prefilter-progress] force phase switch requested; continuing with implemented post A/B phases.")
                break
            if not valid_pool:
                unlocked = unlocked_next
                if unlocked >= len(rank_lift):
                    break
                continue
            best_paths = list(valid_pool)
            cur_best = float(best_paths[0]["ratio"]) if best_paths else -np.inf
            if phase_a and (not force_phase_b_requested) and cur_best <= prev_best + 1e-12:
                break
            prev_best = cur_best
            unlocked = unlocked_next
    except KeyboardInterrupt:
        print("[prefilter] interrupted; checkpoint saved.")
        _save_progress(valid_pool, progress_state)

    # Phase C: family-top-pool beam search
    pool_c = _build_unlocked_pool(
        miner_mod=miner,
        rank_lists=[rank_lift],
        all_candidate_cols=cols,
        unlocked_next=len(rank_lift),
        step_size=int(args.step_size),
        binary_cap_per_list_block=10 ** 9,
        binary_anchor_lookahead_blocks=int(args.binary_anchor_lookahead_blocks),
        binary_cap_per_block=10 ** 9,
    )
    if len(pool_c) == 0:
        print("[prefilter-phase-c] skipped: empty family-top pool")
    else:
        idxs_c = list(range(len(pool_c)))
        seed_rules = sorted(valid_pool, key=lambda z: (-float(z["ratio"]), -int(z["pos_hits"]), int(z["neg_hits"])))[: max(1, int(args.phase_c_beam_width))]
        cond_to_idx_c = {(str(c["col"]), str(c["op"]), float(c["value"])): i for i, c in enumerate(pool_c)}
        mapped_beam: list[tuple[int, ...]] = []
        for s in seed_rules:
            mapped = []
            ok = True
            for c in s.get("conds", []):
                key = (str(c["col"]), str(c["op"]), float(c["value"]))
                idx = cond_to_idx_c.get(key)
                if idx is None:
                    ok = False
                    break
                mapped.append(int(idx))
            if ok and mapped:
                mapped_beam.append(tuple(sorted(set(mapped))))
        if not mapped_beam:
            mapped_beam = [tuple([i]) for i in idxs_c[: max(1, min(int(args.phase_c_beam_width), len(idxs_c)))]]
        beam = mapped_beam
        beam = [b for b in beam if 1 <= len(b) <= int(args.phase_c_max_conds)]
        print(f"[prefilter-phase-c] start beam={len(beam)} pool={len(pool_c)}")
        phase_c_was_active = True
        phase_c_level = 0
        force_phase_d_after_c = False
        while beam:
            phase_c_level += 1
            level_t0 = time.perf_counter()
            seeds_at_level_start = len(beam)
            generated_extensions = 0
            evaluated_combos = 0
            truncated = False
            out_c: list[dict] = []
            def _iter_cands_c():
                for base in beam:
                    cset = set(base)
                    add_cands = [x for x in idxs_c if x not in cset]
                    for add_k in range(max(1, int(args.phase_c_add_min)), max(1, int(args.phase_c_add_max)) + 1):
                        if len(base) + add_k > int(args.phase_c_max_conds) or len(base) + add_k > int(args.max_path_conds):
                            continue
                        for adds in itertools.combinations(add_cands, add_k):
                            yield tuple(sorted(cset | set(adds)))
            had_any = False
            for chunk in _batched(_iter_cands_c(), max(64, int(args.batch_size))):
                had_any = True
                cap = int(args.phase_c_max_generated_per_level)
                if cap > 0 and generated_extensions + len(chunk) >= cap:
                    chunk = chunk[: max(0, cap - generated_extensions)]
                    truncated = True
                generated_extensions += len(chunk)
                for cb in chunk:
                    evaluated_combos += 1
                    rr = evaluate_combo(cb, pool_c, None, source="C")
                    if rr is not None:
                        out_c.append(rr)
                if truncated:
                    break
            if not had_any:
                print(f"[prefilter-phase-c] level={phase_c_level} depth_range=add{int(args.phase_c_add_min)}..add{int(args.phase_c_add_max)} seeds={seeds_at_level_start} pool={len(pool_c)} estimated_generated=unknown generated=0 evaluated=0 truncated={truncated} rejected_duplicates=unknown rejected_invalid_rules=0 new_valid=0 beam_kept=0 valid_pool={len(valid_pool)} elapsed={time.perf_counter() - level_t0:.2f}s")
                break
            if not out_c:
                print(f"[prefilter-phase-c] level={phase_c_level} depth_range=add{int(args.phase_c_add_min)}..add{int(args.phase_c_add_max)} seeds={seeds_at_level_start} pool={len(pool_c)} estimated_generated=unknown generated={generated_extensions} evaluated={evaluated_combos} truncated={truncated} rejected_duplicates=unknown rejected_invalid_rules={max(0, evaluated_combos - len(out_c))} new_valid=0 beam_kept=0 valid_pool={len(valid_pool)} elapsed={time.perf_counter() - level_t0:.2f}s")
                break
            valid_before = len(valid_pool)
            valid_pool.extend(out_c)
            valid_pool = _dedupe_mask(valid_pool)
            valid_pool = sorted(valid_pool, key=lambda z: (-float(z["ratio"]), -int(z["pos_hits"]), int(z["neg_hits"])))[: int(args.max_valids)]
            _save_progress(valid_pool, progress_state)
            beam_rules = sorted(out_c, key=lambda z: (-float(z["ratio"]), -int(z["pos_hits"]), int(z["neg_hits"])))[: int(args.phase_c_beam_width)]
            beam = [tuple(sorted(int(x) for x in r.get("_combo", tuple()))) for r in beam_rules]
            best_ratio = float(valid_pool[0]["ratio"]) if valid_pool else float("nan")
            print(f"[prefilter-phase-c] level={phase_c_level} depth_range=add{int(args.phase_c_add_min)}..add{int(args.phase_c_add_max)} seeds={seeds_at_level_start} pool={len(pool_c)} estimated_generated=unknown generated={generated_extensions} evaluated={evaluated_combos} truncated={truncated} rejected_duplicates=unknown rejected_invalid_rules={max(0, evaluated_combos - len(out_c))} new_valid={max(0, len(valid_pool) - valid_before)} beam_kept={len(beam)} valid_pool={len(valid_pool)} elapsed={time.perf_counter() - level_t0:.2f}s best_ratio={best_ratio:.6g}")
            cmd = _read_control_command(args.control_file)
            if cmd == "export_now":
                print("[prefilter-control] export_now (phase C)"); _save_progress(valid_pool, progress_state); _write_control_none(args.control_file)
            elif cmd == "stop_after_batch":
                print("[prefilter-control] stop_after_batch requested in phase C; stopping after beam level"); _save_progress(valid_pool, progress_state); _write_control_none(args.control_file); stop_after_batch_requested = True; break
            elif cmd == "force_phase_c":
                print("[prefilter-control] force_phase_c ignored (already in phase C)"); _write_control_none(args.control_file)
            elif cmd == "force_phase_d":
                print("[prefilter-control] force_phase_d"); force_phase_d_after_c = True; _write_control_none(args.control_file); break
            elif cmd == "disable_early_stop":
                print("[prefilter-control] disable_early_stop"); args.early_stop_window_combos = 10 ** 18; _write_control_none(args.control_file)
            elif cmd == "enable_early_stop":
                print("[prefilter-control] enable_early_stop"); args.early_stop_window_combos = int(original_early_stop_window_combos); _write_control_none(args.control_file)
        print("[prefilter-phase-c] done")

    # Phase D: full tick beam search
    d_valid_pool: list[dict] = []
    if stop_after_batch_requested:
        print("[prefilter-phase-d] skipped: stop_after_batch requested before phase D")
    elif not tick_refined_mode:
        print("[prefilter-phase-d] skipped: no tick metrics available")
    else:
        tick_pool_raw = list(phase_d_pool_base)
        tick_pool = [r for r in tick_pool_raw if _has_full_tick_metrics(r)]
        print(f"[prefilter-phase-d] tick_candidates_available={len(tick_pool_raw)} tick_candidates_with_full_metrics={len(tick_pool)} (source=current tick_refine_scope full tick candidates)")
        if not tick_pool:
            print("[prefilter-phase-d] skipped: tick pool empty")
        else:
            tick_pool = sorted(tick_pool, key=lambda z: (-float(z.get("tick_single_ratio", -np.inf)), -int(z.get("tick_single_pos_hits", z.get("_single_pos_hits", 0)))))
            idxs_d = list(range(len(tick_pool)))
            start_depth = max(1, min(int(args.phase_d_start_conds), int(args.phase_d_max_conds), int(args.max_path_conds), len(idxs_d)))
            d_beam: list[tuple[int, ...]] = []
            phase_d_level = 0

            def _phase_d_control() -> bool:
                nonlocal stop_after_batch_requested
                cmd = _read_control_command(args.control_file)
                if cmd == "export_now":
                    print("[prefilter-control] export_now (phase D)"); _save_progress(valid_pool + d_valid_pool, progress_state); _write_control_none(args.control_file)
                elif cmd == "stop_after_batch":
                    print("[prefilter-control] stop_after_batch requested in phase D; stopping after beam level"); _save_progress(valid_pool + d_valid_pool, progress_state); _write_control_none(args.control_file); stop_after_batch_requested = True; return True
                elif cmd == "force_phase_c":
                    print("[prefilter-control] force_phase_c ignored (already in phase D)"); _write_control_none(args.control_file)
                elif cmd == "force_phase_d":
                    print("[prefilter-control] force_phase_d ignored (already in phase D)"); _write_control_none(args.control_file)
                elif cmd == "disable_early_stop":
                    print("[prefilter-control] disable_early_stop"); args.early_stop_window_combos = 10 ** 18; _write_control_none(args.control_file)
                elif cmd == "enable_early_stop":
                    print("[prefilter-control] enable_early_stop"); args.early_stop_window_combos = int(original_early_stop_window_combos); _write_control_none(args.control_file)
                return False

            def _d_sort_key(r: dict) -> tuple[float, int, int, int]:
                return (-float(r["ratio"]), -int(r["pos_hits"]), int(r["neg_hits"]), len(tuple(r.get("_combo", tuple()))))

            def _keep_d_pool(rows: list[dict]) -> list[dict]:
                rows = _dedupe_mask(rows)
                return sorted(rows, key=_d_sort_key)[: int(args.max_valids)]

            # level=0: evaluate all start-depth combinations directly, streamingly.
            level_t0 = time.perf_counter()
            generated_extensions = 0
            evaluated_combos = 0
            truncated = False
            out_d: list[dict] = []
            for chunk in _batched(itertools.combinations(idxs_d, start_depth), max(64, int(args.batch_size))):
                cap = int(args.phase_d_max_generated_per_level)
                if cap > 0 and generated_extensions + len(chunk) >= cap:
                    chunk = chunk[: max(0, cap - generated_extensions)]
                    truncated = True
                generated_extensions += len(chunk)
                for cb in chunk:
                    evaluated_combos += 1
                    rr = evaluate_combo(tuple(sorted(cb)), tick_pool, None, source="D")
                    if rr is not None:
                        out_d.append(rr)
                if truncated:
                    break
            if out_d:
                before = len(d_valid_pool)
                d_valid_pool = _keep_d_pool(d_valid_pool + out_d)
                _save_progress(valid_pool + d_valid_pool, progress_state)
                beam_rules = sorted(_dedupe_mask(out_d), key=_d_sort_key)[: int(args.phase_d_beam_width)]
                d_beam = [tuple(sorted(int(x) for x in r.get("_combo", tuple()))) for r in beam_rules]
                best_ratio = float(d_valid_pool[0]["ratio"]) if d_valid_pool else float("nan")
                print(f"[prefilter-phase-d] level=0 depth={start_depth} seeds=all pool={len(tick_pool)} estimated_generated=unknown generated={generated_extensions} evaluated={evaluated_combos} truncated={truncated} rejected_duplicates=unknown rejected_invalid_rules={max(0, evaluated_combos - len(out_d))} new_valid={max(0, len(d_valid_pool) - before)} beam_kept={len(d_beam)} d_valid_pool={len(d_valid_pool)} elapsed={time.perf_counter() - level_t0:.2f}s best_ratio={best_ratio:.6g}")
            else:
                print(f"[prefilter-phase-d] level=0 depth={start_depth} seeds=all pool={len(tick_pool)} estimated_generated=unknown generated={generated_extensions} evaluated={evaluated_combos} truncated={truncated} rejected_duplicates=unknown rejected_invalid_rules={max(0, evaluated_combos)} new_valid=0 beam_kept=0 d_valid_pool={len(d_valid_pool)} elapsed={time.perf_counter() - level_t0:.2f}s")
                _save_progress(valid_pool + d_valid_pool, progress_state)
            if _phase_d_control():
                d_beam = []

            while d_beam and not stop_after_batch_requested:
                phase_d_level += 1
                level_t0 = time.perf_counter()
                seeds_at_level_start = len(d_beam)
                generated_extensions = 0
                evaluated_combos = 0
                truncated = False
                out_d = []
                current_min_base_len = min((len(b) for b in d_beam), default=0)
                current_max_base_len = max((len(b) for b in d_beam), default=0)
                min_depth = current_min_base_len + max(1, int(args.phase_d_add_min))
                hard_max_depth = min(int(args.phase_d_max_conds), int(args.max_path_conds))
                max_depth = min(current_max_base_len + max(1, int(args.phase_d_add_max)), hard_max_depth)
                if min_depth > max_depth:
                    print("[prefilter-phase-d] done: max depth reached")
                    break

                def _iter_cands_d():
                    for base in d_beam:
                        cset = set(base)
                        add_cands = [x for x in idxs_d if x not in cset]
                        for add_k in range(max(1, int(args.phase_d_add_min)), max(1, int(args.phase_d_add_max)) + 1):
                            if len(base) + add_k > int(args.phase_d_max_conds) or len(base) + add_k > int(args.max_path_conds):
                                continue
                            for adds in itertools.combinations(add_cands, add_k):
                                yield tuple(sorted(cset | set(adds)))

                had_any = False
                for chunk in _batched(_iter_cands_d(), max(64, int(args.batch_size))):
                    had_any = True
                    cap = int(args.phase_d_max_generated_per_level)
                    if cap > 0 and generated_extensions + len(chunk) >= cap:
                        chunk = chunk[: max(0, cap - generated_extensions)]
                        truncated = True
                    generated_extensions += len(chunk)
                    for cb in chunk:
                        evaluated_combos += 1
                        rr = evaluate_combo(cb, tick_pool, None, source="D")
                        if rr is not None:
                            out_d.append(rr)
                    if truncated:
                        break
                depth_txt = f"depth={min_depth}" if min_depth == max_depth else f"depth_range={min_depth}..{max_depth}"
                if (not had_any) or (not out_d):
                    print(f"[prefilter-phase-d] level={phase_d_level} {depth_txt} seeds={seeds_at_level_start} pool={len(tick_pool)} estimated_generated=unknown generated={generated_extensions} evaluated={evaluated_combos} truncated={truncated} rejected_duplicates=unknown rejected_invalid_rules={max(0, evaluated_combos)} new_valid=0 beam_kept=0 d_valid_pool={len(d_valid_pool)} elapsed={time.perf_counter() - level_t0:.2f}s")
                    _save_progress(valid_pool + d_valid_pool, progress_state)
                    _phase_d_control()
                    break
                before = len(d_valid_pool)
                d_valid_pool = _keep_d_pool(d_valid_pool + out_d)
                _save_progress(valid_pool + d_valid_pool, progress_state)
                beam_rules = sorted(_dedupe_mask(out_d), key=_d_sort_key)[: int(args.phase_d_beam_width)]
                d_beam = [tuple(sorted(int(x) for x in r.get("_combo", tuple()))) for r in beam_rules]
                best_ratio = float(d_valid_pool[0]["ratio"]) if d_valid_pool else float("nan")
                print(f"[prefilter-phase-d] level={phase_d_level} {depth_txt} seeds={seeds_at_level_start} pool={len(tick_pool)} estimated_generated=unknown generated={generated_extensions} evaluated={evaluated_combos} truncated={truncated} rejected_duplicates=unknown rejected_invalid_rules={max(0, evaluated_combos - len(out_d))} new_valid={max(0, len(d_valid_pool) - before)} beam_kept={len(d_beam)} d_valid_pool={len(d_valid_pool)} elapsed={time.perf_counter() - level_t0:.2f}s best_ratio={best_ratio:.6g}")
                if _phase_d_control():
                    break
            print("[prefilter-phase-d] done")

    non_d_pool = [r for r in valid_pool if str(r.get("search_source", "")) != "D"]
    d_pool = [r for r in d_valid_pool if str(r.get("search_source", "")) == "D"] + [r for r in valid_pool if str(r.get("search_source", "")) == "D"]

    def _is_binned_continuous_eq(c: dict) -> bool:
        col = str(c.get("col"))
        op = str(c.get("op"))
        if op != "==":
            return False
        ftype = str(meta.get(col, {}).get("feature_type", ""))
        return ftype == "continuous"

    def _better_score(a: dict | None, b: dict | None) -> bool:
        if a is None:
            return False
        if b is None:
            return True
        ka = (float(a["ratio"]), int(a["pos_hits"]), -int(a["neg_hits"]))
        kb = (float(b["ratio"]), int(b["pos_hits"]), -int(b["neg_hits"]))
        return ka > kb

    def _mask_cond_to_interval(c: dict) -> tuple[int, int]:
        if "lo_bin" in c and "hi_bin" in c:
            return int(c["lo_bin"]), int(c["hi_bin"])
        v = int(round(float(c.get("value", 0.0))))
        return v, v

    def _apply_neighbor_merge(rule: dict) -> dict:
        conds = [dict(x) for x in rule.get("conds", [])]
        base_mask = _build_mask_from_conds(conds)
        best_sc = _score_from_mask(base_mask)
        if best_sc is None:
            return rule
        changed = True
        while changed:
            changed = False
            for i, c in enumerate(conds):
                if not _is_binned_continuous_eq(c):
                    continue
                col = str(c["col"])
                eff = int(meta.get(col, {}).get("effective_bin_count", 0) or 0)
                if eff <= 1:
                    continue
                lo, hi = _mask_cond_to_interval(c)
                local_best_sc = best_sc
                local_best = None
                if lo > 1:
                    cand = dict(c)
                    cand["lo_bin"] = int(lo - 1)
                    cand["hi_bin"] = int(hi)
                    cand["value"] = float(cand["lo_bin"])
                    trial_conds = [dict(x) for x in conds]
                    trial_conds[i] = cand
                    sc = _score_from_mask(_build_mask_from_conds(trial_conds))
                    if _better_score(sc, local_best_sc):
                        local_best_sc = sc
                        local_best = cand
                if hi < eff:
                    cand = dict(c)
                    cand["lo_bin"] = int(lo)
                    cand["hi_bin"] = int(hi + 1)
                    cand["value"] = float(cand["lo_bin"])
                    trial_conds = [dict(x) for x in conds]
                    trial_conds[i] = cand
                    sc = _score_from_mask(_build_mask_from_conds(trial_conds))
                    if _better_score(sc, local_best_sc):
                        local_best_sc = sc
                        local_best = cand
                if local_best is not None:
                    conds[i] = local_best
                    best_sc = local_best_sc
                    changed = True
        merged_rule = dict(rule)
        merged_rule["conds"] = conds
        merged_rule.update(best_sc)
        final_mask = _build_mask_from_conds(conds)
        merged_rule["_mask_hash"] = _mask_hash(final_mask)
        final_sc = _score_from_mask(final_mask)
        if final_sc is not None:
            merged_rule.update(final_sc)
        return merged_rule

    def _prepare_export_group(rows: list[dict], group_name: str) -> tuple[list[dict], bool]:
        prepared = [_attach_decode_info(_apply_neighbor_merge(r)) for r in rows]
        prepared = _dedupe_mask(prepared)
        train_ranked_group = sorted(prepared, key=lambda z: (-float(z["ratio"]), -int(z["pos_hits"]), int(z["neg_hits"])))
        train_rank_map_group = {id(r): i for i, r in enumerate(train_ranked_group)}
        shortlist_n = max(100, int(args.top_paths) * 4)
        shortlist_group = train_ranked_group[:shortlist_n]
        shortlist_group = [r for r in shortlist_group if (int(r["test_pos_hits"]) + int(r["test_neg_hits"])) > 0]
        shortlist_group = sorted(
            shortlist_group,
            key=lambda z: (
                -float(z["test_ratio"]),
                -int(z["test_pos_hits"]),
                int(z["test_neg_hits"]),
                int(train_rank_map_group.get(id(z), 10 ** 9)),
            ),
        )
        export_group = shortlist_group[: int(args.top_paths)]
        fallback_used = False
        if not export_group and train_ranked_group:
            export_group = train_ranked_group[: int(args.top_paths)]
            fallback_used = True
            print(f"[prefilter-fallback] Exporting {len(export_group)} {group_name} train-valid rules (fallback: valid>0 but test_hits=0).")
        return export_group, fallback_used

    non_d_export, non_d_fallback = _prepare_export_group(non_d_pool, "non_d")
    d_export, d_fallback = _prepare_export_group(d_pool, "D")
    best_paths = non_d_export + d_export
    fallback_export_used = bool(non_d_fallback or d_fallback)
    if bool(args.cluster_only_lower_entry):
        for r in best_paths:
            le_mask = _build_mask_from_conds(r.get("conds", []))
            le_sc = _score_from_mask(le_mask, only_lower_entry=False, enforce_filters=False)
            if le_sc is None:
                r["le_off_train_ratio"] = float("nan")
                r["le_off_test_ratio"] = float("nan")
                r["le_off_total_hits"] = 0
            else:
                r["le_off_train_ratio"] = float(le_sc.get("ratio", np.nan))
                r["le_off_test_ratio"] = float(le_sc.get("test_ratio", np.nan))
                r["le_off_total_hits"] = int(le_sc.get("pos_hits", 0)) + int(le_sc.get("neg_hits", 0)) + int(le_sc.get("test_pos_hits", 0)) + int(le_sc.get("test_neg_hits", 0))
    if not best_paths:
        print("[prefilter-warning] No rule with test hits found for final CSV export.")

    out_json = {
        "version": 2,
        "settings": {
            "tp": tps,
            "tp_weights": tp_w.tolist(),
            "sl": float(args.sl),
            "hold": int(args.hold),
            "trail": int(args.trail),
            "trail_activate": float(args.trail_activate),
            "trail_offset": float(args.trail_offset),
            "trail_factor": float(args.trail_factor),
            "include_unrealized_at_test_end": int(bool(args.include_unrealized_at_test_end)),
            "train_frac": float(args.train_frac),
            "wf_folds": int(args.wf_folds),
        },
        "rules": best_paths,
        "fallback_export_used": int(bool(fallback_export_used)),
    }
    _atomic_write_text(args.out_rules_json, json.dumps(out_json, ensure_ascii=False, indent=2))

    rows = _rules_to_rows(best_paths, is_fallback_export=fallback_export_used)
    _validate_rows(rows)
    if rows:
        _atomic_write_csv(args.out_rules_csv, pd.DataFrame(rows))
    else:
        _atomic_write_csv(args.out_rules_csv, pd.DataFrame(columns=csv_columns))
    if bool(args.debug_reject_stats):
        print("[prefilter-reject-stats-total] " + " ".join([f"{k}={v}" for k, v in reject_stats.items()]))
    print(f"Saved {len(rows)} rules -> {args.out_rules_json} and {args.out_rules_csv}")


if __name__ == "__main__":
    main()
