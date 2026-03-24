#!/usr/bin/env python3
"""Build a compact integer-binned parquet from the normal XAUUSD features parquet."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
except ModuleNotFoundError:
    np = None
    pd = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build compact integer bins for miner candidate features.")
    p.add_argument("--features", type=Path, required=True, help="Normal features parquet/csv.gz path")
    p.add_argument("--out-binned", type=Path, required=True, help="Output parquet path")
    p.add_argument("--bins", type=int, default=20, help="Target number of bins for continuous columns")
    return p.parse_args()


def load_features(path: Path) -> pd.DataFrame:
    if str(path).lower().endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, compression="infer")
    if isinstance(df.index, pd.DatetimeIndex):
        return df.sort_index()
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        return df.set_index("datetime").sort_index()
    first = str(df.columns[0]).lower()
    if first.startswith("unnamed"):
        df[df.columns[0]] = pd.to_datetime(df[df.columns[0]], errors="coerce")
        return df.set_index(df.columns[0]).sort_index()
    raise ValueError("Features must include a datetime index/column.")


def build_metadata_frame(df: pd.DataFrame) -> pd.DataFrame:
    meta: dict[str, pd.Series] = {
        "datetime": pd.Series(df.index, index=df.index),
        "date": pd.Series(df.index.strftime("%Y-%m-%d"), index=df.index),
        "time": pd.Series(df.index.strftime("%H:%M:%S"), index=df.index),
    }
    for col in ("open", "high", "low", "close", "volume"):
        if col in df.columns:
            meta[col] = pd.to_numeric(df[col], errors="coerce")
    return pd.DataFrame(meta, index=df.index)


def build_candidate_columns(df: pd.DataFrame) -> list[str]:
    out: list[str] = []
    for c in df.columns:
        if c in {"open", "high", "low", "close", "volume"} or c.startswith("fwd_ret_"):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        if c.startswith(("open_tf", "high_tf", "low_tf", "close_tf", "volume_tf", "ema", "vwap_tf", "regime_")):
            continue
        if c.startswith(("delta_", "dist_", "dist_vwap", "rsi", "adx", "plus_di", "minus_di", "dx", "macd", "vol_z", "candle_", "break_", "fvg_", "session_")):
            out.append(c)
    return out


def bin_series(s: pd.Series, bins: int, dtype) -> pd.Series:
    valid = s.dropna()
    if valid.empty:
        return pd.Series(0, index=s.index, dtype=dtype)
    uniq = int(valid.nunique(dropna=True))
    if uniq <= 1:
        enc = pd.Series(1, index=valid.index, dtype=np.int32)
    elif uniq <= bins:
        levels = np.sort(valid.unique())
        mapper = {float(v): i + 1 for i, v in enumerate(levels.tolist())}
        enc = valid.map(mapper).astype(np.int32)
    else:
        ranked = valid.rank(method="first")
        enc = pd.qcut(ranked, q=bins, labels=False, duplicates="drop").astype(np.int32) + 1
    out = pd.Series(0, index=s.index, dtype=np.int32)
    out.loc[enc.index] = enc.to_numpy(dtype=np.int32, copy=False)
    return out.astype(dtype)


def main() -> None:
    args = parse_args()
    if np is None or pd is None:
        raise SystemExit("Missing dependencies: python -m pip install pandas numpy")
    if args.bins < 2:
        raise ValueError("--bins must be >= 2")

    df = load_features(args.features)
    cols = build_candidate_columns(df)
    dtype = np.uint8 if args.bins <= np.iinfo(np.uint8).max else np.uint16

    binned_cols: dict[str, pd.Series] = {}
    for c in cols:
        binned_cols[c] = bin_series(pd.to_numeric(df[c], errors="coerce"), args.bins, dtype)

    out = pd.concat([build_metadata_frame(df), pd.DataFrame(binned_cols, index=df.index)], axis=1)

    args.out_binned.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out_binned, index=False)
    print(f"Saved binned parquet: {args.out_binned}")
    print(f"Rows={len(out)} cols={len(out.columns)} dtype={np.dtype(dtype).name}")


if __name__ == "__main__":
    main()
