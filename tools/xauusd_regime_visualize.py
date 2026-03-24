#!/usr/bin/env python3
"""Quick visual inspection tool for XAUUSD regime heuristics."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import pandas as pd
except ModuleNotFoundError:
    plt = None
    pd = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize extracted direction/volatility regime columns against price.")
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("xauusd_regime_visualization.png"))
    p.add_argument("--tail-rows", type=int, default=5000)
    return p.parse_args()


def load_features(path: Path, tail_rows: int) -> pd.DataFrame:
    if str(path).lower().endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, compression="infer")
    if not isinstance(df.index, pd.DatetimeIndex):
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
            df = df.set_index("datetime")
        else:
            df[df.columns[0]] = pd.to_datetime(df[df.columns[0]], errors="coerce")
            df = df.set_index(df.columns[0])
    df = df.sort_index()
    if tail_rows > 0 and len(df) > tail_rows:
        df = df.tail(tail_rows)
    return df


def main() -> None:
    args = parse_args()
    if plt is None or pd is None:
        raise SystemExit("Missing dependencies: python -m pip install pandas matplotlib")

    df = load_features(args.features, args.tail_rows)
    if "close" not in df.columns:
        raise ValueError("features file must contain close column")

    fig, (ax_price, ax_vol) = plt.subplots(2, 1, figsize=(16, 8), sharex=True, gridspec_kw={"height_ratios": [4, 1]})
    ax_price.plot(df.index, df["close"], color="black", linewidth=1.0, label="close")

    dir_specs = [
        ("regime_dir_bull", "#d8f5d0", "bull"),
        ("regime_dir_bear", "#f8d0d0", "bear"),
        ("regime_dir_sideways", "#ececec", "sideways"),
    ]
    for col, color, label in dir_specs:
        if col in df.columns:
            mask = pd.to_numeric(df[col], errors="coerce").fillna(0) > 0.5
            ax_price.fill_between(df.index, df["close"].min(), df["close"].max(), where=mask, color=color, alpha=0.35, label=label)

    ax_price.set_title("XAUUSD price with directional regime background")
    ax_price.legend(loc="upper left", ncol=4)
    ax_price.grid(alpha=0.2)

    vol_cols = ["regime_vol_high", "regime_vol_normal", "regime_vol_low"]
    for level, col in enumerate(vol_cols, start=1):
        if col in df.columns:
            mask = pd.to_numeric(df[col], errors="coerce").fillna(0) > 0.5
            ax_vol.plot(df.index[mask], [level] * int(mask.sum()), linestyle="", marker="|", markersize=10, label=col)

    ax_vol.set_yticks([1, 2, 3], ["high", "normal", "low"])
    ax_vol.set_title("Volatility regimes")
    ax_vol.grid(alpha=0.2)
    ax_vol.legend(loc="upper left", ncol=3)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    print(f"Saved regime visualization: {args.out}")


if __name__ == "__main__":
    main()
