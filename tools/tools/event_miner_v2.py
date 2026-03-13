import os
import itertools
import numpy as np
import pandas as pd

# ==========================================================
# CONFIG
# ==========================================================

CSV_PATH = r"C:\Users\heith\Desktop\Backtest\backtrader-development\feature_scan_features.csv"

CONFIGS = [
    (5,  0.0010, 0.0010),
    (5,  0.0020, 0.0015),
    (15, 0.0015, 0.0015),
    (15, 0.0025, 0.0020),
]

LAG = 0
TEST_FRACTION = 0.20

N_BINS = 5
D_WINDOWS = [1, 2, 3, 5]   # dynamic lookbacks (minutes)

TOP_BINS_PER_FEATURE = 1
TOP_FEATURES_FOR_RULES = 30
MIN_COUNT_BIN = 2000
MIN_COUNT_RULE = 1500

TOP_K_PER_DAY = 10
COOLDOWN_MIN = 10

BASE_FEATURES = [
    "dist_ema20_tf1", "rsi7_tf1", "macd_hist_tf1",
    "dist_ema20_tf5", "rsi7_tf5", "macd_hist_tf5",
    "dist_ema20_tf15", "rsi7_tf15", "macd_hist_tf15",
    "dist_support50_tf15", "dist_resist50_tf15",
    "adx14_tf15", "vol_z_tf15",
]

# ==========================================================
# FIRST HIT LABEL
# ==========================================================

def first_hit_label(close: np.ndarray, horizon: int, tp: float, sl: float) -> np.ndarray:
    n = len(close)
    y = np.zeros(n, dtype=np.int8)
    for i in range(n - horizon - 1):
        entry = close[i]
        tp_level = entry * (1 + tp)
        sl_level = entry * (1 - sl)
        future = close[i+1:i+1+horizon]
        hit = 0
        for p in future:
            if p >= tp_level:
                hit = 1
                break
            if p <= sl_level:
                hit = 0
                break
        y[i] = hit
    return y

# ==========================================================
# DYNAMIC FEATURES (FAST: build dict -> concat once)
# ==========================================================

def build_dynamic_features(df: pd.DataFrame, cols: list[str], windows: list[int]) -> tuple[pd.DataFrame, list[str]]:
    newcols = {}
    dyn_names = []

    for col in cols:
        if col not in df.columns:
            continue

        s = df[col]

        # multi-window deltas + acc
        for w in windows:
            name_d = f"{col}_d{w}"
            newcols[name_d] = s - s.shift(w)
            dyn_names.append(name_d)

            if w >= 2:
                name_a = f"{col}_acc{w}"
                newcols[name_a] = (s - s.shift(1)) - (s.shift(1) - s.shift(2))
                dyn_names.append(name_a)

        # turn pattern (fixed 3-down then up)
        name_t = f"{col}_turn"
        newcols[name_t] = (
            (s.shift(3) > s.shift(2)) &
            (s.shift(2) > s.shift(1)) &
            (s > s.shift(1))
        ).astype(np.int8)
        dyn_names.append(name_t)

    if newcols:
        df = pd.concat([df, pd.DataFrame(newcols, index=df.index)], axis=1)

    return df, dyn_names

# ==========================================================
# BINNING (Train edges)
# ==========================================================

def make_train_bins(series: pd.Series, q: int):
    """
    Returns:
      edges: array of bin edges (len = k+1)
      codes_train: int codes for train
    """
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < MIN_COUNT_BIN:
        return None, None

    # quantile edges
    qs = np.linspace(0, 1, q + 1)
    edges = s.quantile(qs).to_numpy()

    # make edges strictly increasing (remove duplicates)
    edges = np.unique(edges)

    # need at least 3 edges => 2 bins
    if len(edges) < 3:
        return None, None

    # codes on full series (including NaNs -> code = -1)
    full = pd.to_numeric(series, errors="coerce")
    b = pd.cut(full, bins=edges, include_lowest=True, duplicates="drop")
    codes = b.cat.codes.to_numpy()  # -1 for NaN/out
    return edges, codes

def apply_bins(series: pd.Series, edges: np.ndarray):
    full = pd.to_numeric(series, errors="coerce")
    b = pd.cut(full, bins=edges, include_lowest=True, duplicates="drop")
    return b.cat.codes.to_numpy()

# ==========================================================
# LIFT TABLE (Train only)
# ==========================================================

def lift_table_train(df_train: pd.DataFrame, label_col: str, features: list[str]) -> pd.DataFrame:
    base = float(df_train[label_col].mean())
    y = df_train[label_col].to_numpy(dtype=np.int8)

    rows = []
    for f in features:
        if f not in df_train.columns:
            continue

        edges, codes = make_train_bins(df_train[f], N_BINS)
        if edges is None:
            continue

        # ignore -1
        mask_ok = codes >= 0
        if mask_ok.sum() < MIN_COUNT_BIN:
            continue

        for code in np.unique(codes[mask_ok]):
            m = (codes == code)
            cnt = int(m.sum())
            if cnt < MIN_COUNT_BIN:
                continue
            rate = float(y[m].mean())
            lift = rate / base if base > 0 else np.nan

            rows.append({
                "feature": f,
                "bin_code": int(code),
                "count": cnt,
                "event_rate": rate,
                "lift": lift,
                "base_rate": base,
                "edges": edges.tolist(),  # store edges for reproducibility
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # pick best bins by event_rate then count
    return out.sort_values(["event_rate", "lift", "count"], ascending=[False, False, False])

def build_best_bins(lift_df: pd.DataFrame) -> dict[str, list[int]]:
    """
    Returns best allowed bin codes per feature (on TRAIN binning).
    """
    best = {}
    for feat, g in lift_df.groupby("feature"):
        best[feat] = g.head(TOP_BINS_PER_FEATURE)["bin_code"].astype(int).tolist()
    return best

def select_top_features_for_rules(lift_df: pd.DataFrame, top_n: int) -> list[str]:
    agg = (lift_df
           .sort_values(["event_rate", "lift", "count"], ascending=[False, False, False])
           .groupby("feature", as_index=False)
           .first())
    agg = agg.sort_values(["event_rate", "lift", "count"], ascending=[False, False, False])
    return agg["feature"].head(top_n).tolist()

def build_edges_map(lift_df: pd.DataFrame) -> dict[str, np.ndarray]:
    """
    For each feature, take the first row's edges (same edges for that feature).
    """
    edges_map = {}
    for feat, g in lift_df.groupby("feature"):
        edges = g.iloc[0]["edges"]
        edges_map[feat] = np.array(edges, dtype=float)
    return edges_map

# ==========================================================
# RULE SEARCH (Train) using bin codes (Train edges)
# ==========================================================

def rule_search_train(df_train: pd.DataFrame, label_col: str, features_for_rules: list[str],
                      best_bins: dict[str, list[int]], edges_map: dict[str, np.ndarray]) -> pd.DataFrame:
    base = float(df_train[label_col].mean())
    y = df_train[label_col].to_numpy(dtype=np.int8)

    # precompute codes for each feature on train
    codes_map = {}
    for f in features_for_rules:
        if f not in df_train.columns:
            continue
        edges = edges_map.get(f)
        if edges is None:
            continue
        codes_map[f] = apply_bins(df_train[f], edges)

    feats = [f for f in codes_map.keys() if f in best_bins]
    if len(feats) < 2:
        return pd.DataFrame()

    results = []
    n = len(df_train)

    for r in [2, 3, 4]:
        for combo in itertools.combinations(feats, r):
            mask = np.ones(n, dtype=bool)
            for f in combo:
                allowed = np.array(best_bins[f], dtype=np.int16)
                mask &= np.isin(codes_map[f], allowed)
            cnt = int(mask.sum())
            if cnt < MIN_COUNT_RULE:
                continue
            rate = float(y[mask].mean())
            lift = rate / base if base > 0 else np.nan
            results.append({
                "rule_size": r,
                "features": "|".join(combo),
                "count": cnt,
                "event_rate": rate,
                "lift": lift,
                "base_rate": base
            })

    out = pd.DataFrame(results)
    if out.empty:
        return out
    return out.sort_values(["event_rate", "count"], ascending=[False, False])

# ==========================================================
# APPLY RULE ON TEST (using TRAIN edges)
# ==========================================================

def apply_rule_hits(df_any: pd.DataFrame, rule_feats: list[str],
                    best_bins: dict[str, list[int]], edges_map: dict[str, np.ndarray]) -> pd.DataFrame:
    # full match required
    n = len(df_any)
    mask = np.ones(n, dtype=bool)
    for f in rule_feats:
        edges = edges_map.get(f)
        if edges is None or f not in df_any.columns:
            return df_any.iloc[0:0].copy()  # empty
        codes = apply_bins(df_any[f], edges)
        allowed = np.array(best_bins[f], dtype=np.int16)
        mask &= np.isin(codes, allowed)
    return df_any.loc[mask].copy()

def pick_topk_per_day(df_sig: pd.DataFrame, score_col: str) -> pd.DataFrame:
    df_sig = df_sig.sort_values("datetime").copy()
    df_sig["date"] = df_sig["datetime"].dt.date
    kept = []
    for d, g in df_sig.groupby("date", sort=True):
        g = g.sort_values(score_col, ascending=False)
        last_t = None
        n = 0
        for idx, row in g.iterrows():
            t = row["datetime"]
            if last_t is not None and (t - last_t).total_seconds() < COOLDOWN_MIN * 60:
                continue
            kept.append(idx)
            last_t = t
            n += 1
            if n >= TOP_K_PER_DAY:
                break
    return df_sig.loc[kept].drop(columns=["date"])

# ==========================================================
# MAIN
# ==========================================================

def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(CSV_PATH)

    usecols = ["datetime", "close"] + BASE_FEATURES
    df = pd.read_csv(CSV_PATH, usecols=lambda c: c in set(usecols), low_memory=False)

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["datetime", "close"]).sort_values("datetime").reset_index(drop=True)

    for c in BASE_FEATURES:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # build dynamic features fast (no fragmentation)
    df, dyn_cols = build_dynamic_features(df, BASE_FEATURES, D_WINDOWS)
    ALL_FEATURES = BASE_FEATURES + dyn_cols

    # apply LAG shift
    if LAG > 0:
        shifted = {}
        for f in ALL_FEATURES:
            if f in df.columns:
                shifted[f] = df[f].shift(LAG)
        df = df.drop(columns=[f for f in ALL_FEATURES if f in df.columns])
        df = pd.concat([df, pd.DataFrame(shifted, index=df.index)], axis=1)

    df = df.dropna().reset_index(drop=True)

    split_idx = int(len(df) * (1.0 - TEST_FRACTION))
    df_train = df.iloc[:split_idx].copy()
    df_test = df.iloc[split_idx:].copy()

    for (H, TP, SL) in CONFIGS:
        label = f"FH_H{H}_TP{TP:.4f}_SL{SL:.4f}"

        df_train[label] = first_hit_label(df_train["close"].to_numpy(float), H, TP, SL)
        df_test[label] = first_hit_label(df_test["close"].to_numpy(float), H, TP, SL)

        base_train = float(df_train[label].mean())
        base_test = float(df_test[label].mean())

        print("\n===================================")
        print(f"{label} (LAG={LAG})")
        print("Base winrate train:", round(base_train, 6), " | test:", round(base_test, 6))

        lifts = lift_table_train(df_train, label, ALL_FEATURES)
        if lifts.empty:
            print("No bins found on train.")
            continue

        # save lift table (train)
        lifts.to_csv(f"top_feature_bins_{label}.csv", index=False)

        best_bins = build_best_bins(lifts)
        top_feats = select_top_features_for_rules(lifts, TOP_FEATURES_FOR_RULES)
        edges_map = build_edges_map(lifts)

        rules = rule_search_train(df_train, label, top_feats, best_bins, edges_map)
        if rules.empty:
            print("No rules found on train.")
            continue

        rules.to_csv(f"top_rules_{label}.csv", index=False)

        best = rules.iloc[0]
        rule_feats = best["features"].split("|")

        print("Best rule (train):", best["features"])
        print("Rule event_rate train:", round(float(best["event_rate"]), 6),
              " lift:", round(float(best["lift"]), 2),
              " count:", int(best["count"]))

        # apply on test (using train edges)
        hits_test = apply_rule_hits(df_test, rule_feats, best_bins, edges_map)
        if len(hits_test) == 0:
            print("Rule produces 0 hits on test.")
            continue

        rule_test_rate = float(hits_test[label].mean())
        print("Rule event_rate test:", round(rule_test_rate, 6), "  hits:", int(len(hits_test)))

        # density control: score = rule matched (all feats) -> all equal; keep for interface consistency
        hits_test = hits_test[["datetime", "close", label]].copy()
        hits_test["score"] = 1

        selected = pick_topk_per_day(hits_test, "score")
        if len(selected) > 0:
            med = float(selected.groupby(selected["datetime"].dt.date).size().median())
            final = float(selected[label].mean())
            print("Signals/day (median, test):", med)
            print("Final winrate on selected (test):", round(final, 6))
        else:
            print("No signals selected after Top-K + cooldown (test).")

if __name__ == "__main__":
    main()