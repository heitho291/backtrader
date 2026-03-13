import pandas as pd
import numpy as np

PATH = "feature_scan_features.csv"  # deine Datei
df = pd.read_csv(PATH, sep=",", low_memory=False)

df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
df = df.dropna(subset=["datetime", "close"]).sort_values("datetime")
df["date"] = df["datetime"].dt.date

# --- Warmup-Filter: entferne Zeilen, wo Indikatoren noch nicht "real" sind
# (bei dir sind am Anfang viele RSI leer/0)
core_cols = ["dist_ema20_tf1","dist_ema20_tf5","rsi7_tf1","rsi7_tf5","macd_hist_tf1"]
for c in core_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=["dist_ema20_tf1","dist_ema20_tf5","rsi7_tf1","rsi7_tf5","macd_hist_tf1"])

# ---------- 1) Dip-Definition ----------
# Dip, wenn dist_ema20_tf5 im unteren Quantil liegt UND rsi7_tf5 im unteren Quantil
q_dist = df["dist_ema20_tf5"].quantile(0.15)   # bottom 15%
q_rsi  = df["rsi7_tf5"].quantile(0.20)         # bottom 20%

df["is_dip"] = (df["dist_ema20_tf5"] <= q_dist) & (df["rsi7_tf5"] <= q_rsi)

# optional Trend-Kontext: nur Dip-Buys wenn tf60 nicht bearish
if "dist_ema20_tf60" in df.columns:
    df["dist_ema20_tf60"] = pd.to_numeric(df["dist_ema20_tf60"], errors="coerce")
    df["trend_ok"] = df["dist_ema20_tf60"] >= 0
else:
    df["trend_ok"] = True

df["dip_ok"] = df["is_dip"] & df["trend_ok"]

# ---------- 2) Trigger-Definition ----------
# Trigger A: EMA20 reclaim auf tf1 (dist_ema20_tf1 kreuzt von <0 nach >=0)
df["ema_reclaim"] = (df["dist_ema20_tf1"].shift(1) < 0) & (df["dist_ema20_tf1"] >= 0)

# Trigger B: MACD hist "turn": 3 steigende Werte (Momentum dreht)
df["macd_turn"] = (df["macd_hist_tf1"] > df["macd_hist_tf1"].shift(1)) & \
                  (df["macd_hist_tf1"].shift(1) > df["macd_hist_tf1"].shift(2)) & \
                  (df["macd_hist_tf1"].shift(2) > df["macd_hist_tf1"].shift(3))

# Trigger C: RSI turn: 3 steigende Werte nach lokalem Tief
df["rsi_turn"] = (df["rsi7_tf1"] > df["rsi7_tf1"].shift(1)) & \
                 (df["rsi7_tf1"].shift(1) > df["rsi7_tf1"].shift(2)) & \
                 (df["rsi7_tf1"].shift(2) > df["rsi7_tf1"].shift(3))

# Entry-Signal: Dip ok und (mindestens 1 Trigger)
df["entry_signal"] = df["dip_ok"] 
# & (df["ema_reclaim"] | df["macd_turn"] | df["rsi_turn"])

# ---------- 3) Trade-Auswertung: First-Hit (take x% before stop y%) ----------
H_MIN = 60
TP = 0.0003    # +0.03%
SL = 0.0010    # -0.10%
COOLDOWN = 15 # Minuten
MAX_TRADES_PER_DAY = 12

df = df.reset_index(drop=True)

def first_hit(i):
    entry_px = df.loc[i, "close"]
    end_i = min(i + H_MIN, len(df)-1)
    # path
    px = df.loc[i+1:end_i, "close"].values
    if len(px) == 0:
        return None
    # thresholds
    tp_level = entry_px * (1 + TP)
    sl_level = entry_px * (1 - SL)
    # first hit
    for p in px:
        if p >= tp_level:
            return 1
        if p <= sl_level:
            return 0
    # neither hit -> label as 0 (oder None)
    return 0

# Trade-Picks: pro Tag limitiert + Cooldown
entries = []
for d, g in df.groupby("date", sort=True):
    idx = g.index[g["entry_signal"]].tolist()
    last_t = None
    picked = []
    for i in idx:
        t = df.loc[i, "datetime"]
        if last_t is not None:
            if abs((t - last_t).total_seconds()) < COOLDOWN*60:
                continue
        picked.append(i)
        last_t = t
        if len(picked) >= MAX_TRADES_PER_DAY:
            break
    entries.extend(picked)

res = df.loc[entries, ["datetime","close","dist_ema20_tf5","rsi7_tf5","dist_ema20_tf1","rsi7_tf1","macd_hist_tf1"]].copy()
res["win"] = [first_hit(i) for i in entries]
res = res.dropna(subset=["win"])

print("Trades:", len(res))
print("Winrate:", res["win"].mean())

res.to_csv("dipbuy_entries_firsthit.csv", index=False)

# ---------- 4) Support/Resistance-Zonen Analyse (optional) ----------
# Beispiel: Nähe zu Support50_tf15 (quantilbasiert)
if "dist_support50_tf15" in df.columns:
    df["dist_support50_tf15"] = pd.to_numeric(df["dist_support50_tf15"], errors="coerce")
    sup_q = df["dist_support50_tf15"].quantile(0.15)
    df["near_support"] = df["dist_support50_tf15"] <= sup_q
    # Winrate nur wenn near_support
    res2_idx = [i for i in entries if df.loc[i, "near_support"]]
    if res2_idx:
        w = np.mean([first_hit(i) for i in res2_idx])
        print("Winrate near_support:", w, "n=", len(res2_idx))