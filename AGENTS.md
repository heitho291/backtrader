# AGENTS.md

## Allgemeine Arbeitsregeln fuer dieses Repository

- Arbeite minimal-invasiv.
- Aendere nur Dateien und Logikbereiche, die im jeweiligen Task ausdruecklich erlaubt sind.
- Keine grossen Refactors, wenn nur ein gezielter Fix oder ein gezieltes neues Tool beauftragt ist.
- Keine bestehenden CLI-Parameter umbenennen, ausser der Task verlangt es ausdruecklich.
- Keine Ergebnisdateien, Caches, grosse Datendateien, Logs, NPZ-, Parquet-, CSV- oder JSON-Outputs committen.
- Kein git add . verwenden.
- Wenn eine Aenderung an Punkt A wahrscheinlich auch Punkt B oder C betrifft, zuerst die abhaengigen Codepfade nennen und dann nur minimal konsistent anpassen.
- Wenn ein Task unklar, zu gross oder riskant ist, stoppen und im Abschlussbericht erklaeren, statt halb geraten zu implementieren.
- Diagnosen sollen standardmaessig nur ins Terminal printen. Keine zusaetzlichen Diagnose-Dateien erzeugen, ausser der Task verlangt ausdruecklich Output-Dateien.

## Projektstruktur und Verantwortlichkeiten

- Feature-Extraction, Binning, Miner, Optimizer, Prefilter und neue Spezial-Tools getrennt halten.
- Keine Scoring-, Simulations-, Label-, Cache-, Tick-Outcome- oder Phase-Semantik aendern, ausser der Task verlangt genau das.
- Neue Tools unter tools/ sollen klare CLI-Argumente, deterministische Outputs und explizite Output-Pfade haben.
- Bestehende Pipeline-Teile nicht stillschweigend zurueckbauen.

## XAUUSD-Regel- und Feature-Konventionen

- Fuer generalisierbare XAUUSD-Regeln standardmaessig keine absoluten Preislevel verwenden.
- Rohe Preislevel wie open, high, low, close, open_tf*, high_tf*, low_tf*, close_tf*, rohe ema*, rohe sma*, rohe ma* und rohe vwap* nicht als Rule-Kandidaten verwenden, ausser der Task erlaubt es ausdruecklich.
- Bevorzuge relative oder normalisierte Features wie dist_*, dist_ema*, dist_sma*, dist_vwap*, rsi*, adx*, plus_di*, minus_di*, dx*, atr*, macd*, vol_z*, break_*, fvg_*, liq_sweep_*, ms_*, bos_*, choch_*, Struktur- und Volatilitaetsfeatures.

## Checks

- Nach Aenderungen an Python-Dateien immer python -m py_compile fuer jede betroffene Datei ausfuehren.
- Immer git diff --check und git diff --stat ausfuehren.
- Im Abschlussbericht klar nennen:
  1. Basis-Commit vor Aenderung
  2. geaenderte Dateien
  3. vollstaendig umgesetzte Punkte
  4. teilweise umgesetzte Punkte
  5. nicht umgesetzte Punkte
  6. ausgefuehrte Checks
  7. Commit-Hash, falls ein Commit erstellt wurde
  8. bewusste Nicht-Aenderungen
