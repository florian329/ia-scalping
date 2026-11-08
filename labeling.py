"""Labellisation des donnees de ticks depuis un fichier d'annotations de trades.

Chaque ligne d'annotation decrit un trade reel (timestamp, actif, cote, prix
d'entree, prix de sortie). La politique de labellisation retenue :

- classe 1 (Long) / 2 (Short) : le dernier tick observable AVANT le timestamp
  d'entree du trade. C'est le seul tick ou une decision d'entree est honnete
  (aucune fuite d'information : on n'utilise pas le future du trade).
- classe 0 (Attente) : tous les autres ticks.

Option "window" : etiqueter tous les ticks dans les `window_ticks` precedant
l'entree, pour densifier les classes positives.
"""

import numpy as np
import pandas as pd

LONG = 1
SHORT = 2
FLAT = 0


def load_annotations(path):
    ann = pd.read_csv(path)
    required = {"timestamp", "asset", "side", "entry_price", "exit_price"}
    missing = required - set(ann.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans {path}: {missing}")
    ann = ann.copy()
    ann["timestamp"] = pd.to_datetime(ann["timestamp"], utc=True)
    ann["side"] = ann["side"].str.upper().str.strip()
    bad = ~ann["side"].isin(["BUY", "SELL"])
    if bad.any():
        raise ValueError(f"Cotes invalides (attendu BUY/SELL): {ann.loc[bad, 'side'].unique()}")
    return ann


def label_ticks(df, annotations, asset, window_ticks=0, pnl_weights=False):
    """Retourne df avec une colonne 'label' (0/1/2) et un masque 'is_entry_tick'.

    pnl_weights=True ajoute une colonne 'sample_weight' : les ticks etiquetes
    sont ponderes par la magnitude du PnL du trade annote (un trade a +3 ticks
    pese plus qu'un trade a +0.5), normalisee pour un poids moyen de 1 sur les
    classes positives. Les ticks 'Attente' gardent un poids de 1.
    """
    out = df.copy().reset_index(drop=True)
    out["label"] = FLAT
    out["is_entry_tick"] = False
    out["sample_weight"] = 1.0
    if "timestamp" not in out.columns:
        raise ValueError("df doit contenir une colonne 'timestamp' (datetime, UTC)")
    ticks = pd.to_datetime(out["timestamp"], utc=True).astype("int64").to_numpy()

    ann = annotations[annotations["asset"] == asset].copy()
    ann["ts_int"] = pd.to_datetime(ann["timestamp"], utc=True).astype("int64")
    for _, row in ann.iterrows():
        idx = int(np.searchsorted(ticks, int(row["ts_int"]), side="right")) - 1
        if idx < 0:
            continue
        start = max(0, idx - window_ticks)
        out.loc[start:idx, "label"] = LONG if row["side"] == "BUY" else SHORT
        out.loc[idx, "is_entry_tick"] = True
        if pnl_weights:
            direction = 1.0 if row["side"] == "BUY" else -1.0
            trade_pnl = direction * (float(row["exit_price"]) - float(row["entry_price"]))
            out.loc[start:idx, "sample_weight"] = abs(trade_pnl)
    if pnl_weights:
        positive = out.loc[out["label"] != FLAT, "sample_weight"]
        if len(positive) and positive.mean() > 0:
            out.loc[out["label"] != FLAT, "sample_weight"] = (
                positive / positive.mean()
            )
    return out


def class_distribution(labels):
    unique, counts = np.unique(labels, return_counts=True)
    return dict(zip(unique.tolist(), counts.tolist()))


def compute_class_weights(labels, positive_boost=3.5, cap=20.0):
    """Poids = racine de l'inverse de frequence (attenuation du desequilibre
    extreme), boost optionnel sur les classes positives, plafond `cap`."""
    labels = np.asarray(labels)
    weights = np.ones(3)
    total = len(labels)
    for c in range(3):
        n_c = max(1, int((labels == c).sum()))
        inv = np.sqrt(total / (3 * n_c))
        weights[c] = min(inv * (positive_boost if c in (LONG, SHORT) else 1.0), cap)
    return weights / weights.mean()
