"""Detection de derive (drift) : le modele entraene sur une periode se degrade
silencieusement sur une autre. Deux detecteurs complementaires :

- WinRateDriftMonitor : derive de performance — compare le win rate glissant
  des trades live au win rate de reference (backtest). Alerte et gel si
  l'ecart depasse le seuil.
- FeatureDriftMonitor : derive de donnees — test de Kolmogorov-Smirnov par
  feature entre la distribution de reference (train) et la fenetre recente.
  Detecte un changement de regime avant meme que les pertes s'accumulent.
"""

from collections import deque

import numpy as np
from scipy.stats import ks_2samp


class WinRateDriftMonitor:
    def __init__(self, reference_win_rate, window=50, min_trades=20, abs_threshold=0.15):
        if not 0.0 <= reference_win_rate <= 1.0:
            raise ValueError("reference_win_rate doit etre entre 0 et 1")
        self.reference_win_rate = reference_win_rate
        self.window = window
        self.min_trades = min_trades
        self.abs_threshold = abs_threshold
        self.results = deque(maxlen=window)

    def record(self, net_pnl):
        """Enregistre le resultat net d'un trade. Retourne True si derive."""
        self.results.append(1 if net_pnl > 0 else 0)
        return self.check()

    def current_win_rate(self):
        if not self.results:
            return None
        return float(np.mean(self.results))

    def check(self):
        if len(self.results) < self.min_trades:
            return False
        current = self.current_win_rate()
        drifted = abs(current - self.reference_win_rate) >= self.abs_threshold
        if drifted and current < self.reference_win_rate:
            return True
        return drifted

    def status(self):
        return {
            "n_trades": len(self.results),
            "current_win_rate": self.current_win_rate(),
            "reference_win_rate": self.reference_win_rate,
            "drifted": self.check() if len(self.results) >= self.min_trades else False,
        }


class FeatureDriftMonitor:
    def __init__(self, reference_df, columns, p_value_threshold=0.01):
        if not columns:
            raise ValueError("columns ne peut pas etre vide")
        self.columns = list(columns)
        self.p_value_threshold = p_value_threshold
        self.reference = {
            c: reference_df[c].dropna().to_numpy(dtype=np.float64) for c in self.columns
        }

    def ks_statistics(self, recent_df):
        """Statistique KS et p-value par feature (fenetre recente vs reference)."""
        out = {}
        for c in self.columns:
            recent = recent_df[c].dropna().to_numpy(dtype=np.float64)
            if len(recent) < 30:
                out[c] = {"stat": None, "p_value": None, "n_recent": len(recent)}
                continue
            stat, p = ks_2samp(self.reference[c], recent)
            out[c] = {"stat": float(stat), "p_value": float(p), "n_recent": len(recent)}
        return out

    def drifted_features(self, recent_df):
        """Liste des features dont la distribution a significativement derive."""
        stats = self.ks_statistics(recent_df)
        drifted = []
        for c, s in stats.items():
            if s["p_value"] is not None and s["p_value"] < self.p_value_threshold:
                drifted.append(c)
        return drifted, stats

    def drifted(self, recent_df):
        drifted, _ = self.drifted_features(recent_df)
        return len(drifted) > 0, drifted
