"""Filtres d'execution BTC (spec PDF section 5.1, volet Bitcoin).

- Anti-spoofing : l'OBI n'utilise que les niveaux <= max_book_depth (10).
  Les gros ordres fantomes places loin du prix pour manipuler l'imbalance
  sont ignores.
- Seuil d'OBI optionnel et blackout de news : parametres generiques
  disponibles si besoin.
"""

from datetime import datetime

ASSET_FILTER_DEFAULTS = {
    "BTC": {"max_book_depth": 10, "obi_threshold": None},
}


class AssetFilters:
    def __init__(self, asset="BTC", obi_threshold="default", news_blackouts=None, max_book_depth="default"):
        if asset not in ASSET_FILTER_DEFAULTS:
            raise ValueError(f"Actif non supporte : {asset}. Seul le BTC est supporte.")
        defaults = ASSET_FILTER_DEFAULTS[asset]
        self.asset = asset
        self.obi_threshold = defaults["obi_threshold"] if obi_threshold == "default" else obi_threshold
        self.max_book_depth = defaults["max_book_depth"] if max_book_depth == "default" else max_book_depth
        self.news_blackouts = news_blackouts or []

    def is_news_blackout(self, timestamp):
        if not self.news_blackouts or timestamp is None:
            return False
        ts = timestamp
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        for start, end in self.news_blackouts:
            if start <= ts <= end:
                return True
        return False

    def obi_allowed(self, obi_value, side):
        """side : +1 (long) ou -1 (short)."""
        if self.obi_threshold is None:
            return True
        if side > 0:
            return obi_value > self.obi_threshold
        return obi_value < -self.obi_threshold

    def trade_allowed(self, side, obi_value, timestamp=None):
        """Retourne (autorise, raison_du_refus)."""
        if self.is_news_blackout(timestamp):
            return False, "blackout_news"
        if not self.obi_allowed(obi_value, side):
            return False, "obi_sous_seuil"
        return True, None

    def effective_depth(self, requested_depth):
        if self.max_book_depth is None:
            return requested_depth
        return min(requested_depth, self.max_book_depth)


def filters_for_asset(asset="BTC", **kwargs):
    return AssetFilters(asset, **kwargs)
