"""Dimensionnement de position base sur le risque, avec valeur du point.

Corrections vs version initiale :
- Prise en compte de la valeur du point (multiplicateur de contrat) par actif :
  NQ = 20 $/point, ES = 50 $/point, BTC = 1 $/$, XAU = 100 $/lot (100 oz).
- Plafond de taille et stop minimum pour eviter les quantites absurdes.
"""

import numpy as np

POINT_VALUES = {
    "BTC": 1.0,
    "NQ": 20.0,
    "ES": 50.0,
    "XAU": 100.0,
}


class ScalpingMoneyManager:
    def __init__(
        self,
        risk_per_trade=0.005,
        point_value=1.0,
        min_qty=0.0,
        max_qty=None,
        min_stop_distance=1e-6,
        max_leverage=1.0,
    ):
        self.risk_per_trade = risk_per_trade
        self.point_value = point_value
        self.min_qty = min_qty
        self.max_qty = max_qty
        self.min_stop_distance = min_stop_distance
        self.max_leverage = max_leverage

    def calculate_position_size(self, current_capital, atr, atr_multiplier=1.5, price=None):
        """Quantite (unites ou contrats) telle que stop_distance * point_value
        * qty <= capital * risk_per_trade.

        Si price est fourni, la quantite est plafonnee par le notionnel :
        qty <= capital * max_leverage / price. Sans ce plafond, un stop tres
        serré (ATR faible) genererait des tailles absurdes par rapport au
        capital reellement disponible."""
        if atr is None or float(atr) <= 0.0:
            qty = self.min_qty
        else:
            stop_distance = max(float(atr) * atr_multiplier, self.min_stop_distance)
            amount_to_risk = current_capital * self.risk_per_trade
            qty = amount_to_risk / (stop_distance * self.point_value)
        if price is not None and price > 0:
            qty = min(qty, current_capital * self.max_leverage / price)
        if self.max_qty is not None:
            qty = min(qty, self.max_qty)
        return max(qty, self.min_qty)

    def stop_distance(self, atr, atr_multiplier=1.5):
        return max(float(atr) * atr_multiplier, self.min_stop_distance)


def money_manager_for_asset(asset, **kwargs):
    return ScalpingMoneyManager(point_value=POINT_VALUES.get(asset, 1.0), **kwargs)
