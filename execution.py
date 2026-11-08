"""Logique d'execution unifiee : frais, TP/SL, ordres limites maker.

Unique implementation consommee par le backtester, l'environnement RL et la
boucle live — garantit que les trois se comportent de facon identique.
"""

from dataclasses import dataclass

from config import AssetCosts, ExecutionConfig


def fees_for(costs: AssetCosts, price: float, qty: float, fee_rate: float | None = None) -> float:
    """Frais pour une execution : commission fixe par unite + taux sur le notionnel.

    fee_rate=None -> taux taker (commission_rate) ; sinon le taux fourni
    (typiquement le taux maker sur un remplissage limite).
    """
    rate = costs.commission_rate if fee_rate is None else fee_rate
    return costs.commission_per_qty * qty + rate * price * qty


def taker_entry(costs: AssetCosts, side: int, price: float) -> float:
    """Prix d'entree au marche : half-spread + slippage defavorable."""
    half = costs.spread / 2.0
    return price + half + costs.slippage if side > 0 else price - half - costs.slippage


def taker_exit(costs: AssetCosts, side: int, price: float) -> float:
    """Prix de sortie au marche : slippage defavorable."""
    return price - costs.slippage if side > 0 else price + costs.slippage


def tp_distance(
    cfg: ExecutionConfig,
    costs: AssetCosts,
    entry: float,
    atr: float,
    entry_fee_rate: float | None,
    taker_entry_fill: bool,
) -> float:
    """Distance du take-profit : max(multiple ATR, plancher k x frais aller-retour).

    Le plancher garantit qu'un trade gagnant paie plusieurs trades perdants ;
    sans lui, un TP serre remplit le courtier, pas le compte.
    """
    atr_dist = atr * cfg.atr_multiplier_tp
    if cfg.min_tp_fee_multiple <= 0:
        return atr_dist
    exit_rate = costs.maker_fee_rate if cfg.execution == "maker" else costs.commission_rate
    entry_rate = costs.commission_rate if entry_fee_rate is None else entry_fee_rate
    entry_cost = (costs.spread / 2.0 + costs.slippage) if taker_entry_fill else 0.0
    round_trip = (entry_rate + exit_rate) * entry + entry_cost
    return max(atr_dist, cfg.min_tp_fee_multiple * round_trip)


def compute_stop_tp(
    cfg: ExecutionConfig,
    costs: AssetCosts,
    side: int,
    entry: float,
    atr: float,
    entry_fee_rate: float | None,
    taker_entry_fill: bool,
) -> tuple[float, float]:
    """Retourne (stop, take_profit) pour une position ouverte a `entry`."""
    stop_dist = atr * cfg.atr_multiplier_sl
    tp_dist = tp_distance(cfg, costs, entry, atr, entry_fee_rate, taker_entry_fill)
    if side > 0:
        return entry - stop_dist, entry + tp_dist
    return entry + stop_dist, entry - tp_dist


@dataclass
class MakerOrder:
    """Ordre limite en attente au carnet."""

    side: int
    limit: float
    placed_at: int
    replacements: int = 0

    def crossed_by(self, price: float) -> bool:
        if self.side > 0:
            return price <= self.limit
        return price >= self.limit


class MakerOrderBook:
    """Gestion d'un ordre limite unique : remplissage, re-placement, annulation.

    Politique de re-placement (queue chasing) : a chaque timeout, l'ordre est
    re-place un spread plus pres du mid (sans jamais le traverser), jusqu'a
    `replacements_max` fois.
    """

    def __init__(self, cfg: ExecutionConfig, costs: AssetCosts):
        self.cfg = cfg
        self.costs = costs
        self.order: MakerOrder | None = None
        self.fills = 0
        self.cancels = 0
        self.replacements = 0

    def place(self, side: int, limit: float, tick: int) -> None:
        self.order = MakerOrder(side=side, limit=limit, placed_at=tick)

    def has_pending(self) -> bool:
        return self.order is not None

    def clear(self) -> None:
        self.order = None

    def update(
        self,
        price: float,
        tick: int,
        best_bid: float | None,
        best_ask: float | None,
    ) -> MakerOrder | None:
        """Avance l'ordre d'un tick. Retourne l'ordre s'il est rempli, sinon None.

        L'appelant est responsable d'ouvrir la position avec `order.limit`.
        """
        order = self.order
        if order is None:
            return None
        if order.crossed_by(price):
            self.fills += 1
            self.order = None
            return order
        if tick - order.placed_at < self.cfg.maker_timeout_ticks:
            return None
        if order.replacements < self.cfg.maker_replacements_max:
            step = self.costs.spread
            if order.side > 0:
                ask_ref = best_ask if best_ask is not None else price + self.costs.spread / 2.0
                order.limit = min(order.limit + step, ask_ref)
            else:
                bid_ref = best_bid if best_bid is not None else price - self.costs.spread / 2.0
                order.limit = max(order.limit - step, bid_ref)
            order.placed_at = tick
            order.replacements += 1
            self.replacements += 1
            return None
        self.order = None
        self.cancels += 1
        return None
