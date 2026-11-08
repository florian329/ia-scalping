"""Configuration centrale : une seule source de verite pour les couts et les
parametres d'execution. Tout le reste du pipeline importe d'ici.

Avant : les frais etaient dans backtester, les valeurs de point dans
money_management, les multiplieres ATR par defaut dupliques dans trois
fichiers. Une seule edition oubliee creait une divergence backtest/live.
"""

from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class AssetCosts:
    """Couts d'execution d'un actif, en prix ou en fraction du notionnel."""

    spread: float
    slippage: float
    commission_per_qty: float = 0.0
    commission_rate: float = 0.0
    maker_fee_rate: float = 0.0

    def taker_rate(self) -> float:
        return self.commission_rate

    def maker_rate(self) -> float:
        return self.maker_fee_rate

    def with_slippage(self, multiplier: float) -> "AssetCosts":
        return replace(self, slippage=self.slippage * multiplier)


BTC_COSTS = AssetCosts(
    spread=0.01,
    slippage=0.01,
    commission_per_qty=0.0,
    commission_rate=0.001,
    maker_fee_rate=0.0002,
)

ASSET_COST_MODELS = {"BTC": BTC_COSTS}

MIN_TP_FEE_MULTIPLE = 3.0


@dataclass
class ExecutionConfig:
    """Parametres d'execution d'une session (backtest, RL ou live)."""

    atr_multiplier_sl: float = 1.69
    atr_multiplier_tp: float = 2.42
    max_holding_ticks: int = 36
    execution: str = "taker"
    maker_timeout_ticks: int = 10
    maker_replacements_max: int = 3
    min_tp_fee_multiple: float = MIN_TP_FEE_MULTIPLE
    confidence_threshold: float = 0.0
    sequence_length: int = 20
    initial_capital: float = 10000.0
    risk_per_trade: float = 0.005
    max_leverage: float = 1.0
    max_trades: int | None = None
    max_consecutive_losses: int = 3
    cooldown_ticks: int = 50

    def validate(self) -> None:
        if self.execution not in ("taker", "maker"):
            raise ValueError(f"Mode d'execution inconnu : {self.execution}")
        if self.atr_multiplier_sl <= 0 or self.atr_multiplier_tp <= 0:
            raise ValueError("Les multiplieurs ATR doivent etre positifs")
        if self.max_holding_ticks < 1:
            raise ValueError("max_holding_ticks doit etre >= 1")
