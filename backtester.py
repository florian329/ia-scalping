"""Backtester realiste pour le scalping IA BTC.

Refactor en classe BacktestEngine : l'etat de la simulation (capital, position,
ordre maker en attente, compteurs de protection) est explicite, la logique de
couts/TP/SL vient du module execution.py (partage avec l'env RL et le live),
et run_backtest() reste un wrapper compatible avec l'API historique.
"""

import numpy as np
import pandas as pd
import torch

from asset_filters import AssetFilters
from config import ASSET_COST_MODELS, ExecutionConfig
from dataset import BOOK_FEATURES, SEQ_FEATURES
from execution import (
    MakerOrderBook,
    compute_stop_tp,
    fees_for,
    taker_entry,
    taker_exit,
)
from money_management import ScalpingMoneyManager
from model_pytorch import predict_action

FLAT, LONG, SHORT = 0, 1, -1


class Trade:
    def __init__(self, side: int, entry_index: int, entry_price: float, qty: float,
                 stop_price: float, tp_price: float):
        self.side = side
        self.entry_index = entry_index
        self.entry_price = entry_price
        self.qty = qty
        self.stop_price = stop_price
        self.tp_price = tp_price
        self.exit_index = None
        self.exit_price = None
        self.exit_reason = None
        self.net_pnl = 0.0

    @property
    def direction(self) -> int:
        return 1 if self.side == LONG else -1

    def pnl_fill(self, fill_price: float) -> float:
        gross = (fill_price - self.entry_price) * self.qty
        return gross if self.side == LONG else -gross


class _MarketContext:
    """Tableaux derives du DataFrame, partages par les methodes du moteur."""

    def __init__(self, df: pd.DataFrame, seq_features, norm_stats):
        book = df[BOOK_FEATURES].to_numpy(dtype=np.float32)
        if norm_stats is not None:
            mean, std = norm_stats
            book = (book - mean) / (std + 1e-8)
        self.book = book
        self.seq = df[seq_features].to_numpy(dtype=np.float32)
        self.prices = df["last_price"].to_numpy(dtype=np.float64)
        self.atr = (
            df["ATR_Tick"].to_numpy(dtype=np.float64)
            if "ATR_Tick" in df
            else np.full(len(df), 1.0)
        )
        self.raw_obi = df["OBI"].to_numpy(dtype=np.float64) if "OBI" in df else np.zeros(len(df))
        self.timestamps = (
            df["timestamp"].to_numpy() if "timestamp" in df else [None] * len(df)
        )
        self.best_bid = df["best_bid"].to_numpy(dtype=np.float64) if "best_bid" in df else None
        self.best_ask = df["best_ask"].to_numpy(dtype=np.float64) if "best_ask" in df else None


class BacktestEngine:
    """Moteur de simulation : configuration explicite, etat explicite.

    Toute la logique de couts/TP/SL/remplissage maker est deleguee a
    execution.py — la meme que celle de l'environnement RL et de la boucle live.
    """

    def __init__(
        self,
        model,
        cfg: ExecutionConfig,
        costs,
        money_manager: ScalpingMoneyManager | None = None,
        device: str = "cpu",
        norm_stats=None,
        seq_features=None,
        filters: AssetFilters | None = None,
        vectorized: bool = True,
        inference_chunk_size: int = 4096,
    ):
        cfg.validate()
        self.model = model
        self.cfg = cfg
        self.costs = costs
        self.device = device
        self.norm_stats = norm_stats
        self.seq_features = seq_features or SEQ_FEATURES
        self.filters = filters
        self.vectorized = vectorized
        self.inference_chunk_size = inference_chunk_size
        self.money_manager = money_manager or ScalpingMoneyManager(
            risk_per_trade=cfg.risk_per_trade, point_value=1.0, max_leverage=cfg.max_leverage
        )
        self.ctx: _MarketContext | None = None
        self.actions_cache = None
        self.probs_cache = None
        self.trades: list[Trade] = []
        self.capital = cfg.initial_capital
        self.total_fees = 0.0
        self.position: Trade | None = None
        self.order_book: MakerOrderBook | None = None
        self.consecutive_losses = 0
        self.cooldown_until = -1

    def run(self, df: pd.DataFrame) -> dict:
        cfg = self.cfg
        self.ctx = _MarketContext(df, self.seq_features, self.norm_stats)
        self.capital = cfg.initial_capital
        self.trades = []
        self.total_fees = 0.0
        self.position = None
        self.order_book = MakerOrderBook(cfg, self.costs)
        self.consecutive_losses = 0
        self.cooldown_until = -1
        equity_curve = np.full(len(df), np.nan)
        equity_curve[: cfg.sequence_length] = self.capital

        self.actions_cache, self.probs_cache = (None, None)
        if self.vectorized:
            self.actions_cache, self.probs_cache = _batch_predict(
                self.model,
                self.ctx.book,
                self.ctx.seq,
                cfg.sequence_length,
                self.device,
                self.inference_chunk_size,
            )

        for t in range(cfg.sequence_length, len(df)):
            price = self.ctx.prices[t]
            self._manage_open_position(t, price)
            if self.position is not None:
                unrealized = (
                    (price - self.position.entry_price)
                    * self.position.qty
                    * self.position.direction
                )
                equity_curve[t] = self.capital + unrealized
            else:
                equity_curve[t] = self.capital
            self._update_pending_order(t, price)
            self._maybe_enter(t, price)

        if self.position is not None:
            fill = taker_exit(self.costs, self.position.side, self.ctx.prices[-1])
            self._settle(self.position, len(self.ctx.prices) - 1, fill, "end_of_data", taker_fees=True)

        report = _build_report(self.trades, equity_curve, cfg.initial_capital, self.total_fees)
        report["execution"] = cfg.execution
        report["maker_fills"] = self.order_book.fills
        report["maker_cancels"] = self.order_book.cancels
        report["maker_replacements"] = self.order_book.replacements
        return report

    def _manage_open_position(self, t: int, price: float) -> None:
        pos = self.position
        if pos is None:
            return
        costs = self.costs
        maker_exit = self.cfg.execution == "maker"
        if pos.side == LONG:
            if price <= pos.stop_price:
                fill = taker_exit(costs, LONG, pos.stop_price)
                self._settle(pos, t, fill, "stop", taker_fees=True)
            elif price >= pos.tp_price:
                self._settle(
                    pos,
                    t,
                    pos.tp_price,
                    "take_profit",
                    fee_rate=costs.maker_fee_rate if maker_exit else None,
                    taker_fees=not maker_exit,
                )
        else:
            if price >= pos.stop_price:
                fill = taker_exit(costs, SHORT, pos.stop_price)
                self._settle(pos, t, fill, "stop", taker_fees=True)
            elif price <= pos.tp_price:
                self._settle(
                    pos,
                    t,
                    pos.tp_price,
                    "take_profit",
                    fee_rate=costs.maker_fee_rate if maker_exit else None,
                    taker_fees=not maker_exit,
                )
        if self.position is not None and t - self.position.entry_index >= self.cfg.max_holding_ticks:
            fill = taker_exit(self.costs, self.position.side, price)
            self._settle(self.position, t, fill, "timeout", taker_fees=True)

    def _settle(self, trade: Trade, t: int, fill: float, reason: str, fee_rate=None, taker_fees=False) -> None:
        costs = self.costs
        rate = costs.commission_rate if (fee_rate is None and taker_fees) else fee_rate
        fees = fees_for(costs, fill, trade.qty, fee_rate=rate)
        net = trade.pnl_fill(fill) - fees
        self.capital += net
        self.total_fees += fees
        trade.exit_index = t
        trade.exit_price = fill
        trade.exit_reason = reason
        trade.net_pnl = net
        self.trades.append(trade)
        self.position = None
        if net < 0:
            self.consecutive_losses += 1
            if self.consecutive_losses >= self.cfg.max_consecutive_losses:
                self.cooldown_until = trade.entry_index + self.cfg.cooldown_ticks
                self.consecutive_losses = 0
        else:
            self.consecutive_losses = 0

    def _update_pending_order(self, t: int, price: float) -> None:
        if self.position is not None:
            if self.order_book.has_pending():
                self.order_book.clear()
            return
        order = self.order_book.update(
            price,
            t,
            self.ctx.best_bid[t] if self.ctx.best_bid is not None else None,
            self.ctx.best_ask[t] if self.ctx.best_ask is not None else None,
        )
        if order is not None:
            self._open_position(
                order.side,
                t,
                entry_price=order.limit,
                fee_rate=self.costs.maker_fee_rate,
                taker_entry_fill=False,
            )

    def _maybe_enter(self, t: int, price: float) -> None:
        cfg = self.cfg
        if self.position is not None or self.order_book.has_pending():
            return
        if t < self.cooldown_until:
            return
        if cfg.max_trades is not None and len(self.trades) >= cfg.max_trades:
            return
        if self.actions_cache is not None:
            k = t - cfg.sequence_length
            action = int(self.actions_cache[k])
            probs = self.probs_cache[k]
        else:
            action, probs = predict_action(
                self.model,
                self.ctx.book[t - 1],
                self.ctx.seq[t - cfg.sequence_length : t],
                device=self.device,
            )
        if probs[action] < cfg.confidence_threshold or action not in (1, 2):
            return
        side = 1 if action == 1 else -1
        if self.filters is not None:
            allowed, _ = self.filters.trade_allowed(
                side, self.ctx.raw_obi[t - 1], self.ctx.timestamps[t - 1]
            )
            if not allowed:
                return
        if cfg.execution == "maker":
            if side > 0:
                if self.ctx.best_bid is not None:
                    limit = float(self.ctx.best_bid[t])
                else:
                    limit = price - self.costs.spread / 2.0
            else:
                if self.ctx.best_ask is not None:
                    limit = float(self.ctx.best_ask[t])
                else:
                    limit = price + self.costs.spread / 2.0
            self.order_book.place(side, limit, t)
        else:
            entry = taker_entry(self.costs, side, price)
            self._open_position(side, t, entry_price=entry, fee_rate=None, taker_entry_fill=True)

    def _open_position(self, side: int, t: int, entry_price: float, fee_rate, taker_entry_fill: bool) -> None:
        atr_value = self.ctx.atr[t - 1]
        qty = self.money_manager.calculate_position_size(
            self.capital, atr_value, self.cfg.atr_multiplier_sl, price=entry_price
        )
        if qty <= 0:
            return
        stop, tp = compute_stop_tp(
            self.cfg,
            self.costs,
            side,
            entry_price,
            atr_value,
            entry_fee_rate=fee_rate,
            taker_entry_fill=taker_entry_fill,
        )
        fees = fees_for(self.costs, entry_price, qty, fee_rate=fee_rate)
        self.capital -= fees
        self.total_fees += fees
        self.position = Trade(
            LONG if side > 0 else SHORT, t, entry_price, qty, stop, tp
        )


def _batch_predict(model, book, seq, sequence_length, device, chunk_size=4096):
    """Inference vectorisee : predit toutes les actions en paquets. Meme resultat
    que le mode sequentiel (le modele est sans etat entre les ticks)."""
    n = len(book)
    n_steps = n - sequence_length
    if n_steps <= 0:
        return np.zeros(0, dtype=np.int64), np.zeros((0, 3), dtype=np.float32)
    actions = np.zeros(n_steps, dtype=np.int64)
    probs = np.zeros((n_steps, 3), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, n_steps, chunk_size):
            end = min(start + chunk_size, n_steps)
            idx = np.arange(start, end) + sequence_length - 1
            book_chunk = torch.tensor(book[idx], dtype=torch.float32).to(device)
            seq_chunk = np.stack([seq[k - sequence_length + 1 : k + 1] for k in idx])
            seq_chunk = torch.tensor(seq_chunk, dtype=torch.float32).to(device)
            logits = model(book_chunk, seq_chunk)
            chunk_probs = torch.softmax(logits, dim=1).cpu().numpy()
            actions[start:end] = chunk_probs.argmax(axis=1)
            probs[start:end] = chunk_probs
    return actions, probs


def _build_report(trades, equity_curve, initial_capital, total_fees):
    equity = equity_curve[~np.isnan(equity_curve)]
    final_capital = float(equity[-1]) if len(equity) else initial_capital
    pnls = np.array([getattr(tr, "net_pnl", tr.pnl_fill(tr.exit_price)) for tr in trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    gross_win = wins.sum()
    gross_loss = abs(losses.sum())
    returns = equity[1:] / equity[:-1] - 1.0 if len(equity) > 1 else np.array([])
    sharpe = (
        float(returns.mean() / (returns.std() + 1e-12) * np.sqrt(252 * 6.5 * 60))
        if len(returns) > 1 and returns.std() > 0
        else 0.0
    )
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / (peak + 1e-12)
    return {
        "initial_capital": initial_capital,
        "final_capital": final_capital,
        "total_return_pct": (final_capital / initial_capital - 1.0) * 100.0,
        "n_trades": len(trades),
        "win_rate": float((pnls > 0).mean() * 100.0) if len(pnls) else 0.0,
        "profit_factor": float(gross_win / gross_loss) if gross_loss > 0 else float("inf"),
        "expectancy_per_trade": float(pnls.mean()) if len(pnls) else 0.0,
        "max_drawdown_pct": float(dd.min() * 100.0) if len(dd) else 0.0,
        "sharpe": sharpe,
        "total_fees": total_fees,
        "trades": trades,
        "equity_curve": equity_curve,
    }


def print_report(report):
    print("=" * 60)
    print(f"Capital initial   : {report['initial_capital']:>12.2f} $")
    print(f"Capital final     : {report['final_capital']:>12.2f} $")
    print(f"Rendement total   : {report['total_return_pct']:>11.2f} %")
    print(f"Nombre de trades  : {report['n_trades']:>12d}")
    print(f"Win rate          : {report['win_rate']:>11.2f} %")
    profit_factor = report["profit_factor"]
    pf_display = "inf" if profit_factor == float("inf") else f"{profit_factor:.2f}"
    print(f"Profit factor     : {pf_display:>12}")
    print(f"Expectancy/trade : {report['expectancy_per_trade']:>11.2f} $")
    print(f"Max drawdown      : {report['max_drawdown_pct']:>11.2f} %")
    print(f"Sharpe (annualise): {report['sharpe']:>12.2f}")
    print(f"Frais totaux      : {report['total_fees']:>11.2f} $")
    print("=" * 60)


def run_backtest(
    model,
    df,
    asset="BTC",
    money_manager=None,
    sequence_length=20,
    device="cpu",
    initial_capital=10000.0,
    atr_multiplier_sl=1.69,
    atr_multiplier_tp=2.42,
    max_holding_ticks=36,
    norm_stats=None,
    seq_features=None,
    confidence_threshold=0.0,
    filters=None,
    max_trades=None,
    max_consecutive_losses=3,
    cooldown_ticks=50,
    vectorized=True,
    inference_chunk_size=4096,
    execution="taker",
    maker_timeout_ticks=10,
    min_tp_fee_multiple=3.0,
    maker_replacements_max=3,
):
    """Wrapper compatible : construit ExecutionConfig + BacktestEngine et lance le run."""
    if asset != "BTC":
        raise ValueError("Seul le BTC est supporte par cette strategie.")
    costs = ASSET_COST_MODELS[asset]
    cfg = ExecutionConfig(
        atr_multiplier_sl=atr_multiplier_sl,
        atr_multiplier_tp=atr_multiplier_tp,
        max_holding_ticks=max_holding_ticks,
        execution=execution,
        maker_timeout_ticks=maker_timeout_ticks,
        maker_replacements_max=maker_replacements_max,
        min_tp_fee_multiple=min_tp_fee_multiple,
        confidence_threshold=confidence_threshold,
        sequence_length=sequence_length,
        initial_capital=initial_capital,
        max_trades=max_trades,
        max_consecutive_losses=max_consecutive_losses,
        cooldown_ticks=cooldown_ticks,
    )
    engine = BacktestEngine(
        model,
        cfg,
        costs,
        money_manager=money_manager,
        device=device,
        norm_stats=norm_stats,
        seq_features=seq_features,
        filters=filters,
        vectorized=vectorized,
        inference_chunk_size=inference_chunk_size,
    )
    return engine.run(df)
