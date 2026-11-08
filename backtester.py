"""Backtester corrige pour le scalping IA.

Corrections vs version initiale :
1. La quantite de position provient du money manager (plus de P&L "1 unite fixe").
2. TP/SL bases sur l'ATR + timeout : les trades ne sont plus fermés au tick
   suivant quoi qu'il arrive.
3. Modele de couts par actif : spread, commission par unite/contrat, slippage.
4. Anti look-ahead : la decision au tick t utilise les features <= t-1.
5. Metriques completes : Sharpe, drawdown max, profit factor, expectancy,
   win rate, frais totaux.
"""

import numpy as np
import pandas as pd
import torch

from money_management import ScalpingMoneyManager
from model_pytorch import predict_action
from dataset import BOOK_FEATURES, SEQ_FEATURES

FLAT, LONG, SHORT = 0, 1, -1

ASSET_COST_MODELS = {
    "BTC": {"spread": 1.0, "commission_per_qty": 0.0, "slippage": 0.5},
    "NQ": {"spread": 0.25, "commission_per_qty": 2.5, "slippage": 0.25},
    "ES": {"spread": 0.25, "commission_per_qty": 2.5, "slippage": 0.25},
    "XAU": {"spread": 0.30, "commission_per_qty": 0.50, "slippage": 0.15},
}


class Trade:
    def __init__(self, side, entry_index, entry_price, qty, stop_price, tp_price):
        self.side = side
        self.entry_index = entry_index
        self.entry_price = entry_price
        self.qty = qty
        self.stop_price = stop_price
        self.tp_price = tp_price
        self.exit_index = None
        self.exit_price = None
        self.exit_reason = None

    @property
    def pnl(self):
        if self.exit_price is None:
            return 0.0
        gross = (self.exit_price - self.entry_price) * self.qty
        return gross if self.side == LONG else -gross

    @property
    def direction(self):
        return 1 if self.side == LONG else -1

    def pnl_fill(self, fill_price):
        gross = (fill_price - self.entry_price) * self.qty
        return gross if self.side == LONG else -gross


def run_backtest(
    model,
    df,
    asset="BTC",
    money_manager=None,
    sequence_length=20,
    device="cpu",
    initial_capital=10000.0,
    atr_multiplier_sl=1.5,
    atr_multiplier_tp=2.0,
    max_holding_ticks=100,
    norm_stats=None,
    seq_features=None,
    confidence_threshold=0.0,
):
    """Simule la strategie sur df (features calculees par pipeline_data.build_features).

    Retourne un dict : capital final, metriques et liste des trades.
    """
    seq_features = seq_features or SEQ_FEATURES
    costs = ASSET_COST_MODELS.get(asset, ASSET_COST_MODELS["BTC"])
    if money_manager is None:
        money_manager = ScalpingMoneyManager(
            risk_per_trade=0.005,
            point_value={"BTC": 1.0, "NQ": 20.0, "ES": 50.0, "XAU": 100.0}.get(asset, 1.0),
        )

    book = df[BOOK_FEATURES].to_numpy(dtype=np.float32)
    if norm_stats is not None:
        mean, std = norm_stats
        book = (book - mean) / (std + 1e-8)
    seq = df[seq_features].to_numpy(dtype=np.float32)
    prices = df["last_price"].to_numpy(dtype=np.float64)
    atr = df["ATR_Tick"].to_numpy(dtype=np.float64) if "ATR_Tick" in df else np.full(len(df), 1.0)

    capital = initial_capital
    position = None
    trades = []
    equity_curve = np.full(len(df), np.nan)
    equity_curve[:sequence_length] = capital
    total_fees = 0.0

    def close_position(trade, exit_index, exit_price, reason):
        nonlocal capital, position, total_fees
        slip = costs["slippage"]
        fill = exit_price - slip if trade.side == LONG else exit_price + slip
        fees = costs["commission_per_qty"] * trade.qty
        capital += trade.pnl_fill(fill) - fees
        total_fees += fees + slip * trade.qty * money_manager.point_value
        trade.exit_index = exit_index
        trade.exit_price = fill
        trade.exit_reason = reason
        trades.append(trade)
        position = None

    def open_position(side, t, action_price):
        nonlocal capital, position, total_fees
        qty = money_manager.calculate_position_size(capital, atr[t - 1])
        if qty <= 0:
            return
        half_spread = costs["spread"] / 2.0
        slip = costs["slippage"]
        if side == LONG:
            entry = action_price + half_spread + slip
            stop = entry - money_manager.stop_distance(atr[t - 1], atr_multiplier_sl)
            tp = entry + money_manager.stop_distance(atr[t - 1], atr_multiplier_sl) * (
                atr_multiplier_tp / atr_multiplier_sl
            )
        else:
            entry = action_price - half_spread - slip
            stop = entry + money_manager.stop_distance(atr[t - 1], atr_multiplier_sl)
            tp = entry - money_manager.stop_distance(atr[t - 1], atr_multiplier_sl) * (
                atr_multiplier_tp / atr_multiplier_sl
            )
        fees = costs["commission_per_qty"] * qty
        capital -= fees
        total_fees += fees
        position = Trade(side, t, entry, qty, stop, tp)

    for t in range(sequence_length, len(df)):
        price = prices[t]
        if position is not None:
            if position.side == LONG:
                if price <= position.stop_price:
                    close_position(position, t, position.stop_price, "stop")
                elif price >= position.tp_price:
                    close_position(position, t, position.tp_price, "take_profit")
            else:
                if price >= position.stop_price:
                    close_position(position, t, position.stop_price, "stop")
                elif price <= position.tp_price:
                    close_position(position, t, position.tp_price, "take_profit")
            if position is not None and t - position.entry_index >= max_holding_ticks:
                close_position(position, t, price, "timeout")

        if position is None:
            action, probs = predict_action(
                model, book[t - 1], seq[t - sequence_length : t], device=device
            )
            if probs[action] >= confidence_threshold and action in (1, 2):
                open_position(LONG if action == 1 else SHORT, t, price)

        if position is not None:
            unrealized = (price - position.entry_price) * position.qty * position.direction
            equity_curve[t] = capital + unrealized
        else:
            equity_curve[t] = capital

    if position is not None:
        close_position(position, len(df) - 1, prices[-1], "end_of_data")

    return _build_report(trades, equity_curve, initial_capital, total_fees)


def _build_report(trades, equity_curve, initial_capital, total_fees):
    equity = equity_curve[~np.isnan(equity_curve)]
    final_capital = float(equity[-1]) if len(equity) else initial_capital
    pnls = np.array([tr.pnl_fill(tr.exit_price) for tr in trades])
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
    print(f"Profit factor     : {report['profit_factor']:>12.2f}")
    print(f"Expectancy/trade  : {report['expectancy_per_trade']:>11.2f} $")
    print(f"Max drawdown      : {report['max_drawdown_pct']:>11.2f} %")
    print(f"Sharpe (annualise): {report['sharpe']:>12.2f}")
    print(f"Frais totaux      : {report['total_fees']:>11.2f} $")
    print("=" * 60)
