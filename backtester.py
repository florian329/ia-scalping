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
from asset_filters import AssetFilters

FLAT, LONG, SHORT = 0, 1, -1

ASSET_COST_MODELS = {
    "BTC": {
        "spread": 0.01,
        "slippage": 0.01,
        "commission_per_qty": 0.0,
        "commission_rate": 0.001,
        "maker_fee_rate": 0.0002,
    },
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
    """Simule la strategie sur df (features calculees par pipeline_data.build_features).

    Retourne un dict : capital final, metriques et liste des trades.
    """
    seq_features = seq_features or SEQ_FEATURES
    costs = ASSET_COST_MODELS.get(asset, ASSET_COST_MODELS["BTC"])
    if asset != "BTC":
        raise ValueError("Seul le BTC est supporte par cette strategie.")
    if money_manager is None:
        money_manager = ScalpingMoneyManager(risk_per_trade=0.005, point_value=1.0)

    book = df[BOOK_FEATURES].to_numpy(dtype=np.float32)
    if norm_stats is not None:
        mean, std = norm_stats
        book = (book - mean) / (std + 1e-8)
    seq = df[seq_features].to_numpy(dtype=np.float32)
    prices = df["last_price"].to_numpy(dtype=np.float64)
    atr = df["ATR_Tick"].to_numpy(dtype=np.float64) if "ATR_Tick" in df else np.full(len(df), 1.0)
    raw_obi = df["OBI"].to_numpy(dtype=np.float64) if "OBI" in df else np.zeros(len(df))
    timestamps = df["timestamp"].to_numpy() if "timestamp" in df else [None] * len(df)

    capital = initial_capital
    position = None
    trades = []
    equity_curve = np.full(len(df), np.nan)
    equity_curve[:sequence_length] = capital
    total_fees = 0.0
    consecutive_losses = 0
    cooldown_until = -1

    actions_cache, probs_cache = None, None
    if vectorized:
        actions_cache, probs_cache = _batch_predict(
            model, book, seq, sequence_length, device, inference_chunk_size
        )

    def _fees_for(price, qty, fee_rate=None):
        rate = costs.get("commission_rate", 0.0) if fee_rate is None else fee_rate
        return (
            costs.get("commission_per_qty", 0.0) * qty
            + rate * price * qty
        )

    def _tp_distance(entry, atr_value, entry_fee_rate, exit_fee_rate, taker_entry):
        """TP = max(multiple ATR, plancher k x couts aller-retour).

        Sans plancher, un TP serre peut etre plus petit que les frais : chaque
        trade gagnant remplit le courtier, pas le compte."""
        atr_dist = atr_value * atr_multiplier_sl * (atr_multiplier_tp / atr_multiplier_sl)
        if min_tp_fee_multiple <= 0:
            return atr_dist
        entry_cost = (costs["spread"] / 2.0 + costs["slippage"]) if taker_entry else 0.0
        fees_price = (entry_fee_rate + exit_fee_rate) * entry
        round_trip = fees_price + entry_cost
        return max(atr_dist, min_tp_fee_multiple * round_trip)

    def close_position(trade, exit_index, exit_price, reason):
        nonlocal capital, position, total_fees, consecutive_losses, cooldown_until
        maker_exit = execution == "maker" and reason == "take_profit"
        if maker_exit:
            fill = exit_price
            fees = _fees_for(fill, trade.qty, fee_rate=costs.get("maker_fee_rate", 0.0))
            slip_cost = 0.0
        else:
            slip = costs["slippage"]
            fill = exit_price - slip if trade.side == LONG else exit_price + slip
            fees = _fees_for(fill, trade.qty)
            slip_cost = slip * trade.qty * money_manager.point_value
        net = trade.pnl_fill(fill) - fees
        capital += net
        total_fees += fees + slip_cost
        if net < 0:
            consecutive_losses += 1
            if consecutive_losses >= max_consecutive_losses:
                cooldown_until = exit_index + cooldown_ticks
                consecutive_losses = 0
        else:
            consecutive_losses = 0
        trade.exit_index = exit_index
        trade.exit_price = fill
        trade.exit_reason = reason
        trades.append(trade)
        position = None

    def open_position(side, t, action_price, entry_override=None, fee_rate=None):
        nonlocal capital, position, total_fees
        qty = money_manager.calculate_position_size(
            capital, atr[t - 1], price=entry_override if entry_override is not None else action_price
        )
        if qty <= 0:
            return
        half_spread = costs["spread"] / 2.0
        slip = costs["slippage"]
        if entry_override is not None:
            entry = entry_override
        elif side == LONG:
            entry = action_price + half_spread + slip
        else:
            entry = action_price - half_spread - slip
        entry_fee_rate = (
            costs.get("commission_rate", 0.0)
            if fee_rate is None
            else fee_rate
        )
        exit_fee_rate = (
            costs.get("maker_fee_rate", 0.0)
            if execution == "maker"
            else costs.get("commission_rate", 0.0)
        )
        taker_entry = fee_rate is None
        tp_dist = _tp_distance(
            entry, atr[t - 1], entry_fee_rate, exit_fee_rate, taker_entry
        )
        if side == LONG:
            stop = entry - money_manager.stop_distance(atr[t - 1], atr_multiplier_sl)
            tp = entry + tp_dist
        else:
            stop = entry + money_manager.stop_distance(atr[t - 1], atr_multiplier_sl)
            tp = entry - tp_dist
        fees = _fees_for(entry, qty, fee_rate=fee_rate)
        capital -= fees
        total_fees += fees
        position = Trade(side, t, entry, qty, stop, tp)

    best_bid = df["best_bid"].to_numpy(dtype=np.float64) if "best_bid" in df else None
    best_ask = df["best_ask"].to_numpy(dtype=np.float64) if "best_ask" in df else None
    pending = None
    maker_fills = 0
    maker_cancels = 0
    maker_replacements = 0

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

        if position is not None:
            unrealized = (price - position.entry_price) * position.qty * position.direction
            equity_curve[t] = capital + unrealized
        else:
            equity_curve[t] = capital
        if pending is not None:
            if position is not None:
                pending = None
                maker_cancels += 1
            else:
                filled = False
                if pending["side"] == LONG and price <= pending["limit"]:
                    filled = True
                elif pending["side"] == SHORT and price >= pending["limit"]:
                    filled = True
                elif t - pending["placed_at"] >= maker_timeout_ticks:
                    if pending["replacements"] < maker_replacements_max:
                        step = costs["spread"]
                        if pending["side"] == LONG:
                            ask_ref = best_ask[t] if best_ask is not None else price + costs["spread"] / 2.0
                            pending["limit"] = min(pending["limit"] + step, ask_ref)
                        else:
                            bid_ref = best_bid[t] if best_bid is not None else price - costs["spread"] / 2.0
                            pending["limit"] = max(pending["limit"] - step, bid_ref)
                        pending["placed_at"] = t
                        pending["replacements"] += 1
                        maker_replacements += 1
                    else:
                        pending = None
                        maker_cancels += 1
                if filled:
                    open_position(
                        pending["side"],
                        t,
                        price,
                        entry_override=pending["limit"],
                        fee_rate=costs.get("maker_fee_rate", 0.0),
                    )
                    if position is not None:
                        maker_fills += 1
                    pending = None

        in_cooldown = t < cooldown_until
        quota_reached = max_trades is not None and len(trades) >= max_trades
        if position is None and pending is None and not in_cooldown and not quota_reached:
            if actions_cache is not None:
                k = t - sequence_length
                action = int(actions_cache[k])
                probs = probs_cache[k]
            else:
                action, probs = predict_action(
                    model, book[t - 1], seq[t - sequence_length : t], device=device
                )
            if probs[action] >= confidence_threshold and action in (1, 2):
                side = 1 if action == 1 else -1
                allowed = True
                if filters is not None:
                    allowed, _ = filters.trade_allowed(side, raw_obi[t - 1], timestamps[t - 1])
                if allowed:
                    if execution == "maker":
                        if side > 0:
                            limit = best_bid[t] if best_bid is not None else price - costs["spread"] / 2.0
                        else:
                            limit = best_ask[t] if best_ask is not None else price + costs["spread"] / 2.0
                        pending = {
                            "side": LONG if side > 0 else SHORT,
                            "limit": float(limit),
                            "placed_at": t,
                            "replacements": 0,
                        }
                    else:
                        open_position(LONG if side > 0 else SHORT, t, price)

    if position is not None:
        close_position(position, len(df) - 1, prices[-1], "end_of_data")

    report = _build_report(trades, equity_curve, initial_capital, total_fees)
    report["execution"] = execution
    report["maker_fills"] = maker_fills
    report["maker_cancels"] = maker_cancels
    report["maker_replacements"] = maker_replacements
    return report


def _batch_predict(model, book, seq, sequence_length, device, chunk_size=4096):
    """Inference vectorisee : predit toutes les actions en paquets au lieu d'un
    appel par tick. ~50x plus rapide sur de gros volumes ; meme resultat que le
    mode sequentiel (le modele est sans etat entre les ticks)."""
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
