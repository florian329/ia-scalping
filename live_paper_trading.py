"""Boucle live paper trading, structure reelle (CCXT sandbox ou flux injecte).

La version initiale etait un stub vide (sleep + break). Cette version :
- consomme un flux de ticks via un `feed` injectable (CCXT sandbox, websocket,
  ou replay de CSV) ;
- maintient un historique glissant et calcule les features a chaque tick ;
- applique les memes regles de sizing / TP / SL que le backtester ;
- integre un kill-switch (perte journaliere max) et la mesure de latence
  d'inference.
"""

import csv
import logging
import os
import time
from collections import deque

import numpy as np

from backtester import LONG, SHORT, Trade
from config import ASSET_COST_MODELS
from execution import fees_for, taker_entry, taker_exit
from dataset import BOOK_FEATURES, SEQ_FEATURES
from model_pytorch import predict_action
from money_management import money_manager_for_asset
from pipeline_data import build_features

logger = logging.getLogger(__name__)


class TradeJournal:
    """Journalisation CSV de chaque trade cloture."""

    FIELDNAMES = [
        "timestamp", "asset", "side", "entry_price", "exit_price",
        "qty", "pnl", "exit_reason", "capital_after",
    ]

    def __init__(self, path="trades_journal.csv", asset="BTC"):
        self.path = path
        self.asset = asset
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=self.FIELDNAMES).writeheader()

    def log(self, trade, capital_after, timestamp=None):
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
            writer.writerow({
                "timestamp": timestamp,
                "asset": self.asset,
                "side": "LONG" if trade.side == LONG else "SHORT",
                "entry_price": trade.entry_price,
                "exit_price": trade.exit_price,
                "qty": trade.qty,
                "pnl": trade.pnl_fill(trade.exit_price),
                "exit_reason": trade.exit_reason,
                "capital_after": capital_after,
            })


class LatencyMonitor:
    """Statistiques de latence d'inference (moyenne / p95 / max)."""

    def __init__(self):
        self.samples = []

    def record(self, milliseconds):
        if milliseconds is not None:
            self.samples.append(milliseconds)

    def stats(self):
        if not self.samples:
            return {"mean_ms": None, "p95_ms": None, "max_ms": None, "n": 0}
        arr = np.array(self.samples)
        return {
            "mean_ms": float(arr.mean()),
            "p95_ms": float(np.percentile(arr, 95)),
            "max_ms": float(arr.max()),
            "n": len(arr),
        }


class LivePaperTradingLoop:
    def __init__(
        self,
        model,
        asset="BTC",
        feed=None,
        device="cpu",
        initial_capital=10000.0,
        sequence_length=20,
        atr_multiplier_sl=1.69,
        atr_multiplier_tp=2.42,
        max_holding_ticks=36,
        max_daily_loss_pct=2.0,
        confidence_threshold=0.5,
        norm_stats=None,
        on_trade=None,
        poll_seconds=0.2,
        filters=None,
        journal_path=None,
        max_trades=None,
        max_consecutive_losses=3,
        cooldown_ticks=50,
        drift_monitor=None,
        execution="taker",
        maker_timeout_ticks=10,
        min_tp_fee_multiple=3.0,
        maker_replacements_max=3,
    ):
        self.model = model
        self.asset = asset
        self.feed = feed
        self.device = device
        self.capital = initial_capital
        self.initial_capital = initial_capital
        self.sequence_length = sequence_length
        self.atr_multiplier_sl = atr_multiplier_sl
        self.atr_multiplier_tp = atr_multiplier_tp
        self.max_holding_ticks = max_holding_ticks
        self.max_daily_loss_pct = max_daily_loss_pct
        self.confidence_threshold = confidence_threshold
        self.norm_stats = norm_stats
        self.on_trade = on_trade
        self.poll_seconds = poll_seconds
        self.filters = filters
        self.journal = TradeJournal(journal_path, asset=asset) if journal_path else None
        self.latency = LatencyMonitor()
        self.max_trades = max_trades
        self.max_consecutive_losses = max_consecutive_losses
        self.cooldown_ticks = cooldown_ticks
        self.consecutive_losses = 0
        self.cooldown_until = -1
        self.tick_count = 0
        self.drift_monitor = drift_monitor
        self.execution = execution
        self.maker_timeout_ticks = maker_timeout_ticks
        self.min_tp_fee_multiple = min_tp_fee_multiple
        self.maker_replacements_max = maker_replacements_max
        self.pending = None
        self.maker_fills = 0
        self.maker_cancels = 0
        self.maker_replacements = 0

        if asset != "BTC":
            raise ValueError("Seul le BTC est supporte par cette strategie.")
        self.money_manager = money_manager_for_asset(asset, risk_per_trade=0.005)
        self.costs = ASSET_COST_MODELS["BTC"]
        self.raw_history = deque(maxlen=max(sequence_length * 4, 200))
        self.position = None
        self.trades = []
        self.running = False
        self.last_inference_ms = None

    def _features_ready(self):
        return len(self.raw_history) >= self.sequence_length * 2

    def _build_frame(self):
        import pandas as pd

        return pd.DataFrame(list(self.raw_history))

    def _kill_switch_triggered(self):
        return (self.capital / self.initial_capital - 1.0) <= -(self.max_daily_loss_pct / 100.0)

    def _manage_position(self, price):
        pos = self.position
        if pos is None:
            return
        if pos.side == LONG:
            if price <= pos.stop_price:
                self._close(price if price > pos.stop_price else pos.stop_price, "stop")
                return
            if price >= pos.tp_price:
                self._close(pos.tp_price, "take_profit")
                return
        else:
            if price >= pos.stop_price:
                self._close(pos.stop_price, "stop")
                return
            if price <= pos.tp_price:
                self._close(price if price < pos.tp_price else pos.tp_price, "take_profit")
                return
        if len(self.raw_history) - pos.entry_index >= self.max_holding_ticks:
            self._close(price, "timeout")

    def _fees(self, price, qty, fee_rate=None):
        return fees_for(self.costs, price, qty, fee_rate=fee_rate)

    def _close(self, fill_price, reason):
        pos = self.position
        maker_exit = self.execution == "maker" and reason == "take_profit"
        if maker_exit:
            fill = fill_price
            fees = self._fees(fill, pos.qty, fee_rate=self.costs.maker_fee_rate)
        else:
            fill = taker_exit(self.costs, pos.side, fill_price)
            fees = self._fees(fill, pos.qty)
        self.capital += pos.pnl_fill(fill) - fees
        pos.exit_price = fill
        pos.exit_reason = reason
        self.trades.append(pos)
        net = pos.pnl_fill(fill) - fees
        if net < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        if self.consecutive_losses >= self.max_consecutive_losses:
            self.cooldown_until = self.tick_count + self.cooldown_ticks
            self.consecutive_losses = 0
            logger.warning(
                "%d pertes consecutives : gel des trades jusqu'au tick %d",
                self.max_consecutive_losses, self.cooldown_until,
            )
        if self.drift_monitor is not None and self.drift_monitor.record(net):
            self.cooldown_until = self.tick_count + self.cooldown_ticks
            logger.warning("derive du win rate : %s — gel des trades", self.drift_monitor.status())
        if self.journal:
            self.journal.log(pos, self.capital, timestamp=self._last_timestamp())
        if self.on_trade:
            self.on_trade(pos)
        self.position = None

    def _last_timestamp(self):
        if not self.raw_history:
            return None
        return self.raw_history[-1].get("timestamp")

    def _maybe_open(self, price, t_index):
        frame = self._build_frame()
        feats = build_features(frame)
        book = feats[BOOK_FEATURES].to_numpy(dtype=np.float32)
        seq = feats[SEQ_FEATURES].to_numpy(dtype=np.float32)
        if self.norm_stats is not None:
            mean, std = self.norm_stats
            book = (book - mean) / (std + 1e-8)
        start = time.perf_counter()
        action, probs = predict_action(
            self.model, book[-1], seq[-self.sequence_length :], device=self.device
        )
        self.last_inference_ms = (time.perf_counter() - start) * 1000.0
        self.latency.record(self.last_inference_ms)
        if action not in (1, 2) or probs[action] < self.confidence_threshold:
            return
        if self.filters is not None:
            obi_raw = float(feats["OBI"].iloc[-1])
            ts = self._last_timestamp()
            allowed, refused = self.filters.trade_allowed(1 if action == 1 else -1, obi_raw, ts)
            if not allowed:
                return
        atr = float(feats["ATR_Tick"].iloc[-1])
        qty = self.money_manager.calculate_position_size(self.capital, atr, self.atr_multiplier_sl)
        if qty <= 0:
            return
        side = LONG if action == 1 else SHORT
        last_tick = self.raw_history[-1] if self.raw_history else {}
        if self.execution == "maker":
            if side == LONG:
                limit = float(last_tick.get("best_bid", price - self.costs.spread / 2.0))
            else:
                limit = float(last_tick.get("best_ask", price + self.costs.spread / 2.0))
            self.pending = {
                "side": side,
                "limit": limit,
                "placed_at": self.tick_count,
                "replacements": 0,
            }
            return
        stop_dist = self.money_manager.stop_distance(atr, self.atr_multiplier_sl)
        entry = taker_entry(self.costs, side, price)
        atr_tp = atr * self.atr_multiplier_tp
        if self.min_tp_fee_multiple > 0:
            round_trip = 2 * self.costs.commission_rate * price + self.costs.spread / 2.0 + self.costs.slippage
            atr_tp = max(atr_tp, self.min_tp_fee_multiple * round_trip)
        if side == LONG:
            stop = entry - stop_dist
            tp = entry + atr_tp
        else:
            stop = entry + stop_dist
            tp = entry - atr_tp
        self.capital -= self._fees(entry, qty)
        self.position = Trade(side, t_index, entry, qty, stop, tp)

    def _check_pending(self, price, t_index):
        if self.pending is None:
            return
        if self.position is not None:
            self.pending = None
            self.maker_cancels += 1
            return
        order = self.pending
        filled = (
            (order["side"] == LONG and price <= order["limit"])
            or (order["side"] == SHORT and price >= order["limit"])
        )
        if filled:
            atr = self._current_atr()
            qty = self.money_manager.calculate_position_size(
                self.capital, atr, self.atr_multiplier_sl, price=order["limit"]
            )
            if qty > 0:
                stop_dist = self.money_manager.stop_distance(atr, self.atr_multiplier_sl)
                entry = order["limit"]
                atr_tp = atr * self.atr_multiplier_tp
                if self.min_tp_fee_multiple > 0:
                    exit_rate = self.costs.maker_fee_rate if self.execution == "maker" else self.costs.commission_rate
                    round_trip = (self.costs.maker_fee_rate + exit_rate) * entry
                    atr_tp = max(atr_tp, self.min_tp_fee_multiple * round_trip)
                if order["side"] == LONG:
                    stop = entry - stop_dist
                    tp = entry + atr_tp
                else:
                    stop = entry + stop_dist
                    tp = entry - atr_tp
                self.capital -= self._fees(
                    entry, qty, fee_rate=self.costs.maker_fee_rate
                )
                self.position = Trade(order["side"], t_index, entry, qty, stop, tp)
                self.maker_fills += 1
            self.pending = None
        elif self.tick_count - order["placed_at"] >= self.maker_timeout_ticks:
            if order["replacements"] < self.maker_replacements_max:
                step = self.costs.spread
                last_tick = self.raw_history[-1] if self.raw_history else {}
                if order["side"] == LONG:
                    ask_ref = float(last_tick.get("best_ask", price + self.costs.spread))
                    order["limit"] = min(order["limit"] + step, ask_ref)
                else:
                    bid_ref = float(last_tick.get("best_bid", price - self.costs.spread))
                    order["limit"] = max(order["limit"] - step, bid_ref)
                order["placed_at"] = self.tick_count
                order["replacements"] += 1
                self.maker_replacements += 1
            else:
                self.pending = None
                self.maker_cancels += 1

    def _current_atr(self):
        frame = self._build_frame()
        feats = build_features(frame)
        return float(feats["ATR_Tick"].iloc[-1])

    def step(self, tick, t_index=None):
        """Traite un tick brut (dict avec last_price, last_volume, bid/ask sizes...)."""
        self.raw_history.append(tick)
        self.tick_count += 1
        if not self._features_ready():
            return
        price = tick["last_price"]
        self._manage_position(price)
        if self.position is not None:
            self.pending = None
        self._check_pending(price, t_index or len(self.raw_history))
        if self.position is None and self.pending is None and not self._kill_switch_triggered():
            if self.tick_count < self.cooldown_until:
                return
            if self.max_trades is not None and len(self.trades) >= self.max_trades:
                return
            self._maybe_open(price, t_index or len(self.raw_history))

    def run(self, max_ticks=None):
        """Boucle principale : consomme le feed jusqu'a epuisement."""
        self.running = True
        n = 0
        try:
            for tick in self.feed:
                self.step(tick, t_index=n)
                n += 1
                if max_ticks and n >= max_ticks:
                    break
                if self._kill_switch_triggered():
                    logger.warning("Kill-switch declenche : arret de la boucle.")
                    break
                if self.poll_seconds:
                    time.sleep(self.poll_seconds)
        except KeyboardInterrupt:
            logger.info("Interruption manuelle.")
        finally:
            if self.position is not None:
                self._close(self.raw_history[-1]["last_price"], "shutdown")
            self.running = False
        result = {
            "capital": self.capital,
            "n_trades": len(self.trades),
            "trades": self.trades,
            "last_inference_ms": self.last_inference_ms,
            "latency": self.latency.stats(),
            "execution": self.execution,
            "maker_fills": self.maker_fills,
            "maker_cancels": self.maker_cancels,
            "maker_replacements": self.maker_replacements,
        }
        if hasattr(self.feed, "reconnect_count"):
            result["reconnect_count"] = self.feed.reconnect_count
        return result
