"""Boucle live paper trading, structure reelle (CCXT sandbox ou flux injecte).

La version initiale etait un stub vide (sleep + break). Cette version :
- consomme un flux de ticks via un `feed` injectable (CCXT sandbox, websocket,
  ou replay de CSV) ;
- maintient un historique glissant et calcule les features a chaque tick ;
- applique les memes regles de sizing / TP / SL que le backtester ;
- integre un kill-switch (perte journaliere max) et la mesure de latence
  d'inference.
"""

import time
from collections import deque

import numpy as np

from backtester import ASSET_COST_MODELS, LONG, SHORT, Trade
from dataset import BOOK_FEATURES, SEQ_FEATURES
from model_pytorch import predict_action
from money_management import money_manager_for_asset
from pipeline_data import build_features


class LivePaperTradingLoop:
    def __init__(
        self,
        model,
        asset="BTC",
        feed=None,
        device="cpu",
        initial_capital=10000.0,
        sequence_length=20,
        atr_multiplier_sl=1.5,
        atr_multiplier_tp=2.0,
        max_holding_ticks=100,
        max_daily_loss_pct=2.0,
        confidence_threshold=0.5,
        norm_stats=None,
        on_trade=None,
        poll_seconds=0.2,
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

        self.money_manager = money_manager_for_asset(asset, risk_per_trade=0.005)
        self.costs = ASSET_COST_MODELS.get(asset, ASSET_COST_MODELS["BTC"])
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

    def _close(self, fill_price, reason):
        pos = self.position
        slip = self.costs["slippage"]
        fill = fill_price - slip if pos.side == LONG else fill_price + slip
        fees = self.costs["commission_per_qty"] * pos.qty
        self.capital += pos.pnl_fill(fill) - fees
        pos.exit_price = fill
        pos.exit_reason = reason
        self.trades.append(pos)
        if self.on_trade:
            self.on_trade(pos)
        self.position = None

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
        if action not in (1, 2) or probs[action] < self.confidence_threshold:
            return
        atr = float(feats["ATR_Tick"].iloc[-1])
        qty = self.money_manager.calculate_position_size(self.capital, atr, self.atr_multiplier_sl)
        if qty <= 0:
            return
        half_spread = self.costs["spread"] / 2.0
        slip = self.costs["slippage"]
        side = LONG if action == 1 else SHORT
        stop_dist = self.money_manager.stop_distance(atr, self.atr_multiplier_sl)
        if side == LONG:
            entry = price + half_spread + slip
            stop = entry - stop_dist
            tp = entry + stop_dist * (self.atr_multiplier_tp / self.atr_multiplier_sl)
        else:
            entry = price - half_spread - slip
            stop = entry + stop_dist
            tp = entry - stop_dist * (self.atr_multiplier_tp / self.atr_multiplier_sl)
        self.capital -= self.costs["commission_per_qty"] * qty
        self.position = Trade(side, t_index, entry, qty, stop, tp)

    def step(self, tick, t_index=None):
        """Traite un tick brut (dict avec last_price, last_volume, bid/ask sizes...)."""
        self.raw_history.append(tick)
        if not self._features_ready():
            return
        price = tick["last_price"]
        self._manage_position(price)
        if self.position is None and not self._kill_switch_triggered():
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
                    print("Kill-switch declenche : arret de la boucle.")
                    break
                if self.poll_seconds:
                    time.sleep(self.poll_seconds)
        except KeyboardInterrupt:
            print("Interruption manuelle.")
        finally:
            if self.position is not None:
                self._close(self.raw_history[-1]["last_price"], "shutdown")
            self.running = False
        return {
            "capital": self.capital,
            "n_trades": len(self.trades),
            "trades": self.trades,
            "last_inference_ms": self.last_inference_ms,
        }
