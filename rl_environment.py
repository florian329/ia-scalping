"""Environnement de reinforcement learning pour la phase 2 (spec PDF section 4).

Interface type Gymnasium (reset/step) sans dependance externe :
- Observation : features carnet (vecteur) + fenetre de price action (sequence).
- Actions : 0 = Attente, 1 = Long, 2 = Short.
- Une seule position a la fois, TP/SL bases ATR, couts identiques au backtester.
- Reward = profit_net - penalite_temps * ticks_detenus - penalite_drawdown * drawdown.
"""

import numpy as np

from dataset import BOOK_FEATURES, SEQ_FEATURES
from execution import (
    MakerOrderBook,
    compute_stop_tp,
    fees_for,
    taker_entry,
    taker_exit,
)
from config import ExecutionConfig

FLAT, LONG, SHORT = 0, 1, -1


class ScalpingEnv:
    def __init__(
        self,
        df,
        asset="BTC",
        _validate_asset=True,
        sequence_length=20,
        initial_capital=10000.0,
        risk_per_trade=0.005,
        atr_multiplier_sl=1.69,
        atr_multiplier_tp=2.42,
        spread=1.0,
        commission_per_qty=0.0,
        commission_rate=0.0,
        slippage=0.5,
        time_penalty=0.001,
        drawdown_penalty=0.01,
        max_holding_ticks=100,
        norm_stats=None,
        filters=None,
        min_qty=0.0,
        reward_scale=1.0,
        execution="taker",
        maker_fee_rate=0.0002,
        maker_timeout_ticks=10,
        min_tp_fee_multiple=3.0,
        maker_replacements_max=3,
    ):
        if asset != "BTC":
            raise ValueError("Seul le BTC est supporte par cette strategie.")
        self.point_value = 1.0
        self.sequence_length = sequence_length
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.atr_multiplier_sl = atr_multiplier_sl
        self.atr_multiplier_tp = atr_multiplier_tp
        self.spread = spread
        self.commission_per_qty = commission_per_qty
        self.commission_rate = commission_rate
        self.slippage = slippage
        self.max_leverage = 1.0
        self.time_penalty = time_penalty
        self.drawdown_penalty = drawdown_penalty
        self.max_holding_ticks = max_holding_ticks
        self.filters = filters
        self.min_qty = min_qty
        self.reward_scale = reward_scale
        self.execution = execution
        self.maker_fee_rate = maker_fee_rate
        self.maker_timeout_ticks = maker_timeout_ticks
        self.min_tp_fee_multiple = min_tp_fee_multiple
        self.maker_replacements_max = maker_replacements_max
        self.pending = None

        book = df[BOOK_FEATURES].to_numpy(dtype=np.float32)
        if norm_stats is not None:
            mean, std = norm_stats
            book = (book - mean) / (std + 1e-8)
        self.book = book
        self.seq = df[SEQ_FEATURES].to_numpy(dtype=np.float32)
        self.prices = df["last_price"].to_numpy(dtype=np.float64)
        self.atr = df["ATR_Tick"].to_numpy(dtype=np.float64) if "ATR_Tick" in df else np.full(len(df), 1.0)
        self.raw_obi = df["OBI"].to_numpy(dtype=np.float64) if "OBI" in df else np.zeros(len(df))
        self.timestamps = df["timestamp"].to_numpy() if "timestamp" in df else [None] * len(df)
        self.n = len(self.prices)

    @property
    def observation_space(self):
        return {
            "book": (len(BOOK_FEATURES),),
            "seq": (self.sequence_length, self.seq.shape[1]),
        }

    @property
    def action_space(self):
        return (3,)

    def _observation(self):
        t = self.current_index
        return {
            "book": self.book[t - 1],
            "seq": self.seq[t - self.sequence_length : t],
        }

    def reset(self):
        self.current_index = self.sequence_length
        self.capital = self.initial_capital
        self.peak_capital = self.initial_capital
        self.position = None
        self.pending = None
        self.order_book = None
        self.entry_price = 0.0
        self.stop_price = 0.0
        self.tp_price = 0.0
        self.qty = 0.0
        self.entry_tick = 0
        self.realized_pnl = 0.0
        return self._observation()

    def _qty(self):
        atr = self.atr[self.current_index - 1]
        price = self.prices[self.current_index]
        if atr <= 0:
            qty = self.min_qty
        else:
            stop_dist = atr * self.atr_multiplier_sl
            qty = (self.capital * self.risk_per_trade) / (stop_dist * self.point_value)
        if price > 0:
            qty = min(qty, self.capital * self.max_leverage / price)
        return max(self.min_qty, qty)

    def _fees(self, price, qty, fee_rate=None):
        rate = self.commission_rate if fee_rate is None else fee_rate
        return self.commission_per_qty * qty + rate * price * qty

    def _open(self, side, price, entry_override=None, fee_rate=None):
        self.qty = self._qty()
        if self.qty <= 0:
            return
        if entry_override is not None:
            entry = entry_override
        else:
            entry = taker_entry(self._costs(), side, price)
        atr_value = self.atr[self.current_index - 1]
        cfg = self._exec_cfg()
        stop, tp = compute_stop_tp(
            cfg,
            self._costs(),
            side,
            entry,
            atr_value,
            entry_fee_rate=fee_rate,
            taker_entry_fill=fee_rate is None,
        )
        self.entry_price = entry
        self.stop_price = stop
        self.tp_price = tp
        self.position = side
        self.entry_tick = self.current_index
        self.capital -= fees_for(self._costs(), entry, self.qty, fee_rate=fee_rate)

    def _close(self, fill_price, fee_rate=None):
        gross = (fill_price - self.entry_price) * self.qty
        pnl = gross if self.position == LONG else -gross
        net = pnl - fees_for(self._costs(), fill_price, self.qty, fee_rate=fee_rate)
        self.capital += net
        self.realized_pnl += net
        self.position = None
        return net

    def _costs(self):
        from config import ASSET_COST_MODELS

        return ASSET_COST_MODELS["BTC"]

    def _exec_cfg(self):
        return ExecutionConfig(
            atr_multiplier_sl=self.atr_multiplier_sl,
            atr_multiplier_tp=self.atr_multiplier_tp,
            execution=self.execution,
            min_tp_fee_multiple=self.min_tp_fee_multiple,
            maker_timeout_ticks=self.maker_timeout_ticks,
            maker_replacements_max=self.maker_replacements_max,
        )

    def step(self, action):
        t = self.current_index
        price = self.prices[t]
        step_reward = 0.0
        done = False

        maker_tp = self.execution == "maker"
        if self.position == LONG:
            if price <= self.stop_price:
                step_reward += self._close(taker_exit(self._costs(), LONG, self.stop_price))
            elif price >= self.tp_price:
                step_reward += self._close(
                    self.tp_price,
                    fee_rate=self.maker_fee_rate if maker_tp else None,
                )
        elif self.position == SHORT:
            if price >= self.stop_price:
                step_reward += self._close(taker_exit(self._costs(), SHORT, self.stop_price))
            elif price <= self.tp_price:
                step_reward += self._close(
                    self.tp_price,
                    fee_rate=self.maker_fee_rate if maker_tp else None,
                )

        if self.position is not None:
            held_ticks = t - self.entry_tick
            step_reward -= self.time_penalty * held_ticks
            if held_ticks >= self.max_holding_ticks:
                step_reward += self._close(taker_exit(self._costs(), self.position, price))

        if self.order_book is None:
            self.order_book = MakerOrderBook(self._exec_cfg(), self._costs())
        if self.pending is not None and self.position is None:
            filled_order = self.order_book.update(price, t, None, None)
            if filled_order is not None:
                self._open(
                    filled_order.side,
                    price,
                    entry_override=filled_order.limit,
                    fee_rate=self.maker_fee_rate,
                )
                self.pending = None
            elif not self.order_book.has_pending():
                self.pending = None
        if self.position is not None:
            self.pending = None

        if self.position is None and self.pending is None and action in (1, 2):
            side = 1 if action == 1 else -1
            allowed = True
            if self.filters is not None:
                allowed, _ = self.filters.trade_allowed(
                    side, self.raw_obi[t - 1], self.timestamps[t - 1]
                )
            if allowed:
                if self.execution == "maker":
                    half = self.spread / 2.0
                    limit = price - half if side > 0 else price + half
                    self.order_book.place(side, limit, t)
                    self.pending = self.order_book.order
                else:
                    self._open(LONG if side > 0 else SHORT, price)

        self.peak_capital = max(self.peak_capital, self.capital)
        drawdown = (self.peak_capital - self.capital) / self.peak_capital
        step_reward -= self.drawdown_penalty * drawdown

        self.current_index += 1
        if self.current_index >= self.n:
            done = True
            if self.position is not None:
                step_reward += self._close(taker_exit(self._costs(), self.position, self.prices[-1]))

        return (
            self._observation(),
            float(step_reward * self.reward_scale),
            done,
            {"capital": self.capital},
        )


class MultiSegmentEnv:
    """Enveloppe plusieurs ScalpingEnv (segments distincts du train).

    reset() passe au segment suivant : l'agent PPO n'optimise jamais un seul
    scenario, ce qui limite le surapprentissage du backtest. L'interface est
    identique a ScalpingEnv (reset/step/observation_space).
    """

    def __init__(self, envs):
        if not envs:
            raise ValueError("MultiSegmentEnv necessite au moins un environnement")
        self.envs = list(envs)
        self.current = 0
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

    @property
    def capital(self):
        return self.envs[self.current].capital

    def reset(self):
        self.current = (self.current + 1) % len(self.envs)
        return self.envs[self.current].reset()

    def step(self, action):
        return self.envs[self.current].step(action)
