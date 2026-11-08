"""Environnement de reinforcement learning pour la phase 2 (spec PDF section 4).

Interface type Gymnasium (reset/step) sans dependance externe :
- Observation : features carnet (vecteur) + fenetre de price action (sequence).
- Actions : 0 = Attente, 1 = Long, 2 = Short.
- Une seule position a la fois, TP/SL bases ATR, couts identiques au backtester.
- Reward = profit_net - penalite_temps * ticks_detenus - penalite_drawdown * drawdown.
"""

import numpy as np

from dataset import BOOK_FEATURES, SEQ_FEATURES

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
        atr_multiplier_sl=1.5,
        atr_multiplier_tp=2.0,
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

    def _fees(self, price, qty):
        return (
            self.commission_per_qty * qty
            + self.commission_rate * price * qty
        )

    def _open(self, side, price):
        self.qty = self._qty()
        if self.qty <= 0:
            return
        half = self.spread / 2.0
        if side == LONG:
            self.entry_price = price + half + self.slippage
            self.stop_price = self.entry_price - self.atr[self.current_index - 1] * self.atr_multiplier_sl
            self.tp_price = self.entry_price + self.atr[self.current_index - 1] * self.atr_multiplier_sl * (
                self.atr_multiplier_tp / self.atr_multiplier_sl
            )
        else:
            self.entry_price = price - half - self.slippage
            self.stop_price = self.entry_price + self.atr[self.current_index - 1] * self.atr_multiplier_sl
            self.tp_price = self.entry_price - self.atr[self.current_index - 1] * self.atr_multiplier_sl * (
                self.atr_multiplier_tp / self.atr_multiplier_sl
            )
        self.position = side
        self.entry_tick = self.current_index
        self.capital -= self._fees(self.entry_price, self.qty)

    def _close(self, fill_price):
        gross = (fill_price - self.entry_price) * self.qty
        pnl = gross if self.position == LONG else -gross
        net = pnl - self._fees(fill_price, self.qty)
        self.capital += net
        self.realized_pnl += net
        self.position = None
        return net

    def step(self, action):
        t = self.current_index
        price = self.prices[t]
        step_reward = 0.0
        done = False

        if self.position == LONG:
            if price <= self.stop_price:
                step_reward += self._close(self.stop_price - self.slippage)
            elif price >= self.tp_price:
                step_reward += self._close(self.tp_price)
        elif self.position == SHORT:
            if price >= self.stop_price:
                step_reward += self._close(self.stop_price + self.slippage)
            elif price <= self.tp_price:
                step_reward += self._close(self.tp_price)

        if self.position is not None:
            held_ticks = t - self.entry_tick
            step_reward -= self.time_penalty * held_ticks
            if held_ticks >= self.max_holding_ticks:
                fill = price - self.slippage if self.position == LONG else price + self.slippage
                step_reward += self._close(fill)

        if self.position is None and action in (1, 2):
            side = 1 if action == 1 else -1
            allowed = True
            if self.filters is not None:
                allowed, _ = self.filters.trade_allowed(
                    side, self.raw_obi[t - 1], self.timestamps[t - 1]
                )
            if allowed:
                self._open(LONG if action == 1 else SHORT, price)

        self.peak_capital = max(self.peak_capital, self.capital)
        drawdown = (self.peak_capital - self.capital) / self.peak_capital
        step_reward -= self.drawdown_penalty * drawdown

        self.current_index += 1
        if self.current_index >= self.n:
            done = True
            if self.position is not None:
                fill = self.prices[-1] - self.slippage if self.position == LONG else self.prices[-1] + self.slippage
                step_reward += self._close(fill)

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
