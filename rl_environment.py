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
        half = self.spread / 2.0
        if entry_override is not None:
            self.entry_price = entry_override
        elif side == LONG:
            self.entry_price = price + half + self.slippage
        else:
            self.entry_price = price - half - self.slippage
        atr_value = self.atr[self.current_index - 1]
        atr_tp = atr_value * self.atr_multiplier_tp
        if self.min_tp_fee_multiple > 0:
            entry_fee_rate = self.maker_fee_rate if fee_rate is not None else self.commission_rate
            exit_fee_rate = self.maker_fee_rate if self.execution == "maker" else self.commission_rate
            entry_cost = 0.0 if fee_rate is not None else self.spread / 2.0 + self.slippage
            round_trip = (entry_fee_rate + exit_fee_rate) * self.entry_price + entry_cost
            atr_tp = max(atr_tp, self.min_tp_fee_multiple * round_trip)
        if side == LONG:
            self.stop_price = self.entry_price - atr_value * self.atr_multiplier_sl
            self.tp_price = self.entry_price + atr_tp
        else:
            self.stop_price = self.entry_price + atr_value * self.atr_multiplier_sl
            self.tp_price = self.entry_price - atr_tp
        self.position = side
        self.entry_tick = self.current_index
        self.capital -= self._fees(self.entry_price, self.qty, fee_rate=fee_rate)

    def _close(self, fill_price, fee_rate=None):
        gross = (fill_price - self.entry_price) * self.qty
        pnl = gross if self.position == LONG else -gross
        net = pnl - self._fees(fill_price, self.qty, fee_rate=fee_rate)
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
                maker_tp = self.execution == "maker"
                step_reward += self._close(
                    self.tp_price,
                    fee_rate=self.maker_fee_rate if maker_tp else None,
                )
        elif self.position == SHORT:
            if price >= self.stop_price:
                step_reward += self._close(self.stop_price + self.slippage)
            elif price <= self.tp_price:
                maker_tp = self.execution == "maker"
                step_reward += self._close(
                    self.tp_price,
                    fee_rate=self.maker_fee_rate if maker_tp else None,
                )

        if self.position is not None:
            held_ticks = t - self.entry_tick
            step_reward -= self.time_penalty * held_ticks
            if held_ticks >= self.max_holding_ticks:
                fill = price - self.slippage if self.position == LONG else price + self.slippage
                step_reward += self._close(fill)

        if self.pending is not None and self.position is None:
            if (
                (self.pending["side"] == LONG and price <= self.pending["limit"])
                or (self.pending["side"] == SHORT and price >= self.pending["limit"])
            ):
                self._open(
                    self.pending["side"],
                    price,
                    entry_override=self.pending["limit"],
                    fee_rate=self.maker_fee_rate,
                )
            elif t - self.pending["placed_at"] >= self.maker_timeout_ticks:
                if self.pending["replacements"] < self.maker_replacements_max:
                    step = self.spread
                    if self.pending["side"] == LONG:
                        self.pending["limit"] = self.pending["limit"] + step
                    else:
                        self.pending["limit"] = self.pending["limit"] - step
                    self.pending["placed_at"] = t
                    self.pending["replacements"] += 1
                else:
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
                    self.pending = {
                        "side": LONG if side > 0 else SHORT,
                        "limit": limit,
                        "placed_at": t,
                        "replacements": 0,
                    }
                else:
                    self._open(LONG if side > 0 else SHORT, price)

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
