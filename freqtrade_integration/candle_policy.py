"""Shared candle risk/cost assumptions, labels and conservative OHLC replay.

Decisions use a closed candle; entry is the next open. Touching both barriers
in a candle loses (stop first). Limit fills and funding are validated separately
by FreqTrade: this OHLC replay does not prove queue position or actual fills.
"""
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class CandlePolicy:
    leverage: float = 10.0
    stoploss: float = -0.02  # margin risk, as interpreted by FreqTrade
    roi_steps: tuple = ((0, 0.02), (30, 0.01), (60, 0.005))
    horizon: int = 15
    fee_entry: float = 0.0002
    fee_exit: float = 0.0002
    fee_stop: float = 0.0005
    slippage_bps: float = 1.0
    cooldown_candles: int = 5
    loss_guard_count: int = 4
    loss_guard_lookback: int = 60
    loss_guard_pause: int = 30
    drawdown_limit: float = 0.05
    drawdown_lookback: int = 240
    drawdown_pause: int = 60

    def __post_init__(self):
        if self.leverage < 1 or not -1 < self.stoploss < 0 or self.horizon < 1:
            raise ValueError('Invalid leverage, stoploss or holding horizon')
        if any(not 0 <= rate < 0.05 for rate in (self.fee_entry, self.fee_exit, self.fee_stop)):
            raise ValueError('Invalid fee rate')
        if not 0 <= self.slippage_bps <= 100 or not self.roi_steps or self.roi_steps[0][0] != 0:
            raise ValueError('Invalid slippage or ROI schedule')
        if list(self.roi_steps) != sorted(self.roi_steps) or any(t < 0 or r < 0 for t, r in self.roi_steps):
            raise ValueError('ROI schedule must be ordered, nonnegative and start at zero')

    @property
    def lookahead(self):
        return self.horizon + 1  # next open + complete holding interval

    def roi_at(self, minutes):
        return next(value for start, value in reversed(self.roi_steps) if minutes >= start)

    def protections(self):
        return [dict(method='CooldownPeriod', stop_duration_candles=self.cooldown_candles),
                dict(method='StoplossGuard', lookback_period_candles=self.loss_guard_lookback,
                     trade_limit=self.loss_guard_count, stop_duration_candles=self.loss_guard_pause,
                     only_per_pair=False),
                dict(method='MaxDrawdown', lookback_period_candles=self.drawdown_lookback,
                     trade_limit=5, max_allowed_drawdown=self.drawdown_limit, calculation_mode='ratios',
                     stop_duration_candles=self.drawdown_pause)]

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        if 'roi_steps' in values:
            values['roi_steps'] = tuple(tuple(step) for step in values['roi_steps'])
        return cls(**values)


@dataclass
class CandleTrade:
    decision_index: int
    entry_index: int
    exit_index: int
    side: str
    reason: str
    entry_price: float
    exit_price: float
    net_return: float  # return on margin, including both fees
    fee_return: float


class CandleMarket:
    def __init__(self, df):
        self.open = df['open'].to_numpy(float)
        self.high = df['high'].to_numpy(float)
        self.low = df['low'].to_numpy(float)
        self.minutes = (df['date'].dt.as_unit('ns').astype('int64').to_numpy() / 60e9 if 'date' in df
                        else np.arange(len(df), dtype=float))
    def __len__(self):
        return len(self.open)


def simulate_trade(df, decision, side, policy, cost_multiplier=1.0):
    market = df if isinstance(df, CandleMarket) else CandleMarket(df)
    return _simulate_trade(market, decision, side, policy, cost_multiplier)


def _simulate_trade(market, decision, side, policy, cost_multiplier=1.0):
    if side not in ('long', 'short') or cost_multiplier <= 0:
        raise ValueError('Invalid side or cost multiplier')
    end = decision + policy.lookahead
    if decision < 0 or end >= len(market):
        raise ValueError('Complete future unavailable for trade')
    sign = 1 if side == 'long' else -1
    entry_idx = decision + 1
    slip = policy.slippage_bps * 1e-4 * cost_multiplier
    entry = float(market.open[entry_idx]) * (1 + sign * slip)
    entry_fee, take_fee, stop_fee = (r * cost_multiplier for r in
                                    (policy.fee_entry, policy.fee_exit, policy.fee_stop))
    stop = entry * (1 + sign * policy.stoploss / policy.leverage)
    price, reason, exit_idx, fee = float(market.open[end]), 'timeout', end, stop_fee
    for j in range(entry_idx, end):
        minutes = market.minutes[j] - market.minutes[entry_idx]
        roi = policy.roi_at(minutes) / policy.leverage
        target = (entry * (1 + entry_fee) * (1 + roi) / (1 - take_fee) if sign == 1
                  else entry * (1 - entry_fee) * (1 - roi) / (1 + take_fee))
        stopped = market.low[j] <= stop if sign == 1 else market.high[j] >= stop
        targeted = market.high[j] >= target if sign == 1 else market.low[j] <= target
        if stopped:
            price = min(float(market.open[j]), stop) if sign == 1 else max(float(market.open[j]), stop)
            reason, exit_idx, fee = 'stop_loss', j, stop_fee
            break
        if targeted:
            price, reason, exit_idx, fee = target, 'roi', j, take_fee
            break
    if reason != 'roi':
        price *= 1 - sign * slip
    net = ((price * (1 - fee) / (entry * (1 + entry_fee)) - 1) if sign == 1
           else (1 - price * (1 + fee) / (entry * (1 - entry_fee)))) * policy.leverage
    fees = (entry_fee + fee * price / entry) * policy.leverage
    return CandleTrade(decision, entry_idx, exit_idx, side, reason, entry, price, float(net), float(fees))


def triple_barrier_labels(df, policy):
    market = CandleMarket(df)
    labels = np.full(len(df), -1, dtype=np.int64)
    for i in range(max(0, len(df) - policy.lookahead)):
        long = simulate_trade(market, i, 'long', policy)
        short = simulate_trade(market, i, 'short', policy)
        # Ambiguous paths profitable in both directions do not teach a direction.
        labels[i] = 1 if long.net_return > 0 >= short.net_return else (
            2 if short.net_return > 0 >= long.net_return else 0)
    return labels
