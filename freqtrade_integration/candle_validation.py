"""Validation-only decisions and side-by-side, cost-aware candle evaluation."""
from dataclasses import asdict

import numpy as np
import torch

from freqtrade_integration.candle_policy import CandleMarket, simulate_trade


def collect_logits(model, prepared, indices):
    chunks = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(indices), 512):
            ix = indices[start:start + 512]
            chunks.append(model(torch.tensor(prepared.book[ix]),
                                torch.tensor(prepared.sequences[ix])).cpu().numpy())
    return np.concatenate(chunks) if chunks else np.zeros((0, 3))


def softmax(logits, temperature=1.):
    values = np.asarray(logits, dtype=float) / temperature
    values -= values.max(axis=1, keepdims=True)
    values = np.exp(values)
    return values / values.sum(axis=1, keepdims=True)


def temperature_from_logits(logits, labels):
    if not len(labels):
        raise ValueError('Calibration requires validation labels')
    grid = np.geomspace(0.25, 8, 81)
    scores = [-np.log(softmax(logits, t)[np.arange(len(labels)), labels].clip(1e-12)).mean() for t in grid]
    return float(grid[np.argmin(scores)])


def replay_signals(df, probabilities, indices, policy, thresholds, cost_multiplier=1.):
    """One position at a time; conservative OHLC fills, matching cooldown/guards."""
    market = CandleMarket(df)
    trades, allowed = [], 0
    for i in indices:
        i = int(i)
        if i < allowed or i + policy.lookahead >= len(df):
            continue
        action = int(np.argmax(probabilities[i]))
        if action not in (1, 2):
            continue
        side = 'long' if action == 1 else 'short'
        if probabilities[i, action] <= thresholds[side]:
            continue
        trade = simulate_trade(market, i, side, policy, cost_multiplier)
        trades.append(trade)
        allowed = trade.exit_index + policy.cooldown_candles
        recent = [t for t in trades if t.exit_index >= trade.exit_index - policy.loss_guard_lookback]
        if sum(t.reason == 'stop_loss' and t.net_return < 0 for t in recent) >= policy.loss_guard_count:
            allowed = max(allowed, trade.exit_index + policy.loss_guard_pause)
        recent = [t for t in trades if t.exit_index >= trade.exit_index - policy.drawdown_lookback]
        if len(recent) >= 5:
            cumulative = np.r_[0., np.cumsum([t.net_return for t in recent])]
            drawdown = np.max(np.maximum.accumulate(cumulative) - cumulative)
            if drawdown > policy.drawdown_limit:
                allowed = max(allowed, trade.exit_index + policy.drawdown_pause)
    return trades


def expectancy_interval(values, draws=500, seed=42):
    """Circular moving-block bootstrap of net trade returns, not iid candles."""
    values = np.asarray(values, float)
    if len(values) < 4:
        return None
    rng = np.random.default_rng(seed)
    block = max(2, int(np.sqrt(len(values))))
    starts = rng.integers(0, len(values), (draws, int(np.ceil(len(values) / block))))
    indices = (starts[..., None] + np.arange(block)) % len(values)
    means = values[indices.reshape(draws, -1)[:, :len(values)]].mean(axis=1)
    return np.quantile(means, [.025, .975]).tolist()


def _metrics(trades):
    returns = np.array([t.net_return for t in trades], float)
    gain, loss = returns[returns > 0].sum(), -returns[returns < 0].sum()
    equity = np.cumprod(np.r_[1., np.maximum(0., 1 + .99 * returns)])
    return {'n_trades': len(trades), 'expectancy': float(returns.mean()) if len(returns) else None,
            'expectancy_ci95': expectancy_interval(returns),
            'profit_factor': float(gain / loss) if loss else None,
            'return_pct': float((equity[-1] - 1) * 100),
            'max_drawdown_pct': float(np.max(1 - equity / np.maximum.accumulate(equity)) * 100),
            'win_rate': float((returns > 0).mean()) if len(returns) else None,
            'fee_return_sum': float(sum(t.fee_return for t in trades))}


def summarize_trades(trades):
    return {**_metrics(trades), 'directions': {side: _metrics([t for t in trades if t.side == side])
                                             for side in ('long', 'short')}}


def select_thresholds(probabilities, df, indices, policy, min_trades=20,
                      grid=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9), with_diagnostics=False):
    if min_trades < 1:
        raise ValueError('min_trades must be positive')
    trials = []
    result = {'long': 1., 'short': 1.}  # disabled unless validation supports a side
    for side in result:
        best, best_score = 1., 0.
        for threshold in grid:
            thresholds = {'long': 1., 'short': 1., side: float(threshold)}
            trades = replay_signals(df, probabilities, indices, policy, thresholds)
            stats = _metrics(trades)
            trials.append({'side': side, 'threshold': float(threshold), **stats})
            ci = stats['expectancy_ci95']
            if len(trades) >= min_trades and ci is not None and ci[0] > best_score:
                best, best_score = float(threshold), ci[0]
        result[side] = best
    return (result, trials) if with_diagnostics else result


def fit_decision(model, prepared, labels, df, policy, min_trades=20):
    validation = prepared.splits['validation']
    mid = len(validation) // 2
    calibration = validation[:max(0, mid - policy.lookahead)]
    selection = validation[mid:]
    if not len(calibration) or not len(selection):
        raise ValueError('Not enough validation data to separate calibration and selection')
    temperature = temperature_from_logits(collect_logits(model, prepared, calibration), labels[calibration])
    probabilities = np.zeros((len(df), 3)); probabilities[:, 0] = 1.
    probabilities[selection] = softmax(collect_logits(model, prepared, selection), temperature)
    thresholds, trials = select_thresholds(probabilities, df, selection, policy, min_trades, with_diagnostics=True)
    return {'temperature': temperature, 'thresholds': thresholds, 'min_validation_trades': min_trades,
            'calibration_first_index': int(calibration[0]), 'calibration_last_index': int(calibration[-1]),
            'selection_first_index': int(selection[0]), 'selection_last_index': int(selection[-1]),
            'test_used': False, 'validation_threshold_trials': trials, 'criterion': 'positive 95% block-bootstrap lower bound on validation'}


def trade_records(trades):
    return [asdict(t) for t in trades]
