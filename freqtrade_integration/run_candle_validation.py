"""Purged walk-forward validation; writes candidates, never touches the live bot.

Example: python -m freqtrade_integration.run_candle_validation --data ticks.feather
 --out outputs/new-run --train-days 21 --validation-days 7 --test-days 7 --folds 3
"""
import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from calibration import expected_calibration_error
from freqtrade_integration.candle_policy import CandlePolicy
from freqtrade_integration.candle_validation import (
    collect_logits, fit_decision, replay_signals, softmax, summarize_trades, trade_records,
)
from freqtrade_integration.train_candle_model import load_candles, train_candidate

ROOT = Path(__file__).resolve().parents[1]


def classification(probabilities, labels):
    predicted = probabilities.argmax(axis=1)
    result = {'ece': expected_calibration_error(probabilities, labels)}
    for value, name in enumerate(('wait', 'long', 'short')):
        actual, selected = labels == value, predicted == value
        correct = int((actual & selected).sum())
        result[name] = {'support': int(actual.sum()), 'predictions': int(selected.sum()),
                        'precision': correct / int(selected.sum()) if selected.sum() else None,
                        'recall': correct / int(actual.sum()) if actual.sum() else None}
    return result


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False))


def freqtrade_backtest(model_dir, data_dir, start, end, out, fee=0.0005):
    """Separate config, model path and result directory. No trading DB is opened."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    config = json.loads((ROOT / 'freqtrade_integration/config_ia_scalping.json').read_text())
    config['api_server'] = {'enabled': False, 'listen_ip_address': '127.0.0.1', 'listen_port': 8081,
                            'username': 'offline-validation', 'password': 'unused-disabled-api',
                            'jwt_secret_key': 'unused-offline-validation-api-is-disabled'}
    config['telegram'] = {'enabled': False, 'token': '', 'chat_id': ''}
    config['exchange']['key'] = config['exchange']['secret'] = ''
    config['dry_run'] = True
    config['db_url'] = 'sqlite:///' + str(out / 'unused-simulation.sqlite')
    config['fee'] = fee
    config['ia_model_dir'] = str(Path(model_dir).resolve())
    config_path = out / 'config.json'
    write_json(config_path, config)
    timerange = f'{start.strftime("%Y%m%dT%H%M")}-{end.strftime("%Y%m%dT%H%M")}'
    command = [sys.executable, '-m', 'freqtrade', 'backtesting', '--config', str(config_path),
               '--userdir', str(out / 'user_data'), '--datadir', str(Path(data_dir).resolve()),
               '--strategy-path', str(ROOT / 'freqtrade_integration'), '--strategy', 'IaScalpingStrategy',
               '--timerange', timerange, '--cache', 'none', '--enable-protections',
               '--export', 'trades', '--backtest-directory', str(out / 'results')]
    (out / 'user_data').mkdir()
    (out / 'results').mkdir()
    env = dict(os.environ, IA_SCALPING_MODEL_DIR=str(Path(model_dir).resolve()), OMP_NUM_THREADS='2')
    with (out / 'command.log').open('w') as log:
        code = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT).returncode
    write_json(out / 'command.json', {'argv': command, 'returncode': code, 'dry_run': True})
    return read_freqtrade_result(out, code)


def read_freqtrade_result(directory, returncode):
    directory = Path(directory)
    base = {'returncode': returncode, 'log': str(directory / 'command.log'), 'validated': False}
    text = (directory / 'command.log').read_text()
    if returncode or ' - ERROR - ' in text or 'Configuration error:' in text or 'Traceback (most recent call last)' in text:
        return {**base, 'reason': 'FreqTrade reported an error; exit code alone is insufficient'}
    archives = list((directory / 'results').glob('*.zip')) + list(directory.glob('results-*.zip'))
    for path in sorted(archives, reverse=True):
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                if not name.endswith('.json'):
                    continue
                value = json.loads(archive.read(name))
                if 'strategy' not in value or not isinstance(value['strategy'], dict):
                    continue
                stats = value['strategy'].get('IaScalpingStrategy')
                if stats is not None:
                    return {**base, 'validated': True, 'result': str(path),
                            'metrics': {key: stats.get(key) for key in
                                        ['total_trades', 'profit_total', 'profit_factor',
                                         'max_drawdown_account', 'trade_count_long', 'trade_count_short']}}
    if 'No trades made.' in text and 'STRATEGY SUMMARY' in text and 'IaScalpingStrategy' in text:
        return {**base, 'validated': True, 'zero_trades_confirmed_by_engine': True,
                'metrics': {'total_trades': 0, 'profit_total': 0.0, 'profit_factor': None,
                            'max_drawdown_account': 0.0, 'trade_count_long': 0, 'trade_count_short': 0}}
    return {**base, 'reason': 'No structured result or explicit zero-trade engine summary was exported'}


def complete_utc_days(df):
    if not len(df):
        raise ValueError('No candles supplied')
    cutoff = (df.date.iloc[-1] + pd.Timedelta(minutes=1)).normalize()
    start = df.date.iloc[0].normalize()
    if df.date.iloc[0] != start:
        start += pd.Timedelta(days=1)
    complete = df[(df.date >= start) & (df.date < cutoff)].reset_index(drop=True)
    if not len(complete):
        raise ValueError('No complete UTC day available')
    return complete, cutoff


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--out', required=True, help='New result directory; refuses overwrite')
    parser.add_argument('--train-days', type=int, default=21)
    parser.add_argument('--validation-days', type=int, default=7)
    parser.add_argument('--test-days', type=int, default=7)
    parser.add_argument('--folds', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=6)
    parser.add_argument('--hidden-dim', type=int, default=32)
    parser.add_argument('--min-trades', type=int, default=20)
    parser.add_argument('--freqtrade-data-dir')
    parser.add_argument('--fee', type=float, default=0.0005, help='Conservative uniform fee per leg')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    torch.set_num_threads(2)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    df = load_candles(args.data)
    # Use complete UTC days only, and report this cutoff explicitly.
    df, cutoff = complete_utc_days(df)
    train_rows, val_rows, test_rows = (d * 1440 for d in (args.train_days, args.validation_days, args.test_days))
    span = train_rows + val_rows + test_rows
    if min(train_rows, val_rows, test_rows, args.folds) < 1 or len(df) < span + (args.folds - 1) * test_rows:
        raise ValueError('Not enough complete 1m days for the requested walk-forward protocol')
    policy = CandlePolicy(fee_entry=args.fee, fee_exit=args.fee, fee_stop=args.fee)
    summaries, all_selected, all_reference = [], [], []
    for fold in range(args.folds):
        end = len(df) - (args.folds - fold - 1) * test_rows
        segment = df.iloc[end - span:end].reset_index(drop=True)
        directory = out / f'fold-{fold + 1}'
        logging.info('Fold %d/%d: %s -> %s', fold + 1, args.folds, segment.date.iloc[0], segment.date.iloc[-1])
        model, prepared, labels, metadata = train_candidate(
            segment, directory, policy=policy, epochs=args.epochs, hidden_dim=args.hidden_dim,
            train_ratio=train_rows / span, val_ratio=val_rows / span)
        decision = fit_decision(model, prepared, labels, segment, policy, min_trades=args.min_trades)
        write_json(directory / 'decision.json', decision)
        test_indices = prepared.splits['test']
        logits = collect_logits(model, prepared, test_indices)
        probabilities = np.zeros((len(segment), 3)); probabilities[:, 0] = 1
        probabilities[test_indices] = softmax(logits, decision['temperature'])
        selected = replay_signals(segment, probabilities, test_indices, policy, decision['thresholds'])
        reference_probs = probabilities.copy(); reference_probs[test_indices] = softmax(logits)
        reference = replay_signals(segment, reference_probs, test_indices, policy, {'long': .4, 'short': .4})
        stress = {}
        for multiple in (1., 1.5, 2., 3.):
            trades = replay_signals(segment, probabilities, test_indices, policy,
                                    decision['thresholds'], cost_multiplier=multiple)
            stress[str(multiple)] = summarize_trades(trades)
        report = {'fold': fold + 1, 'training': metadata, 'decision': decision,
                  'test_classification': classification(probabilities[test_indices], labels[test_indices]),
                  'test_selected': summarize_trades(selected),
                  'test_reference_fixed_0_4': summarize_trades(reference),
                  'cost_stress': stress,
                  'limitations': ['OHLC fills are approximations; no queue reconstruction',
                                  'Funding evaluated by FreqTrade, not predicted by OHLC replay',
                                  'Fixed 0.4 reference is this candidate, not the deployed model']}
        write_json(directory / 'report.json', report)
        pd.DataFrame(trade_records(selected)).to_csv(directory / 'test_trades.csv', index=False)
        pd.DataFrame(trade_records(reference)).to_csv(directory / 'test_reference_trades.csv', index=False)
        if args.freqtrade_data_dir:
            report['freqtrade'] = freqtrade_backtest(directory, args.freqtrade_data_dir,
                segment.date.iloc[test_indices[0]], segment.date.iloc[-1] + pd.Timedelta(minutes=1),
                directory / 'freqtrade', fee=args.fee)
            write_json(directory / 'report.json', report)
        summaries.append(report)
        all_selected += selected
        all_reference += reference
        logging.info('Fold %d done: selected %d trades; fixed reference %d', fold + 1, len(selected), len(reference))
        write_json(out / 'progress.json', {'completed_folds': fold + 1, 'planned_folds': args.folds})
    aggregate = summarize_trades(all_selected)
    lower = aggregate['expectancy_ci95']
    qualified = all(r.get('freqtrade', {}).get('validated') and r['freqtrade']['metrics']['profit_total'] > 0 for r in summaries) and len(all_selected) >= 100 and lower is not None and lower[0] > 0 and all(
        r['test_selected']['n_trades'] >= args.min_trades and r['test_selected']['expectancy'] > 0 for r in summaries)
    result = {'source': str(Path(args.data).resolve()), 'source_sha256': hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
              'complete_utc_days_cutoff': str(cutoff), 'source_start': str(df.date.iloc[0]),
              'source_end': str(df.date.iloc[-1]), 'candles': len(df),
              'execution_policy': policy.to_dict(), 'folds': summaries,
              'aggregate_selected': aggregate, 'aggregate_reference': summarize_trades(all_reference),
              'profitability_demonstrated': qualified, 'automatic_promotion': False,
              'active_bot_or_model_modified': False,
              'verdict': 'candidate requires independent paper confirmation' if qualified else
                         'profitability not demonstrated; do not promote on these results'}
    write_json(out / 'summary.json', result)
    print(json.dumps({key: result[key] for key in ['aggregate_selected', 'aggregate_reference', 'verdict']}, indent=2))


if __name__ == '__main__':
    main()
