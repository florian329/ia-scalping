"""Train an isolated BTC candle candidate with purged, causal validation."""
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from freqtrade_integration.candle_features import BOOK_FEATURES, SEQ_FEATURES, build_features, _build_seq_windows
from freqtrade_integration.candle_policy import CandlePolicy, triple_barrier_labels
from freqtrade_integration.candle_training import prepare_inputs
from model_pytorch import ScalpingMultimodalNet
from training_engine import train_model

build_seq_windows = _build_seq_windows


def load_candles(path):
    path = Path(path)
    if path.suffix == '.feather':
        df = pd.read_feather(path)
    elif path.suffix == '.json':
        df = pd.read_json(path, orient='values')
        df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        df['date'] = pd.to_datetime(df['date'], unit='ms', utc=True)
    else:
        df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'], utc=True, format='ISO8601')
    df = df.sort_values('date').reset_index(drop=True)
    if (df['date'].dt.floor('min') != df['date']).any():
        raise ValueError('Candles must start on UTC minute boundaries')
    if df['date'].duplicated().any() or (df['date'].diff().dropna() != pd.Timedelta(minutes=1)).any():
        raise ValueError('Candles must be unique and contiguous at 1 minute')
    ohlcv = df[['open', 'high', 'low', 'close', 'volume']].to_numpy(float)
    if not np.isfinite(ohlcv).all() or (ohlcv[:, :4] <= 0).any() or (ohlcv[:, 4] < 0).any():
        raise ValueError('Invalid OHLCV values')
    if (df.high < df[['open', 'close']].max(axis=1)).any() or (df.low > df[['open', 'close']].min(axis=1)).any():
        raise ValueError('Inconsistent OHLC price range')
    return df


def build_labels(df, horizon=15, fee_rate=None, policy=None):
    if policy is None:
        fees = {} if fee_rate is None else dict(fee_entry=fee_rate, fee_exit=fee_rate, fee_stop=fee_rate)
        policy = CandlePolicy(horizon=horizon, **fees)
    return triple_barrier_labels(df, policy)


def train_candidate(df, out, policy=None, epochs=10, hidden_dim=64, seed=42,
                    train_ratio=0.7, val_ratio=0.15):
    policy = policy or CandlePolicy()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)  # never overwrite an active model or previous run
    torch.manual_seed(seed)
    np.random.seed(seed)
    features = build_features(df, version=2)
    labels = build_labels(df, policy=policy)
    prepared = prepare_inputs(features, horizon=policy.lookahead,
                              train_ratio=train_ratio, val_ratio=val_ratio)
    train_idx, val_idx = prepared.splits['train'], prepared.splits['validation']
    assert (labels[np.concatenate(list(prepared.splits.values()))] >= 0).all()
    def loader(indices):
        return DataLoader(TensorDataset(torch.tensor(prepared.book[indices]),
                          torch.tensor(prepared.sequences[indices]), torch.tensor(labels[indices])),
                          batch_size=256, shuffle=False)
    counts = np.bincount(labels[train_idx], minlength=3)
    weights = np.sqrt(len(train_idx) / (3 * np.maximum(counts, 1))).clip(0.25, 5.0)
    model = ScalpingMultimodalNet(len(BOOK_FEATURES), len(SEQ_FEATURES), hidden_dim)
    best_loss = train_model(model, loader(train_idx), loader(val_idx), epochs=epochs, lr=0.001,
                           patience=3, seed=seed, class_weights=weights,
                           checkpoint_path=out / 'best_checkpoint.pth')
    metadata = {
        'format_version': 2, 'feature_version': 2, 'book_dim': len(BOOK_FEATURES),
        'seq_dim': len(SEQ_FEATURES), 'hidden_dim': hidden_dim, 'sequence_length': 20,
        'execution_policy': policy.to_dict(), 'seed': seed, 'best_val_loss': best_loss,
        'train_class_counts': counts.tolist(), 'train_class_weights': weights.tolist(),
        'splits': {name: {'first': int(idx[0]), 'last': int(idx[-1]), 'n': len(idx),
                          'start': str(df.date.iloc[idx[0]]), 'end': str(df.date.iloc[idx[-1]])}
                   for name, idx in prepared.splits.items()},
        'training_label_future_end': str(df.date.iloc[train_idx[-1] + policy.lookahead]),
        'test_used_for_training_or_threshold_selection': False,
    }
    torch.save({**metadata, 'model_state_dict': model.state_dict()}, out / 'candle_model.pth')
    norm = {key: value.tolist() for key, value in prepared.norm.items()}
    norm.update(mean=norm['book_mean'], std=norm['book_std'])
    (out / 'norm_stats.json').write_text(json.dumps(norm, indent=2))
    (out / 'training_metadata.json').write_text(json.dumps(metadata, indent=2))
    return model, prepared, labels, metadata


def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--horizon', type=int, default=15)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--leverage', type=float, default=10)
    parser.add_argument('--fee-entry', type=float, default=0.0002)
    parser.add_argument('--fee-exit', type=float, default=0.0002)
    parser.add_argument('--fee-stop', type=float, default=0.0005)
    parser.add_argument('--slippage-bps', type=float, default=1)
    parser.add_argument('--min-validation-trades', type=int, default=20)
    parser.add_argument('--out', required=True, help='NEW directory for the candidate; existing directories are refused')
    args = parser.parse_args()
    torch.set_num_threads(2)
    policy = CandlePolicy(horizon=args.horizon, leverage=args.leverage, fee_entry=args.fee_entry,
                          fee_exit=args.fee_exit, fee_stop=args.fee_stop, slippage_bps=args.slippage_bps)
    df = load_candles(args.data)
    model, prepared, labels, metadata = train_candidate(df, args.out, policy, args.epochs, args.hidden_dim)
    from freqtrade_integration.candle_validation import fit_decision
    decision = fit_decision(model, prepared, labels, df, policy, min_trades=args.min_validation_trades)
    (Path(args.out) / 'decision.json').write_text(json.dumps(decision, indent=2))
    print(json.dumps({'training': metadata, 'decision': decision}, indent=2))
    print('Candidate saved; the active bot/model was not changed.')


if __name__ == '__main__':
    main()
