"""Causal preprocessing: fit statistics on purged training samples only."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from freqtrade_integration.candle_features import BOOK_FEATURES, SEQ_FEATURES, _build_seq_windows


def purged_split_indices(n, horizon, warmup=80, train_ratio=0.7, val_ratio=0.15):
    if horizon < 1 or not 0 < train_ratio < train_ratio + val_ratio < 1:
        raise ValueError('Invalid horizon or chronological split ratios')
    train_end, val_end = int(n * train_ratio), int(n * (train_ratio + val_ratio))
    parts = {
        'train': np.arange(warmup, train_end - horizon),
        'validation': np.arange(max(warmup, train_end), val_end - horizon),
        'test': np.arange(max(warmup, val_end), n - horizon),
    }
    if any(len(part) == 0 for part in parts.values()):
        raise ValueError('Not enough candles for purged train/validation/test splits')
    return parts


@dataclass
class PreparedCandles:
    book: np.ndarray
    sequences: np.ndarray
    splits: dict
    norm: dict


def prepare_inputs(features, horizon=15, train_ratio=0.7, val_ratio=0.15, window=20):
    parts = purged_split_indices(len(features), horizon, 60 + window, train_ratio, val_ratio)
    book = features[BOOK_FEATURES].to_numpy(dtype=np.float64)
    seq = features[SEQ_FEATURES].to_numpy(dtype=np.float64)
    finite = np.isfinite(book).all(axis=1) & np.isfinite(seq).all(axis=1)
    complete_window = pd.Series(finite.astype(int)).rolling(window, min_periods=window).sum().eq(window).to_numpy()
    parts = {name: indices[complete_window[indices]] for name, indices in parts.items()}
    if any(len(indices) == 0 for indices in parts.values()):
        raise ValueError('No finite complete sequence in one of the chronological splits')
    train = parts['train']
    norm = {}
    for prefix, values in [('book', book), ('seq', seq)]:
        norm[prefix + '_mean'] = values[train].mean(axis=0)
        norm[prefix + '_std'] = np.maximum(values[train].std(axis=0), 1e-8)
    normalized_book, normalized_seq = normalize_inputs(book, seq, norm)
    return PreparedCandles(normalized_book, _build_seq_windows(normalized_seq, window), parts, norm)


def normalize_inputs(book, seq, norm):
    def transform(values, prefix):
        return np.nan_to_num((values - np.asarray(norm[prefix + '_mean'])) /
                             np.asarray(norm[prefix + '_std']), nan=0, posinf=0, neginf=0).astype(np.float32)
    return transform(book, 'book'), transform(seq, 'seq')


def infer_probabilities(model, df, norm, feature_version=2, window=20, temperature=1.0):
    """Same preprocessing as training, with no positional assumption about df.index."""
    import torch
    from freqtrade_integration.candle_features import build_features
    if temperature <= 0:
        raise ValueError('Temperature must be positive')
    features = build_features(df, version=feature_version)
    book = features[BOOK_FEATURES].to_numpy(float)
    seq = features[SEQ_FEATURES].to_numpy(float)
    finite = np.isfinite(book).all(axis=1) & np.isfinite(seq).all(axis=1)
    complete = pd.Series(finite.astype(int)).rolling(window, min_periods=window).sum().eq(window).to_numpy()
    indices = np.flatnonzero(complete)
    if feature_version == 1:
        book = np.nan_to_num((book - np.asarray(norm['book_mean'])) /
                             (np.asarray(norm['book_std']) + 1e-8)).astype(np.float32)
        seq = np.nan_to_num(seq).astype(np.float32)
    else:
        book, seq = normalize_inputs(book, seq, norm)
    windows = _build_seq_windows(seq, window)
    probabilities = np.zeros((len(df), 3), dtype=np.float32)
    probabilities[:, 0] = 1.
    model.eval()
    with torch.no_grad():
        for start in range(0, len(indices), 512):
            ix = indices[start:start + 512]
            # Fixed batch shape prevents GEMM/LSTM roundoff from depending on
            # how many future rows happen to be present in a backtest prefix.
            padded_book = np.zeros((512, book.shape[1]), dtype=np.float32)
            padded_seq = np.zeros((512, window, seq.shape[1]), dtype=np.float32)
            padded_book[:len(ix)] = book[ix]
            padded_seq[:len(ix)] = windows[ix]
            logits = model(torch.tensor(padded_book), torch.tensor(padded_seq))
            probabilities[ix] = torch.softmax(logits / temperature, dim=1).cpu().numpy()[:len(ix)]
    return probabilities
