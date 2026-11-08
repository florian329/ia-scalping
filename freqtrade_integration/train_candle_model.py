"""Entrainement du modele candle pour la strategie FreqTrade.

FreqTrade fournit des candles OHLCV (pas de carnet L2) : ce script entraine le
meme reseau (ScalpingMultimodalNet) sur des features candle, avec un labelling
triple-barriere simple rendant compte des frais :

- classe 1 (long)  : rendement futur sur `horizon` candles > seuil (+frais) ;
- classe 2 (short) : rendement futur < -seuil ;
- classe 0         : attente.

Usage (apres `freqtrade download-data`) :
    python freqtrade_integration/train_candle_model.py \
        --data user_data/data/binance/BTC_USDT-1m.feather \
        --epochs 10 --out freqtrade_integration/model
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from candle_features import BOOK_FEATURES, SEQ_FEATURES, build_features as _bf
from model_pytorch import ScalpingMultimodalNet
from training_engine import train_model

FEE_RATE = 0.001

def build_seq_windows(seq: np.ndarray, window: int = 20) -> np.ndarray:
    """Fenetres glissantes de taille fixe ; le debut est complete par repetition."""
    n = len(seq)
    if n == 0:
        return np.zeros((0, window, seq.shape[1] if seq.ndim > 1 else 1), dtype=np.float32)
    padded = np.concatenate([np.repeat(seq[:1], window - 1, axis=0), seq], axis=0)
    return np.stack([padded[i : i + window] for i in range(n)]).astype(np.float32)



def load_candles(path: str) -> pd.DataFrame:
    suffix = Path(path).suffix
    if suffix == ".feather":
        df = pd.read_feather(path)
    elif suffix == ".json":
        df = pd.read_json(path, orient="values")
        df.columns = ["date", "open", "high", "low", "close", "volume"]
        df["date"] = pd.to_datetime(df["date"], utc=True)
    else:
        df = pd.read_csv(path)
    return df


def build_labels(df: pd.DataFrame, horizon: int, fee_rate: float) -> np.ndarray:
    close = df["close"].astype(float).to_numpy()
    future = close[horizon:] / close[:-horizon] - 1.0
    future = np.concatenate([future, np.zeros(horizon)])
    threshold = 3 * fee_rate
    labels = np.zeros(len(df), dtype=np.int64)
    labels[future > threshold] = 1
    labels[future < -threshold] = 2
    return labels


def main():
    parser = argparse.ArgumentParser(description="Entrainement du modele candle (FreqTrade)")
    parser.add_argument("--data", required=True, help="Candles (feather/csv/json) BTC/USDT 1m")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=15, help="Horizon de labellisation en candles")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--out", default=str(Path(__file__).parent / "model"))
    args = parser.parse_args()

    df = load_candles(args.data)
    feats = _bf(df)
    labels = build_labels(df, args.horizon, FEE_RATE)
    feats = feats.iloc[60:].reset_index(drop=True)
    labels = labels[60:]

    valid = ~feats[BOOK_FEATURES + SEQ_FEATURES].isna().any(axis=1)
    feats = feats[valid].reset_index(drop=True)
    labels = labels[valid]

    book = feats[BOOK_FEATURES].to_numpy(dtype=np.float32)
    seq = feats[SEQ_FEATURES].to_numpy(dtype=np.float32)
    mean, std = book.mean(axis=0), book.std(axis=0) + 1e-8
    book = (book - mean) / std

    seq_windows = build_seq_windows(seq)

    n = len(labels)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)
    from torch.utils.data import DataLoader, TensorDataset

    train_ds = TensorDataset(
        torch.tensor(book[:n_train]), torch.tensor(seq_windows[:n_train]), torch.tensor(labels[:n_train])
    )
    val_ds = TensorDataset(
        torch.tensor(book[n_train : n_train + n_val]),
        torch.tensor(seq_windows[n_train : n_train + n_val]),
        torch.tensor(labels[n_train : n_train + n_val]),
    )
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    unique, counts = np.unique(labels[:n_train], return_counts=True)
    dist = dict(zip(unique.tolist(), counts.tolist()))
    print(f"Distribution des classes (train) : {dist}")

    model = ScalpingMultimodalNet(
        order_book_dim=len(BOOK_FEATURES), sequence_dim=len(SEQ_FEATURES), hidden_dim=args.hidden_dim
    )
    import torch as _torch
    weights = _torch.tensor([0.1, 2.0, 2.0])
    train_model(model, train_loader, val_loader, epochs=args.epochs, lr=0.001, patience=3, class_weights=weights)

    os.makedirs(args.out, exist_ok=True)
    model_path = Path(args.out) / "candle_model.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "book_dim": len(BOOK_FEATURES),
            "seq_dim": len(SEQ_FEATURES),
            "hidden_dim": args.hidden_dim,
        },
        model_path,
    )
    with open(Path(args.out) / "norm_stats.json", "w") as f:
        json.dump({"mean": mean.tolist(), "std": std.tolist()}, f)
    print(f"Modele sauvegarde : {model_path}")


if __name__ == "__main__":
    main()
