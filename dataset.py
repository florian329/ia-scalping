"""Construction des tenseurs d'entree et split CHRONOLOGIQUE train/val/test.

Le split est strictement temporel (jamais aleatoire) conformement a la spec :
30% des donnees les plus recentes sont reservees a la validation finale.
"""

import numpy as np
import torch
from torch.utils.data import TensorDataset

BOOK_FEATURES = ["OBI", "Spread_BPS", "VWAP_Dev", "ATR_Tick"]
SEQ_FEATURES_SHORT = ["Log_Return_10", "VW_Momentum"]
SEQ_FEATURES = ["Log_Return_10", "Log_Return_30", "Log_Return_60", "VW_Momentum"]


def build_window_dataset(df, sequence_length=20, seq_features=None, with_weights=False):
    """Genere les fenetres glissantes pour le modele a deux branches.

    Anti look-ahead : la fenetre se terminant au tick t-1 sert a predire
    l'action executee au tick t. Le label du tick t-1 est la cible.
    with_weights=True ajoute un tenseur de poids par echantillon (colonne
    'sample_weight' produite par labeling.label_ticks avec pnl_weights=True).
    """
    seq_features = seq_features or SEQ_FEATURES
    book = df[BOOK_FEATURES].to_numpy(dtype=np.float32)
    seq = df[seq_features].to_numpy(dtype=np.float32)
    labels = df["label"].to_numpy(dtype=np.int64)
    weights = (
        df["sample_weight"].to_numpy(dtype=np.float32)
        if "sample_weight" in df.columns
        else np.ones(len(df), dtype=np.float32)
    )

    mean = book.mean(axis=0)
    std = book.std(axis=0) + 1e-8
    book = (book - mean) / std

    X_book, X_seq, y, w = [], [], [], []
    for t in range(sequence_length, len(df)):
        X_book.append(book[t - 1])
        X_seq.append(seq[t - sequence_length : t])
        y.append(labels[t - 1])
        w.append(weights[t - 1])
    if with_weights:
        return (
            torch.tensor(np.stack(X_book), dtype=torch.float32),
            torch.tensor(np.stack(X_seq), dtype=torch.float32),
            torch.tensor(np.array(y), dtype=torch.long),
            (mean, std),
            torch.tensor(np.array(w), dtype=torch.float32),
        )
    return (
        torch.tensor(np.stack(X_book), dtype=torch.float32),
        torch.tensor(np.stack(X_seq), dtype=torch.float32),
        torch.tensor(np.array(y), dtype=torch.long),
        (mean, std),
    )


def chronological_split(dataset, train_ratio=0.7, val_ratio=0.15):
    """Split temporel : train | val | test, dans cet ordre chronologique."""
    n = len(dataset.tensors[0])
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = TensorDataset(*[t[:n_train] for t in dataset.tensors])
    val = TensorDataset(*[t[n_train : n_train + n_val] for t in dataset.tensors])
    test = TensorDataset(*[t[n_train + n_val :] for t in dataset.tensors])
    return train, val, test


def make_loaders(train, val, test, batch_size=64):
    from torch.utils.data import DataLoader

    return (
        DataLoader(train, batch_size=batch_size, shuffle=False),
        DataLoader(val, batch_size=batch_size, shuffle=False),
        DataLoader(test, batch_size=batch_size, shuffle=False),
    )
