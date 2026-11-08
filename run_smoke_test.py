"""Smoke test de bout en bout : donnees -> features -> labels -> entrainement -> backtest."""

import numpy as np
import pandas as pd
import torch

from data_generator import generate_sample_data
from dataset import (
    BOOK_FEATURES,
    SEQ_FEATURES,
    build_window_dataset,
    chronological_split,
    make_loaders,
)
from evaluation_metrics import evaluate_and_plot, per_class_accuracy
from labeling import compute_class_weights, label_ticks, load_annotations
from model_pytorch import ScalpingMultimodalNet
from pipeline_data import build_features
from training_engine import train_model


def main():
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    df, ann = generate_sample_data()
    print(f"Ticks generes : {len(df)} | Annotations : {len(ann)}")

    df = build_features(df)
    df = label_ticks(df, ann, asset="BTC", window_ticks=5)
    print(f"Distribution des classes : {df['label'].value_counts().to_dict()}")

    x_book, x_seq, y, norm_stats = build_window_dataset(df)
    n = len(y)
    train_ds, val_ds, test_ds = chronological_split(
        __import__("torch.utils.data", fromlist=["TensorDataset"]).TensorDataset(x_book, x_seq, y)
    )
    print(f"Split chronologique : train={len(train_ds)} | val={len(val_ds)} | test={len(test_ds)}")

    train_loader, val_loader, test_loader = make_loaders(train_ds, val_ds, test_ds, batch_size=128)
    weights = compute_class_weights(y[: len(train_ds)].numpy())
    print(f"Poids de classes calcules : {np.round(weights, 3).tolist()}")

    model = ScalpingMultimodalNet(
        order_book_dim=x_book.shape[1], sequence_dim=x_seq.shape[2], hidden_dim=64
    )
    train_model(model, train_loader, val_loader, epochs=10, class_weights=weights, patience=3)

    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for b_book, b_seq, b_targets in test_loader:
            logits = model(b_book, b_seq)
            y_pred.extend(logits.argmax(dim=1).tolist())
            y_true.extend(b_targets.tolist())
    evaluate_and_plot(y_true, y_pred, save_path="confusion_matrix.png")
    print(f"Accuracy par classe : {per_class_accuracy(y_true, y_pred)}")

    from backtester import run_backtest, print_report

    n_train_val = len(train_ds) + len(val_ds)
    test_start = n_train_val
    df_test = df.iloc[test_start:].reset_index(drop=True)
    report = run_backtest(
        model,
        df_test,
        asset="BTC",
        norm_stats=norm_stats,
        confidence_threshold=0.0,
    )
    print_report(report)
    reasons = pd.Series([t.exit_reason for t in report["trades"]]).value_counts().to_dict()
    print(f"Raisons de sortie : {reasons}")
    print("Smoke test termine avec succes.")


if __name__ == "__main__":
    main()
