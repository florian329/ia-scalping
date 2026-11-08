"""Generateur de donnees synthetiques de test (ticks + carnet 5 niveaux + annotations)."""

import numpy as np
import pandas as pd


def generate_sample_data(
    filename="market_data_sample.csv",
    annotations_filename="annotations_sample.csv",
    n_ticks=5000,
    seed=42,
    asset="BTC",
):
    rng = np.random.default_rng(seed)
    start_price = 98000.0
    drift = np.concatenate(
        [np.zeros(n_ticks // 4), np.full(n_ticks // 4, 0.8), np.zeros(n_ticks // 4), np.full(n_ticks - 3 * (n_ticks // 4), -0.8)]
    )
    noise = rng.normal(0, 3.0, n_ticks)
    prices = start_price + np.cumsum(drift + noise)

    half_spread = 0.5
    data = {
        "timestamp": pd.date_range("2026-01-05 14:00:00", periods=n_ticks, freq="10ms", tz="UTC"),
        "last_price": prices,
        "last_volume": rng.integers(1, 10, n_ticks),
        "best_bid": prices - half_spread,
        "best_ask": prices + half_spread,
    }
    for i in range(1, 6):
        base = rng.integers(10, 100, n_ticks).astype(float)
        data[f"bid_size_{i}"] = base * (1.0 + 0.05 * i)
        data[f"ask_size_{i}"] = rng.integers(10, 100, n_ticks).astype(float) * (1.0 + 0.05 * i)

    regime = np.sign(drift)
    for i in range(1, 6):
        data[f"bid_size_{i}"] = np.where(regime > 0, data[f"bid_size_{i}"] * 2.0, data[f"bid_size_{i}"])
        data[f"ask_size_{i}"] = np.where(regime < 0, data[f"ask_size_{i}"] * 2.0, data[f"ask_size_{i}"])

    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)

    ann_idx = np.arange(50, n_ticks - 100, 100)
    annotations = []
    for k, idx in enumerate(ann_idx):
        long_trade = regime[idx + 5] >= 0
        annotations.append(
            {
                "timestamp": df["timestamp"].iloc[idx + 1],
                "asset": asset,
                "side": "BUY" if long_trade else "SELL",
                "entry_price": float(df["last_price"].iloc[idx + 1]),
                "exit_price": float(df["last_price"].iloc[idx + 30]),
                "volume": 1.0,
            }
        )
    ann_df = pd.DataFrame(annotations)
    ann_df.to_csv(annotations_filename, index=False)
    print(f"Fichiers de test generes : {filename}, {annotations_filename}")
    return df, ann_df


if __name__ == "__main__":
    generate_sample_data()
