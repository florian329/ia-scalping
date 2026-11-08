"""Features candle partagees entre la strategie FreqTrade et l'entrainement.

Module volontairement independant de freqtrade : utilisable pour l'entrainement
sans installer le moteur.
"""

import numpy as np
import pandas as pd

def _build_seq_windows(seq: np.ndarray, window: int = 20) -> np.ndarray:
    n = len(seq)
    if n == 0:
        return np.zeros((0, window, seq.shape[1]), dtype=np.float32)
    padded = np.concatenate([np.repeat(seq[:1], window - 1, axis=0), seq], axis=0)
    return np.stack([padded[i : i + window] for i in range(n)]).astype(np.float32)


BOOK_FEATURES = ["vol_z", "vwap_dev", "atr_norm", "hour_sin", "hour_cos", "vol_change", "range_ratio"]
SEQ_FEATURES = ["log_ret_10", "log_ret_30", "log_ret_60", "vol_mom", "ret_1"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features candle : proxies des features tick du pipeline ia-scalping."""
    close = df["close"].astype(float)
    volume = df["volume"].astype(float)
    out = pd.DataFrame(index=df.index)
    out["date"] = pd.to_datetime(df["date"], utc=True)
    out["ret_1"] = np.log(close / close.shift(1))
    for w in (10, 30, 60):
        out[f"log_ret_{w}"] = np.log(close / close.shift(w))
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    vwap = (typical * volume).cumsum() / (volume.cumsum() + 1e-12)
    out["vwap_dev"] = (close - vwap) / (vwap + 1e-12)
    tr = np.maximum(
        df["high"] - df["low"],
        np.maximum(abs(df["high"] - close.shift(1)), abs(df["low"] - close.shift(1))),
    )
    out["atr_norm"] = tr.rolling(14).mean() / (close + 1e-12)
    out["vol_z"] = (volume - volume.rolling(60).mean()) / (volume.rolling(60).std() + 1e-12)
    out["vol_change"] = volume.pct_change()
    out["vol_mom"] = (volume * np.sign(out["ret_1"].fillna(0))).rolling(10).sum()
    out["range_ratio"] = (close - df["low"]) / (df["high"] - df["low"] + 1e-12)
    hour = out["date"].dt.hour + out["date"].dt.minute / 60.0
    angle = 2 * np.pi * hour / 24.0
    out["hour_sin"] = np.sin(angle)
    out["hour_cos"] = np.cos(angle)
    return out
