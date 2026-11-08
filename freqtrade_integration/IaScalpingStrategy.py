"""Strategie FreqTrade branchant le modele IA scalping (pipeline ia-scalping).

Integration du pipeline ia-scalping dans le moteur FreqTrade :
- les features sont recalculees au niveau CANDLE (FreqTrade ne fournit pas le
  carnet d'ordres L2 — les features order-flow du pipeline tick (OBI, VPIN)
  sont remplacees par leurs equivalents candle, documentes ci-dessous) ;
- le modele PyTorch (ScalpingMultimodalNet) est charge depuis
  freqtrade_integration/model/ et infere en batch sur tout le dataframe ;
- les entrees/sorties utilisent des ordres LIMIT (comportement maker, coherent
  avec le mode execution="maker" du pipeline) ;
- short active (can_short=True) — le modele a trois classes :
  0 = attente, 1 = long, 2 = short.

Correspondance features tick -> candle :
  OBI, OFI, VPIN  ->  proxies de volume (vol_z, vol_change, vol_mom)
  Log_Return_N    ->  Log_Return_N (identique, fenetre en candles)
  VWAP_Dev        ->  VWAP_Dev (identique)
  ATR_Tick        ->  ATR candle (fenetre 14)
  Hour_Sin/Cos    ->  Hour_Sin/Cos (identique)
"""

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
from pandas import DataFrame

from candle_features import BOOK_FEATURES, SEQ_FEATURES, _build_seq_windows
from freqtrade.strategy import IStrategy

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODEL_DIR = Path(__file__).parent / "model"
MODEL_PATH = MODEL_DIR / "candle_model.pth"
NORM_STATS_PATH = MODEL_DIR / "norm_stats.json"




class IaScalpingStrategy(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "1m"
    can_short = True
    process_only_new_candles = True
    startup_candle_count = 60

    minimal_roi = {"0": 0.02, "30": 0.01, "60": 0.005}
    stoploss = -0.02
    trailing_stop = False

    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }

    prob_threshold = 0.4

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._model = None
        self._norm_stats = None
        self._load_model()

    def _load_model(self) -> None:
        if not MODEL_PATH.exists():
            logger.warning(
                "Modele introuvable (%s) : la strategie n'emettra aucun signal. "
                "Lancez freqtrade_integration/train_candle_model.py d'abord.",
                MODEL_PATH,
            )
            return
        from model_pytorch import ScalpingMultimodalNet

        ckpt = torch.load(MODEL_PATH, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt)
        book_dim = int(ckpt.get("book_dim", len(BOOK_FEATURES)))
        seq_dim = int(ckpt.get("seq_dim", len(SEQ_FEATURES)))
        hidden = int(ckpt.get("hidden_dim", 64))
        model = ScalpingMultimodalNet(order_book_dim=book_dim, sequence_dim=seq_dim, hidden_dim=hidden)
        model.load_state_dict(state)
        model.eval()
        self._model = model
        if NORM_STATS_PATH.exists():
            with open(NORM_STATS_PATH) as f:
                stats = json.load(f)
            self._norm_stats = (np.array(stats["mean"]), np.array(stats["std"]))
        logger.info("Modele candle charge (book_dim=%d, seq_dim=%d)", book_dim, seq_dim)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe.copy()
        close = df["close"].astype(float)
        volume = df["volume"].astype(float)

        df["ret_1"] = np.log(close / close.shift(1))
        for w in (10, 30, 60):
            df[f"log_ret_{w}"] = np.log(close / close.shift(w))
        typical = (df["high"] + df["low"] + df["close"]) / 3.0
        vwap = (typical * volume).cumsum() / (volume.cumsum() + 1e-12)
        df["vwap_dev"] = (close - vwap) / (vwap + 1e-12)
        tr = np.maximum(
            df["high"] - df["low"],
            np.maximum(abs(df["high"] - close.shift(1)), abs(df["low"] - close.shift(1))),
        )
        df["atr_norm"] = tr.rolling(14).mean() / (close + 1e-12)
        df["vol_z"] = (volume - volume.rolling(60).mean()) / (volume.rolling(60).std() + 1e-12)
        df["vol_change"] = volume.pct_change()
        df["vol_mom"] = (volume * np.sign(df["ret_1"].fillna(0))).rolling(10).sum()
        df["range_ratio"] = (close - df["low"]) / (df["high"] - df["low"] + 1e-12)
        hour = df["date"].dt.hour + df["date"].dt.minute / 60.0
        angle = 2 * np.pi * hour / 24.0
        df["hour_sin"] = np.sin(angle)
        df["hour_cos"] = np.cos(angle)

        if self._model is None:
            df["model_action"] = 0
            df["model_prob_long"] = 0.0
            df["model_prob_short"] = 0.0
            return df

        valid = df.dropna(subset=SEQ_FEATURES + BOOK_FEATURES).index
        book = df.loc[valid, BOOK_FEATURES].to_numpy(dtype=np.float32)
        if self._norm_stats is not None:
            mean, std = self._norm_stats
            book = (book - mean) / (std + 1e-8)
        seq = df.loc[valid, SEQ_FEATURES].to_numpy(dtype=np.float32)

        actions = np.zeros(len(df), dtype=np.int64)
        probs = np.zeros((len(df), 3), dtype=np.float32)
        if len(valid) > 0:
            with torch.no_grad():
                book_t = torch.tensor(book, dtype=torch.float32)
                seq_windows = _build_seq_windows(seq)
                seq_t = torch.tensor(seq_windows)
                logits = self._model(book_t, seq_t)
                chunk_probs = torch.softmax(logits, dim=1).numpy()
                actions[valid] = chunk_probs.argmax(axis=1)
                probs[valid] = chunk_probs
        df["model_action"] = actions
        df["model_prob_long"] = probs[:, 1]
        df["model_prob_short"] = probs[:, 2]
        return df

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["model_action"] == 1)
            & (dataframe["model_prob_long"] > self.prob_threshold)
            & (dataframe["volume"] > 0),
            "enter_long",
        ] = 1
        dataframe.loc[
            (dataframe["model_action"] == 2)
            & (dataframe["model_prob_short"] > self.prob_threshold)
            & (dataframe["volume"] > 0),
            "enter_short",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["model_action"] != 1) & (dataframe["volume"] > 0),
            "exit_long",
        ] = 1
        dataframe.loc[
            (dataframe["model_action"] != 2) & (dataframe["volume"] > 0),
            "exit_short",
        ] = 1
        return dataframe
