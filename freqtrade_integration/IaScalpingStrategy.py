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

from freqtrade.strategy import DecimalParameter, IStrategy

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from freqtrade_integration.candle_features import BOOK_FEATURES, SEQ_FEATURES
from freqtrade_integration.candle_training import infer_probabilities
from freqtrade_integration.candle_policy import CandlePolicy

MODEL_DIR = Path(__file__).parent / "model"
MODEL_PATH = MODEL_DIR / "candle_model.pth"
NORM_STATS_PATH = MODEL_DIR / "norm_stats.json"




class IaScalpingStrategy(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "1m"
    can_short = True
    process_only_new_candles = True
    startup_candle_count = 80

    minimal_roi = {"0": 0.02, "30": 0.01, "60": 0.005}
    stoploss = -0.02
    trailing_stop = False

    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }

    # Levier futures : 1.0 par defaut (prudent). A augmenter UNIQUEMENT apres
    # un verdict statistique positif (hyperopt_holdout.py / run_validation.py).
    # FreqTrade interprete le stoploss comme risque sur marge : a x10,
    # -2 % de stop correspond a environ -0,2 % de mouvement du sous-jacent, hors frais.
    leverage_value: float = 1.0

    def leverage(self, pair: str, current_time, current_rate: float,
                 proposed_leverage: float, max_leverage: float,
                 entry_tag, side: str, **kwargs) -> float:
        """Levier demande, plafonne par la limite reelle de l'exchange."""
        return min(self.leverage_value, max_leverage)

    entry_threshold = DecimalParameter(
        0.30, 0.70, default=0.40, space="entry", optimize=True
    )

    @property
    def protections(self):
        if self._policy is not None:
            return self._policy.protections()
        return [
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": 5,
            },
            {
                "method": "StoplossGuard",
                "lookback_period_candles": 60,
                "trade_limit": 4,
                "stop_duration_candles": 30,
                "only_per_pair": False,
            },
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": 240,
                "trade_limit": 5,
                "max_allowed_drawdown": 0.05,
                "stop_duration_candles": 60,
            },
        ]

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._model = None
        self._norm_stats = None
        self._norm = None
        self._feature_version = 1
        self._policy = None
        self._temperature = 1.0
        self._thresholds = None
        self._artifact_dir = config.get('ia_model_dir') or os.environ.get('IA_SCALPING_MODEL_DIR')
        self._load_model()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_model"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._load_model()

    def _load_model(self) -> None:
        model_path = Path(self._artifact_dir) / 'candle_model.pth' if self._artifact_dir else MODEL_PATH
        norm_path = model_path.parent / 'norm_stats.json' if self._artifact_dir else NORM_STATS_PATH
        if not model_path.exists():
            logger.warning(
                "Modele introuvable (%s) : la strategie n'emettra aucun signal. "
                "Lancez freqtrade_integration/train_candle_model.py d'abord.",
                model_path,
            )
            return
        from model_pytorch import ScalpingMultimodalNet

        ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
        state = ckpt.get("model_state_dict", ckpt)
        book_dim = int(ckpt.get("book_dim", len(BOOK_FEATURES)))
        seq_dim = int(ckpt.get("seq_dim", len(SEQ_FEATURES)))
        hidden = int(ckpt.get("hidden_dim", 64))
        model = ScalpingMultimodalNet(order_book_dim=book_dim, sequence_dim=seq_dim, hidden_dim=hidden)
        model.load_state_dict(state)
        model.eval()
        self._model = model
        self._feature_version = int(ckpt.get('feature_version', 1))
        if norm_path.exists():
            stats = json.loads(norm_path.read_text())
            self._norm_stats = (np.asarray(stats['mean']), np.asarray(stats['std']))
            self._norm = {'book_mean': stats['mean'], 'book_std': stats['std']}
            if self._feature_version == 2:
                self._norm.update(seq_mean=stats['seq_mean'], seq_std=stats['seq_std'])
        if self._norm is None:
            raise ValueError('Missing training normalization; refusing inference')
        if self._feature_version == 2:
            self._policy = CandlePolicy.from_dict(ckpt['execution_policy'])
            self.stoploss = self._policy.stoploss
            self.minimal_roi = {str(t): roi for t, roi in self._policy.roi_steps}
            self.leverage_value = self._policy.leverage
            decision_path = model_path.parent / 'decision.json'
            if not decision_path.exists():
                raise ValueError('Candidate has no validation-only calibration and thresholds')
            decision = json.loads(decision_path.read_text())
            self._temperature = float(decision['temperature'])
            self._thresholds = {side: float(decision['thresholds'][side]) for side in ('long', 'short')}
            if self._temperature <= 0 or any(not 0 <= v <= 1 for v in self._thresholds.values()):
                raise ValueError('Invalid calibration or directional thresholds')
        logger.info("Modele candle charge (book_dim=%d, seq_dim=%d)", book_dim, seq_dim)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe.copy()
        if self._model is None:
            df['model_action'] = 0
            df['model_prob_long'] = 0.0
            df['model_prob_short'] = 0.0
            return df
        probs = infer_probabilities(self._model, df, self._norm,
                                    feature_version=self._feature_version,
                                    temperature=self._temperature)
        df['model_action'] = probs.argmax(axis=1)
        df['model_prob_long'] = probs[:, 1]
        df['model_prob_short'] = probs[:, 2]
        return df

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        long_threshold = self._thresholds['long'] if self._thresholds else self.entry_threshold.value
        short_threshold = self._thresholds['short'] if self._thresholds else self.entry_threshold.value
        dataframe.loc[
            (dataframe["model_action"] == 1)
            & (dataframe["model_prob_long"] > long_threshold)
            & (dataframe["volume"] > 0),
            "enter_long",
        ] = 1
        dataframe.loc[
            (dataframe["model_action"] == 2)
            & (dataframe["model_prob_short"] > short_threshold)
            & (dataframe["volume"] > 0),
            "enter_short",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if self._policy is not None:
            dataframe['exit_long'] = 0
            dataframe['exit_short'] = 0
            return dataframe
        dataframe.loc[
            (dataframe["model_action"] != 1) & (dataframe["volume"] > 0),
            "exit_long",
        ] = 1
        dataframe.loc[
            (dataframe["model_action"] != 2) & (dataframe["volume"] > 0),
            "exit_short",
        ] = 1
        return dataframe

    def custom_exit(self, pair, trade, current_time, current_rate, current_profit, **kwargs):
        if self._policy is not None:
            minutes = (current_time - trade.open_date_utc).total_seconds() / 60
            if minutes >= self._policy.horizon:
                return 'policy_timeout'
        return None
