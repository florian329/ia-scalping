"""Tests de l'integration FreqTrade (sautes si freqtrade n'est pas installe)."""

import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "freqtrade_integration"))

try:
    import freqtrade  # noqa: F401

    FREQTRADE_AVAILABLE = True
except ImportError:
    FREQTRADE_AVAILABLE = False

from freqtrade_integration.train_candle_model import build_features, build_labels


def make_candles(n=400, seed=0):
    rng = np.random.default_rng(seed)
    close = 76000.0 + np.cumsum(rng.normal(0, 20.0, n))
    df = pd.DataFrame(
        {
            "date": pd.date_range("2026-11-01", periods=n, freq="1min", tz="UTC"),
            "open": close - rng.uniform(0, 10, n),
            "high": close + rng.uniform(0, 15, n),
            "low": close - rng.uniform(0, 15, n),
            "close": close,
            "volume": rng.uniform(1, 100, n),
        }
    )
    return df


class TestCandleFeatures(unittest.TestCase):
    def test_features_generees_sans_nan_apres_chauffe(self):
        feats = build_features(make_candles())
        from freqtrade_integration.IaScalpingStrategy import BOOK_FEATURES, SEQ_FEATURES

        hot = feats.iloc[60:]
        self.assertFalse(hot[BOOK_FEATURES + SEQ_FEATURES].isna().any().any())

    def test_labels_triple_barriere(self):
        df = make_candles()
        labels = build_labels(df, horizon=15, fee_rate=0.001)
        self.assertTrue(set(np.unique(labels)).issubset({0, 1, 2}))
        haussier = make_candles()
        haussier["close"] = 76000.0 + np.arange(len(haussier)) * 50.0
        labels_haussier = build_labels(haussier, horizon=15, fee_rate=0.001)
        self.assertGreater((labels_haussier == 1).sum(), (labels_haussier == 2).sum())

    def test_fenetres_seq_taille_fixe(self):
        from freqtrade_integration.IaScalpingStrategy import _build_seq_windows

        seq = np.random.default_rng(0).normal(size=(50, 5)).astype(np.float32)
        windows = _build_seq_windows(seq)
        self.assertEqual(windows.shape, (50, 20, 5))
        self.assertEqual(_build_seq_windows(np.zeros((0, 5))).shape, (0, 20, 5))


@unittest.skipUnless(FREQTRADE_AVAILABLE, "freqtrade n'est pas installe")
class TestStrategy(unittest.TestCase):
    def _strategy(self):
        from IaScalpingStrategy import IaScalpingStrategy

        config = {
            "strategy": "IaScalpingStrategy",
            "dry_run": True,
            "stake_currency": "USDT",
            "stake_amount": 100,
            "dry_run_wallet": 1000,
            "exchange": {"name": "binance", "pair_whitelist": ["BTC/USDT:USDT"]},
        }
        return IaScalpingStrategy(config)

    def test_populate_indicators_et_trends(self):
        strat = self._strategy()
        df = make_candles(300)
        out = strat.populate_indicators(df, {"pair": "BTC/USDT:USDT"})
        for col in ("model_action", "model_prob_long", "model_prob_short"):
            self.assertIn(col, out.columns)
        out = strat.populate_entry_trend(out, {"pair": "BTC/USDT:USDT"})
        out = strat.populate_exit_trend(out, {"pair": "BTC/USDT:USDT"})
        self.assertIn("enter_long", out.columns)
        self.assertIn("enter_short", out.columns)
        self.assertIn("exit_long", out.columns)
        self.assertIn("exit_short", out.columns)

    def test_sans_modele_aucun_signal(self):
        from IaScalpingStrategy import IaScalpingStrategy

        config = {
            "strategy": "IaScalpingStrategy",
            "dry_run": True,
            "stake_currency": "USDT",
            "stake_amount": 100,
            "dry_run_wallet": 1000,
            "exchange": {"name": "binance", "pair_whitelist": ["BTC/USDT:USDT"]},
        }
        original_exists = IaScalpingStrategy.__module__
        import IaScalpingStrategy as mod

        saved = mod.MODEL_PATH
        mod.MODEL_PATH = mod.MODEL_PATH.parent / "inexistant.pth"
        try:
            strat = IaScalpingStrategy(config)
            self.assertIsNone(strat._model)
            df = make_candles(200)
            out = strat.populate_indicators(df, {"pair": "BTC/USDT:USDT"})
            self.assertEqual(int(out["model_action"].max()), 0)
        finally:
            mod.MODEL_PATH = saved


if __name__ == "__main__":
    unittest.main()
