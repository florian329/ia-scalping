"""Golden test de non-regression numerique.

Scenario deterministe (seed fixe, modele constant) dont les metriques sont
gelees. Toute modification du comportement du backtester, des couts ou de la
logique d'execution qui change ces nombres doit echouer ici — et le changement
doit etre justifie et le golden mis a jour explicitement.
"""

import os
import sys
import unittest

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtester import run_backtest
from pipeline_data import build_features

GOLDEN = {
    "n_trades": 14,
    "final_capital": 9842.3493,
    "total_fees": 166.6770,
    "win_rate": 0.0,
    "maker_fills": 14,
    "maker_cancels": 1,
    "maker_replacements": 3,
}


class AlwaysLong:
    def eval(self):
        return self

    def __call__(self, book, seq):
        logits = torch.zeros((book.shape[0], 3), dtype=torch.float32)
        logits[:, 1] = 10.0
        return logits


def golden_df():
    rng = np.random.default_rng(42)
    n = 500
    prices = 76000.0 + np.cumsum(rng.normal(0, 2.0, n))
    prices[::7] -= 1.0
    data = {
        "timestamp": pd.date_range("2026-01-05", periods=n, freq="10ms", tz="UTC"),
        "last_price": prices,
        "last_volume": rng.integers(1, 10, n).astype(float),
        "best_bid": prices - 0.5,
        "best_ask": prices + 0.5,
    }
    for i in range(1, 6):
        data[f"bid_size_{i}"] = rng.integers(10, 100, n).astype(float)
        data[f"ask_size_{i}"] = rng.integers(10, 100, n).astype(float)
    return build_features(pd.DataFrame(data))


class TestGoldenBacktest(unittest.TestCase):
    def test_metriques_golden_inchangees(self):
        report = run_backtest(
            AlwaysLong(),
            golden_df(),
            asset="BTC",
            initial_capital=10000.0,
            execution="maker",
            max_holding_ticks=100,
            sequence_length=20,
        )
        self.assertEqual(report["n_trades"], GOLDEN["n_trades"])
        self.assertAlmostEqual(report["final_capital"], GOLDEN["final_capital"], places=3)
        self.assertAlmostEqual(report["total_fees"], GOLDEN["total_fees"], places=3)
        self.assertAlmostEqual(report["win_rate"], GOLDEN["win_rate"], places=4)
        self.assertEqual(report["maker_fills"], GOLDEN["maker_fills"])
        self.assertEqual(report["maker_cancels"], GOLDEN["maker_cancels"])
        self.assertEqual(report["maker_replacements"], GOLDEN["maker_replacements"])

    def test_modele_attente_aucun_trade(self):
        report = run_backtest(
            _Constant(0),
            golden_df(),
            asset="BTC",
            execution="maker",
            max_holding_ticks=100,
        )
        self.assertEqual(report["n_trades"], 0)
        self.assertAlmostEqual(report["final_capital"], 10000.0, places=6)


class _Constant(AlwaysLong):
    def __init__(self, action):
        self.action = action

    def __call__(self, book, seq):
        logits = torch.zeros((book.shape[0], 3), dtype=torch.float32)
        logits[:, self.action] = 10.0
        return logits


if __name__ == "__main__":
    unittest.main()
