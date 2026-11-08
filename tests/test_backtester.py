"""Tests unitaires du backtester corrige et du money manager."""

import unittest

import numpy as np
import pandas as pd
import torch

import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtester import run_backtest
from money_management import ScalpingMoneyManager, money_manager_for_asset
from pipeline_data import build_features


class ConstantModel:
    """Faux modele qui predit toujours la meme action (0, 1 ou 2)."""

    def __init__(self, action):
        self.action = action

    def eval(self):
        return self

    def __call__(self, book, seq):
        logits = torch.zeros((book.shape[0], 3), dtype=torch.float32)
        logits[:, self.action] = 10.0
        return logits


def make_df(prices, seed=0):
    n = len(prices)
    rng = np.random.default_rng(seed)
    data = {
        "timestamp": pd.date_range("2026-01-05", periods=n, freq="10ms", tz="UTC"),
        "last_price": np.asarray(prices, dtype=float),
        "last_volume": rng.integers(1, 10, n),
        "best_bid": np.asarray(prices, dtype=float) - 0.5,
        "best_ask": np.asarray(prices, dtype=float) + 0.5,
    }
    for i in range(1, 6):
        data[f"bid_size_{i}"] = rng.integers(10, 100, n).astype(float)
        data[f"ask_size_{i}"] = rng.integers(10, 100, n).astype(float)
    return build_features(pd.DataFrame(data))


class TestBacktester(unittest.TestCase):
    def test_prix_constant_toujours_long_perte_egale_aux_couts(self):
        prices = np.full(300, 98000.0)
        df = make_df(prices)
        report = run_backtest(
            ConstantModel(1),
            df,
            asset="BTC",
            max_holding_ticks=5,
            money_manager=ScalpingMoneyManager(risk_per_trade=0.005, min_qty=0.01),
        )
        self.assertGreater(report["n_trades"], 0)
        self.assertLess(report["final_capital"], report["initial_capital"])
        self.assertGreater(report["total_fees"], 0)
        for tr in report["trades"]:
            self.assertLessEqual(tr.pnl_fill(tr.exit_price), 0.0)

    def test_tendance_haussiere_long_est_rentable_avant_couts(self):
        prices = 98000.0 + np.arange(400) * 2.0
        df = make_df(prices)
        report = run_backtest(
            ConstantModel(1),
            df,
            asset="BTC",
            max_holding_ticks=10,
        )
        self.assertGreater(report["n_trades"], 0)
        gross = sum(tr.pnl_fill(tr.exit_price) for tr in report["trades"])
        self.assertGreater(gross, 0)

    def test_take_profit_declenche(self):
        prices = np.concatenate([np.full(50, 98000.0), np.arange(1, 200) * 5.0 + 98000.0])
        df = make_df(prices)
        report = run_backtest(
            ConstantModel(1),
            df,
            asset="BTC",
            max_holding_ticks=1000,
        )
        reasons = {tr.exit_reason for tr in report["trades"]}
        self.assertIn("take_profit", reasons)

    def test_stop_loss_declenche(self):
        prices = np.concatenate([np.full(50, 98000.0), 98000.0 - np.arange(1, 200) * 5.0])
        df = make_df(prices)
        report = run_backtest(
            ConstantModel(1),
            df,
            asset="BTC",
            max_holding_ticks=1000,
        )
        reasons = {tr.exit_reason for tr in report["trades"]}
        self.assertIn("stop", reasons)

    def test_modele_attente_ne_trade_pas(self):
        prices = 98000.0 + np.cumsum(np.random.default_rng(1).normal(0, 3, 300))
        df = make_df(prices)
        report = run_backtest(ConstantModel(0), df, asset="BTC")
        self.assertEqual(report["n_trades"], 0)
        self.assertAlmostEqual(report["final_capital"], report["initial_capital"], places=6)

    def test_quantite_au_dessus_de_zero_et_ponderee_par_risque(self):
        prices = 98000.0 + np.cumsum(np.random.default_rng(2).normal(0, 3, 300))
        df = make_df(prices)
        mm = ScalpingMoneyManager(risk_per_trade=0.005, point_value=1.0)
        report = run_backtest(ConstantModel(1), df, asset="BTC", money_manager=mm)
        self.assertTrue(all(tr.qty > 0 for tr in report["trades"]))
        capital_ref = report["initial_capital"]
        for tr in report["trades"]:
            stop_dist = abs(tr.entry_price - tr.stop_price)
            risked = stop_dist * tr.qty * mm.point_value
            self.assertLessEqual(risked, capital_ref * 0.005 * 1.01)


class TestMoneyManager(unittest.TestCase):
    def test_valeur_du_point_nq(self):
        mm = money_manager_for_asset("NQ", risk_per_trade=0.005, min_qty=0.0)
        qty = mm.calculate_position_size(10000.0, atr=2.0, atr_multiplier=1.5)
        self.assertAlmostEqual(qty, 10000 * 0.005 / (3.0 * 20.0))

    def test_plafond_max_qty(self):
        mm = ScalpingMoneyManager(risk_per_trade=0.5, point_value=1.0, max_qty=2.0, min_qty=0.0)
        qty = mm.calculate_position_size(1_000_000.0, atr=1.0)
        self.assertEqual(qty, 2.0)

    def test_atr_nul_renvoie_min_qty(self):
        mm = ScalpingMoneyManager(min_qty=0.001)
        self.assertEqual(mm.calculate_position_size(10000.0, atr=0.0), 0.001)


if __name__ == "__main__":
    unittest.main()
