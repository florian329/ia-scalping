"""Tests phase 4 : walk-forward, bootstrap, protections, labels ponderes, multi-segments."""

import os
import sys
import unittest

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtester import run_backtest
from labeling import label_ticks
from money_management import ScalpingMoneyManager
from pipeline_data import build_features
from rl_environment import MultiSegmentEnv, ScalpingEnv
from validation import (
    bootstrap_metrics,
    classify_regimes,
    metrics_by_regime,
    run_cost_sensitivity,
    walk_forward_backtest,
)


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


class ConstantModel:
    def __init__(self, action):
        self.action = action

    def eval(self):
        return self

    def __call__(self, book, seq):
        logits = torch.zeros((book.shape[0], 3), dtype=torch.float32)
        logits[:, self.action] = 10.0
        return logits


class TestWalkForward(unittest.TestCase):
    def test_fenetres_chronologiques(self):
        df = make_df(98000.0 + np.arange(1000) * 2.0)
        reports, summary = walk_forward_backtest(
            ConstantModel(1), df, asset="BTC", n_folds=4, max_holding_ticks=10
        )
        self.assertEqual(len(reports), 4)
        windows = [r["window"] for r in reports]
        self.assertEqual(windows, sorted(windows))
        self.assertEqual(summary["n_folds"], 4)
        self.assertGreater(summary["folds_profitables"], 0)

    def test_aucun_trade_modele_attente(self):
        df = make_df(98000.0 + np.arange(500) * 2.0)
        reports, summary = walk_forward_backtest(ConstantModel(0), df, n_folds=3)
        self.assertEqual(summary["folds_profitables"], 0)
        self.assertEqual(sum(summary["n_trades"]), 0)


class TestBootstrap(unittest.TestCase):
    def test_ic_contient_la_valeur_ponctuelle(self):
        rng = np.random.default_rng(0)
        pnls = list(rng.normal(1.0, 2.0, 200))
        boot = bootstrap_metrics(pnls, n_bootstrap=500)
        self.assertLess(boot["expectancy"][0], boot["expectancy_point"])
        self.assertGreater(boot["expectancy"][1], boot["expectancy_point"])
        self.assertEqual(boot["n_trades"], 200)

    def test_aucun_trade(self):
        boot = bootstrap_metrics([], n_bootstrap=100)
        self.assertEqual(boot["n_trades"], 0)
        self.assertIsNone(boot["expectancy"][0])

    def test_perte_systematique_ic_negatif(self):
        boot = bootstrap_metrics([-1.0] * 50, n_bootstrap=200)
        self.assertLess(boot["expectancy"][1], 0.0)


class TestProtectionsBacktest(unittest.TestCase):
    def test_quota_max_trades(self):
        df = make_df(98000.0 + np.arange(400) * 2.0)
        report = run_backtest(ConstantModel(1), df, asset="BTC", max_holding_ticks=5, max_trades=3)
        self.assertEqual(report["n_trades"], 3)

    def test_gel_apres_pertes_consecutives(self):
        prices = np.concatenate([np.full(60, 98000.0), 98000.0 - np.arange(1, 300) * 5.0])
        df = make_df(prices)
        mm = ScalpingMoneyManager(risk_per_trade=0.005, min_qty=0.01)
        report_sans = run_backtest(
            ConstantModel(1), df, asset="BTC", money_manager=mm, max_holding_ticks=1000,
            max_consecutive_losses=10 ** 6, cooldown_ticks=0,
        )
        report_avec = run_backtest(
            ConstantModel(1), df, asset="BTC", money_manager=mm, max_holding_ticks=1000,
            max_consecutive_losses=3, cooldown_ticks=50,
        )
        self.assertGreater(report_sans["n_trades"], report_avec["n_trades"])

    def test_prix_constant_perte_limitee_par_quota(self):
        df = make_df(np.full(300, 98000.0))
        mm = ScalpingMoneyManager(risk_per_trade=0.005, min_qty=0.01)
        report = run_backtest(
            ConstantModel(1), df, asset="BTC", money_manager=mm, max_holding_ticks=5, max_trades=5
        )
        self.assertEqual(report["n_trades"], 5)
        self.assertLess(report["final_capital"], report["initial_capital"])


class TestLabelsPonderes(unittest.TestCase):
    def _annotations(self):
        return pd.DataFrame(
            [
                {"timestamp": pd.Timestamp("2026-01-05 00:00:00.5", tz="UTC"), "asset": "BTC",
                 "side": "BUY", "entry_price": 100.0, "exit_price": 103.0},
                {"timestamp": pd.Timestamp("2026-01-05 00:00:01.5", tz="UTC"), "asset": "BTC",
                 "side": "SELL", "entry_price": 100.0, "exit_price": 100.4},
            ]
        )

    def test_poids_proportionnels_au_pnl(self):
        n = 300
        df = make_df(98000.0 + np.arange(n) * 0.001)
        out = label_ticks(df, self._annotations(), "BTC", pnl_weights=True)
        self.assertIn("sample_weight", out.columns)
        w1 = out.loc[(out["label"] == 1), "sample_weight"].iloc[0]
        w2 = out.loc[(out["label"] == 2), "sample_weight"].iloc[0]
        self.assertAlmostEqual(w1 / w2, 3.0 / 0.4, places=2)
        self.assertAlmostEqual(out.loc[out["label"] == 0, "sample_weight"].mean(), 1.0)

    def test_dataset_avec_poids(self):
        from dataset import build_window_dataset

        n = 200
        df = make_df(98000.0 + np.arange(n) * 0.001)
        out = label_ticks(df, self._annotations(), "BTC", pnl_weights=True)
        result = build_window_dataset(out, with_weights=True)
        self.assertEqual(len(result), 5)
        x_book, x_seq, y, norm, w = result
        self.assertEqual(len(w), len(y))
        self.assertTrue((w > 0).all())


class TestMultiSegmentEnv(unittest.TestCase):
    def test_rotation_des_segments(self):
        df1 = make_df(98000.0 + np.arange(300) * 2.0, seed=1)
        df2 = make_df(98000.0 - np.arange(300) * 2.0, seed=2)
        envs = [ScalpingEnv(df1, sequence_length=20), ScalpingEnv(df2, sequence_length=20)]
        multi = MultiSegmentEnv(envs)
        obs = multi.reset()
        self.assertEqual(multi.current, 1)
        obs2, reward, done, info = multi.step(0)
        self.assertEqual(obs2["book"].shape, (4,))
        obs = multi.reset()
        self.assertEqual(multi.current, 0)

    def test_vide_rejete(self):
        with self.assertRaises(ValueError):
            MultiSegmentEnv([])


class TestCostSensitivity(unittest.TestCase):
    def test_tous_multiplicateurs(self):
        df = make_df(98000.0 + np.arange(400) * 2.0)
        results = run_cost_sensitivity(
            ConstantModel(1), df, asset="BTC", slippage_multipliers=(0.5, 1.0, 2.0), max_holding_ticks=10
        )
        self.assertEqual(len(results), 3)
        self.assertEqual([r["slippage_multiplier"] for r in results], [0.5, 1.0, 2.0])
        rendements = [r["total_return_pct"] for r in results]
        self.assertGreaterEqual(rendements[0], rendements[2])
        import backtester

        self.assertEqual(backtester.ASSET_COST_MODELS["BTC"]["slippage"], 0.5)

    def test_regimes(self):
        prices = np.concatenate(
            [np.full(150, 98000.0), 98000.0 + np.arange(150) * 3.0, np.full(150, 98450.0)]
        )
        df = make_df(prices)
        trend, vol = classify_regimes(df)
        self.assertIn("hausse", trend)
        self.assertIn("range", trend)
        report = run_backtest(ConstantModel(1), df, asset="BTC", max_holding_ticks=10)
        regimes = metrics_by_regime(report, df)
        total = sum(m["n_trades"] for m in regimes.values())
        self.assertEqual(total, report["n_trades"])


if __name__ == "__main__":
    unittest.main()
