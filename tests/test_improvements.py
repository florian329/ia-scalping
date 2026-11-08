"""Tests unitaires : filtres par actif, environnement RL, PPO, feeds, journal."""

import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from asset_filters import AssetFilters, filters_for_asset
from backtester import run_backtest
from data_feed import CSVReplayFeed, ListFeed, ReconnectingFeed
from live_paper_trading import LatencyMonitor, TradeJournal
from pipeline_data import build_features
from rl_environment import ScalpingEnv


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


class TestAssetFilters(unittest.TestCase):
    def test_seuil_obi_es(self):
        f = filters_for_asset("ES")
        self.assertEqual(f.obi_threshold, 0.85)
        self.assertFalse(f.obi_allowed(0.5, side=1)[0] if isinstance(f.obi_allowed(0.5, 1), tuple) else f.obi_allowed(0.5, 1))
        self.assertTrue(f.obi_allowed(0.9, 1))
        self.assertFalse(f.obi_allowed(0.9, -1))
        self.assertTrue(f.obi_allowed(-0.9, -1))

    def test_blackout_xau(self):
        from datetime import datetime, timedelta, timezone

        start = datetime(2026, 1, 5, 14, 25, tzinfo=timezone.utc)
        end = start + timedelta(minutes=30)
        f = filters_for_asset("XAU", news_blackouts=[(start, end)])
        inside = start + timedelta(minutes=5)
        outside = end + timedelta(minutes=5)
        self.assertFalse(f.trade_allowed(1, 0.0, inside)[0])
        self.assertTrue(f.trade_allowed(1, 0.0, outside)[0])

    def test_profondeur_max_btc(self):
        f = filters_for_asset("BTC")
        self.assertEqual(f.effective_depth(20), 10)
        self.assertEqual(f.effective_depth(5), 5)

    def test_backtester_respecte_le_filtre_es(self):
        import torch

        class AlwaysLong:
            def eval(self):
                return self

            def __call__(self, book, seq):
                logits = torch.zeros((book.shape[0], 3))
                logits[:, 1] = 10.0
                return logits

        prices = 98000.0 + np.arange(300) * 2.0
        df = make_df(prices)
        df["OBI"] = 0.5
        report_no_filter = run_backtest(AlwaysLong(), df, asset="ES")
        report_filtered = run_backtest(AlwaysLong(), df, asset="ES", filters=filters_for_asset("ES"))
        self.assertGreater(report_no_filter["n_trades"], 0)
        self.assertEqual(report_filtered["n_trades"], 0)


class TestScalpingEnv(unittest.TestCase):
    def test_reset_et_shapes(self):
        df = make_df(98000.0 + np.cumsum(np.random.default_rng(0).normal(0, 3, 400)))
        env = ScalpingEnv(df, sequence_length=20)
        obs = env.reset()
        self.assertEqual(obs["book"].shape, (4,))
        self.assertEqual(obs["seq"].shape, (20, 4))
        self.assertEqual(env.action_space, (3,))

    def test_reward_sur_tendance_haussiere(self):
        df = make_df(98000.0 + np.arange(400) * 3.0)
        env = ScalpingEnv(df, sequence_length=20, time_penalty=0.0, drawdown_penalty=0.0)
        env.reset()
        total_reward = 0.0
        done = False
        while not done:
            obs, reward, done, info = env.step(1)
            total_reward += reward
        self.assertGreater(total_reward, 0.0)
        self.assertGreater(env.capital, env.initial_capital)

    def test_penalite_temps(self):
        df = make_df(np.full(400, 98000.0))
        env = ScalpingEnv(df, sequence_length=20, time_penalty=0.01, drawdown_penalty=0.0, min_qty=0.01)
        env.reset()
        total_reward = 0.0
        done = False
        while not done:
            _, reward, done, _ = env.step(1)
            total_reward += reward
        self.assertLess(total_reward, 0.0)


class TestPPO(unittest.TestCase):
    def test_une_iteration_ppo_tourne(self):
        from ppo_agent import PPOAgent

        df = make_df(98000.0 + np.cumsum(np.random.default_rng(1).normal(0, 3, 600)))
        env = ScalpingEnv(df, sequence_length=20)
        agent = PPOAgent(env, hidden_dim=32)
        history = agent.train(n_iterations=2, n_steps_per_rollout=128, verbose_every=0)
        self.assertEqual(len(history), 2)
        for stats in history:
            self.assertTrue(np.isfinite(stats["policy_loss"]))
            self.assertTrue(np.isfinite(stats["value_loss"]))
        agent.save(os.path.join(tempfile.gettempdir(), "ppo_test_policy.pth"))


class TestFeeds(unittest.TestCase):
    def test_list_feed(self):
        ticks = [{"last_price": 1.0, "last_volume": 2}, {"last_price": 1.1, "last_volume": 3}]
        self.assertEqual(len(list(ListFeed(ticks))), 2)

    def test_csv_replay(self):
        path = os.path.join(tempfile.gettempdir(), "replay_test.csv")
        make_df(98000.0 + np.arange(50)).to_csv(path, index=False)
        ticks = list(CSVReplayFeed(path))
        self.assertEqual(len(ticks), 50)
        self.assertIn("last_price", ticks[0])

    def test_reconnecting_feed(self):
        attempts = {"n": 0}

        def factory():
            def gen():
                attempts["n"] += 1
                if attempts["n"] == 1:
                    yield {"last_price": 1.0, "last_volume": 1}
                    raise ConnectionError("boom")
                yield {"last_price": 2.0, "last_volume": 1}

            return gen()

        feed = ReconnectingFeed(factory, base_delay=0.01, max_retries=3)
        ticks = list(feed)
        self.assertEqual(len(ticks), 2)
        self.assertEqual(feed.reconnect_count, 1)


class TestJournalEtLatence(unittest.TestCase):
    def test_journal_csv(self):
        from backtester import LONG, Trade

        path = os.path.join(tempfile.gettempdir(), "journal_test.csv")
        if os.path.exists(path):
            os.remove(path)
        journal = TradeJournal(path, asset="BTC")
        trade = Trade(LONG, 10, 98001.0, 0.5, 97999.0, 98005.0)
        trade.exit_price = 98003.0
        trade.exit_reason = "take_profit"
        journal.log(trade, 10005.0, timestamp="2026-01-05")
        journal.log(trade, 10010.0, timestamp="2026-01-05")
        df = pd.read_csv(path)
        self.assertEqual(len(df), 2)
        self.assertEqual(df.loc[0, "asset"], "BTC")
        self.assertGreater(float(df.loc[0, "pnl"]), 0.0)

    def test_latence(self):
        monitor = LatencyMonitor()
        monitor.record(5.0)
        monitor.record(7.0)
        stats = monitor.stats()
        self.assertEqual(stats["n"], 2)
        self.assertAlmostEqual(stats["mean_ms"], 6.0)
        self.assertGreaterEqual(stats["p95_ms"], 6.0)


if __name__ == "__main__":
    unittest.main()
