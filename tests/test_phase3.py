"""Tests phase 3 : connecteurs CME/IBKR/OANDA et boucle live de bout en bout."""

import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from asset_filters import filters_for_asset
from data_feed import ListFeed
from live_paper_trading import LivePaperTradingLoop
from dataset import ORDER_BOOK_DIM, SEQUENCE_DIM
from model_pytorch import ScalpingMultimodalNet
from pipeline_data import build_features


def make_ticks(n=600, seed=0):
    rng = np.random.default_rng(seed)
    prices = 98000.0 + np.cumsum(rng.normal(0, 3, n))
    ticks = []
    for i, p in enumerate(prices):
        tick = {
            "timestamp": pd.Timestamp("2026-01-05", tz="UTC") + pd.Timedelta(milliseconds=10 * i),
            "last_price": float(p),
            "last_volume": float(rng.integers(1, 10)),
            "best_bid": float(p) - 0.5,
            "best_ask": float(p) + 0.5,
        }
        for j in range(1, 6):
            tick[f"bid_size_{j}"] = float(rng.integers(10, 100))
            tick[f"ask_size_{j}"] = float(rng.integers(10, 100))
        ticks.append(tick)
    return ticks


class TestLiveLoopEndToEnd(unittest.TestCase):
    def test_boucle_complete_avec_journal(self):
        torch.manual_seed(7)
        model = ScalpingMultimodalNet(order_book_dim=ORDER_BOOK_DIM, sequence_dim=SEQUENCE_DIM, hidden_dim=32)
        model.eval()
        ticks = make_ticks(600)
        journal_path = os.path.join(tempfile.gettempdir(), "live_journal_test.csv")
        if os.path.exists(journal_path):
            os.remove(journal_path)

        from data_feed import CSVReplayFeed

        csv_path = os.path.join(tempfile.gettempdir(), "live_ticks.csv")
        pd.DataFrame(ticks).to_csv(csv_path, index=False)

        loop = LivePaperTradingLoop(
            model,
            asset="BTC",
            feed=CSVReplayFeed(csv_path),
            initial_capital=10000.0,
            sequence_length=20,
            confidence_threshold=0.0,
            journal_path=journal_path,
            poll_seconds=0.0,
        )
        result = loop.run()
        self.assertEqual(result["n_trades"], len(loop.trades))
        self.assertGreater(result["latency"]["n"], 0)
        self.assertGreater(result["latency"]["mean_ms"], 0.0)
        self.assertTrue(os.path.exists(journal_path))
        if result["n_trades"] > 0:
            journal_df = pd.read_csv(journal_path)
            self.assertEqual(len(journal_df), result["n_trades"])
            self.assertIn("pnl", journal_df.columns)

    def test_seuil_obi_bloque_les_trades_en_live(self):
        torch.manual_seed(7)
        model = ScalpingMultimodalNet(order_book_dim=ORDER_BOOK_DIM, sequence_dim=SEQUENCE_DIM, hidden_dim=32)
        model.eval()
        ticks = make_ticks(400)
        for tick in ticks:
            tick["bid_size_1"] = 50.0
            tick["ask_size_1"] = 50.0
        loop = LivePaperTradingLoop(
            model,
            asset="BTC",
            feed=ListFeed(ticks),
            initial_capital=10000.0,
            confidence_threshold=0.0,
            filters=filters_for_asset("BTC", obi_threshold=0.99),
            poll_seconds=0.0,
        )
        result = loop.run()
        self.assertEqual(result["n_trades"], 0)


if __name__ == "__main__":
    unittest.main()
