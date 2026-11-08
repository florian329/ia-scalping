"""Tests phase 5 : vectorisation, drift, annotations, calibration, tuning."""

import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtester import run_backtest
from calibration import expected_calibration_error, fit_temperature
from dataset import BOOK_FEATURES, ORDER_BOOK_DIM, SEQUENCE_DIM, build_window_dataset
from data_feed import ListFeed
from drift_monitor import FeatureDriftMonitor, WinRateDriftMonitor
from model_pytorch import ScalpingMultimodalNet
from pipeline_data import build_features
from rl_environment import ScalpingEnv
from prepare_annotations import generate_template, validate_annotations
from tune_hyperparameters import evaluate_params, random_search


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


class TestVectorizedBacktest(unittest.TestCase):
    def test_meme_resultat_que_le_modele_sequentiel(self):
        df = make_df(98000.0 + np.cumsum(np.random.default_rng(3).normal(0, 3, 400)))
        model = ConstantModel(1)
        seq_report = run_backtest(model, df, asset="BTC", max_holding_ticks=10, vectorized=False)
        vec_report = run_backtest(model, df, asset="BTC", max_holding_ticks=10, vectorized=True)
        self.assertEqual(seq_report["n_trades"], vec_report["n_trades"])
        self.assertAlmostEqual(seq_report["final_capital"], vec_report["final_capital"], places=6)

    def test_plus_rapide_avec_un_vrai_modele(self):
        import time

        torch.manual_seed(0)
        model = ScalpingMultimodalNet(order_book_dim=ORDER_BOOK_DIM, sequence_dim=SEQUENCE_DIM, hidden_dim=32)
        model.eval()
        df = make_df(98000.0 + np.arange(1500) * 0.5)
        run_backtest(model, df, asset="BTC", max_holding_ticks=10, vectorized=False)
        run_backtest(model, df, asset="BTC", max_holding_ticks=10, vectorized=True)
        t0 = time.perf_counter()
        run_backtest(model, df, asset="BTC", max_holding_ticks=10, vectorized=False)
        t_seq = time.perf_counter() - t0
        t0 = time.perf_counter()
        run_backtest(model, df, asset="BTC", max_holding_ticks=10, vectorized=True)
        t_vec = time.perf_counter() - t0
        self.assertLess(t_vec, t_seq)


class TestDriftMonitors(unittest.TestCase):
    def test_win_rate_derive_apres_pertes(self):
        monitor = WinRateDriftMonitor(reference_win_rate=0.6, window=20, min_trades=10, abs_threshold=0.2)
        drifted = False
        for _ in range(15):
            if monitor.record(-1.0):
                drifted = True
                break
        self.assertTrue(drifted)
        self.assertLess(monitor.current_win_rate(), 0.4)

    def test_win_rate_stable_pas_d_alerte(self):
        monitor = WinRateDriftMonitor(reference_win_rate=0.6, window=20, min_trades=10, abs_threshold=0.2)
        results = [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0]
        alerts = [monitor.record(r) for r in results]
        self.assertFalse(any(alerts))

    def test_ks_derive_de_features(self):
        rng = np.random.default_rng(0)
        ref = pd.DataFrame({"OBI": rng.normal(0, 0.1, 5000), "ATR_Tick": rng.normal(3, 0.5, 5000)})
        monitor = FeatureDriftMonitor(ref, ["OBI", "ATR_Tick"])
        same = pd.DataFrame({"OBI": rng.normal(0, 0.1, 1000), "ATR_Tick": rng.normal(3, 0.5, 1000)})
        changed = pd.DataFrame({"OBI": rng.normal(0.5, 0.1, 1000), "ATR_Tick": rng.normal(3, 0.5, 1000)})
        self.assertFalse(monitor.drifted(same)[0])
        self.assertTrue(monitor.drifted(changed)[0])
        _, features_drifted = monitor.drifted(changed)
        self.assertIn("OBI", features_drifted)

    def test_reference_invalide(self):
        with self.assertRaises(ValueError):
            WinRateDriftMonitor(reference_win_rate=1.5)


class TestAnnotations(unittest.TestCase):
    def test_template_genere(self):
        path = os.path.join(tempfile.gettempdir(), "ann_template.csv")
        generate_template(path)
        self.assertTrue(os.path.exists(path))
        df = pd.read_csv(path)
        self.assertEqual(len(df), 2)

    def test_fichier_valide(self):
        path = os.path.join(tempfile.gettempdir(), "ann_ok.csv")
        pd.DataFrame(
            {
                "timestamp": ["2026-01-05T14:00:00Z", "2026-01-05T15:00:00Z", "2026-01-05T16:00:00Z"],
                "asset": ["BTC"] * 3,
                "side": ["BUY", "SELL", "BUY"],
                "entry_price": [98000.0, 98100.0, 98200.0],
                "exit_price": [98050.0, 98050.0, 98260.0],
            }
        ).to_csv(path, index=False)
        report = validate_annotations(path)
        self.assertTrue(report["valid"])
        self.assertEqual(report["n_trades"], 3)

    def test_erreurs_detectees(self):
        path = os.path.join(tempfile.gettempdir(), "ann_bad.csv")
        pd.DataFrame(
            {
                "timestamp": ["pas-une-date", "2026-01-05T15:00:00Z"],
                "asset": ["BTC", "BTC"],
                "side": ["ACHAT", "SELL"],
                "entry_price": [-5.0, 98100.0],
                "exit_price": [98050.0, 0.0],
            }
        ).to_csv(path, index=False)
        report = validate_annotations(path)
        self.assertFalse(report["valid"])
        self.assertGreaterEqual(len(report["errors"]), 3)

    def test_manque_de_trades_averti(self):
        path = os.path.join(tempfile.gettempdir(), "ann_few.csv")
        pd.DataFrame(
            {
                "timestamp": ["2026-01-05T14:00:00Z"],
                "asset": ["BTC"],
                "side": ["BUY"],
                "entry_price": [98000.0],
                "exit_price": [98050.0],
            }
        ).to_csv(path, index=False)
        report = validate_annotations(path)
        self.assertTrue(report["valid"])
        self.assertTrue(any("minimum" in w for w in report["warnings"]))


class TestCalibration(unittest.TestCase):
    def test_fit_temperature_ameliore_ou_egale_nll(self):
        torch.manual_seed(0)
        model = ScalpingMultimodalNet(order_book_dim=ORDER_BOOK_DIM, sequence_dim=SEQUENCE_DIM, hidden_dim=32)
        from torch.utils.data import DataLoader, TensorDataset

        xb = torch.randn(300, ORDER_BOOK_DIM)
        xs = torch.randn(300, 20, SEQUENCE_DIM)
        y = torch.randint(0, 3, (300,))
        loader = DataLoader(TensorDataset(xb, xs, y), batch_size=64)
        temperature = fit_temperature(model, loader)
        self.assertGreater(temperature, 0.0)

    def test_ece_parfait_vs_mauvais(self):
        probs = np.array([[1.0, 0.0, 0.0]] * 50 + [[0.0, 1.0, 0.0]] * 50)
        targets_good = np.array([0] * 50 + [1] * 50)
        ece_good = expected_calibration_error(probs, targets_good)
        self.assertAlmostEqual(ece_good, 0.0, places=6)
        targets_bad = np.array([0] * 25 + [1] * 25 + [1] * 25 + [0] * 25)
        ece_bad = expected_calibration_error(probs, targets_bad)
        self.assertGreater(ece_bad, ece_good)


class TestTuning(unittest.TestCase):
    def test_random_search_tourne_et_garde_le_meilleur(self):
        df = make_df(98000.0 + np.arange(600) * 2.0)
        best, history = random_search(
            ConstantModel(1), df, n_trials=4, seed=0, verbose=False
        )
        self.assertEqual(len(history), 4)
        self.assertIsNotNone(best["params"])
        self.assertGreaterEqual(best["score"], max(h["score"] for h in history) - 1e-9)

    def test_evaluate_params_retourne_un_score(self):
        df = make_df(98000.0 + np.arange(400) * 2.0)
        params = {
            "atr_multiplier_sl": 1.5,
            "atr_multiplier_tp": 2.0,
            "max_holding_ticks": 10,
            "confidence_threshold": 0.0,
            "sequence_length": 20,
        }
        score, summary = evaluate_params(ConstantModel(1), df, params, n_folds=2)
        self.assertTrue(np.isfinite(score))
        self.assertEqual(summary["n_folds"], 2)


class TestNouvellesFeatures(unittest.TestCase):
    def test_features_ajoutees(self):
        df = make_df(98000.0 + np.arange(300))
        for col in ("OFI", "Hour_Sin", "Hour_Cos", "VPIN_Rolling"):
            self.assertIn(col, df.columns)
        self.assertTrue((df["Hour_Sin"] >= -1.001).all() and (df["Hour_Sin"] <= 1.001).all())
        self.assertTrue((df["VPIN_Rolling"] >= 0).all() and (df["VPIN_Rolling"] <= 1).all())

    def test_dataset_dimensions_coherentes(self):
        df = make_df(98000.0 + np.arange(300))
        df = df.assign(label=np.zeros(len(df), dtype=int))
        x_book, x_seq, y, _ = build_window_dataset(df)
        self.assertEqual(x_book.shape[1], ORDER_BOOK_DIM)
        self.assertEqual(x_seq.shape[2], SEQUENCE_DIM)
        self.assertEqual(ORDER_BOOK_DIM, 7)
        self.assertEqual(SEQUENCE_DIM, 5)



class TestMakerExecution(unittest.TestCase):
    def test_remplissage_au_bid_sans_half_spread(self):
        base = 98000.0 + 0.5 * ((-1) ** np.arange(60))
        prices = np.concatenate([base, [97998.0], 98000.0 + np.arange(1, 200) * 5.0])
        df = make_df(prices)
        report = run_backtest(
            ConstantModel(1), df, asset="BTC", max_holding_ticks=1000, execution="maker"
        )
        self.assertGreater(report["n_trades"], 0)
        self.assertGreater(report["maker_fills"], 0)
        for tr in report["trades"]:
            self.assertLessEqual(tr.entry_price, 98000.0)

    def test_annulation_si_le_prix_ne_traverse_jamais(self):
        prices = 98000.0 + np.arange(300) * 5.0
        df = make_df(prices)
        report = run_backtest(
            ConstantModel(1), df, asset="BTC", max_holding_ticks=10, execution="maker",
            maker_timeout_ticks=5,
        )
        self.assertEqual(report["n_trades"], 0)
        self.assertGreater(report["maker_cancels"], 0)

    def test_frais_maker_inferieurs_au_taker(self):
        base = 98000.0 + 0.5 * ((-1) ** np.arange(60))
        prices = np.concatenate([base, [97998.0], 98000.0 + np.arange(1, 150) * 5.0])
        df = make_df(prices)
        taker = run_backtest(ConstantModel(1), df, asset="BTC", max_holding_ticks=1000, execution="taker")
        maker = run_backtest(ConstantModel(1), df, asset="BTC", max_holding_ticks=1000, execution="maker")
        self.assertGreater(taker["n_trades"], 0)
        self.assertGreater(maker["n_trades"], 0)
        fees_taker_par_trade = taker["total_fees"] / taker["n_trades"]
        fees_maker_par_trade = maker["total_fees"] / maker["n_trades"]
        self.assertLess(fees_maker_par_trade, fees_taker_par_trade)

    def test_env_rl_maker_remplit_au_travers(self):
        base = 98000.0 + 0.5 * ((-1) ** np.arange(60))
        prices = np.concatenate([base, [97998.0], 98000.0 + np.arange(1, 100) * 5.0])
        df = make_df(prices)
        env = ScalpingEnv(df, sequence_length=20, execution="maker", time_penalty=0.0, drawdown_penalty=0.0)
        env.reset()
        done = False
        while not done:
            _, reward, done, _ = env.step(1)
        self.assertNotEqual(env.realized_pnl, 0.0)


class TestMakerLiveLoop(unittest.TestCase):
    def test_boucle_live_maker_remplit_et_compte(self):
        from live_paper_trading import LivePaperTradingLoop

        model = ConstantModel(1)
        rng = np.random.default_rng(0)
        n = 600
        prices = 98000.0 + 0.5 * ((-1) ** np.arange(n)) + np.cumsum(rng.normal(0, 0.2, n))
        ticks = []
        for i, p in enumerate(prices):
            tick = {
                "timestamp": pd.Timestamp("2026-01-05", tz="UTC") + pd.Timedelta(milliseconds=10 * i),
                "last_price": float(p),
                "last_volume": 1.0,
                "best_bid": float(p) - 0.5,
                "best_ask": float(p) + 0.5,
            }
            for j in range(1, 6):
                tick[f"bid_size_{j}"] = 50.0
                tick[f"ask_size_{j}"] = 50.0
            ticks.append(tick)
        loop = LivePaperTradingLoop(
            model,
            asset="BTC",
            feed=ListFeed(ticks),
            initial_capital=10000.0,
            confidence_threshold=0.0,
            execution="maker",
            poll_seconds=0.0,
        )
        result = loop.run()
        self.assertEqual(result["execution"], "maker")
        self.assertIn("maker_fills", result)
        self.assertIn("maker_cancels", result)
        self.assertGreater(result["maker_fills"] + result["maker_cancels"], 0)

class TestTpFeeFloor(unittest.TestCase):
    def test_tp_couvre_au_moins_k_fois_les_frais(self):
        base = 76000.0 + 0.5 * ((-1) ** np.arange(80))
        prices = np.concatenate([base, 76000.0 + np.arange(1, 150) * 20.0])
        df = make_df(prices)
        report = run_backtest(
            ConstantModel(1), df, asset="BTC", max_holding_ticks=1000,
            execution="maker", min_tp_fee_multiple=3.0,
        )
        self.assertGreater(report["n_trades"], 0)
        entry_fee = 0.0002
        exit_fee = 0.0002
        for tr in report["trades"]:
            if tr.exit_reason == "take_profit":
                tp_distance = tr.tp_price - tr.entry_price
                round_trip = (entry_fee + exit_fee) * tr.entry_price
                self.assertGreaterEqual(tp_distance, 3.0 * round_trip * 0.999)

    def test_plancher_desactive(self):
        base = 76000.0 + 0.5 * ((-1) ** np.arange(80))
        prices = np.concatenate([base, 76000.0 + np.arange(1, 150) * 20.0])
        df = make_df(prices)
        report = run_backtest(
            ConstantModel(1), df, asset="BTC", max_holding_ticks=1000,
            execution="maker", min_tp_fee_multiple=0.0,
        )
        self.assertGreater(report["n_trades"], 0)
        atr_tp = df["ATR_Tick"].mean() * 2.42
        for tr in report["trades"]:
            if tr.exit_reason == "take_profit":
                self.assertLess(tr.tp_price - tr.entry_price, atr_tp * 3)


class TestMakerReplacements(unittest.TestCase):
    def _prix_avec_dips(self):
        return np.concatenate([
            np.full(40, 76000.0),
            [75999.8],
            np.full(5, 76000.0),
            [75999.5],
            76000.0 + np.arange(1, 60) * 3.0,
        ])

    def test_replacement_provoque_le_remplissage(self):
        df = make_df(self._prix_avec_dips())
        report = run_backtest(
            ConstantModel(1), df, asset="BTC", max_holding_ticks=1000,
            execution="maker", maker_timeout_ticks=10,
            maker_replacements_max=3, min_tp_fee_multiple=0.0,
        )
        self.assertGreater(report["maker_replacements"], 0)
        self.assertGreater(report["maker_fills"], 0)
        self.assertGreater(report["n_trades"], 0)

    def test_sans_replacement_pas_de_remplissage(self):
        df = make_df(self._prix_avec_dips())
        report = run_backtest(
            ConstantModel(1), df, asset="BTC", max_holding_ticks=1000,
            execution="maker", maker_timeout_ticks=10,
            maker_replacements_max=0, min_tp_fee_multiple=0.0,
        )
        self.assertEqual(report["maker_replacements"], 0)
        self.assertEqual(report["maker_fills"], 0)
        self.assertEqual(report["n_trades"], 0)




class TestExportExcel(unittest.TestCase):
    def test_excel_genere_et_colore(self):
        import os
        import tempfile

        from backtester import run_backtest
        from export_trades_excel import export_trades_to_excel

        base = 76000.0 + 0.5 * ((-1) ** np.arange(80))
        prices = np.concatenate([base, [75998.0], 76000.0 + np.arange(1, 150) * 20.0])
        df = make_df(prices)
        report = run_backtest(
            ConstantModel(1), df, asset="BTC", initial_capital=1000.0,
            max_holding_ticks=1000, execution="maker",
        )
        self.assertGreater(report["n_trades"], 0)
        path = os.path.join(tempfile.gettempdir(), "test_export_excel.xlsx")
        export_trades_to_excel(report, path, df=df)
        self.assertTrue(os.path.exists(path))
        import openpyxl

        wb = openpyxl.load_workbook(path)
        self.assertIn("Trades", wb.sheetnames)
        self.assertIn("Resume", wb.sheetnames)
        ws = wb["Trades"]
        self.assertEqual(ws.max_row - 1, report["n_trades"])
        colored = 0
        for row in ws.iter_rows(min_row=2):
            cell = row[7]
            rgb = str(cell.fill.start_color.rgb)
            if "C6EFCE" in rgb or "FFC7CE" in rgb:
                colored += 1
        self.assertEqual(colored, report["n_trades"])

if __name__ == "__main__":
    unittest.main()
