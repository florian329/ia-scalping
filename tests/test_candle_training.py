"""Behavioral regressions for the BTC candle training pipeline."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'freqtrade_integration')]


class TestPurgedPreprocessing(unittest.TestCase):
    def test_training_labels_end_before_validation_and_last_future_is_excluded(self):
        from freqtrade_integration.candle_training import purged_split_indices
        parts = purged_split_indices(1000, horizon=15, warmup=80)
        self.assertLess(parts['train'][-1] + 15, parts['validation'][0])
        self.assertLess(parts['validation'][-1] + 15, parts['test'][0])
        self.assertLess(parts['test'][-1] + 15, 1000)

    def test_future_values_cannot_change_training_normalization(self):
        from candle_features import BOOK_FEATURES, SEQ_FEATURES
        from freqtrade_integration.candle_training import prepare_inputs
        rng = np.random.default_rng(5)
        features = pd.DataFrame(rng.normal(size=(1000, len(set(BOOK_FEATURES + SEQ_FEATURES)))),
                                columns=list(dict.fromkeys(BOOK_FEATURES + SEQ_FEATURES)))
        before = prepare_inputs(features, horizon=15)
        changed = features.copy()
        changed.iloc[700:] += 100000
        after = prepare_inputs(changed, horizon=15)
        np.testing.assert_array_equal(before.norm['book_mean'], after.norm['book_mean'])
        np.testing.assert_array_equal(before.norm['seq_mean'], after.norm['seq_mean'])
        idx = before.splits['train']
        np.testing.assert_array_equal(before.book[idx], after.book[idx])
        np.testing.assert_array_equal(before.sequences[idx], after.sequences[idx])

class TestBestCheckpoint(unittest.TestCase):
    def test_model_is_restored_to_best_validation_epoch(self):
        import tempfile
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from training_engine import train_model
        class BiasModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = torch.nn.Parameter(torch.zeros(3))
            def forward(self, book, seq):
                return self.bias.expand(len(book), 3)
        book, seq = torch.zeros(8, 7), torch.zeros(8, 20, 5)
        train = DataLoader(TensorDataset(book, seq, torch.ones(8, dtype=torch.long)), batch_size=8)
        val = DataLoader(TensorDataset(book, seq, torch.full((8,), 2, dtype=torch.long)), batch_size=8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'best.pth'
            model = BiasModel()
            train_model(model, train, val, epochs=5, lr=0.1, patience=2, checkpoint_path=path)
            state = torch.load(path, weights_only=False)
            self.assertEqual(state['epoch'], 1)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, state['model_state_dict'][key], rtol=0, atol=0)

class TestTrueBarriers(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({'date': pd.date_range('2026-01-01', periods=6, freq='1min', tz='UTC'),
                             'open': [100.] * 6, 'high': [100.] * 6,
                             'low': [100.] * 6, 'close': [100.] * 6, 'volume': [1.] * 6})
    def policy(self):
        from freqtrade_integration.candle_policy import CandlePolicy
        return CandlePolicy(leverage=1., stoploss=-.01, roi_steps=((0, .01),), horizon=2,
                            fee_entry=0., fee_exit=0., fee_stop=0., slippage_bps=0.)
    def test_intrabar_long_target_and_short_target(self):
        from freqtrade_integration.candle_policy import triple_barrier_labels
        df = self.frame(); df.loc[1, 'high'] = 102.
        self.assertEqual(triple_barrier_labels(df, self.policy())[0], 1)
        df = self.frame(); df.loc[1, 'low'] = 98.
        self.assertEqual(triple_barrier_labels(df, self.policy())[0], 2)
    def test_same_candle_stop_and_target_is_pessimistic(self):
        from freqtrade_integration.candle_policy import simulate_trade
        df = self.frame(); df.loc[1, ['high', 'low']] = [102., 98.]
        for side in ['long', 'short']:
            trade = simulate_trade(df, 0, side, self.policy())
            self.assertEqual(trade.reason, 'stop_loss')
            self.assertLess(trade.net_return, 0.)
    def test_entry_uses_next_open_and_gap_stop_is_not_guaranteed(self):
        from freqtrade_integration.candle_policy import simulate_trade
        df = self.frame(); df.loc[1, ['open', 'high', 'low', 'close']] = [105., 105.1, 104.9, 105.]
        df.loc[2, ['open', 'high', 'low', 'close']] = [95., 95., 94., 95.]
        trade = simulate_trade(df, 0, 'long', self.policy())
        self.assertEqual(trade.entry_price, 105.)
        self.assertEqual(trade.exit_price, 95.)
    def test_tail_without_full_future_is_unlabelled(self):
        from freqtrade_integration.candle_policy import triple_barrier_labels
        labels = triple_barrier_labels(self.frame(), self.policy())
        np.testing.assert_array_equal(labels[-3:], [-1, -1, -1])

class TestTrainExecutionParity(unittest.TestCase):
    def test_version_two_features_do_not_depend_on_distant_prefix(self):
        from candle_features import build_features, BOOK_FEATURES, SEQ_FEATURES
        n = 500
        close = 100 + np.sin(np.arange(n) / 10)
        df = pd.DataFrame({'date': pd.date_range('2026-01-01', periods=n, freq='1min', tz='UTC'),
                           'open': close, 'high': close + .1, 'low': close - .1,
                           'close': close, 'volume': np.arange(n) + 1.})
        full = build_features(df, version=2)
        sliced = build_features(df.iloc[250:].reset_index(drop=True), version=2)
        np.testing.assert_allclose(full.iloc[-1][BOOK_FEATURES + SEQ_FEATURES].to_numpy(float),
                                   sliced.iloc[-1][BOOK_FEATURES + SEQ_FEATURES].to_numpy(float), atol=1e-10)
    def test_inference_accepts_shifted_index_and_matches_causal_prefix(self):
        import torch
        from freqtrade_integration.candle_training import infer_probabilities
        from candle_features import BOOK_FEATURES, SEQ_FEATURES
        n = 240
        close = 100 + np.sin(np.arange(n) / 10)
        df = pd.DataFrame({'date': pd.date_range('2026-01-01', periods=n, freq='1min', tz='UTC'),
                           'open': close, 'high': close + .1, 'low': close - .1,
                           'close': close, 'volume': np.arange(n) + 1.}, index=np.arange(n) + 1000)
        class Model(torch.nn.Module):
            def forward(self, book, seq):
                return torch.stack([book[:, 0], seq[:, -1, 0], book[:, 1]], dim=1)
        norm = {'book_mean': np.zeros(len(BOOK_FEATURES)), 'book_std': np.ones(len(BOOK_FEATURES)),
                'seq_mean': np.zeros(len(SEQ_FEATURES)), 'seq_std': np.ones(len(SEQ_FEATURES))}
        model = Model()
        batch = infer_probabilities(model, df, norm)
        prefix = infer_probabilities(model, df.iloc[:160], norm)
        np.testing.assert_allclose(batch[:160], prefix, atol=1e-7)

class TestDirectionalValidation(unittest.TestCase):
    def test_losing_side_is_disabled_instead_of_forcing_shorts(self):
        from freqtrade_integration.candle_validation import select_thresholds
        from freqtrade_integration.candle_policy import CandlePolicy
        df = TestTrueBarriers().frame()
        # Every trade loses to costs on a flat market, even with high confidence.
        probs = np.tile([.01, .01, .98], (len(df), 1))
        choice = select_thresholds(probs, df, np.array([0, 1, 2]), CandlePolicy(horizon=1), min_trades=1)
        self.assertEqual(choice['short'], 1.)
        self.assertEqual(choice['long'], 1.)
    def test_metrics_are_net_and_report_sides_separately(self):
        from freqtrade_integration.candle_validation import summarize_trades
        from freqtrade_integration.candle_policy import CandleTrade
        trades = [CandleTrade(0, 1, 2, 'long', 'roi', 100, 101, .01, .002),
                  CandleTrade(3, 4, 5, 'short', 'stop_loss', 100, 101, -.02, .003)]
        report = summarize_trades(trades)
        self.assertAlmostEqual(report['expectancy'], -.005)
        self.assertEqual(report['directions']['long']['n_trades'], 1)
        self.assertLess(report['directions']['short']['expectancy'], 0)
    def test_calibration_and_selection_never_use_test_indices(self):
        import torch
        from freqtrade_integration.candle_validation import fit_decision
        from freqtrade_integration.candle_training import prepare_inputs
        from freqtrade_integration.candle_features import build_features
        from freqtrade_integration.candle_policy import CandlePolicy
        n = 700
        close = 100 + np.sin(np.arange(n) / 9)
        df = pd.DataFrame({'date': pd.date_range('2026-01-01', periods=n, freq='1min', tz='UTC'),
                           'open': close, 'high': close + .1, 'low': close - .1,
                           'close': close, 'volume': np.ones(n)})
        prepared = prepare_inputs(build_features(df), horizon=4)
        class Model(torch.nn.Module):
            def forward(self, book, seq):
                return torch.stack([book[:, 0], book[:, 1], -book[:, 1]], dim=1)
        labels = np.arange(n) % 3
        before = fit_decision(Model(), prepared, labels, df, CandlePolicy(horizon=3), min_trades=2)
        changed = labels.copy(); changed[prepared.splits['test']] = 2
        after = fit_decision(Model(), prepared, changed, df, CandlePolicy(horizon=3), min_trades=2)
        self.assertEqual(before, after)
        self.assertLess(before['selection_last_index'] + 4, prepared.splits['test'][0])

class TestCandidateArtifact(unittest.TestCase):
    def test_exported_candidate_reproduces_training_inference_and_policy(self):
        import json
        import tempfile
        import torch
        from datetime import timedelta
        from types import SimpleNamespace
        from freqtrade_integration.train_candle_model import train_candidate
        from freqtrade_integration.candle_policy import CandlePolicy
        from freqtrade_integration.candle_validation import fit_decision, collect_logits, softmax
        from IaScalpingStrategy import IaScalpingStrategy
        n = 700
        close = 100 + np.sin(np.arange(n) / 9)
        df = pd.DataFrame({'date': pd.date_range('2026-01-01', periods=n, freq='1min', tz='UTC'),
                           'open': close, 'high': close + .1, 'low': close - .1,
                           'close': close, 'volume': np.arange(n) % 13 + 1.})
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'candidate'
            policy = CandlePolicy(horizon=3, leverage=10)
            model, prepared, labels, meta = train_candidate(df, path, policy, epochs=2, hidden_dim=8)
            decision = fit_decision(model, prepared, labels, df, policy, min_trades=2)
            (path / 'decision.json').write_text(json.dumps(decision))
            strategy = IaScalpingStrategy({'dry_run': True, 'stake_currency': 'USDT',
                'stake_amount': 100, 'ia_model_dir': str(path),
                'exchange': {'name': 'binance', 'pair_whitelist': ['BTC/USDT:USDT']}})
            out = strategy.populate_indicators(df.set_axis(np.arange(n) + 400), {})
            ix = prepared.splits['test']
            expected = softmax(collect_logits(model, prepared, ix), decision['temperature'])
            np.testing.assert_allclose(out['model_prob_long'].iloc[ix], expected[:, 1], atol=1e-6)
            np.testing.assert_allclose(out['model_prob_short'].iloc[ix], expected[:, 2], atol=1e-6)
            self.assertEqual(strategy.leverage_value, policy.leverage)
            self.assertEqual(strategy.stoploss, policy.stoploss)
            self.assertEqual(strategy.protections, policy.protections())
            fake_trade = SimpleNamespace(open_date_utc=df.date.iloc[100])
            self.assertIsNone(strategy.custom_exit('', fake_trade, fake_trade.open_date_utc + timedelta(minutes=2), 0, 0))
            self.assertEqual(strategy.custom_exit('', fake_trade, fake_trade.open_date_utc + timedelta(minutes=3), 0, 0), 'policy_timeout')
            with self.assertRaises(FileExistsError):
                train_candidate(df, path, policy, epochs=1, hidden_dim=8)

class TestFreqtradeResultValidation(unittest.TestCase):
    def test_zero_exit_code_without_result_is_not_a_success(self):
        import tempfile
        from freqtrade_integration.run_candle_validation import read_freqtrade_result
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / 'command.log').write_text('ERROR - Configuration error: token is required')
            result = read_freqtrade_result(path, returncode=0)
            self.assertFalse(result['validated'])
            self.assertIn('reason', result)

class TestRuntimeBoundaries(unittest.TestCase):
    def test_microsecond_and_nanosecond_dates_have_same_holding_minutes(self):
        from freqtrade_integration.candle_policy import CandleMarket
        df = TestTrueBarriers().frame()
        df['date'] = df['date'].dt.as_unit('us')
        market = CandleMarket(df)
        self.assertAlmostEqual(market.minutes[1] - market.minutes[0], 1.)
    def test_structured_backtest_prefix_zip_is_read(self):
        import json
        import tempfile
        import zipfile
        from freqtrade_integration.run_candle_validation import read_freqtrade_result
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / 'command.log').write_text('Backtested: IaScalpingStrategy')
            with zipfile.ZipFile(path / 'results-example.zip', 'w') as archive:
                archive.writestr('result.json', json.dumps({'strategy': {'IaScalpingStrategy':
                    {'total_trades': 12, 'profit_total': -.1}}}))
            result = read_freqtrade_result(path, 0)
            self.assertTrue(result['validated'])
            self.assertEqual(result['metrics']['total_trades'], 12)
    def test_explicit_zero_trade_summary_is_valid_but_not_profitable(self):
        import tempfile
        from freqtrade_integration.run_candle_validation import read_freqtrade_result
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / 'command.log').write_text('No trades made.\nSTRATEGY SUMMARY\nIaScalpingStrategy')
            result = read_freqtrade_result(path, 0)
            self.assertTrue(result['validated'])
            self.assertEqual(result['metrics']['profit_total'], 0.)


class TestExactPrefixInference(unittest.TestCase):
    def test_real_lstm_probabilities_are_exactly_prefix_invariant(self):
        import torch
        from model_pytorch import ScalpingMultimodalNet
        from candle_features import BOOK_FEATURES, SEQ_FEATURES
        from freqtrade_integration.candle_training import infer_probabilities
        n = 800
        close = 100 + np.sin(np.arange(n) / 9)
        df = pd.DataFrame({'date': pd.date_range('2026-01-01', periods=n, freq='1min', tz='UTC'),
                           'open': close, 'high': close + .1, 'low': close - .1,
                           'close': close, 'volume': np.arange(n) % 13 + 1.})
        torch.manual_seed(42)
        model = ScalpingMultimodalNet(len(BOOK_FEATURES), len(SEQ_FEATURES), 32)
        norm = {'book_mean': np.zeros(7), 'book_std': np.ones(7),
                'seq_mean': np.zeros(5), 'seq_std': np.ones(5)}
        full = infer_probabilities(model, df, norm)
        prefix = infer_probabilities(model, df.iloc[:190], norm)
        np.testing.assert_array_equal(full[:190], prefix)


class TestProtectionParity(unittest.TestCase):
    def test_ratio_drawdown_guard_matches_freqtrade_default(self):
        from freqtrade_integration.candle_policy import CandlePolicy
        from freqtrade_integration.candle_validation import replay_signals
        from freqtrade.data.metrics import calculate_max_drawdown
        n = 100
        df = pd.DataFrame({'date': pd.date_range('2026-01-01', periods=n, freq='1min', tz='UTC'),
                           'open': np.full(n, 100.), 'high': np.full(n, 100.),
                           'low': np.full(n, 98.), 'close': np.full(n, 100.), 'volume': np.ones(n)})
        policy = CandlePolicy(leverage=1, stoploss=-.0102, horizon=1, fee_entry=0, fee_exit=0,
                              fee_stop=0, slippage_bps=0, cooldown_candles=0, loss_guard_count=100)
        probs = np.tile([.01, .98, .01], (n, 1))
        trades = replay_signals(df, probs, np.arange(10), policy, {'long':.4, 'short':1.})
        self.assertEqual(len(trades), 5)
        measured = calculate_max_drawdown(pd.DataFrame({'close_date': df.date[:5], 'close_profit': [-.0102]*5}), value_col='close_profit')
        self.assertAlmostEqual(measured.drawdown_abs, .051)


class TestStandaloneTrainingCLI(unittest.TestCase):
    def test_cli_exports_validation_decision_required_by_strategy(self):
        import tempfile
        import subprocess
        import json
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            n = 700; close = 100 + np.sin(np.arange(n) / 9)
            pd.DataFrame({'date': pd.date_range('2026-01-01', periods=n, freq='1min', tz='UTC'),
                          'open': close, 'high': close+.1, 'low': close-.1, 'close': close,
                          'volume': np.arange(n)%13+1.}).to_csv(path/'candles.csv', index=False)
            run = subprocess.run([sys.executable, '-m', 'freqtrade_integration.train_candle_model',
                                  '--data', str(path/'candles.csv'), '--out', str(path/'model'),
                                  '--epochs', '1', '--hidden-dim', '8', '--horizon', '3'],
                                 cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertTrue((path/'model'/'decision.json').exists())
            decision = json.loads((path/'model'/'decision.json').read_text())
            self.assertFalse(decision['test_used'])


class TestCompleteDayCutoff(unittest.TestCase):
    def test_complete_last_day_is_kept_and_partial_last_day_is_removed(self):
        from freqtrade_integration.run_candle_validation import complete_utc_days
        dates = pd.date_range('2026-01-01', periods=2880, freq='1min', tz='UTC')
        full, cutoff = complete_utc_days(pd.DataFrame({'date': dates}))
        self.assertEqual(len(full), 2880)
        self.assertEqual(str(cutoff), '2026-01-03 00:00:00+00:00')
        partial, _ = complete_utc_days(pd.DataFrame({'date': dates[:2000]}))
        self.assertEqual(len(partial), 1440)


if __name__ == '__main__':
    unittest.main()
