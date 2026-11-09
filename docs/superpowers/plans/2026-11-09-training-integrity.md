# Training integrity Implementation Plan

> For agentic workers: use superpowers:executing-plans. User explicitly requested executing these five corrections in order; no additional design interview.

Goal: correct and validate the five BTC training defects while keeping the active bot in simulation.
Architecture: shared candle training/evaluation and execution-policy helpers, isolated artifacts, backward-compatible strategy loader.
Spec: docs/superpowers/specs/2026-11-09-training-integrity-design.md

Global constraints: BTC only, dry_run true, no active-model overwrite, train-only statistics, purged boundaries, untouched test for threshold selection, no secrets in git.
Review focus: future values affecting train statistics; labels crossing split boundaries; stop/TP hit in one candle; final epoch worse than checkpoint; threshold chosen from test; confidence insufficient for one direction; dataframe index offsets; recursive features depending on missing history.

- [ ] Task 1: write failing tests for purged train/val/test indices and train-only normalizer; implement preprocessing in candle_training.py and expose the training CLI's split metadata.
- [ ] Task 2: test optimizer iterations with worsening validation; restore best checkpoint in training_engine.py and give candidate training its own checkpoint path.
- [ ] Task 3: write trajectory tests long/short, gaps, simultaneous barriers and unavailable future; implement candle_policy.py and true labels in train_candle_model.py.
- [ ] Task 4: share features and policy with IaScalpingStrategy, preserve legacy artifacts, version feature definition, match timeout/SL/ROI/fees/leverage; test batch-vs-prefix and non-zero indices, run all existing tests.
- [ ] Task 5: fit calibration and directional thresholds on validation only; implement candle_validation.py and CLI for purged walk-forward, bootstrap/cost stress and JSON/CSV results; validate on real futures candles and FreqTrade without the active bot database.

Execution uses the managed isolated worktree and Python 3.12 venv. Original Python 3.9 baseline fails for unsupported type syntax and obsolete FreqTrade; run baseline again in the compatible isolated runtime. Independent whole-change review after implementation; fix important findings before delivery. No automatic promotion of a candidate whose out-of-sample net results are unproven.
