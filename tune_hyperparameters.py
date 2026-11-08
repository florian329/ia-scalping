"""Recherche d'hyperparametres, objectif = expectancy out-of-sample.

Utilise Optuna si disponible (pip install optuna), sinon bascule
automatiquement sur une recherche aleatoire avec la meme interface.
Les hyperparametres tunes sont ceux de l'EXECUTION (SL/TP ATR, timeout,
seuil de confiance, fenetre de sequence) — pas ceux du reseau : minimiser
la loss n'est pas maximiser l'expectancy.

Usage :
    python3 tune_hyperparameters.py --csv market_data_sample.csv --model best_scalping_model.pth --trials 30
"""

import argparse

import numpy as np
import pandas as pd
import torch

from backtester import run_backtest
from pipeline_data import build_features
from run_live import load_model
from validation import walk_forward_backtest

SEARCH_SPACE = {
    "atr_multiplier_sl": (1.0, 2.5),
    "atr_multiplier_tp": (1.0, 3.5),
    "max_holding_ticks": (30, 200),
    "confidence_threshold": (0.3, 0.7),
    "sequence_length": (10, 40),
}

DEFAULTS = {
    "atr_multiplier_sl": 1.5,
    "atr_multiplier_tp": 2.0,
    "max_holding_ticks": 100,
    "confidence_threshold": 0.4,
    "sequence_length": 20,
}


def sample_params(rng):
    return {
        "atr_multiplier_sl": float(rng.uniform(*SEARCH_SPACE["atr_multiplier_sl"])),
        "atr_multiplier_tp": float(rng.uniform(*SEARCH_SPACE["atr_multiplier_tp"])),
        "max_holding_ticks": int(rng.integers(*SEARCH_SPACE["max_holding_ticks"])),
        "confidence_threshold": float(rng.uniform(*SEARCH_SPACE["confidence_threshold"])),
        "sequence_length": int(rng.integers(*SEARCH_SPACE["sequence_length"])),
    }


def evaluate_params(model, df, params, n_folds=3, norm_stats=None, filters=None):
    """Objectif : expectancy moyenne sur les fenetres de validation walk-forward."""
    _, summary = walk_forward_backtest(
        model,
        df,
        n_folds=n_folds,
        norm_stats=norm_stats,
        filters=filters,
        confidence_threshold=params["confidence_threshold"],
        sequence_length=params["sequence_length"],
        atr_multiplier_sl=params["atr_multiplier_sl"],
        atr_multiplier_tp=params["atr_multiplier_tp"],
        max_holding_ticks=params["max_holding_ticks"],
    )
    expectancies = [e for e in summary["expectancies"] if np.isfinite(e)]
    if not expectancies:
        return -1e9, summary
    return float(np.mean(expectancies)), summary


def random_search(model, df, n_trials=20, seed=42, norm_stats=None, filters=None, verbose=True):
    rng = np.random.default_rng(seed)
    best = {"score": -1e18, "params": None, "summary": None}
    history = []
    for trial in range(1, n_trials + 1):
        params = sample_params(rng)
        score, summary = evaluate_params(model, df, params, norm_stats=norm_stats, filters=filters)
        history.append({"trial": trial, "score": score, **params})
        if score > best["score"]:
            best = {"score": score, "params": params, "summary": summary}
        if verbose:
            print(
                f"Essai {trial:>3d}/{n_trials} | expectancy moyenne={score:>9.2f} $ "
                f"| sl={params['atr_multiplier_sl']:.2f} tp={params['atr_multiplier_tp']:.2f} "
                f"| conf={params['confidence_threshold']:.2f}"
            )
    return best, history


def optuna_search(model, df, n_trials=20, seed=42, norm_stats=None, filters=None, verbose=True):
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    history = []

    def objective(trial):
        params = {
            "atr_multiplier_sl": trial.suggest_float("atr_multiplier_sl", *SEARCH_SPACE["atr_multiplier_sl"]),
            "atr_multiplier_tp": trial.suggest_float("atr_multiplier_tp", *SEARCH_SPACE["atr_multiplier_tp"]),
            "max_holding_ticks": trial.suggest_int("max_holding_ticks", *SEARCH_SPACE["max_holding_ticks"]),
            "confidence_threshold": trial.suggest_float("confidence_threshold", *SEARCH_SPACE["confidence_threshold"]),
            "sequence_length": trial.suggest_int("sequence_length", *SEARCH_SPACE["sequence_length"]),
        }
        score, summary = evaluate_params(model, df, params, norm_stats=norm_stats, filters=filters)
        history.append({"trial": trial.number, "score": score, **params})
        if verbose:
            print(f"Essai {trial.number + 1:>3d}/{n_trials} | expectancy moyenne={score:>9.2f} $")
        return score

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)
    return {"score": study.best_value, "params": study.best_params, "summary": None}, history


def main():
    parser = argparse.ArgumentParser(description="Tuning des hyperparametres d'execution")
    parser.add_argument("--csv", default="market_data_sample.csv")
    parser.add_argument("--asset", default="BTC")
    parser.add_argument("--model", default="best_scalping_model.pth")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-dim", type=int, default=64)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = build_features(df)
    model = load_model(args.model, hidden_dim=args.hidden_dim)

    from dataset import BOOK_FEATURES

    book = df[BOOK_FEATURES].to_numpy(dtype=np.float32)
    norm_stats = (book.mean(axis=0), book.std(axis=0))

    from asset_filters import filters_for_asset

    filters = filters_for_asset(args.asset)

    try:
        import optuna  # noqa: F401

        print("Moteur : Optuna (TPE)")
        best, history = optuna_search(
            model, df, n_trials=args.trials, seed=args.seed, norm_stats=norm_stats, filters=filters
        )
    except ImportError:
        print("Optuna non installe : recherche aleatoire (pip install optuna pour le TPE)")
        best, history = random_search(
            model, df, n_trials=args.trials, seed=args.seed, norm_stats=norm_stats, filters=filters
        )

    print("=" * 62)
    print(f"Meilleur score (expectancy moyenne out-of-sample) : {best['score']:.2f} $")
    print("Meilleurs hyperparametres :")
    for k, v in best["params"].items():
        print(f"  {k:<22} = {v}")
    print("A comparer aux valeurs par defaut :")
    for k, v in DEFAULTS.items():
        print(f"  {k:<22} = {v}")


if __name__ == "__main__":
    main()
