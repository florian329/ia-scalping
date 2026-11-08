"""Validation statistique complète : walk-forward, bootstrap, stress coûts, régimes.

Usage :
    python3 run_validation.py --csv market_data_sample.csv --model best_scalping_model.pth
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch

from asset_filters import filters_for_asset
from backtester import run_backtest
from data_generator import generate_sample_data
from pipeline_data import build_features
from run_live import load_model
from validation import (
    bootstrap_metrics,
    metrics_by_regime,
    print_validation_summary,
    run_cost_sensitivity,
    walk_forward_backtest,
)


def main():
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Validation statistique de la strategie")
    parser.add_argument("--csv", default="market_data_sample.csv")
    parser.add_argument("--asset", default="BTC", choices=["BTC"])
    parser.add_argument("--model", default="best_scalping_model.pth")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--hidden-dim", type=int, default=64)
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"{args.csv} introuvable : generation de donnees synthetiques.")
        generate_sample_data(args.csv, args.csv.replace(".csv", "_ann.csv").replace("_ann", "_annotations") if False else "annotations_sample.csv")
    if not os.path.exists(args.model):
        raise SystemExit(
            f"Modele introuvable : {args.model}. Lancez run_smoke_test.py d'abord."
        )

    df = pd.read_csv(args.csv)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="ISO8601")
    df = build_features(df)
    model = load_model(args.model, hidden_dim=args.hidden_dim)

    from dataset import BOOK_FEATURES

    book = df[BOOK_FEATURES].to_numpy(dtype=np.float32)
    norm_stats = (book.mean(axis=0), book.std(axis=0))
    filters = filters_for_asset(args.asset)

    print("[1/4] Walk-forward...")
    reports, summary = walk_forward_backtest(
        model, df, asset=args.asset, n_folds=args.n_folds, norm_stats=norm_stats, filters=filters
    )

    print("[2/4] Bootstrap sur les trades agreges...")
    all_trades = [tr for r in reports for tr in r["trades"]]
    boot = bootstrap_metrics(all_trades, n_bootstrap=args.n_bootstrap)

    print("[3/4] Sensibilite aux couts...")
    test_window = df.iloc[int(len(df) * 0.6) :].reset_index(drop=True)
    cost_results = run_cost_sensitivity(
        model, test_window, asset=args.asset, norm_stats=norm_stats, filters=filters
    )

    print("[4/4] Metriques par regime de marche...")
    full_report = run_backtest(
        model, test_window, asset=args.asset, norm_stats=norm_stats, filters=filters
    )
    regimes = metrics_by_regime(full_report, test_window)

    print_validation_summary(summary, boot=boot, cost_results=cost_results, regimes=regimes)

    if boot["n_trades"] > 0 and boot["expectancy"][0] is not None:
        if boot["expectancy"][0] > 0:
            print("Conclusion : expectancy positive au seuil de confiance -> bord statistiquement solide.")
        elif boot["expectancy"][1] > 0:
            print("Conclusion : IC de l'expectancy chevauche 0 -> bord NON significatif statistiquement.")
        else:
            print("Conclusion : expectancy negative au seuil de confiance -> strategie non viable.")
    else:
        print("Conclusion : pas assez de trades pour conclure.")


if __name__ == "__main__":
    main()
