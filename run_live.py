"""Point d'entree du paper trading live.

Exemples :
    python3 run_live.py --asset BTC --feed csv --csv market_data_sample.csv
    python3 run_live.py --asset BTC --feed ccxt --exchange binance
    python3 run_live.py --asset NQ --feed cme --csv nq_ticks.csv
    python3 run_live.py --asset XAU --feed oanda --news-blackouts blackouts.csv

Charge le modele (imitation ou PPO), applique les filtres par actif, journalise
les trades et affiche les statistiques de latence en fin de session.
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch

from asset_filters import filters_for_asset
from dataset import BOOK_FEATURES, ORDER_BOOK_DIM, SEQUENCE_DIM
from data_feed import CSVReplayFeed, CCXTFeed
from live_paper_trading import LivePaperTradingLoop
from model_pytorch import ScalpingMultimodalNet
from pipeline_data import build_features


def load_model(path, hidden_dim=64, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt.get("policy_state_dict"))
    model = ScalpingMultimodalNet(order_book_dim=ORDER_BOOK_DIM, sequence_dim=SEQUENCE_DIM, hidden_dim=hidden_dim)
    backbone = state
    prefix = "backbone."
    if any(k.startswith(prefix) for k in state):
        backbone = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    model.load_state_dict(backbone)
    model.eval()
    return model.to(device)


def make_feed(args):
    if args.feed == "csv":
        return CSVReplayFeed(args.csv)
    if args.feed == "ccxt":
        return CCXTFeed(exchange_id=args.exchange, symbol=args.symbol, sandbox=True)
    raise ValueError(f"Feed inconnu : {args.feed}")


def load_norm_stats(df_path):
    df = pd.read_csv(df_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    feats = build_features(df)
    book = feats[BOOK_FEATURES].to_numpy(dtype=np.float32)
    return book.mean(axis=0), book.std(axis=0)


def main():
    parser = argparse.ArgumentParser(description="Live paper trading IA scalping")
    parser.add_argument("--asset", default="BTC", choices=["BTC"])
    parser.add_argument("--feed", default="csv", choices=["csv", "ccxt"])
    parser.add_argument("--csv", default="market_data_sample.csv")
    parser.add_argument("--exchange", default="binance")
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--model", default="best_scalping_model.pth")
    parser.add_argument("--norm-csv", default=None, help="CSV de reference pour la normalisation des features carnet")
    parser.add_argument("--capital", type=float, default=10000.0)
    parser.add_argument("--max-ticks", type=int, default=None)
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--journal", default="trades_journal.csv")
    parser.add_argument("--news-blackouts", default=None, help="CSV start,end (ISO) par ligne")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise SystemExit(
            f"Modele introuvable : {args.model}. Lancez run_smoke_test.py ou train_ppo.py d'abord."
        )
    model = load_model(args.model)
    feed = make_feed(args)

    news_blackouts = None
    if args.news_blackouts:
        news_blackouts = []
        with open(args.news_blackouts) as f:
            for line in f:
                if line.strip():
                    s, e = line.strip().split(",")
                    news_blackouts.append((pd.to_datetime(s, utc=True), pd.to_datetime(e, utc=True)))

    filters = filters_for_asset(args.asset, news_blackouts=news_blackouts)
    norm_stats = load_norm_stats(args.norm_csv or args.csv) if args.feed == "csv" else None

    loop = LivePaperTradingLoop(
        model,
        asset=args.asset,
        feed=feed,
        device="cpu",
        initial_capital=args.capital,
        confidence_threshold=args.confidence,
        norm_stats=norm_stats,
        filters=filters,
        journal_path=args.journal,
        poll_seconds=0.0 if args.feed == "csv" else 0.2,
    )
    result = loop.run(max_ticks=args.max_ticks)

    print(f"Capital final      : {result['capital']:.2f} $")
    print(f"Nombre de trades   : {result['n_trades']}")
    lat = result.get("latency", {})
    print(f"Latence inference  : mean={lat.get('mean_ms')} ms | p95={lat.get('p95_ms')} ms | max={lat.get('max_ms')} ms")
    if "reconnect_count" in result:
        print(f"Reconnexions feed  : {result['reconnect_count']}")
    print(f"Journal des trades : {args.journal}")


if __name__ == "__main__":
    main()
