"""Phase 2 : affinage par renforcement (PPO) du modele d'imitation.

Usage :
    python3 train_ppo.py --csv market_data_sample.csv --iterations 50

Charge les donnees, construit l'environnement ScalpingEnv (couts, TP/SL, filters
par actif), initialise la politique depuis le modele d'imitation pre-entraine
(best_scalping_model.pth) si present, puis lance PPO.
"""

import argparse

import numpy as np
import pandas as pd

from asset_filters import filters_for_asset
from pipeline_data import build_features
from ppo_agent import ActorCritic, PPOAgent
from dataset import ORDER_BOOK_DIM, SEQUENCE_DIM
from rl_environment import ScalpingEnv


def main():
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Entrainement PPO pour le scalping IA")
    parser.add_argument("--csv", default="market_data_sample.csv")
    parser.add_argument("--asset", default="BTC", choices=["BTC"])
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--steps-per-rollout", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=20)
    parser.add_argument("--imitation-weights", default="best_scalping_model.pth")
    parser.add_argument("--out", default="ppo_scalping_policy.pth")
    parser.add_argument("--news-blackouts", default=None, help="CSV start,end (ISO) par ligne")
    parser.add_argument("--reward-scale", type=float, default=0.001, help="Normalisation du reward (stabilise la value loss)")
    parser.add_argument("--execution", default="taker", choices=["taker", "maker"])
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = build_features(df)

    news_blackouts = None
    if args.news_blackouts:
        news_blackouts = []
        with open(args.news_blackouts) as f:
            for line in f:
                if line.strip():
                    s, e = line.strip().split(",")
                    news_blackouts.append((pd.to_datetime(s, utc=True), pd.to_datetime(e, utc=True)))

    filters = filters_for_asset(args.asset, news_blackouts=news_blackouts)
    env = ScalpingEnv(
        df,
        asset=args.asset,
        sequence_length=args.sequence_length,
        filters=filters,
        reward_scale=args.reward_scale,
        execution=args.execution,
    )

    import os
    import torch

    from model_pytorch import ScalpingMultimodalNet

    base_model = None
    if os.path.exists(args.imitation_weights):
        ckpt = torch.load(args.imitation_weights, map_location="cpu")
        base_model = ScalpingMultimodalNet(
            order_book_dim=ORDER_BOOK_DIM, sequence_dim=SEQUENCE_DIM, hidden_dim=args.hidden_dim
        )
        base_model.load_state_dict(ckpt["model_state_dict"])
        print(f"Politique initialisee depuis {args.imitation_weights} (epoque {ckpt.get('epoch', '?')})")
    else:
        print(f"Pas de poids d'imitation trouves ({args.imitation_weights}) : politique initialisee aleatoirement.")

    agent = PPOAgent(env, hidden_dim=args.hidden_dim, device="cpu")
    if base_model is not None:
        agent.policy = ActorCritic(
            ORDER_BOOK_DIM, SEQUENCE_DIM, hidden_dim=args.hidden_dim, base_model=base_model
        ).to(agent.device)
        agent.optimizer = torch.optim.Adam(agent.policy.parameters(), lr=3e-4)

    history = agent.train(
        n_iterations=args.iterations,
        n_steps_per_rollout=args.steps_per_rollout,
        verbose_every=max(1, args.iterations // 10),
    )
    agent.save(args.out)

    rewards = [h["mean_reward"] for h in history]
    print(f"Reward moyen : debut={np.mean(rewards[:3]):.4f} | fin={np.mean(rewards[-3:]):.4f}")
    print(f"Capital final de l'environnement : {env.capital:.2f} $")


if __name__ == "__main__":
    main()
