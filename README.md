# IA Scalping Multimodal — version corrigée

Pipeline complet de scalping IA (BTC, NQ, ES, XAU) : carnet d'ordres + price action,
modèle hybride MLP + LSTM, apprentissage par imitation, backtester réaliste,
money management basé risque, boucle live paper trading.

## Correctifs par rapport à la version initiale

| Problème initial | Correction |
|---|---|
| Backtester : position fermée au tick suivant, sans TP/SL | TP/SL basés ATR + timeout (`backtester.py`) |
| P&L en unités de prix, taille de position jamais utilisée | Quantité issue du money manager, P&L = qty × point_value × Δprix |
| Coûts fixes identiques pour tous les actifs | `ASSET_COST_MODELS` par actif (spread, commission, slippage) |
| Look-ahead (features de t pour trader à t) | Décision à t basée sur les features ≤ t-1 |
| Aucune métrique | Sharpe, max drawdown, profit factor, expectancy, win rate, frais |
| Labels jamais définis | `labeling.py` : labellisation depuis le CSV d'annotations, anti-fuite |
| Split aléatoire impossible à contrôler | `dataset.py` : split strictement chronologique 70/15/15 |
| `BatchNorm1d` dangereux en prod (batch=1) | `LayerNorm` sur la branche carnet |
| Poids de classes arbitraires [1, 3.5, 3.5] | Poids calculés depuis la distribution réelle (racine inv. fréq., plafonnés) |
| Money manager sans valeur du point | NQ=20 $/pt, ES=50 $/pt, XAU=100 $/lot, plafond qty, stop min |
| Live trading = stub vide | `live_paper_trading.py` : feed injectable, TP/SL, kill-switch, latence |
| Features promises absentes (VWAP, ATR, spread, momentum 10/30/60) | Toutes implémentées dans `pipeline_data.py` |

## Modules (phase 1 — corrections)

- `pipeline_data.py` — OBI multi-profondeur, spread, log-returns 10/30/60, momentum signé, écart VWAP, ATR tick.
- `labeling.py` — labels 0/1/2 depuis les annotations, fenêtre optionnelle, poids de classes.
- `dataset.py` — fenêtres glissantes anti look-ahead, split chronologique, loaders.
- `model_pytorch.py` — réseau hybride MLP + LSTM, inférence single-tick.
- `training_engine.py` — CrossEntropy pondérée, AdamW, checkpoint, early stopping.
- `evaluation_metrics.py` — rapport de classification, matrice de confusion.
- `money_management.py` — sizing par risque avec valeur du point par actif.
- `backtester.py` — simulation avec coûts réalistes, TP/SL ATR, métriques complètes.
- `live_paper_trading.py` — boucle temps réel avec feed injectable et kill-switch.
- `data_generator.py` — données synthétiques de test (ticks + carnet 5 niveaux + annotations).
- `run_smoke_test.py` — pipeline de bout en bout.

## Modules (phase 2 — améliorations)

- `asset_filters.py` — filtres par actif de la spec : OBI > 0.85 pour ES, blackout news pour XAU (fenêtres configurables), anti-spoofing BTC (profondeur carnet plafonnée à 10 niveaux). Intégrés au backtester et à la boucle live.
- `rl_environment.py` — environnement RL (interface reset/step type Gym, sans dépendance) : TP/SL ATR, coûts, pénalité de temps de rétention et de drawdown conformes à la fonction de reward de la spec.
- `ppo_agent.py` — PPO compact en PyTorch pur (clip surrogate, GAE, entropy bonus), politique initialisable depuis les poids d'imitation.
- `train_ppo.py` — script de la phase 2 : `python3 train_ppo.py --csv market_data_sample.csv --iterations 50`.
- `data_feed.py` — connecteurs : `CSVReplayFeed`, `ListFeed`, `ReconnectingFeed` (backoff exponentiel), `CCXTFeed` (sandbox, import paresseux), `TardisCSVFeed` (carnet L2 + trades agrégés).
- `live_paper_trading.py` (durci) — journalisation CSV des trades (`TradeJournal`), statistiques de latence moyenne/p95/max (`LatencyMonitor`), filtres par actif, support des feeds avec reconnexion.
- `tests/test_improvements.py` — 13 tests supplémentaires (22 au total).

## Tests

```bash
python3 tests/test_backtester.py    # 9 tests : backtester + money management
python3 tests/test_improvements.py  # 13 tests : filtres, RL, PPO, feeds, journal
```

## Usage

```bash
pip install numpy pandas torch scikit-learn matplotlib
python3 tests/test_backtester.py   # tests unitaires
python3 run_smoke_test.py         # pipeline complet sur donnees synthetiques
```

## Avertissements connus

- Le connecteur CME/IBKR (NQ/ES) et le flux XAU interbancaire restent à brancher
  (interface `data_feed.py` prête ; `TardisCSVFeed` couvre déjà BTC).
- La phase PPO est validée mécaniquement (22 tests verts, entraînement stable avec
  `reward_scale`) mais n'a été démontrée que sur données synthétiques.
- Le smoke test produit un résultat négatif : c'est attendu. Les données sont un random
  walk sans signal, et ~950 $ de frais sur 212 trades illustrent pourquoi le scalping
  exige un bord brut très élevé avant coûts. Ne pas extrapoler sur données réelles.
- Le trading comporte des risques élevés ; valider sur paper trading ≥ 4 semaines
  avant toute mise en réel.
