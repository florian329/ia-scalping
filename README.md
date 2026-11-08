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

## Modules (phase 3 — connecteurs et mise en production)

- `CMECSVFeed` (`data_feed.py`) — ticks NQ/ES au format CSV FirstRateData (DateTime,Price,Volume) ; carnet synthétisé (le flux L2 CME réel nécessite IBKRFeed ou MDP3).
- `IBKRFeed` (`data_feed.py`) — profondeur L2 CME via TWS/IB Gateway (`pip install ib_insync`, import paresseux).
- `OandaXAUFeed` (`data_feed.py`) — XAU/USD spot interbancaire via l'API REST OANDA v20 (stdlib uniquement, clé via `OANDA_API_KEY`, environnement practice par défaut).
- `run_live.py` — point d'entrée du paper trading : `python3 run_live.py --asset BTC --feed csv --csv market_data_sample.csv --model best_scalping_model.pth` ; charge le modèle (imitation ou PPO), applique les filtres, journalise, affiche la latence moyenne/p95/max.

## Modules (phase 4 — validation statistique et protections)

- `validation.py` — **walk-forward** (n fenêtres de test chronologiques), **bootstrap** (IC 95 % sur expectancy, win rate, profit factor), **sensibilité aux coûts** (slippage ×0,5 à ×3), **métriques par régime** (tendance × volatilité).
- `run_validation.py` — validation complète en une commande : `python3 run_validation.py --csv market_data_sample.csv --model best_scalping_model.pth` ; conclut explicitement si le bord est statistiquement significatif.
- **Protections** (backtester + boucle live) : quota de trades max (`max_trades`), **gel après N pertes consécutives** (`max_consecutive_losses` + `cooldown_ticks`), en plus du kill-switch perte journalière.
- **Labels pondérés par PnL** (`labeling.label_ticks(pnl_weights=True)`) : un trade annoté à +3 ticks pèse plus qu'un trade à +0,5 ; support des poids par échantillon dans `dataset` et `training_engine`.
- `MultiSegmentEnv` (`rl_environment.py`) : le PPO tourne sur plusieurs segments du train en rotation — pas de surapprentissage d'un seul scénario.

## Modules (phase 5 — performance, drift, données, calibration, tuning)

- **Backtest vectorisé** (`backtester.py`, `vectorized=True` par défaut) : l'inférence se fait en paquets au lieu d'un appel par tick — **gain mesuré ×41** (10,5 s → 0,26 s sur 20 000 ticks), résultat identique au mode séquentiel.
- **Détection de dérive** (`drift_monitor.py`) : `WinRateDriftMonitor` (win rate glissant vs référence backtest, gel automatique dans la boucle live) et `FeatureDriftMonitor` (test de Kolmogorov-Smirnov par feature — détecte le changement de régime avant les pertes).
- **Validation d'annotations** (`prepare_annotations.py`) : template CSV, contrôle des timestamps/côtés/prix/doublons, couverture temporelle contre les données de marché. `python3 prepare_annotations.py --ann mes_trades.csv --market-csv ticks.csv`.
- **Nouvelles features** (`pipeline_data.py`) : **OFI** (variation dynamique de l'OBI), **time-of-day** (encodage cyclique sin/cos), **VPIN roulant** (fraction du volume agresseur — toxicité du flux). Le modèle passe à 7 features carnet + 5 séquentielles.
- **Calibration** (`calibration.py`) : temperature scaling appris sur la validation — les seuils de confiance deviennent interprétables ; ECE pour mesurer la calibration.
- **Tuning** (`tune_hyperparameters.py`) : recherche d'hyperparamètres d'exécution (SL/TP ATR, timeout, seuil, fenêtre) avec Optuna si disponible, sinon recherche aléatoire — objectif = **expectancy out-of-sample**, pas la loss.
- **CI** (`.github/workflows/tests.yml`) : les 57 tests tournent à chaque push/PR ; `requirements.txt` verrouille les dépendances.

## Tests

```bash
python3 tests/test_backtester.py    # 9 tests : backtester + money management
python3 tests/test_improvements.py  # 13 tests : filtres, RL, PPO, feeds, journal
python3 tests/test_phase3.py       # 5 tests : CME CSV, OANDA, boucle live de bout en bout
python3 tests/test_phase4.py       # 14 tests : walk-forward, bootstrap, protections, labels pondérés, multi-segments
python3 tests/test_phase5.py       # 16 tests : vectorisation, drift, annotations, calibration, tuning, nouvelles features
```
57 tests au total, tous verts.

## Usage

```bash
pip install numpy pandas torch scikit-learn matplotlib
python3 tests/test_backtester.py   # tests unitaires
python3 run_smoke_test.py         # pipeline complet sur donnees synthetiques
```

## Avertissements connus

- `CMECSVFeed` synthétise la profondeur du carnet (FirstRateData ne fournit pas le L2) :
  pour de l'order flow CME réel en profondeur, utiliser `IBKRFeed`.
- La phase PPO est validée mécaniquement (27 tests verts, entraînement stable avec
  `reward_scale`) mais n'a été démontrée que sur données synthétiques.
- Le smoke test produit un résultat négatif : c'est attendu. Les données sont un random
  walk sans signal, et ~950 $ de frais sur 212 trades illustrent pourquoi le scalping
  exige un bord brut très élevé avant coûts. Ne pas extrapoler sur données réelles.
- Le trading comporte des risques élevés ; valider sur paper trading ≥ 4 semaines
  avant toute mise en réel.
