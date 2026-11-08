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

## Modules

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
- `tests/test_backtester.py` — 9 tests unitaires.

## Usage

```bash
pip install numpy pandas torch scikit-learn matplotlib
python3 tests/test_backtester.py   # tests unitaires
python3 run_smoke_test.py         # pipeline complet sur donnees synthetiques
```

## Avertissements connus

- La phase RL/PPO de la spec n'est pas implémentée (imitation seule).
- Les seuils par actif du PDF (OBI > 0.85 pour ES, blackout news XAU, filtre spoofing BTC)
  ne sont pas encore câblés dans la boucle live.
- Le smoke test produit un résultat négatif : c'est attendu. Les données sont un random
  walk sans signal, et ~950 $ de frais sur 212 trades illustrent pourquoi le scalping
  exige un bord brut très élevé avant coûts. Ne pas extrapoler sur données réelles.
- Le trading comporte des risques élevés ; valider sur paper trading ≥ 4 semaines
  avant toute mise en réel.
