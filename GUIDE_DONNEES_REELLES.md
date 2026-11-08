# Guide — Passage aux données réelles

## 1. Collecter des ticks réels (BTC, sans clé API)

```bash
python3 binance_collector.py --duration 600 --out btc_real_ticks.csv
```
Collecte le carnet d'ordres Binance (20 niveaux) + les trades agrégés, au format
du pipeline. Pour un entraînement sérieux : viser plusieurs heures
(`--duration 3600`), idéalement sur plusieurs sessions différentes.

## 2. Préparer vos annotations (trades issus de vos vidéos)

```bash
python3 prepare_annotations.py --template --out annotations_template.csv
```

Format attendu (une ligne par trade) :

| timestamp (ISO 8601 UTC) | asset | side | entry_price | exit_price | volume |
|---|---|---|---|---|---|
| 2026-11-08T14:07:05.441Z | BTC | SELL | 76420.50 | 76395.00 | 1.5 |

Règles importantes :
- **timestamps en UTC** (convertir depuis l'heure de votre plateforme de trading) ;
- le timestamp doit être celui de l'**entrée** du trade ;
- minimum recommandé : **30-50 trades par actif** pour l'apprentissage par imitation ;
- valider avant entraînement :
  ```bash
  python3 prepare_annotations.py --ann mes_trades.csv --market-csv btc_real_ticks.csv
  ```

## 3. Entraîner et valider

```bash
python3 run_smoke_test.py                      # pipeline complet avec vos donnees
python3 train_ppo.py --csv btc_real_ticks.csv --iterations 100
python3 run_validation.py --csv btc_real_ticks.csv --model best_scalping_model.pth
```

## 4. Leçon apprise sur données réelles (échantillon du 2026-11-08)

Sur 3 minutes de BTC/Binance avec les frais spot réels (0,1 % taker) :
- l'agent PPO **apprend de lui-même à ne pas trader** (politique → Attente) ;
- le backtest de contrôle : 9 trades, 0 % de win rate, **178 $ de frais** pour
  10 000 $ de capital, expectancy IC 95 % entièrement négative.

**Interprétation** : chaque aller-retour coûte ~0,2 % du notionnel en frais,
alors que le mouvement moyen par tick est ~0,0001 %. Le scalping tick-à-tick
de BTC spot aux frais standard n'est **pas viable** — ce n'est pas un défaut
du modèle, c'est l'arithmétique. Options réalistes :
1. **Ordres maker** (limit) au lieu de taker : implémenté (`execution="maker"`) —
   frais ÷3 mesurés sur données réelles, adverse selection modélisée ;
2. **Frais réduits** : Binance Futures (0,02-0,05 %) ou statut VIP ;
3. **Horizon plus long** : viser des mouvements ≥ 10× les frais aller-retour ;
4. **Frais négociés** : viser des frais réduits (statut VIP, paiement en BNB)
   pour abaisser le seuil de viabilité.

Le pipeline est conçu pour répondre honnêtement à cette question par actif :
c'est exactement ce que font `run_validation.py` (walk-forward + bootstrap)
et `run_cost_sensitivity` (seuil de rupture en slippage).
