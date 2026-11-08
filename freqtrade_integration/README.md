# Intégration FreqTrade + FreqUI

Le pipeline ia-scalping branché sur le **moteur FreqTrade** avec l'interface **FreqUI** :
le modèle PyTorch (ScalpingMultimodalNet) devient une stratégie FreqTrade, FreqUI fournit
le dashboard (trades, courbe de profit, positions ouvertes, logs, contrôles).

## Contenu

| Fichier | Rôle |
|---|---|
| `IaScalpingStrategy.py` | Stratégie FreqTrade (interface v3, shorts, ordres limites maker) branchant le modèle |
| `train_candle_model.py` | Entraîne le modèle sur des candles OHLCV (labellisation triple-barrière > frais) |
| `config_ia_scalping.json` | Config Binance futures BTC/USDT:USDT, dry-run, API FreqUI sur 127.0.0.1:8080 |
| `model/` | Checkpoint + stats de normalisation (généré par l'entraînement) |

## Adaptation tick → candle (à lire)

FreqTrade travaille en **candles OHLCV** : il ne fournit pas le carnet d'ordres L2.
Les features order-flow du pipeline tick (OBI, OFI, VPIN) sont remplacées par des
proxies de volume ; le reste (log-returns, VWAP dev, ATR, hour-of-day) est identique.
Le pipeline tick reste l'outil de recherche/validation du signal ; FreqTrade est la
couche d'exécution et de monitoring.

## Installation et usage

```bash
# 1. Dépendances
pip install -r requirements.txt
pip install freqtrade

# 2. Données (Binance futures BTC/USDT:USDT, 1m — public, sans clé)
freqtrade create-userdir --userdir user_data
freqtrade download-data --config freqtrade_integration/config_ia_scalping.json \
    --userdir user_data --timeframes 1m --timerange 20261101-

# 3. Entraîner le modèle candle
python freqtrade_integration/train_candle_model.py \
    --data user_data/data/binance/futures/BTC_USDT_USDT-1m-futures.feather \
    --epochs 12

# 4. Backtest FreqTrade avec la stratégie
freqtrade backtesting --config freqtrade_integration/config_ia_scalping.json \
    --userdir user_data --strategy-path freqtrade_integration \
    --strategy IaScalpingStrategy --timerange 20261105-

# 5. Paper trading + FreqUI (dashboard)
freqtrade install-ui --userdir user_data
freqtrade trade --config freqtrade_integration/config_ia_scalping.json \
    --userdir user_data --strategy-path freqtrade_integration \
    --strategy IaScalpingStrategy
# puis ouvrir http://127.0.0.1:8080
```

⚠️ **Sécurité** : l'API serveur écoute sur `127.0.0.1` uniquement (localhost, aucun accès
externe) et `dry_run: true` (aucun ordre réel). Changez `password` dans la config avant
toute utilisation, et ne passez `dry_run` à `false` qu'après des semaines de validation.

## Référence mesurée (validation sandbox, 2026-11-08)

- Données : Binance futures BTC/USDT:USDT 1m réelles (7 637 candles d'entraînement,
  backtest sur 3 jours).
- Modèle : 12 epochs, hidden 32, labellisation horizon 15 candles / seuil 3× frais.
- Backtest FreqTrade : **76 trades**, profit factor 0,71, expectancy −0,31 — la chaîne
  complète fonctionne ; la rentabilité dépend du signal (annotations réelles, plus de
  données), pas de l'intégration.

## Hyperopt avec holdout strict (`hyperopt_holdout.py`)

Le piège classique de l'hyperopt : optimiser sur tout l'historique puis s'étonner
que le marché « ait changé ». Le script impose la discipline en 4 phases :

```bash
python freqtrade_integration/hyperopt_holdout.py \
    --train 20261105-20261107 --holdout 20261107-20261108 --epochs 30
```

1. **Hyperopt** sur la période train uniquement (seuil d'entrée, espace `entry`) ;
2. **Export** des meilleurs paramètres vers `user_data/strategies/IaScalpingStrategy.json` ;
3. **Backtest holdout** sur une période jamais vue par l'hyperopt ;
4. **Verdict** : train vs holdout — surajustement détecté = paramètres inutilisables.

Validation sandbox (2026-11-08) : train +1,64 % (36 trades) vs holdout −0,04 % (1 trade)
→ verdict « SURAJUSTEMENT ». Le système vous protège de vous-même.

## Protections intégrées (stratégie)

- **CooldownPeriod** (5 candles) : pause après chaque trade ;
- **StoplossGuard** (4 stops en 60 candles → pause 30 candles) ;
- **MaxDrawdown** (5 % de drawdown en 240 candles → pause 60 candles).

## Alertes Telegram (optionnel)

1. Créez un bot via [@BotFather](https://t.me/botfather) → récupérez le `token` ;
2. Récupérez votre `chat_id` via [@userinfobot](https://t.me/userinfobot) ;
3. Dans `config_ia_scalping.json` :
```json
"telegram": { "enabled": true, "token": "VOTRE_TOKEN", "chat_id": "VOTRE_CHAT_ID" }
```
Vous recevrez chaque trade, chaque anomalie, et pourrez envoyer `/stop` au bot.

## Licences

FreqTrade et FreqUI sont sous **GPL-3.0** : toute distribution combinant ce code avec
FreqTrade devra être sous licence compatible. Usage personnel : sans contrainte.
