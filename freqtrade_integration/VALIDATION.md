# Validation du modèle BTC corrigé

Ce parcours crée des **candidats séparés**. Il n'écrit ni dans le modèle ni dans
la base du bot actif et ne lance jamais `freqtrade trade`. Utiliser Python 3.12,
les dépendances de `requirements.txt` et `freqtrade==2026.10` pour reproduire
l'environnement vérifié. Les données/modèles/configurations de sortie restent
hors Git ; ne jamais copier les secrets du VPS dans une configuration de test.

## Les cinq corrections

1. **Temps et normalisation** : statistiques calculées sur le train uniquement,
   séparation chronologique, purge d'un horizon complet aux frontières. Les
   dernières lignes sans futur complet ont le label -1 et sont exclues. Les
   séquences n'utilisent que les bougies déjà closes.
2. **Meilleur checkpoint** : l'état ayant la meilleure perte de validation est
   restauré avant export. Le checkpoint est propre à chaque candidat ; un
   répertoire de sortie déjà présent est refusé.
3. **Trois barrières réelles** : décision à la clôture, entrée à l'ouverture
   suivante, TP/stop évalués sur high/low, stop prioritaire en cas de collision,
   prise en compte d'un gap défavorable et sortie à l'échéance. Les labels
   comparent les deux directions après frais. Ils ne prouvent pas un fill maker.
4. **Même définition entre apprentissage et stratégie** : features versionnées,
   normalisation des deux branches, politique de levier/stop/ROI/durée et
   protections exportées avec le modèle. La VWAP version 2 est glissante sur
   60 bougies, et le momentum volume est relatif. L'inférence utilise une taille
   de lot fixe complétée avec des zéros : les probabilités ne dépendent plus de
   la longueur future du dataframe par des variations numériques de calcul.
5. **Validation par direction** : température ajustée sur la première moitié de
   la validation, puis seuils long/short sur sa seconde moitié, séparée par purge.
   Un côté est désactivé (seuil 1) s'il n'atteint pas le nombre de trades requis
   et une borne basse bootstrap par blocs positive. Le test n'est pas utilisé
   pour choisir ces valeurs. Les rapports séparent longs/shorts, frais, drawdown
   et incertitude ; aucune promotion automatique n'est effectuée.

Les anciens checkpoints sans version conservent leurs features cumulatives et
leurs sorties sur changement de signal. Les candidats v2 utilisent les sorties
TP/stop/timeout définies par la politique, et non les anciens exits à chaque
changement de classe. Un candidat sans `decision.json` est refusé. Le CLI
standalone et le runner complet produisent tous deux ce fichier à partir de
la validation seulement.

## Commande complète

Télécharger les bougies futures BTC 1m, ainsi que mark/funding nécessaires à
FreqTrade, dans un dossier séparé :

```bash
python -m freqtrade create-userdir --userdir outputs/recherche/user_data
python -m freqtrade download-data \
  --config freqtrade_integration/config_ia_scalping.json \
  --userdir outputs/recherche/user_data --datadir outputs/recherche/data \
  --timeframes 1m --timerange 20260901-20261109

python -m freqtrade_integration.run_candle_validation \
  --data outputs/recherche/data/futures/BTC_USDT_USDT-1m-futures.feather \
  --out outputs/recherche/nouveau-run \
  --train-days 21 --validation-days 7 --test-days 7 --folds 3 \
  --epochs 6 --hidden-dim 32 --fee 0.0005 \
  --freqtrade-data-dir outputs/recherche/data
```

`--fee` fixe un taux uniforme par jambe pour que les labels, le replay et le
backtest FreqTrade partagent la même hypothèse. 0.0005 est un profil conservateur
à cinq points de base par jambe, **pas une attestation du tarif de votre compte**.
Les étiquettes/replays ajoutent aussi un slippage défavorable paramétré ;
FreqTrade ne reconstruit pas la file maker. Le funding est inclus par le moteur
FreqTrade lorsqu'il dispose de son historique, pas par le replay OHLC.
Le verdict économique doit être confirmé avec les frais de la place réellement
utilisée et une nouvelle période indépendante, puis du paper trading.

Chaque fenêtre réentraîne son modèle sur le passé. Les fenêtres de test sont
successives et non superposées ; une ancienne fenêtre peut ensuite devenir du
passé accessible à une fenêtre ultérieure. Les journées UTC partielles sont
exclues. Un fichier de bougies non contigu ou incohérent est refusé.

Sorties : `summary.json`, `fold-*/report.json`, `training_metadata.json`,
`decision.json`, modèles/checkpoints et CSV de trades, puis résultats/logs du
moteur FreqTrade. Un code de sortie zéro ne suffit pas : le runner vérifie le
résultat structuré ou le résumé explicite « No trades made » du moteur.

Pour entraîner un candidat seul :

```bash
python -m freqtrade_integration.train_candle_model --data candles.feather \
  --out outputs/candidat-neuf --epochs 6 --leverage 10
```

Ce CLI permet des taux entrée/sortie/stop distincts et exporte la calibration
sur validation. Fournir un dossier inexistant. Pour charger un candidat dans
un **backtest isolé**, utiliser `IA_SCALPING_MODEL_DIR=/chemin/absolu/candidat`
ou la clé `ia_model_dir` d'une configuration distincte. Ne pas remplacer le
modèle du service actif simplement pour lancer la validation.

## Tests de causalité et limites

Les commandes `lookahead-analysis` et `recursive-analysis` du moteur vérifient
la causalité et la stabilité des indicateurs. Si le candidat désactive les deux
côtés, l'analyse de lookahead ne dispose pas de trades suffisants. Un exemplaire
**de diagnostic**, identifié comme tel et sans effet sur le service actif,
peut utiliser des seuils fixes pour éprouver le chemin d'exécution ; son PnL ne
valide pas les seuils sélectionnés et ne doit pas servir à promouvoir le modèle.

Sept jours positifs ou quelques trades ne démontrent pas la rentabilité. Un
résultat nul dû à zéro trade prouve seulement que la sélection s'abstient.
Les exports ne remplacent pas une période finale indépendante ni les contraintes
réelles de liquidité, latence, financement et exécution.
