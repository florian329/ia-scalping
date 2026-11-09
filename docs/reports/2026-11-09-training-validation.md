# Résultat des cinq corrections — 9 novembre 2026

Les cinq corrections sont implémentées dans la branche de travail ; aucun modèle
ni paramètre du bot actif n'a été remplacé. Le travail et les simulations ont
été exécutés dans un environnement local isolé. Aucun ordre broker n'a été envoyé.

## Vérifications techniques

- 103 tests passent, moteur FreqTrade inclus ; pas de test d'intégration ignoré
  pour absence du moteur dans cet environnement.
- Meilleure époque restaurée avant export, testé avec une validation qui se
  dégrade après la première époque.
- Invariance de la normalisation train sous modification des données futures ;
  purge des labels aux frontières et exclusion de la queue non observable.
- Tests de barrières long/short, collision intrabougie, gap, sortie à échéance,
  unités temporelles ns/us et égalité entraînement/inférence avec index décalé.
- Test réel du calcul MaxDrawdown de FreqTrade : replay aligné sur le mode ratios.
- CLI autonome et runner exportent les décisions ajustées sur validation seule.
- `lookahead-analysis` final : 20 signaux examinés, has_bias=False, aucune entrée,
  sortie ou colonne signalée. Une première analyse avait détecté un écart numérique
  de probabilités (~3e-8) lié à la taille du dernier lot ; il a été reproduit puis
  corrigé par des lots d'inférence de taille fixe, sans arrondir les probabilités.
- `recursive-analysis` final : aucune variance des indicateurs et aucun biais
  indicateur détectés pour 80, 100, 160 et 320 bougies d'amorçage.

Les analyses de causalité ont utilisé un exemplaire distinct de diagnostic à
seuil fixe 0,34 et température 1 pour obtenir assez de trades. Elles ne prouvent
ni la rentabilité ni l'activité du candidat aux seuils sélectionnés.

## Validation économique sur données réelles

99 360 bougies Binance futures BTC/USDT 1m, du 1 septembre au 8 novembre 2026
inclus (journées UTC complètes). Trois fenêtres : 21 jours d'entraînement,
7 jours de validation, 7 jours de test ; réentraînement à chaque fenêtre et
périodes de test successives non superposées. Calibration et sélection des seuils
utilisent des sous-périodes distinctes de validation, purgées. Les statistiques
et les checkpoints sont propres à chaque fenêtre.

Profil déclaré : levier simulé x10, frais conservateurs uniformes de 0,05 % par
jambe, slippage défavorable de 1 point de base dans les labels/replays ; stop,
ROI, timeout et protections versionnés. FreqTrade intègre le financement selon
les données téléchargées ; les replays de labels ne prédisent pas le funding
ni la file d'attente maker. Ce profil ne certifie pas les frais du compte actif.

| Fenêtre | Test UTC | Seuil long / short | Trades FreqTrade | Rendement FreqTrade |
|---|---|---|---:|---:|
| 1 | 2026-10-19 → 2026-10-25 | 1.0 / 1.0 | 0 | 0.00 % |
| 2 | 2026-10-26 → 2026-11-01 | 1.0 / 1.0 | 0 | 0.00 % |
| 3 | 2026-11-02 → 2026-11-08 | 1.0 / 1.0 | 0 | 0.00 % |

Les deux seuils valent 1 : les deux directions sont désactivées dans ces
**candidats seulement**. La validation n'a pas établi de côté avec assez de
trades et une borne basse positive de l'expectancy après frais. Résultat nul
par abstention : ce n'est pas une preuve de rentabilité.

Référence de diagnostic du même modèle corrigé, non calibré, à seuil fixe 0,4 :
28 trades en replay OHLC, rendement composé -23,49 %, profit factor 0,44 ;
IC 95 % par bootstrap de blocs de l'expectancy incluant zéro. Cette référence
n'est pas le modèle déployé et ne mesure pas une amélioration par rapport au VPS.

**Verdict : rentabilité non démontrée ; aucun candidat promu.** Les corrections
rendent les tests plus fiables, elles ne créent pas à elles seules un avantage
prédictif. Une nouvelle période indépendante et du paper trading restent requis
avant une conclusion économique positive. Les corrections de validation ont été
redéroulées sur les mêmes données : ces résultats sont de recherche, pas une
certification finale indépendante.

## Reproduction et artefacts

Voir `freqtrade_integration/VALIDATION.md`. Les modèles, données, exports détaillés,
configurations offline et journaux résident dans `outputs/training-validation-20261109/`
du worktree, exclus de Git. Les verdicts retenus sont ceux de `walk-forward-final`,
`lookahead-final.csv`, `recursive-final.log` et `final-tests.log` ; les premières
exécutions et erreurs de configuration offline sont conservées pour traçabilité.
Le runner vérifie désormais les résultats exportés : un retour CLI à zéro ne
suffit pas à déclarer un backtest réussi.
