# Entraînement BTC : cinq corrections autorisées

Objectif : améliorer la fiabilité du signal et mesurer la rentabilité nette sans passer le bot en réel. L’utilisateur a demandé l’exécution des cinq corrections dans l’ordre après leur explication ; poursuivre sans nouvel entretien de conception.

1. Découpage chronologique purgé selon l’horizon des labels ; normalisation ajustée sur train seulement. Période test réservée au verdict.
2. Restaurer le meilleur checkpoint avant export ; rendre son chemin explicite et conserver sa traçabilité.
3. Vraies barrières profit/stop/temps depuis l’ouverture de la bougie suivant la décision ; traitement pessimiste des collisions et exclusion des dernières lignes sans futur complet.
4. Politique d’exécution partagée (frais, levier, ROI, stop, durée) et mêmes features entre entraînement et stratégie ; compatibilité des anciens checkpoints. Les modèles candidats peuvent utiliser une VWAP glissante causale pour éviter la dépendance au début du dataframe.
5. Calibration et seuils séparés long/short choisis uniquement sur validation, désactivation d’un côté sans preuve suffisante, mesures nettes hors échantillon par direction, stress frais et intervalles d’incertitude.

Approche choisie : conserver PyTorch/FreqTrade et corriger leur interface. Pas d’ajout de PPO, de nouveau courtier ou d’architecture concurrente. Le moteur FreqTrade reste juge de l’exécution ; le simulateur de labels documente ses approximations (fills maker non garantis, funding non prédit).

Validation : tests de non-fuite, meilleur checkpoint, trajectoires intrabougie, cohérence train/inférence, fenêtres causales, évaluation directionnelle ; entraînement de candidats dans un répertoire distinct ; validation chronologique sur données Binance futures réelles, backtest FreqTrade avec protections puis lookahead/recursive-analysis lorsque le nombre de trades permet une conclusion. Aucun ordre forcé, aucune écriture dans la base du bot, aucun écrasement du modèle actif. Rapporter explicitement résultats négatifs, manque de trades ou limites.
