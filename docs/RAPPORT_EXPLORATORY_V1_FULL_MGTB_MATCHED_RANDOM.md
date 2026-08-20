# Rapport exploratoire — MGT-B complet, contrôle aléatoire et Vanilla sur MATH-500

## Résumé

Cette expérience exploratoire compare la génération Vanilla, le contrôleur MGT-B complet (`full_mgtb`) et un contrôle d'intervention aléatoire (`matched_random`) sur les 500 problèmes de MATH-500. Avec une seule réplication, `full_mgtb` obtient une exactitude de **56,2 %**, contre **55,2 %** pour Vanilla et **55,0 %** pour `matched_random`. Le gain apparié de MGT-B sur Vanilla est de **+1,0 point de pourcentage** (14 corrections, 9 régressions), mais son intervalle de confiance bootstrap à 95 % inclut zéro (−0,8 à +2,8 points) et le test de McNemar n'est pas significatif après correction de Holm (*p* = 0,810). Ces résultats constituent donc un signal encourageant, pas une démonstration d'efficacité.

## Question expérimentale

L'expérience cherche à déterminer si des retours arrière déclenchés par le moniteur MGT-B améliorent la résolution mathématique, et si un éventuel gain provient du ciblage des interventions plutôt que du seul fait de régénérer une partie de la trajectoire.

## Méthodologie

- **Jeu de données :** les 500 exemples du split `test` de `HuggingFaceH4/MATH-500`, révision `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be`.
- **Modèle :** `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`, révision `ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562`, quantifié INT4 (`fp4`, calcul `float16`).
- **Génération :** prompt MATH-500 chain-of-thought, maximum de 20 000 nouveaux tokens, une réplication de graine 0.
- **Vanilla :** génération sans alarme ni retour arrière.
- **`full_mgtb` :** score multivarié calculé sur des fenêtres de 64 tokens avec un pas de 32 ; calibration positionnelle et accumulation CUSUM avec remise à zéro. Le seuil sélectionné sur le développement est `h = 11,429`, pour un taux d'alarme sain observé de 4,84 % (cible 5 %). Une alarme déclenche un rollback adaptatif, puis un redécodage à température 0,6 avec pénalité de répétition 1,1 et blocage ciblé de n-grammes. Trois rerolls au maximum sont autorisés.
- **`matched_random` :** le même opérateur de réparation est déclenché à des positions aléatoires. Son profil de nombre d'interventions et de longueurs de rollback est appris sur 100 exemples de développement exécutés avec `full_mgtb` (activation 15 %, surcoût moyen 163,2 tokens), indépendamment des labels du test.
- **Évaluation :** exactitude de la réponse finale, extraction, troncature et comptabilité des tokens. Les comparaisons sont appariées par problème. L'incertitude sur la différence d'exactitude est estimée par 10 000 bootstrap appariés (graine 20260811) ; un test exact de McNemar mesure l'asymétrie corrections/régressions, avec correction de Holm pour les deux comparaisons à Vanilla.
- **Reproductibilité :** configuration `configs/science_campaign/math500_all500_exploratory_ablations.yaml`, commit d'exécution `0fdde7f94619cd9f5fb2923cee33c64bc62202c5`, manifeste gelé `b0302d2b75bcc3980f459abb260e8de8d1671d294b160acbeecfe2c2ea5a9baf` et freeze `d26ba748646beab47be49c3270dc069640c9c649d1423fe54ab03426950f5482`.

## Résultats principaux

| Méthode | Corrects | Exactitude | Écart à Vanilla | IC 95 % de l'écart | Corrections / régressions | *p* McNemar | *p* Holm |
|---|---:|---:|---:|---:|---:|---:|---:|
| Vanilla | 276/500 | 55,2 % | — | — | — | — | — |
| `full_mgtb` | 281/500 | **56,2 %** | **+1,0 pt** | [−0,8 ; +2,8] pts | 14 / 9 | 0,405 | 0,810 |
| `matched_random` | 275/500 | 55,0 % | −0,2 pt | [−2,0 ; +1,4] pts | 9 / 10 | 1,000 | 1,000 |

La comparaison exploratoire directe entre `full_mgtb` et `matched_random` donne **+1,2 point** en faveur de MGT-B (22 corrections, 16 régressions ; IC 95 % [−1,2 ; +3,6] points ; *p* de McNemar = 0,418). Elle n'est pas incluse dans la famille confirmatoire déclarée et reste non significative.

Parmi les 107 problèmes où MGT-B intervient, l'exactitude passe de 34/107 pour la trajectoire Vanilla appariée à 39/107 après intervention : le gain net global de cinq réponses correctes est donc entièrement concentré dans le sous-ensemble ciblé. Sur les 393 problèmes sans alarme, les tokens MGT-B et Vanilla sont identiques dans 100 % des cas, ce qui confirme l'absence de dérive lorsque le contrôleur reste inactif.

## Coût et comportement du contrôleur

| Méthode | Problèmes avec intervention | Tokens échantillonnés moyens | Écart de tokens à Vanilla | Tokens supprimés | Troncature | Extraction réussie |
|---|---:|---:|---:|---:|---:|---:|
| Vanilla | 0/500 (0 %) | 4 332,5 | — | 0 | 0,2 % | 99,6 % |
| `full_mgtb` | 107/500 (21,4 %) | 4 519,4 | +4,3 % | 102 816 | 0,6 % | 99,8 % |
| `matched_random` | 55/500 (11,0 %) | 4 270,5 | −1,4 % | 34 464 | 0,4 % | 99,6 % |

MGT-B déclenche une seule intervention sur 106 problèmes et deux interventions sur un problème. Son surcoût d'échantillonnage est de 93 405 tokens sur l'ensemble du benchmark. La longueur émise totale est néanmoins légèrement inférieure à Vanilla (−0,4 %), car les trajectoires réparées peuvent terminer plus tôt après suppression et redécodage.

## Interprétation et limites

Le résultat est compatible avec un petit bénéfice du ciblage MGT-B : le gain ponctuel est positif face à Vanilla et au contrôle aléatoire, les corrections excèdent les régressions, et les trajectoires sans alarme restent strictement inchangées. L'incertitude demeure toutefois assez large pour inclure une légère dégradation comme un gain utile ; aucune comparaison n'atteint la significativité statistique.

Le contrôle `matched_random` est **apparié ex ante, mais pas ex post**. Son profil provient du développement, où l'activation de MGT-B était de 15 %, puis seulement 11 % des problèmes de test ont effectivement reçu une intervention aléatoire, contre 21,4 % pour `full_mgtb`. Les coûts d'intervention observés ne sont donc pas équivalents sur le test. La comparaison MGT-B–aléatoire mélange encore qualité du ciblage et quantité effective d'intervention ; elle ne permet pas à elle seule d'attribuer causalement le gain au détecteur.

Enfin, l'étude utilise une seule graine, un seul modèle quantifié, un seul benchmark déjà entièrement consommé et un statut explicitement exploratoire. Les analyses par matière sont descriptives et trop petites pour soutenir des conclusions par sous-domaine. Les temps muraux ont été collectés avec six workers parallèles et ne constituent pas un benchmark propre de latence.

## Conclusion

Sur MATH-500, `full_mgtb` améliore ponctuellement l'exactitude de 55,2 % à 56,2 % pour un surcoût d'échantillonnage de 4,3 %. Le contrôle aléatoire atteint 55,0 %, ce qui va dans le sens d'un ciblage utile, mais ni le gain à Vanilla ni l'écart au contrôle aléatoire ne sont statistiquement concluants. La prochaine expérience doit rétablir un contrôle aléatoire strictement apparié au budget d'intervention observé et tester le protocole gelé sur des données nouvelles, avec plusieurs graines.
