# Rapport intermédiaire — ablations MGT-B sur MATH-500

**Date du snapshot :** 30 août 2026, 23:27 (Europe/Paris)  
**Campagne :** `math500_all500_exploratory_v1`  
**Statut scientifique :** exploratoire, non confirmatoire  
**Jeu de test :** les 500 problèmes de `HuggingFaceH4/MATH-500`, une seed (`0`)

## Résumé exécutif

Onze variantes disposent de 500/500 artefacts authentifiés et appariés. Sur les 500 problèmes, Vanilla obtient **55,2 %** (276/500), MGT-B complet **56,2 %** (281/500), et la baseline `self_consistency_5` atteint **70,0 %** (350/500). Son gain de +14,8 points contre Vanilla est statistiquement net (IC bootstrap [11,6 ; 18,0], McNemar brut `p ≈ 2,04 × 10⁻²⁰`) mais coûte environ cinq fois plus de tokens qu'une génération simple. Parmi MGT-B et ses ablations, la meilleure valeur brute reste celle du rollback fixe : **56,8 %** (284/500). Aucun gain d'une variante MGT-B contre Vanilla ne résiste à la correction de multiplicité de Holm ; seule la dégradation de la pénalité de répétition isolée reste significative.

La nouvelle ablation la plus tranchée est `repair_repetition_penalty`. Utilisée seule, la pénalité de répétition obtient **52,2 %** (261/500), soit **−3,0 points contre Vanilla** et **−4,0 points contre MGT-B complet**. Les différences appariées restent significatives après correction de Holm : `p` ajusté ≈ **0,0367** contre Vanilla et ≈ **0,000287** contre MGT-B. Ce résultat indique que la pénalité de répétition n'est pas seulement insuffisante isolément : dans cette configuration, elle dégrade nettement les réponses lorsqu'elle n'est pas accompagnée des autres contraintes du repair.

À l'inverse, `repair_ngram_blocking` obtient **55,4 %** (277/500), soit +0,2 point contre Vanilla et −0,8 point contre MGT-B complet. Ses intervalles incluent zéro et les tests appariés ne montrent pas de différence nette. Le blocage de n-grammes seul conserve donc l'essentiel de l'accuracy, sans reproduire le gain numérique du système complet.

L'ablation la plus informative sur le mécanisme est `repair_rollback_only`. Elle obtient **54,0 %**, soit **−2,2 points par rapport à MGT-B complet**. La comparaison appariée donne 6 corrections contre 17 régressions, un IC bootstrap brut à 95 % de [−4,0 ; −0,4] points et un McNemar exact brut `p = 0,0347`. Cet effet ne reste pas significatif après correction sur les comparaisons d'ablation ; il constitue donc un signal exploratoire, pas une preuve définitive. Il suggère que revenir en arrière sans modifier la politique de redécodage ne suffit pas et peut dégrader les réponses.

Le rollback fixe de 1 024 tokens atteint **56,8 %**, contre 56,2 % pour le rollback adaptatif de MGT-B. La différence directe est faible (+0,6 point ; IC [−1,0 ; +2,2]) : ces données ne démontrent pas l'utilité de la localisation adaptative du point de changement.

Le monitor « répétition seule » atteint **56,0 %** avec seulement 35 problèmes alertés. Face à Vanilla, il corrige 4 réponses et n'en dégrade aucune (`p` exact brut = 0,125). C'est une piste intéressante et parcimonieuse, mais fondée sur quatre paires discordantes seulement.

Enfin, `matched_random` obtient **55,0 %**, proche de Vanilla et inférieur de 1,2 point à MGT-B complet. La direction est compatible avec un monitor informatif, mais l'incertitude est grande. De plus, le random n'intervient que sur 11,0 % des problèmes, contre 21,4 % pour MGT-B : le matching réalisé n'égalise donc pas le taux d'intervention observé sur ce test.

## État des expériences et intégrité

La vérification a relu les artefacts avec `RunStore.valid_artifact`, donc elle contrôle leur identité de run et leurs hashes de contenu ; les nombres ci-dessous ne sont pas de simples comptages de fichiers.

| Variante | Artefacts valides | État au snapshot |
|---|---:|---|
| `vanilla` | 500/500 | complet |
| `full_mgtb` | 500/500 | complet |
| `matched_random` | 500/500 | complet |
| `monitor_repetition_only` | 500/500 | complet |
| `monitor_no_reset` | 500/500 | complet |
| `repair_fixed_rollback` | 500/500 | complet |
| `repair_rollback_only` | 500/500 | complet |
| `repair_temperature` | 500/500 | complet |
| `repair_repetition_penalty` | 500/500 | complet |
| `monitor_entropy_only` | 0/500 | non lancé ; calibration non exploitable |
| `monitor_entropy_repetition` | 0/500 | non lancé ; calibration non exploitable |
| `monitor_global_calibration` | 0/500 | non lancé ; calibration non exploitable |
| `repair_ngram_blocking` | 500/500 | complet |
| `self_consistency_5` | 500/500 | complet |
| `best_of_5_logprob` | 0/500 | non lancé |

Un ancien dossier `monitor_no_reset.kernel-687.39.partial` contient 90 items. Il s'agit d'un reliquat partiel distinct ; les résultats rapportés utilisent exclusivement le run final `monitor_no_reset`, validé à 500/500.

Le fichier `analysis/campaign_results.json` présent au moment du snapshot est **obsolète** : il ne contient que Vanilla, MGT-B complet et matched-random. Les tableaux de ce rapport ont été recalculés directement depuis les onze runs complets authentifiés, sans modifier les artefacts.

## Résultats principaux

| Méthode | Corrects | Accuracy | Écart vs Vanilla | Alertes / rerolls | Tokens échantillonnés moyens | Troncatures |
|---|---:|---:|---:|---:|---:|---:|
| Vanilla | 276/500 | 55,2 % | — | 0 (0,0 %) | 4 332,5 | 1 (0,2 %) |
| MGT-B complet | 281/500 | 56,2 % | +1,0 pt | 107 (21,4 %) | 4 519,4 | 3 (0,6 %) |
| Matched-random | 275/500 | 55,0 % | −0,2 pt | 55 (11,0 %) | 4 270,5 | 2 (0,4 %) |
| Monitor répétition seule | 280/500 | 56,0 % | +0,8 pt | 35 (7,0 %) | 4 321,4 | 2 (0,4 %) |
| Monitor sans reset | 281/500 | 56,2 % | +1,0 pt | 79 (15,8 %) | 4 276,5 | 0 (0,0 %) |
| Rollback fixe | 284/500 | **56,8 %** | **+1,6 pt** | 107 (21,4 %) | 4 552,1 | 6 (1,2 %) |
| Rollback seul | 270/500 | 54,0 % | −1,2 pt | 107 (21,4 %) | 4 629,0 | 1 (0,2 %) |
| Rollback + température | 274/500 | 54,8 % | −0,4 pt | 107 (21,4 %) | 5 032,5 | 18 (3,6 %) |
| Rollback + pénalité de répétition | 261/500 | **52,2 %** | **−3,0 pts** | 107 (21,4 %) | 4 743,3 | 10 (2,0 %) |
| Rollback + blocage de n-grammes | 277/500 | 55,4 % | +0,2 pt | 107 (21,4 %) | 4 665,4 | 1 (0,2 %) |
| Self-consistency (5) | 350/500 | **70,0 %** | **+14,8 pts** | 0 (0,0 %) | 21 867,7 | 1 (0,2 %) |

Les cinq ablations du repair complètes utilisent le même monitor et déclenchent donc sur les mêmes 107 problèmes. Cela rend leur comparaison plus directement causale que les comparaisons entre monitors.

### Comparaisons appariées à Vanilla

Les intervalles sont des IC percentile à 95 % issus de 10 000 bootstraps appariés par problème, seed bootstrap `20260811`. Les `p` indiqués sont les McNemar exacts bilatéraux **bruts**.

| Méthode | Différence | IC bootstrap 95 % | Vanilla faux → méthode vraie | Vanilla vraie → méthode fausse | `p` brut |
|---|---:|---:|---:|---:|---:|
| MGT-B complet | +1,0 pt | [−0,8 ; +2,8] | 14 | 9 | 0,4049 |
| Matched-random | −0,2 pt | [−2,0 ; +1,4] | 9 | 10 | 1,0000 |
| Répétition seule | +0,8 pt | [+0,2 ; +1,6] | 4 | 0 | 0,1250 |
| Sans reset | +1,0 pt | [−0,6 ; +2,8] | 12 | 7 | 0,3593 |
| Rollback fixe | +1,6 pt | [−0,2 ; +3,6] | 16 | 8 | 0,1516 |
| Rollback seul | −1,2 pt | [−2,8 ; +0,4] | 5 | 11 | 0,2101 |
| Rollback + température | −0,4 pt | [−2,4 ; +1,6] | 13 | 15 | 0,8506 |
| Rollback + pénalité de répétition | **−3,0 pts** | **[−5,0 ; −1,0]** | 5 | 20 | **0,00408** |
| Rollback + blocage de n-grammes | +0,2 pt | [−1,4 ; +2,0] | 10 | 9 | 1,0000 |
| Self-consistency (5) | **+14,8 pts** | **[+11,6 ; +18,0]** | 76 | 2 | **2,04 × 10⁻²⁰** |

Après correction de Holm sur ces dix comparaisons contre Vanilla, le gain de self-consistency (`p` ajusté ≈ **2,04 × 10⁻¹⁹**) et la dégradation de la pénalité de répétition (`p` ajusté ≈ **0,0367**) restent significatifs. Tous les autres `p` ajustés valent 1. L'IC bootstrap de « répétition seule » exclut zéro, mais le test exact repose sur seulement quatre paires discordantes et n'est pas significatif, même avant correction. Il faut retenir l'incertitude donnée par l'ensemble des diagnostics, pas sélectionner le seul intervalle favorable.

### Comparaisons directes à MGT-B complet

Ici, une « correction » signifie que MGT-B complet est faux et que l'ablation est vraie ; une « régression » signifie l'inverse.

| Ablation | Écart vs MGT-B | IC bootstrap 95 % | Corrections | Régressions | `p` brut |
|---|---:|---:|---:|---:|---:|
| Matched-random | −1,2 pt | [−3,6 ; +1,2] | 16 | 22 | 0,4177 |
| Répétition seule | −0,2 pt | [−2,0 ; +1,6] | 11 | 12 | 1,0000 |
| Sans reset | 0,0 pt | [−2,2 ; +2,2] | 15 | 15 | 1,0000 |
| Rollback fixe | +0,6 pt | [−1,0 ; +2,2] | 9 | 6 | 0,6072 |
| Rollback seul | **−2,2 pts** | **[−4,0 ; −0,4]** | 6 | 17 | **0,0347** |
| Rollback + température | −1,4 pt | [−3,0 ; +0,2] | 5 | 12 | 0,1435 |
| Rollback + pénalité de répétition | **−4,0 pts** | **[−6,0 ; −2,2]** | 2 | 22 | **0,0000359** |
| Rollback + blocage de n-grammes | −0,8 pt | [−2,8 ; +1,2] | 10 | 14 | 0,5413 |
| Self-consistency (5) | **+13,8 pts** | **[+10,4 ; +17,4]** | 79 | 10 | **1,87 × 10⁻¹⁴** |

Après correction de Holm sur ces neuf comparaisons directes, self-consistency reste nettement supérieur à MGT-B complet (`p` ajusté ≈ **1,69 × 10⁻¹³**) et la pénalité de répétition reste nettement inférieure (`p` ajusté ≈ **0,000287**). Le `p = 0,0347` du rollback seul devient environ `0,243` après ajustement : ce second signal reste suggestif, mais non concluant après multiplicité.

## Résultats par matière

| Méthode | Algèbre (124) | Comptage/proba. (38) | Géométrie (41) | Algèbre interm. (97) | Théorie nombres (62) | Préalgèbre (82) | Précalcul (56) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Vanilla | 69,4 % | 47,4 % | 51,2 % | 43,3 % | 59,7 % | 59,8 % | 41,1 % |
| MGT-B complet | 68,5 % | 47,4 % | 48,8 % | 46,4 % | 59,7 % | 62,2 % | 44,6 % |
| Matched-random | 69,4 % | 47,4 % | 46,3 % | 41,2 % | 64,5 % | 59,8 % | 41,1 % |
| Répétition seule | 70,2 % | 50,0 % | 51,2 % | 43,3 % | 59,7 % | 62,2 % | 41,1 % |
| Sans reset | 71,0 % | 47,4 % | 56,1 % | 46,4 % | 59,7 % | 56,1 % | 42,9 % |
| Rollback fixe | 69,4 % | 47,4 % | 51,2 % | 44,3 % | 61,3 % | 63,4 % | 46,4 % |
| Rollback seul | 70,2 % | 39,5 % | 48,8 % | 40,2 % | 58,1 % | 58,5 % | 44,6 % |
| Rollback + température | 71,0 % | 39,5 % | 48,8 % | 42,3 % | 58,1 % | 61,0 % | 42,9 % |
| Rollback + pénalité de répétition | 66,9 % | 36,8 % | 48,8 % | 39,2 % | 56,5 % | 58,5 % | 41,1 % |
| Rollback + blocage de n-grammes | 71,8 % | 50,0 % | 48,8 % | 41,2 % | 58,1 % | 59,8 % | 42,9 % |
| Self-consistency (5) | 88,7 % | 57,9 % | 63,4 % | 54,6 % | 77,4 % | 72,0 % | 57,1 % |

Ces sous-groupes n'ont pas été prévus comme tests principaux et certains sont petits. Ils servent à localiser les variations, pas à revendiquer des effets par matière.

## Ce que disent les ablations à ce stade

1. **Le simple fait de recommencer n'explique pas le gain.** Matched-random reste au niveau de Vanilla, tandis que MGT-B est orienté positivement. Mais le taux d'intervention aléatoire observé est deux fois plus faible, ce qui empêche une attribution propre et définitive.
2. **Le rollback seul semble insuffisant.** Avec exactement le même monitor, il est inférieur de 2,2 points à MGT-B complet. Les contraintes de redécodage semblent donc avoir une utilité collective.
3. **La pénalité de répétition seule est nuisible dans cette configuration.** Elle perd 4,0 points face à MGT-B, avec 2 corrections pour 22 régressions ; cet effet résiste à la correction de multiplicité. Elle produit aussi 10 troncatures, contre 3 pour MGT-B.
4. **Le blocage de n-grammes seul est proche de Vanilla.** Il atteint 55,4 % et n'est pas distinguable de MGT-B complet dans ce test apparié. Il semble moins risqué que la pénalité de répétition seule, mais son apport causal propre n'est pas établi.
5. **La température réduite seule n'explique pas le système complet.** Elle reste 1,4 point sous MGT-B et provoque beaucoup plus de terminaisons à la limite de tokens (18 contre 3), ainsi qu'un coût moyen de 5 032,5 tokens.
6. **L'avantage du rollback adaptatif n'est pas établi.** Le rollback fixe est numériquement meilleur, mais compatible statistiquement avec MGT-B complet.
7. **La répétition est un signal de monitor prometteur.** Elle reproduit presque l'accuracy du monitor complet avec 35 alertes au lieu de 107. Le nombre de changements de verdict est toutefois trop faible pour conclure.
8. **Le reset CUSUM change les cas, pas l'accuracy totale.** Sans reset et MGT-B finissent tous deux à 56,2 %, mais avec 15 corrections et 15 régressions entre eux. L'égalité des moyennes ne signifie donc pas que les méthodes sont équivalentes problème par problème.

## Calibrations disponibles

La cible de fausse alarme sur développement est 5 %, évaluée sur 62 trajectoires dites saines.

| Calibration | Mode | Seuil `h` | Taux sain obtenu | Exploitable pour un run prioritaire |
|---|---|---:|---:|---|
| Full | positionnelle, CUSUM reset | 11,4293 | 4,84 % (3/62) | oui |
| Répétition seule | positionnelle, CUSUM reset | 11,6623 | 4,84 % (3/62) | oui |
| Sans reset | positionnelle, accumulation sans reset | 6,0692 | 4,84 % (3/62) | oui |
| Entropie seule | positionnelle, CUSUM reset | 18,4207 | 100 % (62/62) | non |
| Entropie + répétition | positionnelle, CUSUM reset | 18,4207 | 100 % (62/62) | non |
| Full globale | globale, CUSUM reset | 18,4207 | 100 % (62/62) | non |

L'échec des trois dernières calibrations est déjà un résultat diagnostique : dans la grille et avec les traces actuelles, elles ne peuvent pas respecter la cible de fausse alarme. Les lancer telles quelles produirait une comparaison non équilibrée.

## Baseline complète : self-consistency à 5 échantillons

`self_consistency_5` génère cinq solutions indépendantes par problème, extrait leurs réponses finales, puis sélectionne la réponse majoritaire. Ses 500 artefacts sont complets et authentifiés.

- accuracy : 350/500 = **70,0 %** ;
- gain apparié contre Vanilla : **+14,8 points**, avec 76 corrections et 2 régressions ;
- gain apparié contre MGT-B complet : **+13,8 points**, avec 79 corrections et 10 régressions ;
- au moins un des cinq candidats est correct sur **79,0 %** des problèmes ; le vote majoritaire en récupère 70,0 %, laissant un écart de sélection de 9 points ;
- coût : **10 933 855 tokens échantillonnés**, soit 21 867,7 par problème et environ **4,84×** le coût total de MGT-B complet ;
- temps mural moyen cumulé des cinq générations : **1 257,5 s** par problème, soit environ 21 minutes ;
- extractabilité : 100 % ; une seule réponse sélectionnée atteint la limite de tokens.

Cette baseline montre qu'un budget massif d'échantillonnage supplémentaire améliore fortement l'accuracy du modèle. Elle ne réfute pas nécessairement l'intérêt d'une intervention ciblée, mais déplace la revendication pertinente : sur ce modèle et ce dataset, MGT-B doit surtout être évalué comme méthode d'efficacité coût–performance, pas comme moyen d'atteindre l'accuracy maximale sans contrainte de calcul.

## Coûts et comportement de génération

| Méthode | Tokens échantillonnés totaux | Tokens supprimés | Surcoût moyen vs Vanilla | Temps mural moyen observé |
|---|---:|---:|---:|---:|
| Vanilla | 2 166 272 | 0 | — | 233,9 s |
| MGT-B complet | 2 259 677 | 102 816 | +4,3 % | 229,2 s |
| Matched-random | 2 135 227 | 34 464 | −1,4 % | 218,6 s |
| Répétition seule | 2 160 684 | 43 712 | −0,3 % | 222,2 s |
| Sans reset | 2 138 256 | 34 720 | −1,3 % | 224,6 s |
| Rollback fixe | 2 276 060 | 98 048 | +5,1 % | 234,1 s |
| Rollback seul | 2 314 509 | 180 576 | +6,8 % | 197,5 s |
| Rollback + température | 2 516 274 | 102 976 | +16,2 % | 235,5 s |
| Rollback + pénalité de répétition | 2 371 653 | 174 784 | +9,5 % | 257,9 s |
| Rollback + blocage de n-grammes | 2 332 678 | 195 296 | +7,7 % | 239,9 s |
| Self-consistency (5) | 10 933 855 | 0 | +404,7 % | 1 257,5 s |

Les temps muraux proviennent de runs exécutés à des moments différents ; ils incluent les conditions de charge de la machine et ne constituent pas un benchmark contrôlé. Les comptes de tokens sont plus fiables pour comparer le coût algorithmique.

## Provenance nécessaire à la reproduction

- Configuration : `configs/science_campaign/math500_all500_exploratory_ablations.yaml`
- Modèle : `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
- Révision modèle : `ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562`
- Quantification : INT4 FP4, double quantification désactivée, calcul FP16
- Génération : prompt `math500_cot`, maximum 20 000 nouveaux tokens
- Fenêtres du monitor : taille 64, stride 32
- Révision MATH-500 : `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be`
- Seed de réplication : `0`
- Commit Git gelé : `0fdde7f94619cd9f5fb2923cee33c64bc62202c5`
- Hash de l'arbre source gelé : `4990cc9e6031000e358905366082846cc6d186f2a30235053e9aa33539d096c2`
- Hash du manifeste : `b0302d2b75bcc3980f459abb260e8de8d1671d294b160acbeecfe2c2ea5a9baf`
- Hash de la configuration de campagne : `fc961f7af6e917c6be368ed133cd9d521bd050d29a35d433dfa234194f4be1f3`
- Hash du freeze : `d26ba748646beab47be49c3270dc069640c9c649d1423fe54ab03426950f5482`
- Environnement gelé : Python 3.10.20, PyTorch 2.13.0+cu130, Transformers 4.57.6, Datasets 5.0.1, bitsandbytes 0.50.1, CUDA 13.0, NVIDIA RTX A5000

Avant l'ajout du présent rapport, le `HEAD` du dépôt était encore le commit gelé et `git status --short` était vide. Le rapport lui-même apparaît désormais comme nouveau fichier non suivi tant qu'il n'est pas ajouté à Git.

## Limites et prochaines décisions

- La campagne est explicitement **exploratoire** : MATH-500 a déjà été observé lors de travaux antérieurs. Les résultats doivent être confirmés sur un test indépendant gelé avant toute revendication forte.
- Une seule seed est utilisée. Le protocole est apparié, mais il ne mesure pas la variabilité entre générations.
- Dix comparaisons complètes sont déjà effectuées contre Vanilla ; le gain de `self_consistency_5` et la dégradation de `repair_repetition_penalty` sont significatifs après correction de Holm.
- Les cinq ablations du repair sont terminées. Elles montrent surtout un effet négatif robuste de la pénalité de répétition isolée ; ni le rollback adaptatif ni les autres contraintes isolées ne démontrent un gain propre sur ce snapshot.
- Il reste à exécuter `best_of_5_logprob` pour déterminer si la confiance moyenne du modèle sélectionne mieux ou moins bien les cinq candidats que le vote majoritaire.
- Les monitors entropie seule, entropie + répétition et calibration globale nécessitent une nouvelle stratégie de calibration ou doivent être rapportés comme échecs de calibration, pas lancés tels quels.
- Une baseline random mieux appariée au **taux effectif** d'intervention de MGT-B serait souhaitable avant de conclure sur la valeur informationnelle du monitor.

## Fichiers de référence encore présents au snapshot

- `outputs/science_campaign/math500_all500_exploratory_v1/freeze/campaign.lock.json`
- `outputs/science_campaign/math500_all500_exploratory_v1/splits/manifest.json`
- `outputs/science_campaign/math500_all500_exploratory_v1/calibration/*/{calibrator,threshold,reference_summary}.json`
- `outputs/science_campaign/math500_all500_exploratory_v1/runs/test/<variante>/`
- `outputs/science_campaign/math500_exploratory_v1/profiles/full_mgtb.json`

Ce rapport est volontairement autonome : les résultats agrégés, statistiques appariées, calibrations, états de complétion et identifiants de provenance nécessaires à leur interprétation y sont conservés même si les artefacts volumineux sont perdus.
