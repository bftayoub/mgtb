# Résultats MATH-500 INT4 — MGT-B avec `log_threshold=10`

## Cadre expérimental

- **Modèle :** `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
- **Précision :** INT4
- **Benchmark :** MATH-500, split `test`
- **Comparaison appariée :** décodage `vanilla` contre `mgtb_v3_window`
- **Longueur maximale :** 20 000 nouveaux tokens
- **Scoring :** égalité exacte après normalisation de la réponse mathématique
- **Artefacts de calibration :** `outputs/calibration/math500_n100/int4/`
- **Seuil testé :** `log_threshold=10`, soit `threshold=exp(10)=22 026,47`

Le seuil 10 est une ablation manuelle utilisant le calibrateur INT4 gelé. Sur les 34 trajectoires saines de calibration, il correspond approximativement au 91,2e percentile des maxima de `logE` et produit 3/34 alertes, soit un taux empirique de 8,8 %. Le seuil automatiquement calibré était plus conservateur : `log_threshold=10,963`, avec 1/34 alerte saine (2,9 %).

## Résultats par run

| Run | N | Vanilla | MGT-B | Gain | Corrections | Régressions |
|---|---:|---:|---:|---:|---:|---:|
| Seed 1 | 200 | 29,5 % | 35,5 % | **+6,0 pts** | 13 | 1 |
| Seed 2 | 100 | 29,0 % | 31,0 % | **+2,0 pts** | 5 | 3 |
| Seed 3 | 200 | 35,0 % | 39,0 % | **+4,0 pts** | 14 | 6 |
| Holdout `unseen109` | 109 | 31,19 % | 34,86 % | **+3,67 pts** | 6 | 2 |

Le holdout contient les 109 problèmes qui n'étaient apparus dans aucun artefact antérieur de calibration ou d'évaluation. Il confirme la direction de l'effet, mais n'est pas significatif seul : test exact de McNemar `p=0,289`.

## Résultat principal dédupliqué

Pour éviter de compter plusieurs fois les mêmes problèmes entre les trois premières seeds, une seule trajectoire est conservée par ancien problème (plus petite seed disponible), puis les 109 problèmes du holdout sont ajoutés.

| Métrique | Vanilla | MGT-B |
|---|---:|---:|
| Problèmes uniques | 467 | 467 |
| Réponses correctes | 146 | **167** |
| Accuracy | 31,26 % | **35,76 %** |
| Gain absolu | — | **+4,50 points** |
| Gain relatif | — | **+14,4 %** |

- **Corrections / régressions :** 29 / 8, soit un ratio de 3,63:1
- **Test exact de McNemar :** `p=0,000753`
- **IC 95 % bootstrap apparié du gain :** environ `[+2,14 ; +7,07]` points

En analyse secondaire, les 609 trajectoires appariées donnent 31,53 % pour vanilla et 35,80 % pour MGT-B (`+4,27` points ; 38 corrections contre 12 régressions). Cette agrégation contient des problèmes répétés et ne doit donc pas être présentée comme 609 observations indépendantes.

## Activation et coût

Sur les 609 paires, MGT-B s'active dans 205 cas (33,66 %). Sur ces cas, l'accuracy passe de 19,02 % avec vanilla à 31,71 % avec MGT-B. Sur les 404 cas sans alerte, les completions vanilla et MGT-B sont identiques.

| Coût moyen sur 609 paires | Vanilla | MGT-B | Variation |
|---|---:|---:|---:|
| Latence | 81,45 s | 100,26 s | **+23,1 %** |
| Tokens conservés | 5 099,95 | 4 887,32 | −4,2 % |
| Événements de génération | 5 099,95 | 5 133,28 | +0,65 % |
| Tokens rééchantillonnés | 0 | 245,96 | +245,96 |

## Paramètres MGT-B et signification

### Fenêtres et score

| Paramètre | Valeur | Signification |
|---|---:|---|
| `window_size` | 64 | Nombre de tokens dans chaque fenêtre analysée. |
| `stride` | 32 | Décalage entre deux fenêtres ; les fenêtres se recouvrent de 32 tokens. |
| `ngram_min`, `ngram_max` | 6, 8 | Tailles des n-grams utilisées pour détecter les répétitions. |
| `exclude_prompt_ngrams` | `true` | Les répétitions déjà présentes dans le prompt ne sont pas imputées à la génération. |

Le score brut d'une fenêtre est :

```text
0.15 × entropie moyenne
+ 0.10 × (− log-probabilité moyenne)
+ 0.20 × taux de répétition
+ 0.35 × confident-loop score
+ 0.18 × hausse locale d'entropie
+ 0.02 × baisse locale d'entropie
```

Le calibrateur ECDF positionnel transforme ce score en p-value en fonction de la position de la fenêtre dans la génération.

### Détecteur séquentiel

| Paramètre | Valeur | Signification |
|---|---:|---|
| `log_threshold` | 10 | Une alerte est émise lorsque l'évidence cumulée vérifie `logE >= 10`. |
| `betting_gammas` | 0.1, 0.3, 0.5, 0.7 | Mélange de fonctions de pari transformant chaque p-value en facteur d'évidence. |
| `p_clip` | `1e-6` | Borne inférieure appliquée aux p-values pour éviter une évidence numérique infinie. |
| `refractory_windows` | 2 | Après un backtracking, deux fenêtres sont ignorées par le détecteur. |

`target_false_alert_rate=0.05` est le paramètre utilisé par la procédure automatique de calibration. Il ne remplace pas le seuil manuel chargé pour cette expérience ; le taux sain observé à `log_threshold=10` est 8,8 %.

### Backtracking et redécodage

| Paramètre | Valeur | Signification |
|---|---:|---|
| `max_rerolls` | 3 | Maximum de trois backtrackings appliqués par génération. |
| `use_adaptive_changepoint` | `true` | Le rollback est placé à partir du dernier retour de `logE` à zéro. |
| `margin_tokens` | 64 | Le rollback est étendu de 64 tokens avant le changement détecté. |
| `fixed_rollback_tokens` | `null` | Aucune longueur fixe : la position adaptative est utilisée. |
| `redecode_temperature` | 0.6 | Température utilisée après une intervention, contre 1.0 avant celle-ci. |
| `repetition_penalty` | 1.1 | Pénalise les tokens déjà générés pendant le redécodage. |
| `use_no_bad_ngrams` | `true` | Interdit de reformer les n-grams fautifs identifiés dans la zone abandonnée. |
| `inject_wait_on_backtrack` | `false` | Aucun texte de type « Wait, re-check » n'est injecté dans ces runs. |

## Conclusion utilisable dans l'article

> Sur 467 problèmes MATH-500 uniques avec DeepSeek-R1-Distill-Qwen-1.5B en INT4, MGT-B avec `log_threshold=10` améliore l'accuracy de 31,26 % à 35,76 %, soit un gain absolu de 4,50 points. L'analyse appariée relève 29 corrections contre 8 régressions (`p=0,000753`). Un holdout indépendant de 109 problèmes jamais utilisés auparavant confirme la direction de l'effet avec un gain de 3,67 points. Les sorties restent strictement identiques à vanilla lorsqu'aucune alerte n'est déclenchée, ce qui localise l'amélioration aux interventions de backtracking.

Fichiers principaux :

- configuration confirmatoire : `configs/tests/math500_unseen109_int4_logthr10.yaml`
- résultats confirmatoires : `outputs/math500_unseen109_int4_logthr10/results.jsonl`
- résumé confirmatoire : `outputs/math500_unseen109_int4_logthr10/summary.json`
- paramètres MGT-B : `configs/mgtb_v3_default.yaml`
- seuil : `outputs/calibration/math500_n100/int4/threshold_log10.json`

## Suite expérimentale vers un article de conférence

Les résultats actuels établissent une preuve de concept convaincante sur un modèle, une précision et un benchmark. Pour défendre une contribution générale dans une conférence, les expériences suivantes sont recommandées.

### Priorité 1 — démontrer la valeur propre du détecteur

Le contrôle le plus important est une baseline de backtracking aléatoire à coût égal. Elle doit reproduire, en moyenne, le taux d'intervention, le nombre de tokens annulés, le nombre de rééchantillonnages et les paramètres de redécodage de MGT-B, mais déclencher les interventions à des positions aléatoires.

Comparer au minimum :

1. `vanilla` ;
2. MGT-B complet ;
3. backtracking aléatoire à budget égal ;
4. backtracking périodique à budget égal ;
5. régénération ou best-of-N avec un budget de tokens comparable ;
6. une baseline simple de self-correction, par exemple une instruction « re-check » à budget égal.

L'objectif est de montrer que MGT-B fait mieux qu'une seconde chance non ciblée, et que le gain vient du signal de détection plutôt que du calcul supplémentaire.

### Priorité 2 — tester la généralisation hors MATH-500

Geler le calibrateur, le seuil et tous les paramètres avant chaque test final. Ajouter au moins un benchmark mathématique indépendant, par exemple :

- GSM8K pour les problèmes arithmétiques courts ;
- AIME ou AMC pour des problèmes plus difficiles à réponse courte ;
- OlympiadBench pour des raisonnements plus longs ;
- un split de MATH distinct de MATH-500, si une séparation calibration/test incontestable est disponible.

Pour chaque benchmark, réserver des ensembles disjoints pour la calibration, le développement des hyperparamètres et le test final. Ne jamais choisir le seuil à partir du test final.

### Priorité 3 — tester plusieurs modèles et précisions

Une matrice expérimentale minimale pourrait être :

| Axe | Conditions suggérées |
|---|---|
| Modèles | modèle actuel + un modèle d'une autre famille |
| Tailles | environ 1–2B + une taille supérieure si les ressources le permettent |
| Précisions | FP16/BF16, INT8 et INT4 |
| Méthodes | vanilla, contrôle à coût égal, MGT-B |

Cette matrice doit déterminer si MGT-B est surtout utile en quantification agressive ou s'il améliore également les modèles non quantifiés. Rapporter les échecs et les configurations où MGT-B n'apporte aucun gain.

### Priorité 4 — ablations du mécanisme

Mesurer séparément l'effet des composants suivants :

- seuil automatiquement calibré `10,963` contre seuil expérimental `10` ;
- plusieurs seuils fixés avant le test afin de tracer accuracy, activation, régressions et coût ;
- rollback adaptatif contre rollback de longueur fixe ;
- redécodage à température 0,6 contre température vanilla ;
- avec et sans pénalité de répétition ;
- avec et sans blocage des n-grams fautifs ;
- avec et sans injection explicite d'une instruction de vérification ;
- différentes tailles de fenêtre et différents strides.

Une courbe accuracy–coût en fonction du seuil serait plus informative qu'un seul point à `log_threshold=10`.

### Priorité 5 — renforcer la calibration

La calibration INT4 actuelle ne contient que 34 trajectoires saines. Pour stabiliser l'estimation du taux de fausse alerte :

- augmenter le nombre de trajectoires saines indépendantes ;
- publier le nombre d'exemples et de fenêtres par bucket positionnel ;
- fournir un intervalle de confiance du taux de fausse alerte ;
- vérifier la calibration sur un jeu sain distinct ;
- étudier la sensibilité au seed et à la composition de l'ensemble de calibration.

Chaque modèle, précision et domaine présentant une distribution différente doit avoir une calibration déclarée et gelée avant l'évaluation.

### Priorité 6 — auditer la qualité du scoring

Le scorer actuel repose sur une égalité exacte après normalisation, et non sur une preuve générale d'équivalence symbolique. Il faut :

- vérifier manuellement, en aveugle par rapport à la méthode, les 37 cas discordants de l'analyse principale ;
- faire annoter au moins un sous-ensemble par deux évaluateurs si possible ;
- rapporter les désaccords et leur résolution ;
- ajouter un scorer symbolique ou officiel lorsque le benchmark le permet ;
- publier les résultats avant et après audit pour montrer leur stabilité.

### Priorité 7 — protocole statistique

Avant les nouvelles expériences, fixer dans un plan d'analyse :

- l'accuracy comme métrique principale ;
- le test exact de McNemar pour les comparaisons appariées ;
- un intervalle de confiance bootstrap clusterisé par problème lorsque plusieurs seeds sont utilisées ;
- la règle de déduplication ;
- le nombre de modèles, benchmarks, seeds et exemples ;
- la correction pour comparaisons multiples si plusieurs variantes sont testées ;
- les critères d'exclusion et la gestion des échecs d'extraction.

Rapporter aussi les tailles d'effet, corrections, régressions et intervalles de confiance : une p-value seule ne suffit pas.

### Priorité 8 — coût et reproductibilité

Mesurer dans des conditions matérielles contrôlées :

- latence moyenne, médiane et percentiles ;
- débit en tokens par seconde ;
- tokens calculés, conservés et abandonnés ;
- mémoire GPU maximale ;
- énergie ou temps GPU total si disponible ;
- coût par réponse correcte supplémentaire.

Pour permettre la reproduction, publier ou archiver :

- le commit exact du code ;
- toutes les configurations YAML ;
- les versions Python, PyTorch, Transformers, bitsandbytes et CUDA ;
- le modèle et sa révision exacte ;
- le matériel utilisé ;
- les seeds et IDs des splits ;
- les calibrateurs, seuils, résultats JSONL et scripts d'analyse.

### Paquet expérimental minimal recommandé

Avant soumission, le minimum prioritaire est :

1. une baseline random-backtrack à coût égal ;
2. une réplication sur au moins un nouveau benchmark avec paramètres gelés ;
3. un second modèle ou une seconde taille de modèle ;
4. une comparaison INT4 contre FP16/BF16 ou INT8 ;
5. une ablation du seuil et des principaux composants ;
6. l'audit manuel des cas corrigés et dégradés ;
7. des mesures de coût contrôlées et un paquet de reproductibilité complet.

Le critère de succès le plus convaincant serait que MGT-B conserve un gain positif avec intervalle de confiance excluant zéro sur plusieurs conditions, tout en surpassant les contrôles à budget égal et en maintenant un compromis accuracy–coût acceptable.
