# Rapport final — Évaluation de MGT-B sur MATH-500 et Omni-MATH

- **Date :** 31 août 2026
- **Modèle évalué :** `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
- **Révision du modèle :** `ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562`
- **Portée :** MATH-500 à trois seeds, ablations MATH-500 à une seed, puis évaluation externe Omni-MATH à trois seeds

## Résumé exécutif

Sur **MATH-500**, la comparaison principale porte sur les mêmes 500 problèmes et les seeds 0, 1 et 2, soit 1 500 générations par méthode. `full_mgtb` atteint **56,40 %**, contre **54,73 %** pour Vanilla et **54,60 %** pour `matched_random`. Le gain apparié de MGT-B face à Vanilla est de **+1,67 point**, avec un IC bootstrap clusterisé à 95 % de **[+0,47 ; +2,80]**. Il est positif sur les trois seeds. Face au contrôle aléatoire, le gain est de **+1,80 point**, IC **[+0,47 ; +3,13]**. Le surcoût de génération de MGT-B est de **+5,14 % de tokens échantillonnés**.

Les ablations MATH-500, exploratoires et limitées à la seed 0, montrent que le résultat ne s'explique pas par un simple rollback. La meilleure variante brute est le rollback fixe, à **56,8 %**, mais elle n'est pas distinguable statistiquement de MGT-B complet, à **56,2 %**. Le rollback seul tombe à **54,0 %**. La pénalité de répétition isolée est nettement nuisible : **52,2 %**, soit −4,0 points face à MGT-B, avec un effet qui résiste à la correction de Holm. Le blocage de n-grammes isolé reste proche de Vanilla, à **55,4 %**. Une baseline beaucoup plus coûteuse, la self-consistency à cinq échantillons, atteint **70,0 %**, au prix d'environ **4,84 fois** les tokens de MGT-B.

Sur **Omni-MATH**, test externe plus difficile et construit sans recouvrement avec MATH-500, aucune généralisation du gain n'est observée. Selon le scoring final Gemini, Vanilla atteint **16,67 %**, `full_mgtb` **16,60 %** et `matched_random` **16,93 %**. L'écart MGT-B − Vanilla est de **−0,07 point**, IC clusterisé **[−0,33 ; +0,20]** : il n'y a ni gain détectable ni dégradation établie. MGT-B intervient sur seulement **4,60 %** des 1 500 unités Omni-MATH et coûte **+2,10 % de tokens** par rapport à Vanilla.

La conclusion générale est donc circonscrite : **MGT-B apporte un gain faible mais reproductible sur MATH-500 dans la configuration testée ; ce gain ne se transfère pas à Omni-MATH.** MGT-B est une piste crédible de contrôle ciblé de la génération, mais les données ne permettent pas de le présenter comme une amélioration générale du raisonnement mathématique.

## 1. Périmètre et niveau de preuve

Les résultats relèvent de trois ensembles distincts.

| Ensemble | Problèmes | Seeds | Méthodes | Statut à retenir |
|---|---:|---:|---|---|
| MATH-500 principal | 500 | 0, 1, 2 | Vanilla, `full_mgtb`, `matched_random` | comparaison principale gelée ; inférence clusterisée par problème |
| MATH-500 ablations | 500 | 0 | 11 méthodes complètes | exploratoire ; comparaisons multiples |
| Omni-MATH externe | 500 | 0, 1, 2 | Vanilla, `full_mgtb`, `matched_random` | campagne prospective, mais scoring final accepté avec déviation à la porte de validation du juge |

Les trois seeds d'un même problème ne sont pas trois observations indépendantes. Elles mesurent la variabilité de génération sur les mêmes 500 problèmes. L'inférence principale utilise donc un **bootstrap apparié clusterisé par problème** : les trois seeds restent ensemble dans chaque cluster rééchantillonné.

Les ablations ont été exécutées sur une autre machine et sont consolidées ici depuis leur rapport authentifié. Elles ne doivent pas être assimilées à une réplication à trois seeds.

## 2. Méthodologie commune

### 2.1 Modèle et génération

Les deux benchmarks utilisent le même modèle et les mêmes paramètres essentiels :

- `DeepSeek-R1-Distill-Qwen-1.5B`, à la révision exacte indiquée en tête du rapport ;
- quantification INT4 FP4, sans double quantification, calcul FP16 ;
- prompt `math500_cot` : résolution pas à pas et réponse finale demandée sous la forme `#### <answer>` ;
- limite de **20 000 nouveaux tokens** par génération ;
- seeds appariées entre méthodes.

Le score MATH-500 repose sur l'extraction de la réponse finale et une normalisation textuelle déterministe : suppression de décorations LaTeX courantes, normalisation de fractions simples, racines, espaces et décimaux, puis égalité des formes normalisées. Ce score est reproductible, mais ne prouve pas l'équivalence symbolique générale.

### 2.2 Contrôleur MGT-B complet

MGT-B surveille la trajectoire de génération sans modifier les poids du modèle. Le monitor calcule des features sur des fenêtres de **64 tokens**, avec un stride de **32 tokens** : entropie moyenne, log-probabilité moyenne, répétition de n-grammes de longueur 6 à 8, boucle confiante et variations locales d'entropie. Le score combine ces composantes avec les poids gelés suivants :

| Feature | Poids |
|---|---:|
| Entropie | 0,15 |
| Log-probabilité | 0,10 |
| Répétition | 0,20 |
| Boucle confiante | 0,35 |
| Hausse locale d'entropie | 0,18 |
| Baisse locale d'entropie | 0,02 |

Le score alimente un détecteur séquentiel de type CUSUM avec reset, calibré pour une cible de 5 % de fausses alertes sur des trajectoires de développement non tronquées. Lorsqu'une alarme survient, le système :

1. localise adaptativement un point de changement ;
2. revient avant ce point, avec une marge de 64 tokens ;
3. redécode à température 0,6 ;
4. applique une pénalité de répétition de 1,1 ;
5. bloque les n-grammes suspects ;
6. autorise au maximum trois rerolls.

Sans alarme, la génération MGT-B doit rester identique token par token à Vanilla. Cette propriété est vérifiée sur toutes les unités sans alarme de la comparaison principale MATH-500.

### 2.3 Contrôles et variantes

- **Vanilla** : génération simple, sans monitor ni retour arrière.
- **`matched_random`** : interventions aléatoires dont la fréquence, les positions et les tailles de rollback sont tirées d'un profil construit sur le développement à partir de MGT-B, sans labels de test ; le repair est identique à celui de MGT-B.
- **Monitors ablatés** : répétition seule ; accumulation sans reset ; les variantes entropie seule, entropie + répétition et calibration globale n'ont pas produit de calibration exploitable.
- **Repairs ablatés** : rollback fixe de 1 024 tokens ; rollback seul ; rollback + température ; rollback + pénalité de répétition ; rollback + blocage de n-grammes.
- **Self-consistency (5)** : cinq générations indépendantes, extraction des cinq réponses, puis vote majoritaire.
- **Best-of-5 logprob** : prévu pour sélectionner parmi cinq candidats par log-probabilité moyenne, mais non exécuté.

### 2.4 Inférence statistique

Les différences sont appariées au niveau problème–seed. Les IC à 95 % proviennent de 10 000 bootstraps. Pour les analyses à trois seeds, le rééchantillonnage est clusterisé par problème. Les tests exacts de McNemar au niveau des unités sont descriptifs ; les comparaisons déclarées contre Vanilla sont corrigées par Holm. Les résultats par matière, domaine ou difficulté sont descriptifs.

## 3. Résultats MATH-500

### 3.1 Comparaison principale à trois seeds

| Seed | Vanilla | `full_mgtb` | `matched_random` | MGT-B − Vanilla | MGT-B − aléatoire |
|---:|---:|---:|---:|---:|---:|
| 0 | 55,2 % | 56,2 % | 55,0 % | +1,0 pt | +1,2 pt |
| 1 | 54,0 % | 56,8 % | 54,0 % | +2,8 pts | +2,8 pts |
| 2 | 55,0 % | 56,2 % | 54,8 % | +1,2 pt | +1,4 pt |
| **Agrégé** | **54,73 %** (821/1 500) | **56,40 %** (846/1 500) | **54,60 %** (819/1 500) | **+1,67 pt** | **+1,80 pt** |

L'écart-type inter-seed est de 0,64 point pour Vanilla, 0,35 pour MGT-B et 0,53 pour `matched_random`. Le signe de l'effet MGT-B est stable sur les trois seeds.

Les artefacts de la seed 0 conservent l'étiquette historique `exploratory`, contrairement aux seeds 1 et 2. Les hashes du manifest et des définitions de Vanilla, `full_mgtb` et `matched_random` sont toutefois identiques entre campagnes, et aucun retuning fondé sur les résultats de test n'a été effectué. La divergence est donc une métadonnée historique, pas une différence de protocole scientifique.

| Comparaison | Corrections / régressions | Différence | IC 95 % clusterisé | McNemar brut | McNemar Holm |
|---|---:|---:|---:|---:|---:|
| `full_mgtb` − Vanilla | 53 / 28 | **+1,67 pt** | **[+0,47 ; +2,80]** | 0,0073 | 0,0146 |
| `matched_random` − Vanilla | 16 / 18 | −0,13 pt | [−0,93 ; +0,67] | 0,8642 | 0,8642 |
| `full_mgtb` − `matched_random` | 67 / 40 | **+1,80 pt** | **[+0,47 ; +3,13]** | 0,0116 | hors famille Holm principale |

Le résultat principal est positif : MGT-B corrige davantage de réponses qu'il n'en dégrade, tandis que les interventions aléatoires ne dépassent pas Vanilla. Le contrôle aléatoire reste néanmoins imparfaitement apparié en fréquence observée : il intervient sur 9,4 % des unités, contre 22,6 % pour MGT-B. Ce décalage est un résultat du profil prospectif, pas une égalisation ex post.

### 3.2 Coût de la comparaison principale

| Méthode | Unités avec intervention | Tokens échantillonnés moyens | Écart vs Vanilla | Troncature | Extraction réussie |
|---|---:|---:|---:|---:|---:|
| Vanilla | 0,0 % | 4 366,9 | — | 0,13 % | 99,73 % |
| `full_mgtb` | 22,6 % | 4 591,3 | **+5,14 %** | 1,20 % | 99,80 % |
| `matched_random` | 9,4 % | 4 360,7 | −0,14 % | 0,27 % | 99,73 % |

Sur 1 500 unités, MGT-B échantillonne 336 543 tokens de plus que Vanilla et supprime 317 312 tokens pendant les rollbacks. Sa longueur finale n'augmente que de 0,29 %, alors que son coût réel de décodage augmente de 5,14 %. Les 1 161 unités MGT-B sans alarme sont identiques token par token à Vanilla.

Le principal coût indésirable est la troncature : 18 cas avec MGT-B, contre 2 avec Vanilla. Les temps muraux ne sont pas interprétés comme un benchmark matériel contrôlé.

### 3.3 Ablations exploratoires à la seed 0

Onze méthodes possèdent 500/500 résultats complets dans le snapshot d'ablation. Le tableau suivant regroupe exactitude, activité du contrôleur et coût de génération.

| Méthode | Corrects | Exactitude | Écart vs Vanilla | Unités alertées | Tokens échantillonnés moyens | Troncatures |
|---|---:|---:|---:|---:|---:|---:|
| Vanilla | 276/500 | 55,2 % | — | 0,0 % | 4 332,5 | 1 |
| MGT-B complet | 281/500 | 56,2 % | +1,0 pt | 21,4 % | 4 519,4 | 3 |
| `matched_random` | 275/500 | 55,0 % | −0,2 pt | 11,0 % | 4 270,5 | 2 |
| Monitor répétition seule | 280/500 | 56,0 % | +0,8 pt | 7,0 % | 4 321,4 | 2 |
| Monitor sans reset | 281/500 | 56,2 % | +1,0 pt | 15,8 % | 4 276,5 | 0 |
| Rollback fixe | 284/500 | **56,8 %** | **+1,6 pt** | 21,4 % | 4 552,1 | 6 |
| Rollback seul | 270/500 | 54,0 % | −1,2 pt | 21,4 % | 4 629,0 | 1 |
| Rollback + température | 274/500 | 54,8 % | −0,4 pt | 21,4 % | 5 032,5 | 18 |
| Rollback + pénalité de répétition | 261/500 | **52,2 %** | **−3,0 pts** | 21,4 % | 4 743,3 | 10 |
| Rollback + blocage de n-grammes | 277/500 | 55,4 % | +0,2 pt | 21,4 % | 4 665,4 | 1 |
| Self-consistency (5) | 350/500 | **70,0 %** | **+14,8 pts** | 0,0 % | 21 867,7 | 1 |

Les cinq ablations du repair utilisent exactement le même monitor et les mêmes 107 cas alertés. Leurs différences isolent donc mieux l'effet de la politique de redécodage que les comparaisons entre monitors.

#### Comparaisons appariées face à Vanilla

| Méthode | Différence | IC bootstrap 95 % | Corrections / régressions | McNemar brut |
|---|---:|---:|---:|---:|
| MGT-B complet | +1,0 pt | [−0,8 ; +2,8] | 14 / 9 | 0,4049 |
| `matched_random` | −0,2 pt | [−2,0 ; +1,4] | 9 / 10 | 1,0000 |
| Monitor répétition seule | +0,8 pt | [+0,2 ; +1,6] | 4 / 0 | 0,1250 |
| Monitor sans reset | +1,0 pt | [−0,6 ; +2,8] | 12 / 7 | 0,3593 |
| Rollback fixe | +1,6 pt | [−0,2 ; +3,6] | 16 / 8 | 0,1516 |
| Rollback seul | −1,2 pt | [−2,8 ; +0,4] | 5 / 11 | 0,2101 |
| Rollback + température | −0,4 pt | [−2,4 ; +1,6] | 13 / 15 | 0,8506 |
| Rollback + pénalité de répétition | **−3,0 pts** | **[−5,0 ; −1,0]** | 5 / 20 | **0,00408** |
| Rollback + blocage de n-grammes | +0,2 pt | [−1,4 ; +2,0] | 10 / 9 | 1,0000 |
| Self-consistency (5) | **+14,8 pts** | **[+11,6 ; +18,0]** | 76 / 2 | **2,04 × 10⁻²⁰** |

Après correction de Holm sur les dix comparaisons contre Vanilla, seuls le gain de self-consistency (`p` ajusté ≈ 2,04 × 10⁻¹⁹) et la dégradation de la pénalité de répétition isolée (`p` ajusté ≈ 0,0367) restent significatifs.

#### Comparaisons directes face à MGT-B complet

| Ablation | Écart vs MGT-B | IC bootstrap 95 % | Corrections / régressions | McNemar brut |
|---|---:|---:|---:|---:|
| `matched_random` | −1,2 pt | [−3,6 ; +1,2] | 16 / 22 | 0,4177 |
| Monitor répétition seule | −0,2 pt | [−2,0 ; +1,6] | 11 / 12 | 1,0000 |
| Monitor sans reset | 0,0 pt | [−2,2 ; +2,2] | 15 / 15 | 1,0000 |
| Rollback fixe | +0,6 pt | [−1,0 ; +2,2] | 9 / 6 | 0,6072 |
| Rollback seul | **−2,2 pts** | **[−4,0 ; −0,4]** | 6 / 17 | **0,0347** |
| Rollback + température | −1,4 pt | [−3,0 ; +0,2] | 5 / 12 | 0,1435 |
| Rollback + pénalité de répétition | **−4,0 pts** | **[−6,0 ; −2,2]** | 2 / 22 | **0,0000359** |
| Rollback + blocage de n-grammes | −0,8 pt | [−2,8 ; +1,2] | 10 / 14 | 0,5413 |
| Self-consistency (5) | **+13,8 pts** | **[+10,4 ; +17,4]** | 79 / 10 | **1,87 × 10⁻¹⁴** |

Après Holm sur ces neuf comparaisons, la pénalité de répétition reste inférieure à MGT-B (`p` ajusté ≈ 0,000287) et self-consistency reste supérieure (`p` ajusté ≈ 1,69 × 10⁻¹³). Le signal défavorable du rollback seul ne résiste pas à la correction (`p` ajusté ≈ 0,243).

### 3.4 Lecture mécanistique des ablations

1. **Le rollback seul ne suffit pas.** Il est inférieur de 2,2 points à MGT-B complet sur les mêmes cas alertés. Le repair doit modifier la continuation, pas seulement effacer des tokens.
2. **La pénalité de répétition ne doit pas être utilisée isolément.** Elle produit 22 régressions pour 2 corrections face à MGT-B et augmente les troncatures.
3. **Le blocage de n-grammes est moins risqué, mais son effet propre n'est pas établi.** Il reste proche de Vanilla et à 0,8 point de MGT-B.
4. **La température réduite seule n'explique pas le système.** Elle coûte le plus parmi les repairs simples et entraîne 18 troncatures.
5. **Le rollback adaptatif n'est pas démontré supérieur.** Le rollback fixe est numériquement meilleur de 0,6 point, mais son IC inclut largement zéro.
6. **Le signal de répétition est un monitor parcimonieux prometteur.** Il atteint 56,0 % avec seulement 35 alertes, mais le résultat repose sur quatre changements favorables seulement.
7. **Le reset modifie les cas sans modifier la moyenne.** Le monitor sans reset et MGT-B obtiennent tous deux 56,2 %, avec pourtant 15 corrections et 15 régressions entre eux.
8. **La self-consistency fixe le plafond de comparaison coût–performance.** Elle apporte un gain massif, mais consomme 10 933 855 tokens, environ 4,84 fois le total de MGT-B.

Parmi les cinq candidats de self-consistency, au moins un est correct sur 79,0 % des problèmes. Le vote majoritaire n'en récupère que 70,0 %, laissant 9 points d'écart de sélection.

### 3.5 Résultats par matière à trois seeds

| Matière | Vanilla | MGT-B | Écart MGT-B − Vanilla |
|---|---:|---:|---:|
| Algèbre (372 unités) | 68,01 % | 69,35 % | +1,34 pt |
| Comptage et probabilités (114) | 42,98 % | 44,74 % | +1,75 pt |
| Géométrie (123) | 47,15 % | 46,34 % | −0,81 pt |
| Algèbre intermédiaire (291) | 42,61 % | 45,36 % | +2,75 pts |
| Théorie des nombres (186) | 62,37 % | 60,22 % | −2,15 pts |
| Préalgèbre (246) | 55,69 % | 60,16 % | +4,47 pts |
| Précalcul (168) | 50,00 % | 52,38 % | +2,38 pts |

Ces écarts sont descriptifs et non corrigés pour multiplicité. Ils localisent l'effet observé, mais ne prouvent pas un bénéfice propre à une matière.

### 3.6 Calibrations et variantes sans résultat

| Calibration MATH-500 | Seuil `h` | Fausse alarme saine | Statut |
|---|---:|---:|---|
| Full, positionnelle, CUSUM reset | 11,4293 | 3/62 = 4,84 % | exploitable |
| Répétition seule, positionnelle | 11,6623 | 3/62 = 4,84 % | exploitable |
| Sans reset, positionnelle | 6,0692 | 3/62 = 4,84 % | exploitable |
| Entropie seule | 18,4207 | 62/62 = 100 % | non exploitable ; non lancée |
| Entropie + répétition | 18,4207 | 62/62 = 100 % | non exploitable ; non lancée |
| Calibration globale | 18,4207 | 62/62 = 100 % | non exploitable ; non lancée |

`best_of_5_logprob` n'a pas été lancé. Il n'existe donc aucun résultat à comparer à self-consistency. Les trois échecs de calibration doivent être rapportés comme tels, sans leur attribuer une exactitude de test.

## 4. Résultats Omni-MATH

### 4.1 Construction prospective du benchmark

La source est `KbsdJames/Omni-MATH`, fichier `Omni-Math.jsonl`, au commit `23be225c8e268df51990f6c5c1448f34d3b56911` et au blob Git `1a9d46a3a2b52992b010152e8e090f5fb7e7cb4a`.

Les énoncés sont normalisés en NFKC, dédupliqués par SHA-256 et comparés aux manifests MATH-500. Tout énoncé déjà présent dans MATH-500 est exclu avant la sélection. Le protocole, gelé avec la seed `20260824`, construit :

- 300 problèmes de référence pour estimer les distributions positionnelles des features ;
- 300 problèmes de développement pour sélectionner le seuil et construire le profil aléatoire ;
- 500 problèmes de test, évalués sous les seeds 0, 1 et 2.

La stratification utilise le couple `(premier domaine officiel, difficulté exacte)`. Les quotas sont proportionnels par la méthode de Hamilton, puis les problèmes sont ordonnés de façon déterministe par hash. Les 500 problèmes de test sont ainsi fixés avant toute génération.

### 4.2 Calibration Omni-MATH

Les poids du monitor, les fenêtres, le CUSUM, la cible de fausses alertes et le repair restent identiques à MATH-500. Seules la distribution empirique des features et la valeur du seuil sont recalibrées, sans lecture des labels de correction.

- 300 trajectoires Vanilla de référence alimentent le calibrateur positionnel ;
- 299 trajectoires de développement non tronquées sont éligibles à la sélection du seuil ;
- seuil retenu : **`h = 47,6874`** ;
- taux de fausse alarme sain observé : **14/299 = 4,68 %** ;
- le profil `matched_random` est construit uniquement depuis MGT-B sur le développement, sans verdicts.

Le seuil Omni-MATH est beaucoup plus élevé que le seuil MATH-500, ce qui reflète un changement important de distribution des scores du monitor.

### 4.3 Scoring final et statut du juge

Le normaliseur MATH-500 n'est pas adapté à l'équivalence générale des réponses Omni-MATH. Un premier scoring local par `KbsdJames/Omni-Judge`, révision `de5bdca15ff3c366b90718c4b4be555d25c655b0`, avait donc été prévu : décodage glouton, code officiel gelé et maximum de 300 tokens par verdict. Comme le désaccord avec des contrôles déterministes était important, un second pipeline aveugle a été utilisé pour la décision finale.

Ce pipeline applique d'abord deux règles certaines : égalité normalisée exacte, ou inégalité entre deux nombres simples. Les cas restants sont jugés par **`gemini-3.5-flash-lite`**, via `google-genai==2.20.0`, à température 0 et avec le niveau de raisonnement `high`, dans un format JSON structuré. Les payloads ne contiennent ni variante, ni seed, ni ancien verdict, ni métrique MGT-B. Les candidats sont anonymisés et, lorsqu'ils sont groupés par problème pour réduire le coût, le prompt interdit explicitement vote, classement ou consensus. `ABSTAIN` est compté comme incorrect.

Répartition des 4 500 décisions finales :

- 2 051 contrôles déterministes ;
- 29 décisions issues du cache Gemini authentifié ;
- 2 420 décisions Gemini groupées.

Le pilote du juge contient 200 cas : 98 contradictions numériques avec l'ancien Omni-Judge, 50 égalités normalisées et 52 réponses symboliques réparties par domaine. Il produit 27 % de `TRUE`, 69 % de `FALSE` et 4 % d'`ABSTAIN`. L'accord entre jugement groupé et individuel est de **94 % sur 50 paires**. Aucune erreur certaine de Gemini n'est observée parmi les arbitrages terminés.

La porte secondaire pré-déclarée n'est cependant pas complète : **21 arbitrages sur 112** avec `gemini-3.5-flash`. La recommandation mécanique du pilote reste donc `NO-GO`. Le rapport complet a été finalisé par décision explicite d'accepter le juge principal, avec le statut **`FINAL_USER_ACCEPTED_JUDGE`**. Il s'agit d'un résultat final pour ce rapport, mais **pas d'un résultat `CONFIRMATORY` au sens de la porte de validation initialement prévue**.

### 4.4 Exactitude Omni-MATH selon le scoring final

| Méthode | TRUE | FALSE | ABSTAIN | Exactitude | Taux d'ABSTAIN |
|---|---:|---:|---:|---:|---:|
| Vanilla | 250 | 1 118 | 132 | **16,67 %** | 8,80 % |
| `full_mgtb` | 249 | 1 117 | 134 | **16,60 %** | 8,93 % |
| `matched_random` | 254 | 1 112 | 134 | **16,93 %** | 8,93 % |

| Comparaison | Corrections / régressions | Différence | IC 95 % clusterisé | McNemar descriptif | Holm |
|---|---:|---:|---:|---:|---:|
| `full_mgtb` − Vanilla | 2 / 3 | **−0,07 pt** | **[−0,33 ; +0,20]** | 1,0000 | 1,0000 |
| `matched_random` − Vanilla | 6 / 2 | +0,27 pt | [−0,07 ; +0,67] | 0,2891 | 0,5781 |
| `full_mgtb` − `matched_random` | 4 / 9 | −0,33 pt | [−0,80 ; +0,13] | 0,2668 | descriptif |

Les trois intervalles incluent zéro. Seulement cinq verdicts diffèrent entre MGT-B et Vanilla sur 1 500 unités : le test ne détecte aucun bénéfice d'exactitude de MGT-B.

### 4.5 Résultats Omni-MATH par seed

| Seed | Vanilla | `full_mgtb` | `matched_random` | MGT-B − Vanilla |
|---:|---:|---:|---:|---:|
| 0 | 14,8 % | 14,8 % | 15,4 % | 0,0 pt |
| 1 | 16,8 % | 16,6 % | 17,0 % | −0,2 pt |
| 2 | 18,4 % | 18,4 % | 18,4 % | 0,0 pt |
| **Agrégé** | **16,67 %** | **16,60 %** | **16,93 %** | **−0,07 pt** |

La hausse commune d'exactitude de la seed 0 à la seed 2 indique un effet de seed sur le niveau absolu. Elle ne crée toutefois aucun avantage relatif pour MGT-B.

### 4.6 Coût et comportement de génération Omni-MATH

Les métriques suivantes ont été recalculées directement depuis les 4 500 artefacts de génération authentifiés.

| Méthode | Interventions | Tokens échantillonnés totaux | Moyenne | Écart vs Vanilla | Tokens supprimés | Troncatures | Extraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| Vanilla | 0/1 500 = 0,00 % | 10 324 437 | 6 883,0 | — | 0 | 5 (0,33 %) | 1 499/1 500 |
| `full_mgtb` | 69/1 500 = **4,60 %** | 10 541 477 | 7 027,7 | **+2,10 %** | 185 184 | 8 (0,53 %) | 1 499/1 500 |
| `matched_random` | 50/1 500 = 3,33 % | 10 401 237 | 6 934,2 | +0,74 % | 94 848 | 7 (0,47 %) | 1 499/1 500 |

MGT-B émet 10 356 293 tokens finaux, soit seulement +0,31 % par rapport aux 10 324 437 tokens Vanilla : l'essentiel du surcoût vient des tokens ensuite supprimés par rollback. Comme l'exactitude est inchangée dans l'incertitude, MGT-B n'améliore pas ici le compromis exactitude–tokens.

Le taux d'intervention tombe de 22,6 % sur MATH-500 à 4,6 % sur Omni-MATH. Cette rareté explique directement le très faible nombre de réponses modifiées, mais elle ne permet pas à elle seule de déterminer si le problème vient du monitor, du seuil recalibré ou du repair.

### 4.7 Sensibilité au juge

L'accord entre le pipeline final Gemini et Omni-Judge est de **70,47 % sur 3 332 unités comparables**. Omni-Judge local avait produit 35,60 % pour Vanilla et 35,53 % pour MGT-B sur leurs 1 500 unités complètes, soit encore une différence de −0,07 point, mais à un niveau absolu d'exactitude très différent. Le run Omni-Judge de `matched_random` n'est complet que pour 419/1 500 unités et ne permet pas de comparaison tripartite valide.

La direction nulle de l'effet MGT-B est donc cohérente entre les deux juges disponibles, mais **l'exactitude absolue d'Omni-MATH est fortement dépendante du juge**. Le résultat principal doit rester rattaché au pipeline Gemini et à son statut de validation explicité ci-dessus.

## 5. Synthèse scientifique

### 5.1 Ce qui est établi

- Sur MATH-500, avec ce modèle 1,5B quantifié en INT4, MGT-B améliore Vanilla de **+1,67 point** sur trois seeds, avec un IC clusterisé entièrement positif.
- MGT-B dépasse également le contrôle `matched_random` de **+1,80 point** sur MATH-500. Le monitor apporte donc de l'information au-delà d'interventions aléatoires selon le profil prospectif utilisé.
- Le coût de MGT-B reste modéré en tokens sur MATH-500 : **+5,14 %**, très inférieur au coût de self-consistency (5).
- La pénalité de répétition isolée est nuisible dans le repair testé. Le système complet dépend vraisemblablement de l'interaction entre rollback et contraintes de redécodage.
- Sur Omni-MATH, aucun avantage de MGT-B n'est détecté : **−0,07 point**, avec un IC étroit autour de zéro et seulement 2 corrections pour 3 régressions.

### 5.2 Ce qui n'est pas établi

- Le gain MATH-500 ne se généralise pas à une autre distribution mathématique plus difficile.
- Le rollback adaptatif n'est pas démontré meilleur que le rollback fixe.
- Aucune feature isolée du monitor n'est démontrée nécessaire et suffisante.
- MGT-B n'atteint pas la meilleure exactitude sans contrainte de calcul : self-consistency (5) est très supérieure sur MATH-500.
- Le résultat Omni-MATH ne satisfait pas la porte confirmatoire complète du juge secondaire, même si le scoring principal a été explicitement accepté comme final.

### 5.3 Interprétation

Les résultats MATH-500 soutiennent l'hypothèse qu'un contrôle ciblé pendant le décodage peut réparer une petite fraction de trajectoires problématiques à faible surcoût. Le contrôle aléatoire n'obtient pas le même gain, et le système ne modifie pas les trajectoires lorsqu'il reste inactif.

Les ablations montrent cependant que la réparation est fragile : supprimer les contraintes de redécodage ou isoler la pénalité de répétition peut annuler le gain, voire dégrader nettement la performance. MGT-B ne se réduit donc pas à « détecter puis recommencer ».

Omni-MATH révèle la limite principale. Après recalibration, le seuil est beaucoup plus élevé et MGT-B n'intervient que sur 4,6 % des unités, contre 22,6 % sur MATH-500. Dans un benchmark où Vanilla n'atteint que 16,67 % selon le juge final, les erreurs semblent majoritairement relever d'un manque de capacité ou de raisonnement, et non des seuls modes de boucle ciblés par le monitor. C'est une interprétation plausible, pas une conclusion causale démontrée.

## 6. Conclusion sur MGT-B

MGT-B est **validé comme amélioration locale et modeste sur MATH-500**, dans une configuration précise : `DeepSeek-R1-Distill-Qwen-1.5B`, INT4, prompt et contrôleur gelés. Son gain de +1,67 point est reproductible sur trois seeds et obtenu avec +5,14 % de tokens. Ce résultat justifie de poursuivre l'étude de la détection en ligne et du repair adaptatif.

En revanche, **MGT-B n'est pas validé comme méthode générale d'amélioration du raisonnement mathématique**. Sur Omni-MATH, il ajoute 2,10 % de tokens sans gain mesurable. La revendication scientifique défendable est donc :

> MGT-B peut améliorer de façon coût-efficace un modèle donné lorsque ses signaux de dérive correspondent aux erreurs du benchmark ; l'effet est dépendant de la distribution et ne remplace ni l'augmentation du budget d'échantillonnage ni une capacité de raisonnement plus forte.

La priorité n'est pas d'ajouter davantage d'ablations post hoc sur MATH-500, mais de comprendre l'échec de transfert : courbes de scores par dataset, analyse des faux négatifs du monitor sur Omni-MATH, réparation conditionnelle à la difficulté, meilleure égalisation du contrôle aléatoire et réplication sur un second modèle ou une autre taille.

## 7. Limites

- Un seul modèle et une seule famille de quantification sont évalués.
- Trois seeds réduisent l'incertitude de génération, mais ne remplacent pas des problèmes indépendants supplémentaires.
- Les ablations utilisent une seule seed et sont exploratoires ; plusieurs analyses ont été observées simultanément.
- Le score textuel MATH-500 ne démontre pas toutes les équivalences symboliques possibles.
- Les taux d'intervention de `matched_random` ne sont pas égaux ex post à ceux de MGT-B.
- Les troncatures sont plus fréquentes avec MGT-B sur les deux benchmarks.
- Omni-MATH dépend d'un juge appris ; son pipeline final n'a pas terminé l'arbitrage secondaire pré-déclaré.
- Les environnements logiciels diffèrent entre les machines MATH-500 et Omni-MATH, même si le modèle, sa révision, la quantification et chaque protocole expérimental sont gelés.

## 8. Provenance et reproductibilité

### MATH-500

- Dataset : `HuggingFaceH4/MATH-500`, révision `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be`.
- Seed 0 et ablations : `outputs/science_campaign/math500_all500_exploratory_v1/`.
- Seeds 1 et 2 : `outputs/science_campaign/math500_all500_confirmatory_seeds_1_2_v1/`.
- Manifest : `b0302d2b75bcc3980f459abb260e8de8d1671d294b160acbeecfe2c2ea5a9baf`.
- Freeze seed 0 : `d26ba748646beab47be49c3270dc069640c9c649d1423fe54ab03426950f5482`.
- Freeze seeds 1–2 : `500b0a9fb09038aa0ccd9fcf46b486fed2840e3c090004ff4893ada5cb490fb4`.
- Environnement d'ablation : Python 3.10.20, PyTorch 2.13.0+cu130, Transformers 4.57.6, Datasets 5.0.1, bitsandbytes 0.50.1, CUDA 13.0, NVIDIA RTX A5000.

### Omni-MATH

- Générations finales : `outputs/science_campaign/omnimath_confirmatory_v1_judge_batch1/`.
- Scoring final : `outputs/gemini_scoring/omnimath_confirmatory_v1/full/`.
- Manifest : `0d84b7670bf640257517e089c51ffac159a144ad1dccce11562a087a467a4c51`.
- Freeze final : `fce1576a35011fc6eb39c25839e997e742379837d6f1e193f217b55ee84d29d8`.
- Commit source gelé : `478c3ec49f65a0d38eadc6cf922b5fe8263df853`.
- Hash du rapport Gemini : `222c16a88ce6c597b361031e7c161d24e0a64a010d5f68664e898db892ef2551`.
- Environnement : Python 3.9.25, PyTorch 2.8.0+cu128, Transformers 4.57.6, Datasets 4.5.0, bitsandbytes 0.48.2, CUDA 12.8, NVIDIA RTX A5000.

### Rapports sources consolidés

- `docs/RAPPORT_TROIS_SEEDS_VANILLA_FULL_MGTB_MATCHED_RANDOM.md`
- `docs/RAPPORT_INTERMEDIAIRE_ABLATIONS_MATH500_2026-08-31.md`
- `docs/OMNIMATH_CONFIRMATORY_V1.md`
- `docs/OMNIMATH_GEMINI_SCORING.md`
- `outputs/gemini_scoring/omnimath_confirmatory_v1/full/REPORT.md`
