# Rapport scientifique — Évaluation prospective de MGT-B en INT4 sur MATH-500

**Date de l'analyse :** 19 août 2026  
**Comparaison principale :** Vanilla INT4 contre MGT-B INT4  
**Modèle :** `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`  
**Jeu de test :** sous-ensemble déterministe de 300 problèmes de MATH-500

## Résumé

Cette expérience évalue si le contrôleur MGT-B améliore la résolution de problèmes mathématiques en détectant les trajectoires de génération probablement dégradées, puis en supprimant et redécodant leur suffixe. Le protocole sépare strictement les données de référence, de développement et de test. Le calibrateur est construit sur 300 problèmes de référence, le seuil d'alarme est choisi sur 100 problèmes de développement, puis les performances sont mesurées une seule fois sur 300 problèmes MATH-500 tenus à l'écart.

MGT-B obtient une exactitude de **60,0 % (180/300)**, contre **55,0 % (165/300)** pour Vanilla. Le gain apparié est de **+5,0 points de pourcentage**, avec un intervalle bootstrap à 95 % de **[+2,0 ; +8,0] points**. Le test exact bilatéral de McNemar donne **p = 0,00260**. MGT-B corrige 19 erreurs de Vanilla et introduit 4 régressions, soit un bénéfice net de 15 réponses correctes.

Le contrôleur intervient sur 82 problèmes (27,3 %). Sur les 218 problèmes sans alarme, les suites de tokens MGT-B et Vanilla sont strictement identiques, conformément à l'invariant prévu. Le gain a un coût : **+8,1 % de tokens échantillonnés**, **+10,2 % de latence moyenne par item**, et 5 troncatures sous MGT-B contre 1 sous Vanilla.

Dans le cadre précis de ce protocole, les données soutiennent donc l'hypothèse que MGT-B améliore l'exactitude. La portée de cette conclusion reste limitée à un modèle, une quantification, un échantillon déterministe de MATH-500 et une génération par problème et par méthode.

## 1. Question de recherche et hypothèses

La question principale est la suivante :

> À modèle, quantification, prompt, budget de génération et seed d'item identiques, le contrôleur MGT-B améliore-t-il l'exactitude par rapport à une génération Vanilla ?

La métrique principale est la différence appariée d'exactitude :

\[
\Delta = \mathrm{Accuracy}_{\mathrm{MGT-B}} - \mathrm{Accuracy}_{\mathrm{Vanilla}}.
\]

L'hypothèse nulle est l'absence de différence systématique entre les deux méthodes sur les mêmes problèmes. L'hypothèse d'intérêt est \(\Delta > 0\).

Les métriques secondaires portent sur les corrections, les régressions, les alarmes, les rerolls, les tokens, la latence, la VRAM, l'extractabilité et les troncatures.

## 2. Méthodologie

### 2.1 Protocole prospectif et séparation des données

Le protocole utilise trois partitions disjointes :

| Rôle | Source | Taille | Usage |
|---|---|---:|---|
| Référence | `EleutherAI/hendrycks_math`, train | 300 | Construction de l'ECDF positionnelle |
| Développement | `EleutherAI/hendrycks_math`, train | 100 | Sélection du seuil d'alarme |
| Test | `HuggingFaceH4/MATH-500`, test | 300 | Comparaison finale Vanilla/MGT-B |

Les révisions des jeux de données sont figées :

- MATH train : `21a5633873b6a120296cce3e2df9d5550074f4a3` ;
- MATH-500 : `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be`.

La sélection repose sur le hash SHA-256 du texte normalisé du problème et sur la seed de protocole `20260811`. Les contenus sont triés par une clé déterministe. Les 300 premiers contenus uniques de MATH train forment la référence, les 100 suivants le développement, et les 300 premiers items triés de MATH-500 forment le test. Des contrôles d'intersection par hash de contenu empêchent les fuites entre rôles.

Le manifest a été construit avant la calibration et le test. Son identifiant interne est :

```text
30ca14e5d4bbeda88347671a4f2e146f981d5bd167f93aa7433a34fb07625813
```

### 2.2 Modèle et génération

Les deux conditions utilisent exactement le même modèle et les mêmes paramètres de base :

| Paramètre | Valeur |
|---|---|
| Modèle | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` |
| Révision | `ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562` |
| Quantification | bitsandbytes INT4, FP4 |
| Double quantification | désactivée |
| Calcul | float16 |
| Stockage | uint8 |
| Style de prompt | `math500_cot` |
| Température Vanilla | 1,0 |
| Budget maximal | 20 000 nouveaux tokens |

Une seule génération est réalisée par problème et par méthode. La seed d'un item est dérivée de manière déterministe de la seed du protocole et de son identifiant stable. Un même problème reçoit la même seed sous Vanilla et MGT-B. L'ordre d'exécution et le nombre de workers ne font pas partie de l'identité scientifique du run.

Les évaluations ont été exécutées séquentiellement, avec six workers parallèles sur une NVIDIA RTX A5000. Chaque item terminé est écrit atomiquement avec son texte, ses tokens, son score, ses traces, sa comptabilité de tokens, son timing et ses identifiants de provenance. Les runs sont reprenables au niveau de l'item.

### 2.3 Signaux et score MGT-B

La génération est inspectée par fenêtres de 64 tokens, avec un stride de 32 tokens. Les répétitions de n-grams de longueurs 6 à 8 sont suivies, en excluant celles provenant uniquement du prompt.

Pour chaque fenêtre \(j\), le vecteur de caractéristiques combine :

- l'entropie moyenne ;
- l'opposé de la log-probabilité moyenne des tokens choisis ;
- le taux de répétition de n-grams ;
- un signal de répétition confiante ;
- les variations positive et négative d'entropie locale.

Le score est :

\[
s_j = 0{,}15\bar H_j + 0{,}10(-\bar\ell_j) + 0{,}20R_j
      + 0{,}35D_j + 0{,}18L_j^+ + 0{,}02L_j^-.
\]

### 2.4 Calibration positionnelle

L'ECDF est estimée séparément dans cinq intervalles de position : `0–512`, `512–1024`, `1024–2048`, `2048–4096` et `4096+`. Seules les trajectoires de référence correctes, extractables et non tronquées alimentent les pools de calibration.

Résultats de la phase de référence :

| Mesure | Valeur |
|---|---:|
| Items terminés | 300/300 |
| Réponses correctes | 158 |
| Réponses extractables | 300 |
| Troncatures | 0 |
| Trajectoires healthy retenues | 158 |
| Fenêtres de calibration | 15 298 |

Répartition des fenêtres :

| Position | Fenêtres | Trajectoires distinctes |
|---|---:|---:|
| 0–512 | 2 209 | 158 |
| 512–1024 | 2 456 | 157 |
| 1024–2048 | 3 758 | 147 |
| 2048–4096 | 3 866 | 92 |
| 4096+ | 3 009 | 32 |

Le calibrateur utilise `p_clip = 10^-6`. Son identifiant est :

```text
9ffdf35fe369dcd8eaf32c07606870b49e7ffbd1a32c7d9a22139e87fcb682ad
```

### 2.5 Statistique séquentielle et seuil

Les p-valeurs empiriques positionnelles sont transformées en e-facteurs par un mélange de paramètres \(\gamma \in \{0{,}1, 0{,}3, 0{,}5, 0{,}7\}\). La statistique cumulée est :

\[
S_j = \max(0,S_{j-1}) + \log(e_j), \qquad S_0=0.
\]

Une alarme est déclenchée lorsque \(S_j \ge h\).

La phase de développement comporte 100 items, dont 62 trajectoires healthy éligibles. Le plus petit seuil de la grille satisfaisant une fréquence d'alarme healthy inférieure ou égale à 5 % a été sélectionné :

| Mesure | Valeur |
|---|---:|
| Seuil \(h\) | 11,42926413138305 |
| Dénominateur healthy | 62 |
| Fréquence d'alarme healthy | 4,84 % (3/62) |
| Cible | ≤ 5 % |
| Avertissement de faible effectif | aucun |

L'identifiant du seuil est :

```text
5cd7c2871695af29a7ba8fff751489555da75780204463359e73feff3ea769a4
```

### 2.6 Intervention après alarme

Lorsqu'une alarme est déclenchée, MGT-B estime un point de changement, étend le rollback de 64 tokens vers l'amont, supprime le suffixe suspect, restaure le préfixe et l'état du cache, puis redécode avec :

- température 0,6 ;
- pénalité de répétition 1,1 ;
- blocage ciblé des n-grams suspects ;
- deux fenêtres réfractaires ;
- au plus trois rerolls ;
- reconstruction de cache `replay_last` ;
- indexation `tracked_windows`.

Aucune injection de prompt n'est utilisée.

### 2.7 Freeze et incident pré-test

Avant le test, deux locks distincts ont figé le manifest, les 300 items test, le modèle, la quantification, les révisions de données, le calibrateur, le seuil, le contrôleur, le scorer, le budget et l'environnement.

| Lock | Identifiant interne |
|---|---|
| Vanilla INT4 | `3eaef38434373bfea1aee94801727bb9a6403564ba8c3e57e9a063664d47b23e` |
| MGT-B INT4 | `6e6ac620425e90925a11321dff492354ab7d2b698df4b32455df18070f814247` |

Une première tentative de lancement a été refusée avant toute génération test : le contrôleur en mémoire contenait les paramètres `betting_gammas` sous forme de tuple Python, tandis que leur sérialisation dans le lock JSON les représentait comme une liste. Les contenus étaient équivalents, mais la comparaison structurelle stricte échouait. La comparaison a été rendue canonique au format JSON, un test de non-régression a été ajouté, puis les locks ont été régénérés avant de consommer un item test. Les freezes finaux enregistrent :

- commit Git : `788a2ab61a3314b64380855b47a4af44bff8523e` ;
- hash de l'arbre source : `fd36348e8dcbb927345b7b7dc67f42c6e1ba52a1b5bab8699225dc7fce90f1ef`.

### 2.8 Analyse statistique

L'analyse est appariée au niveau du problème. Pour chacun des 300 IDs identiques, la variable analysée vaut `1`, `0` ou `−1` selon que MGT-B corrige une erreur, ne change pas le statut, ou introduit une régression.

Deux procédures sont utilisées :

1. un test exact bilatéral de McNemar sur les paires discordantes ;
2. un bootstrap apparié au niveau des problèmes, avec 10 000 réplications et la seed `20260811`, pour l'intervalle de confiance à 95 % de la différence d'exactitude.

Les métriques sont reconstruites depuis les 600 artefacts bruts, et non depuis des logs agrégés pendant l'exécution.

Une vérification finale a confirmé la présence de 300 artefacts Vanilla et 300 artefacts MGT-B, sans aucun hash d'artefact invalide. Les seeds d'item concordent entre les deux méthodes pour les 300 paires.

## 3. Résultats

### 3.1 Résultat principal

| Mesure | Vanilla | MGT-B | Différence |
|---|---:|---:|---:|
| Réponses correctes | 165/300 | 180/300 | +15 |
| Exactitude | 55,0 % | 60,0 % | **+5,0 pp** |
| Erreurs | 135 | 120 | −15 |
| Réduction relative des erreurs | — | — | 11,1 % |

L'intervalle bootstrap apparié à 95 % est **[+2,0 ; +8,0] points de pourcentage**. Il ne contient pas zéro. Le test exact bilatéral de McNemar donne **p = 0,002599**, ce qui indique une asymétrie marquée entre corrections et régressions sous l'hypothèse nulle.

### 3.2 Décomposition appariée

| Statut de la paire | Nombre |
|---|---:|
| Correct sous les deux méthodes | 161 |
| Faux sous Vanilla, correct sous MGT-B | **19** |
| Correct sous Vanilla, faux sous MGT-B | **4** |
| Faux sous les deux méthodes | 116 |
| Total | 300 |

Parmi les 23 paires discordantes, 19 (82,6 %) favorisent MGT-B. Le bénéfice net est de 15 problèmes.

### 3.3 Effet conditionnel aux alarmes

MGT-B a déclenché exactement une alarme et un reroll sur 82 items, et aucune alarme sur 218 items.

| Sous-groupe | Taille | Vanilla | MGT-B | Différence |
|---|---:|---:|---:|---:|
| Items avec alarme | 82 | 29,3 % (24/82) | 47,6 % (39/82) | +18,3 pp |
| Items sans alarme | 218 | 64,7 % (141/218) | 64,7 % (141/218) | 0 pp |

Sur les items avec alarme, MGT-B produit 19 corrections et 4 régressions. Sur les items sans alarme, aucune différence de correction n'est observée et les 218 suites de tokens émises sont identiques entre méthodes. Cette observation valide empiriquement l'invariant de no-alarm identity sur le test réalisé.

L'analyse conditionnelle aux alarmes est descriptive : le sous-groupe est défini après observation par le contrôleur et ne constitue pas un essai séparément randomisé.

### 3.4 Résultats exploratoires par domaine

| Domaine | N | Vanilla correct | MGT-B correct | Gain net | Items avec alarme |
|---|---:|---:|---:|---:|---:|
| Algebra | 64 | 46 | 49 | +3 | 13 |
| Counting and probability | 21 | 7 | 9 | +2 | 11 |
| Geometry | 25 | 16 | 17 | +1 | 6 |
| Intermediate algebra | 63 | 24 | 31 | +7 | 24 |
| Number theory | 37 | 25 | 24 | −1 | 5 |
| Prealgebra | 56 | 29 | 32 | +3 | 13 |
| Precalculus | 34 | 18 | 18 | 0 | 10 |

Le gain le plus important apparaît en algèbre intermédiaire. Ces analyses n'étaient pas la métrique principale, les effectifs sont faibles dans plusieurs catégories et aucune correction pour comparaisons multiples n'est appliquée ; elles doivent donc être considérées comme exploratoires.

### 3.5 Coût de génération

| Mesure | Vanilla | MGT-B | Variation |
|---|---:|---:|---:|
| Tokens échantillonnés | 1 292 246 | 1 397 313 | +105 067 (+8,1 %) |
| Tokens émis/retenus | 1 292 246 | 1 315 457 | +23 211 (+1,8 %) |
| Tokens supprimés | 0 | 81 856 | +81 856 |
| Latence moyenne par item | 230,53 s | 253,97 s | +23,44 s (+10,2 %) |
| Pic VRAM enregistré | 2 281 531 904 o | 2 283 340 288 o | +0,08 % |

Les 82 rollbacks ont une longueur moyenne de 998 tokens et une médiane de 768 tokens. Leur étendue va de 256 à 3 328 tokens. Les alarmes surviennent entre les positions 288 et 12 032, avec une médiane à 3 360 tokens.

La mesure de VRAM est le maximum enregistré dans un processus de génération. Elle ne représente pas la somme de l'occupation des six workers simultanés et ne doit pas être interprétée comme la mémoire totale requise par l'exécution parallèle.

La latence correspond au temps mural enregistré par item. Les deux conditions ont été exécutées séquentiellement avec la même concurrence, mais cette mesure n'est pas issue d'un benchmark alterné et contrôlé ; elle décrit le coût de ce run plutôt qu'une garantie de performance générale.

### 3.6 Extractabilité et troncatures

| Mesure | Vanilla | MGT-B |
|---|---:|---:|
| Réponses extractables | 297/300 (99,0 %) | 298/300 (99,3 %) |
| Troncatures | 1/300 (0,33 %) | 5/300 (1,67 %) |
| Terminaisons EOS | 299 | 295 |
| Terminaisons au budget maximal | 1 | 5 |

L'extractabilité reste élevée dans les deux conditions. En revanche, MGT-B augmente de quatre le nombre de générations atteignant la limite de 20 000 tokens. Cette hausse est cohérente avec les rerolls et l'allongement possible de la trajectoire retenue. Elle doit être suivie dans les réplications futures.

## 4. Interprétation

Trois observations soutiennent l'efficacité du mécanisme dans cette expérience :

1. le gain global de 5 points est positif dans tout l'intervalle bootstrap à 95 % ;
2. le rapport corrections/régressions est fortement favorable, avec 19 corrections pour 4 régressions ;
3. les sorties sans alarme restent token-identiques, de sorte que les différences observées sont localisées aux interventions MGT-B.

Les items détectés sont nettement plus difficiles pour Vanilla que les items sans alarme : 29,3 % d'exactitude contre 64,7 %. Le détecteur identifie donc un sous-ensemble enrichi en trajectoires en échec. Sur ce sous-ensemble, l'intervention récupère 15 succès nets, au prix d'environ 105 000 tokens échantillonnés supplémentaires sur l'ensemble du test.

Le résultat répond positivement à la question principale pour cette configuration. Il ne démontre pas encore que chaque composant du contrôleur est nécessaire, ni que le gain se transpose à d'autres modèles, quantifications, domaines ou seeds.

## 5. Limites et menaces à la validité

### 5.1 Une génération par condition

Le protocole rapide utilise une seule génération par problème et par méthode. Le pairing des seeds réduit la variance de comparaison, mais ne mesure pas la variabilité entre plusieurs tirages sur un même problème. Une réplication multi-seed serait nécessaire pour caractériser cette composante.

### 5.2 Un seul modèle et une seule précision

L'expérience porte uniquement sur DeepSeek-R1-Distill-Qwen-1.5B en INT4 FP4. Les conclusions ne peuvent pas être directement extrapolées à FP16/BF16, à d'autres tailles de modèle ou à d'autres familles.

### 5.3 Sous-ensemble de MATH-500

Le test utilise 300 des 500 problèmes MATH-500, sélectionnés de façon déterministe avant le tuning. L'inférence bootstrap traite ces 300 problèmes comme unités d'analyse, mais la généralisation à l'ensemble des tâches mathématiques reste limitée.

### 5.4 Calibration de développement

Le seuil est sélectionné sur 62 trajectoires healthy. Cet effectif dépasse le seuil d'avertissement prévu par le pipeline, mais demeure modeste. L'estimation de la fréquence d'alarme healthy à 4,84 % correspond à seulement trois trajectoires.

### 5.5 Analyses secondaires

Les résultats par domaine et conditionnels aux alarmes sont descriptifs et postérieurs à la métrique principale. Ils ne doivent pas servir à modifier le contrôleur puis à réévaluer sur les mêmes 300 items.

### 5.6 Coûts système

La latence n'a pas été mesurée dans un benchmark alterné ou avec isolement strict de la charge machine. La VRAM agrégée des six workers n'est pas directement enregistrée. Les chiffres de coût sont donc appropriés pour documenter ce run, mais pas pour conclure finement sur le débit ou la capacité minimale de la GPU.

## 6. Conclusion

Sur les 300 problèmes MATH-500 gelés, MGT-B augmente l'exactitude de **55,0 % à 60,0 %**, soit **+5,0 points** et **15 solutions correctes nettes supplémentaires**. Le signal statistique est robuste dans l'analyse préspécifiée : IC bootstrap 95 % **[+2,0 ; +8,0]** et McNemar exact bilatéral **p = 0,00260**.

Le gain provient exclusivement des items ayant déclenché une intervention. Le contrôleur préserve exactement les sorties Vanilla lorsqu'aucune alarme n'est émise. Le coût observé est une hausse de 8,1 % des tokens échantillonnés et de 10,2 % de la latence moyenne, ainsi qu'une augmentation des troncatures de 1 à 5 items.

Le résultat principal doit désormais être considéré comme figé. Le seuil et le contrôleur ne doivent pas être retunés sur ces 300 items test.

## 7. Étapes scientifiques recommandées

1. **Archiver le résultat principal** avec le manifest, le calibrateur, le seuil, les freezes, les 600 artefacts bruts et ce rapport.
2. **Effectuer une revue qualitative sans retuning** des 19 corrections, 4 régressions et 5 troncatures MGT-B afin de comprendre les modes d'action et d'échec.
3. **Pré-enregistrer une réplication indépendante** avant tout nouveau run : nouvelle partition ou nouveau dataset tenu à l'écart, métrique principale et règles d'arrêt figées.
4. **Ajouter une réplication multi-seed** pour estimer la variabilité stochastique et vérifier la stabilité du gain de 5 points.
5. **Tester la généralisation** sur au moins un autre modèle ou une autre précision.
6. **Conduire des ablations prospectives** sur le point de changement, le rollback adaptatif, le blocage des n-grams et la température de redécodage. Ces ablations doivent employer de nouvelles données de validation/test ou un protocole explicitement séparé.
7. **Mesurer les coûts dans un benchmark dédié**, notamment le débit, la VRAM totale à différents nombres de workers et le coût par correction nette.

## 8. Reproductibilité et artefacts

Environnement enregistré dans les freezes :

| Composant | Version |
|---|---|
| GPU | NVIDIA RTX A5000 |
| CUDA | 12.8 |
| Python | 3.9.25 |
| PyTorch | 2.8.0+cu128 |
| Transformers | 4.57.6 |
| bitsandbytes | 0.48.2 |
| datasets | 4.5.0 |

Artefacts principaux :

| Artefact | Chemin |
|---|---|
| Spécification | `docs/MGTB_FAST_REIMPLEMENTATION_SPEC_v2.md` |
| Manifest | `outputs/science_fast/splits/manifest.json` |
| Résumé référence | `outputs/science_fast/calibration/reference_summary.json` |
| Calibrateur | `outputs/science_fast/calibration/calibrator.json` |
| Seuil | `outputs/science_fast/calibration/threshold.json` |
| Freeze Vanilla | `outputs/science_fast/freeze/vanilla_int4.lock.json` |
| Freeze MGT-B | `outputs/science_fast/freeze/mgtb_int4.lock.json` |
| Artefacts Vanilla | `outputs/science_fast/test/vanilla/items/` |
| Artefacts MGT-B | `outputs/science_fast/test/mgtb/items/` |
| Résultat apparié | `outputs/science_fast/analysis/paired_results.json` |

Commande de reconstruction de l'analyse à partir des artefacts :

```bash
python scripts/analyze_scientific_results.py --config configs/science_fast/analyze.yaml
```

SHA-256 du fichier de résultats appariés :

```text
ab56ecf64efdd3e6d68ad2de9b6a16211352e470a5a4f030d89ee2442c215591
```
