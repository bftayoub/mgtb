# Rapport synthétique — Vanilla, full MGT-B et matched-random sur trois seeds

## Résumé exécutif

Sur MATH-500, avec 500 problèmes évalués sous trois seeds appariées (1 500 unités par méthode), `full_mgtb` atteint **56,40 %** d'exactitude, contre **54,73 %** pour Vanilla et **54,60 %** pour `matched_random`. Le gain de `full_mgtb` est donc de **+1,67 point** face à Vanilla (IC bootstrap clusterisé à 95 % : **[+0,47 ; +2,80]**) et de **+1,80 point** face à `matched_random` (**[+0,47 ; +3,13]**). Le gain est positif sur chacune des trois seeds.

Le résultat fournit une preuve confirmatoire, dans le cadre du protocole gelé, d'un bénéfice reproductible de MGT-B sur MATH-500 avec ce modèle. Le contrôle `matched_random` a été calibré prospectivement à partir d'un run `full_mgtb` sur le développement, indépendamment des labels du test. Il constitue donc bien le contrôle aléatoire prévu par le protocole ; aucune recalibration ex post ni répétition de cette comparaison n'est requise pour clore cette campagne.

## Périmètre et validité de l'agrégation

- Dataset : les 500 problèmes du split test de `HuggingFaceH4/MATH-500`.
- Modèle : `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`, révision `ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562`, en INT4.
- Seeds appariées : 0, 1 et 2 ; mêmes problèmes pour chaque méthode et chaque seed.
- `matched_random` : profil d'intervention construit avant le test à partir de `full_mgtb` sur le split de développement, sans utilisation des labels du test. L'écart entre les taux d'intervention finalement observés sur le test — 22,6 % pour `full_mgtb` et 9,4 % pour `matched_random` — est un résultat du protocole prospectif, pas une erreur de calibration.
- Les hashes du manifeste et des définitions des trois variantes sont identiques entre la seed 0 et les seeds 1–2.
- La seed 0 conserve dans ses artefacts l'étiquette historique `exploratory`. La documentation du dépôt indique qu'il s'agit d'une erreur de métadonnée : les seeds 0, 1 et 2 appartiennent au même plan d'évaluation gelé et aucun retuning du modèle, du contrôleur, de la calibration ou du profil `matched_random` n'a été effectué à partir des résultats. Les modifications de code et commits intermédiaires ont corrigé l'implémentation et la provenance sans modifier le mécanisme scientifique ni les résultats authentifiés. La campagne est donc une **évaluation confirmatoire gelée**, indépendante de tout ajustement post hoc aux résultats du test.

L'inférence principale est un bootstrap apparié clusterisé par problème (10 000 tirages, graine 20260811), afin de ne pas traiter les trois seeds d'un même problème comme indépendantes. Les tests exacts de McNemar au niveau des 1 500 unités sont fournis à titre descriptif.

## Résultats par seed

| Seed | Vanilla | `full_mgtb` | `matched_random` | MGT-B − Vanilla | MGT-B − aléatoire |
|---:|---:|---:|---:|---:|---:|
| 0 | 55,2 % | 56,2 % | 55,0 % | +1,0 pt | +1,2 pt |
| 1 | 54,0 % | 56,8 % | 54,0 % | +2,8 pts | +2,8 pts |
| 2 | 55,0 % | 56,2 % | 54,8 % | +1,2 pt | +1,4 pt |
| **Agrégé** | **54,73 %** | **56,40 %** | **54,60 %** | **+1,67 pt** | **+1,80 pt** |

L'écart-type de l'exactitude entre seeds est de 0,64 point pour Vanilla, 0,35 point pour `full_mgtb` et 0,53 point pour `matched_random`. Le signe du gain de MGT-B est donc stable sur les trois réplications, même si trois seeds restent insuffisantes pour caractériser finement la variance inter-seed.

## Comparaisons appariées

| Comparaison | Corrections / régressions | Différence | IC 95 % clusterisé | McNemar brut | McNemar Holm |
|---|---:|---:|---:|---:|---:|
| `full_mgtb` − Vanilla | 53 / 28 | **+1,67 pt** | **[+0,47 ; +2,80]** | 0,0073 | 0,0146 |
| `matched_random` − Vanilla | 16 / 18 | −0,13 pt | [−0,93 ; +0,67] | 0,8642 | 0,8642 |
| `full_mgtb` − `matched_random` | 67 / 40 | **+1,80 pt** | **[+0,47 ; +3,13]** | 0,0116 | non inclus dans la famille Holm principale |

L'IC clusterisé est l'élément d'inférence à privilégier. Il exclut zéro pour les deux comparaisons impliquant directement `full_mgtb`, mais pas pour `matched_random` face à Vanilla.

## Coût et comportement

| Méthode | Intervention | Tokens échantillonnés moyens | Écart vs Vanilla | Troncature | Extraction |
|---|---:|---:|---:|---:|---:|
| Vanilla | 0 % | 4 366,9 | — | 0,13 % | 99,73 % |
| `full_mgtb` | 22,6 % | 4 591,3 | **+5,14 %** | 1,20 % | 99,80 % |
| `matched_random` | 9,4 % | 4 360,7 | −0,14 % | 0,27 % | 99,73 % |

`full_mgtb` échantillonne 336 543 tokens supplémentaires sur 1 500 unités et supprime 317 312 tokens lors des rollbacks. Sa longueur finale émise n'augmente que de 0,29 % par rapport à Vanilla, mais son coût réel de génération augmente de 5,14 %. Les 1 161 unités sans alarme sont identiques token par token à Vanilla, ce qui confirme que le contrôleur ne modifie pas les trajectoires lorsqu'il reste inactif.

Le taux de troncature de `full_mgtb` augmente à 1,20 % (18 cas), contre 0,13 % pour Vanilla (2 cas). Ce coût de sûreté/longueur doit être analysé et réduit, car il peut masquer une partie du bénéfice ou créer un mode d'échec spécifique.

## Analyse par domaine

Les écarts descriptifs de `full_mgtb` face à Vanilla sont positifs en préalgèbre (+4,47 points), algèbre intermédiaire (+2,75), précalcul (+2,38), probabilités/combinatoire (+1,75) et algèbre (+1,34), mais négatifs en théorie des nombres (−2,15) et géométrie (−0,81). Ces analyses n'ont pas été corrigées pour comparaisons multiples et ne doivent pas être interprétées comme des effets confirmés par domaine.

## Conclusion scientifique

Les trois seeds établissent un résultat confirmatoire sur le périmètre évalué : MGT-B dépasse Vanilla de façon cohérente et l'intervalle de confiance clusterisé exclut l'absence d'effet. `matched_random`, correctement calibré sur le développement avant le test, ne dépasse pas Vanilla, tandis que `full_mgtb` le dépasse de +1,80 point avec un IC clusterisé excluant zéro. Dans le cadre des deux politiques définies prospectivement, ce résultat montre que les interventions guidées par MGT-B sont plus efficaces que les interventions aléatoires calibrées.

La campagne Vanilla–`full_mgtb`–`matched_random` est donc close et n'a pas à être répétée pour corriger son protocole. Ses limites concernent sa **portée**, non sa validité interne : un seul benchmark, un seul modèle 1,5B et une seule précision ont été évalués. Les trois seeds mesurent la robustesse à l'aléa de génération sur les mêmes 500 problèmes ; elles n'ajoutent pas 1 500 problèmes indépendants, ce que l'analyse prend correctement en compte en clusterisant le bootstrap par problème. Enfin, le taux de troncature plus élevé de MGT-B mérite une analyse d'erreurs. Ces limites encadrent la généralisation de la conclusion, sans invalider le résultat confirmatoire obtenu sur MATH-500.

## Prochaines étapes vers un article

1. **Conserver la campagne actuelle comme résultat principal clos.** Archiver son freeze, ses configs, ses calibrateurs, son profil `matched_random`, ses artefacts authentifiés et l'analyse trois-seeds. Ne pas la retuner ni la relancer pour modifier la conclusion.
2. **Élargir la validité externe.** Évaluer prospectivement le même mécanisme sur un second benchmark plus exigeant, par exemple Omni-MATH. Cette extension ne sert pas à « réparer » MATH-500, mais à déterminer si le résultat se généralise à une autre distribution.
3. **Faire les ablations essentielles.** Isoler le détecteur (entropie/répétition/CUSUM), puis l'opérateur de réparation (rollback, température, pénalité, blocage n-gram). Comparer à des baselines de calcul comme self-consistency à coût explicitement rapporté.
4. **Ajouter un second modèle.** Tester une autre taille ou famille de modèle afin de distinguer un effet général de MGT-B d'un effet propre à `DeepSeek-R1-Distill-Qwen-1.5B` INT4.
5. **Renforcer l'analyse.** Conserver le bootstrap clusterisé par problème comme analyse principale, ajouter les tailles d'effet et IC, une sensibilité aux troncatures, et une analyse descriptive des gains/pertes conditionnelle aux interventions. Éviter de sélectionner les analyses par domaine après observation.
6. **Produire les artefacts du papier.** Générer automatiquement une table principale, une courbe exactitude–coût, une figure corrections/régressions, un diagramme de la méthode et une table d'ablations depuis les artefacts authentifiés.
7. **Préparer la reproductibilité.** Archiver code, commits, environnement, manifests, configs gelées, calibrateurs, seeds et script d'analyse ; rédiger une model card expérimentale et une checklist de reproductibilité. Publier les sorties permises par les licences, ou au minimum leurs hashes et agrégats.
8. **Rédiger autour d'une contribution précise.** Structure recommandée : problème et hypothèse ; méthode MGT-B ; protocole apparié et gelé ; résultats efficacité/coût ; ablations ; analyse des échecs ; limites de généralisation. Présenter explicitement les trois seeds comme un seul plan confirmatoire gelé.

## Provenance

- Seed 0 : `outputs/science_campaign/math500_all500_exploratory_v1/`
- Seeds 1–2 : `outputs/science_campaign/math500_all500_confirmatory_seeds_1_2_v1/`
- Configuration multi-seeds : `configs/science_campaign/math500_all500_confirmatory_seeds_1_2.yaml`
- Analyse : agrégation des artefacts authentifiés, bootstrap clusterisé par les 500 problèmes, 10 000 réplications, graine 20260811.
