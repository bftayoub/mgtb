# Plan d'extension du repo — MGT-B / CUSUM-Shaped Monitoring

## Objectif

Faire évoluer le repo pour permettre une **évaluation publiable et reproductible** de MGT-B, sans modifier rétroactivement les résultats existants.

Le principal manque actuel n'est pas une nouvelle méthode, mais une validation expérimentale plus propre : vrai test prospectif, ablations causales, baselines fortes, généralisation et reproductibilité.

Le repo doit conserver les expériences historiques comme **exploratoires** et ajouter un nouveau protocole clairement séparé.

---

## 1. Geler une version scientifique `MGT-B v1.0`

Créer une configuration immuable contenant au minimum :

- window length / stride ;
- longueurs de n-grams ;
- poids des 6 features ;
- calibration positionnelle ;
- mixture betting factors ;
- CUSUM-shaped reset ;
- seuil `h` ;
- rollback margin ;
- nombre maximal de rerolls ;
- refractory period ;
- température de re-decoding ;
- repetition penalty ;
- suspect n-gram blocking ;
- scorer ;
- modèle, révision, quantification et seeds.

Chaque run doit enregistrer la config complète, son hash, la révision Git et l'environnement logiciel.

---

## 2. Mettre en place un vrai protocole `reference / dev / test`

Le repo doit imposer trois partitions disjointes :

- `reference` : construit les distributions empiriques / ECDF ;
- `dev` : choix du seuil et éventuels hyperparamètres ;
- `test` : **jamais utilisé avant l'évaluation finale**.

Contraintes :

- aucune donnée `test` ne peut servir à la calibration, au tuning ou à la sélection de configuration ;
- le test final doit utiliser une configuration déjà gelée ;
- sauvegarder un manifest avec les IDs de chaque partition ;
- ajouter des assertions empêchant toute fuite entre splits.

Le nouveau test prospectif doit devenir l'expérience principale du papier. Les anciens ensembles `chronology-audit` et `historical-coverage` restent disponibles comme analyses historiques.

---

## 3. Expériences prioritaires à implémenter

### P0 — indispensables

#### A. Réplication prospective propre

Comparer sur le nouveau `test` :

- Vanilla ;
- MGT-B v1.0.

Même problème, même seed initiale, même modèle et même budget de sampling.

Reporter : accuracy, corrections, regressions, McNemar exact, bootstrap paired CI, alarm frequency, sampled/emitted/deleted tokens et coût réel.

#### B. Ancien contrôleur vs MGT-B

Rerun le **contrôleur token-level précédent** sur exactement les mêmes IDs, seeds, budgets et scorer :

- Vanilla ;
- Previous controller ;
- MGT-B.

But : montrer si les changements windowing + calibration positionnelle + accumulation + rollback adaptatif améliorent réellement la version antérieure.

#### C. `matched-random intervention`

Ajouter une baseline qui utilise **exactement le même opérateur de réparation que MGT-B**, mais dont les alarmes sont aléatoires.

Idéalement conserver :

- même nombre total d'alarmes ;
- distribution de positions similaire ;
- même rollback/re-decoding ;
- même budget.

But : isoler la valeur informative du monitor, indépendamment du simple fait d'intervenir.

---

### P1 — attribution du mécanisme

#### D. Ablations du monitor

Supporter au minimum :

- entropy-only ;
- repetition-only ;
- entropy + repetition simple ;
- full 6-feature monitor ;
- global calibration vs position-conditional calibration ;
- reset CUSUM-shaped vs non-reset accumulation.

Toutes ces variantes doivent utiliser le **même repair operator** lorsqu'une alarme est déclenchée.

#### E. Ablations de l'intervention

À trigger identique, comparer :

- fixed rollback vs adaptive rollback ;
- rollback seul ;
- rollback + lower temperature ;
- rollback + repetition penalty ;
- rollback + n-gram blocking ;
- full MGT-B repair.

But : déterminer si le gain vient du choix de l'alarme, de la localisation du rollback ou de la politique de re-decoding.

---

### P2 — baselines et généralisation

#### F. Baselines fortes de test-time compute

Ajouter au minimum :

- best-of-N ;
- self-consistency.

Comparer sous un budget transparent :

- même budget maximal de sampled tokens ; ou
- courbe accuracy vs sampled tokens.

Éviter une comparaison où MGT-B et les baselines disposent de budgets implicitement différents.

#### G. Plusieurs seeds

Permettre plusieurs générations par problème.

Minimum souhaitable :

- 3–5 seeds sur un sous-ensemble substantiel ;
- mêmes seeds pour toutes les méthodes comparées.

Reporter la variabilité entre trajectoires, pas seulement entre problèmes.

#### H. Généralisation

Préparer le repo pour lancer sans changement de code :

- un deuxième modèle / une autre famille ;
- un deuxième benchmark de raisonnement ;
- idéalement une comparaison BF16/FP16 vs 8-bit vs 4-bit.

Cela permettra de vérifier si MGT-B est particulièrement utile sous quantification ou s'il s'agit d'un phénomène plus général.

---

## 4. Scoring et métriques

Conserver le scorer historique pour reproductibilité, mais ajouter si possible un scorer mathématique plus robuste.

Pour chaque paire Vanilla / méthode, produire automatiquement :

- exact-normalized accuracy ;
- corrections / regressions ;
- McNemar exact ;
- paired bootstrap 95% CI ;
- extractability ;
- alarm / intervention frequency ;
- sampled, emitted et deleted tokens ;
- nombre de rerolls ;
- position des alarmes et taille des rollbacks ;
- wall-clock latency et, si possible, peak VRAM.

Les statistiques doivent être calculées à partir des raw artifacts, jamais saisies manuellement.

---

## 5. Reproductibilité exigée

Chaque run doit sauvegarder :

- commande exacte ;
- config complète ;
- hash de config ;
- Git commit ;
- modèle + revision ;
- dataset + IDs ;
- seeds ;
- versions Python / PyTorch / Transformers / bitsandbytes / CUDA ;
- GPU ;
- raw generations ;
- traces du monitor ;
- résultats intermédiaires et finaux.

Créer un script unique de type :

```bash
python scripts/run_scientific_experiment.py --config <config.yaml>
```

et un script d'analyse :

```bash
python paper/figures/scripts/analyze_scientific_results.py --manifest <manifest.json>
```

Les résultats de random rollback, periodic rollback, restart et self-correction doivent être **reproduits avec leurs raw artifacts** dans le nouvel environnement, car les artefacts bruts historiques ont été perdus.

---

## 6. Organisation recommandée des configs

Exemple :

```text
configs/science/
  mgtb_v1.yaml
  vanilla.yaml
  previous_controller.yaml
  matched_random.yaml
  monitor_entropy_only.yaml
  monitor_repetition_only.yaml
  monitor_simple_combined.yaml
  monitor_global_calibration.yaml
  monitor_no_reset.yaml
  repair_fixed_rollback.yaml
  repair_rollback_only.yaml
  repair_temp_only.yaml
  repair_penalty_only.yaml
  repair_ngram_only.yaml
  best_of_n.yaml
  self_consistency.yaml
```

Éviter de dupliquer la logique du pipeline : les différences entre expériences doivent être contrôlées par configuration.

---

## 7. Ordre de réalisation recommandé

1. Protocole `reference/dev/test` + contrôles anti-fuite.
2. MGT-B v1.0 gelé + Vanilla.
3. Previous controller.
4. Matched-random intervention.
5. Monitor ablations essentielles.
6. Repair ablations essentielles.
7. Best-of-N / self-consistency.
8. Multi-seed.
9. Deuxième modèle / deuxième dataset / précision supplémentaire.
10. Régénération complète des tableaux et figures du papier depuis les raw artifacts.

---

## Critère de sortie

Le repo est prêt pour une nouvelle soumission lorsque l'on peut lancer une expérience prospective **sans toucher au test pendant le développement**, reconstruire chaque résultat depuis les raw artifacts, comparer MGT-B à ses composants et à des baselines fortes sous budgets comparables, et reproduire la conclusion sur au moins un setting supplémentaire.
