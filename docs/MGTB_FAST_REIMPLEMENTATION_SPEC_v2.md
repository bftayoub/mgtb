# MGT-B — Spécification de réimplémentation rapide

## 0. Objectif

Reconstruire correctement le pipeline scientifique MGT-B après perte du disque / retour du dépôt à une version antérieure.

Priorité immédiate :
1. rétablir l'approche MGT-B telle qu'elle est décrite ici ;
2. rétablir un protocole `reference / development / test` anti-fuite ;
3. lancer en premier **INT4 Vanilla vs INT4 MGT-B** ;
4. permettre la reprise propre de tous les runs interrompus ;
5. conserver les artefacts nécessaires pour reprendre ensuite les ablations et baselines.

---

## 1. Protocole accéléré

Une seule génération par problème et par approche dans chaque phase.

| Rôle | Source | Taille | Passes |
|---|---|---:|---:|
| `reference` | MATH original train | **300 problèmes** | **1 génération / problème** |
| `development` | MATH original train | **100 problèmes** | **1 génération / problème** |
| `test` | MATH-500 test | **300 problèmes** | **1 génération / problème / approche** |

Pour le test :

```text
Vanilla : 1 génération par problème
MGT-B   : 1 génération par problème
```

Pas de multi-seed dans ce protocole.

`300 reference` signifie 300 problèmes lancés. Seules les trajectoires correctes, extractables et non tronquées alimentent ensuite l'ECDF healthy.

---

## 2. Sources et versions immuables

### Modèle

```yaml
base_model: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
model_revision: ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562
quantization: bitsandbytes_int4
bnb_4bit_quant_type: fp4
bnb_4bit_use_double_quant: false
bnb_4bit_compute_dtype: float16
storage_dtype: uint8
```

### Reference / development

```yaml
dataset_name: EleutherAI/hendrycks_math
dataset_revision: 21a5633873b6a120296cce3e2df9d5550074f4a3
split: train
```

### Test

```yaml
dataset_name: HuggingFaceH4/MATH-500
dataset_revision: 6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be
split: test
```

Paramètres communs :

```yaml
prompt_style: math500_cot
max_new_tokens: 20000
vanilla_temperature: 1.0
```

---

## 3. Partitions déterministes

Utiliser :

```yaml
protocol_seed: 20260811
```

Pour chaque problème :

```text
content_sha256 = SHA256(normalize(problem_text))
selection_key = SHA256("20260811|" + content_sha256)
```

La normalisation ne dépend ni de l'ID source ni de la réponse.

### MATH train

1. charger le train à la revision pinée ;
2. calculer les hashes ;
3. rejeter les doublons de contenu pour la sélection scientifique ;
4. trier par `selection_key` ;
5. prendre les 300 premiers contenus uniques comme `reference` ;
6. prendre les 100 suivants comme `development`.

### MATH-500

1. charger les 500 items à la revision pinée ;
2. calculer les hashes ;
3. trier par `selection_key` ;
4. prendre les 300 premiers comme `test`.

Créer le manifest avant tout tuning et y enregistrer source, revision, rôle, IDs, content hashes, selection keys, comptes et hash du manifest.

Assertions obligatoires par `content_sha256` :

```text
reference ∩ development = ∅
reference ∩ test = ∅
development ∩ test = ∅
```

---

## 4. Seed unique et pairing

Utiliser une seule seed déterministe par item :

```text
item_seed = stable_hash(protocol_seed, stable_item_id)
```

Même `item_seed` pour Vanilla et MGT-B sur un même problème.

L'ordre d'exécution, le batching et une reprise de run ne doivent pas modifier le seed d'un item.

---

## 5. Checkpoints et reprise de runs

Tous les runs réels doivent être **reprenables sans recommencer les exemples déjà terminés**.

### Exigences

Après chaque exemple terminé, sauvegarder de manière atomique au minimum :

- `item_id`;
- `content_sha256`;
- `item_seed`;
- statut `completed`;
- génération brute ;
- token IDs ;
- scorer output ;
- métriques de tokens ;
- traces monitor si MGT-B ;
- timing ;
- erreurs éventuelles ;
- hash/config/provenance du run.

Maintenir un index/checkpoint de progression, par exemple :

```text
completed_items.jsonl
run_state.json
```

Au redémarrage :

1. charger le manifest et la config ;
2. vérifier qu'ils correspondent exactement au run interrompu ;
3. lire les items déjà `completed`;
4. ignorer uniquement ceux dont l'artifact final est valide ;
5. reprendre au prochain item non terminé ;
6. ne jamais modifier le seed d'un item repris.

Les writes doivent être atomiques (`tmp` puis rename) pour éviter qu'une coupure laisse un artifact partiel considéré comme terminé.

### Gestion des items interrompus

Un item dont l'artifact final n'est pas marqué `completed` doit être relancé depuis le début avec **le même item seed**.

Il n'est pas nécessaire de reprendre une génération au milieu d'un problème. Le checkpoint requis est **au niveau de l'exemple**, ce qui suffit pour éviter de perdre plusieurs heures de run.

### Compatibilité

Le même mécanisme de reprise doit fonctionner pour :

- reference ;
- development ;
- test Vanilla ;
- test MGT-B.

---

## 6. Monitor MGT-B

Fenêtres :

```yaml
window_length: 64
stride: 32
ngram_lengths: [6, 7, 8]
```

Signaux token-level :

\[
H_t=-\sum_v p_t(v)\log p_t(v)
\]

\[
\ell_t=\log p_t(W_t)
\]

Features fenêtre :

\[
Z_j=[\bar H_j,-\bar\ell_j,R_j,D_j,L_j^+,L_j^-]
\]

avec :

\[
\bar H_j=\frac1{|B_j|}\sum_{t\in B_j}H_t
\]

\(R_j\) = fraction de n-grams générés 6–8 déjà vus dans la génération retenue.

\(D_j\) = confident repetition selon la définition MGT-B décrite dans le repo/papier : gain positif maximal de confiance d'un n-gram répété par rapport à ses occurrences antérieures.

\[
r_j=\log\frac{\bar H_j}{\bar H_{\mathrm{prefix},j}},
\quad
L_j^+=\max(r_j,0),
\quad
L_j^-=\max(-r_j,0)
\]

Score :

\[
s_j =
0.15\bar H_j+
0.10(-\bar\ell_j)+
0.20R_j+
0.35D_j+
0.18L_j^+
+0.02L_j^-.
\]

---

## 7. Calibration positionnelle

Buckets :

```yaml
position_buckets:
  - [0, 512]
  - [512, 1024]
  - [1024, 2048]
  - [2048, 4096]
  - [4096, null]
p_clip: 1.0e-6
```

Healthy reference :
- correct ;
- extractable ;
- non-truncated.

Pour \(s_j\) :

\[
q_j=
\frac{1+\sum_{u\in C_{b(j)}}\mathbf 1\{u\ge s_j\}}
{|C_{b(j)}|+1}.
\]

Si un bucket est vide, conserver le fallback historique vers le pool global.

Le calibrator sauvegarde les pools, nombre de fenêtres, nombre de trajectoires healthy, IDs/hashes retenus, bornes, `p_clip` et provenance complète.

---

## 8. Betting transform et accumulation

```yaml
gammas: [0.1, 0.3, 0.5, 0.7]
```

\[
\tilde q_j=\min(1,\max(q_{\min},q_j))
\]

\[
e_j=\frac1{|\Gamma|}\sum_{\gamma\in\Gamma}\gamma\tilde q_j^{\gamma-1}
\]

\[
S_j=\max(0,S_{j-1})+\log e_j,\qquad S_0=0
\]

Alarme :

\[
S_j\ge h.
\]

Employer `betting_factor`, `CUSUM-shaped statistic`, `empirical threshold`.

---

## 9. Threshold sur 100 development

Le `reference` construit uniquement l'ECDF.

Le `development` utilise l'ECDF gelée pour choisir le threshold.

```yaml
target_healthy_alarm_rate: 0.05
```

Procédure :
1. générer les 100 development, une seule fois chacun ;
2. calculer leurs features ;
3. appliquer le calibrator reference sans le modifier ;
4. identifier les development healthy ;
5. rejouer le statistic ;
6. chercher le plus petit `h` sur la grille pré-définie dont l'alarm rate par trajectoire healthy est ≤ 5 % ;
7. sauvegarder threshold + diagnostics.

Si le dénominateur healthy est faible, afficher un warning mais ne pas modifier adaptativement les tailles.

---

## 10. Intervention MGT-B

```yaml
rollback_margin_tokens: 64
max_rerolls: 3
refractory_windows: 2
redecode_temperature: 0.6
repetition_penalty: 1.1
suspect_ngram_blocking: true
prompt_injection: false
```

À l'alarme :
1. scanner l'historique de \(S_j\) jusqu'au dernier reset/non-positif pertinent ;
2. mapper la fenêtre à une position token ;
3. étendre le rollback 64 tokens plus tôt ;
4. supprimer le suffixe ;
5. restaurer tokens, KV-cache, features, n-grams et détecteur ;
6. reset detector ;
7. appliquer deux fenêtres réfractaires ;
8. re-décoder à T=0.6 avec penalty 1.1 et blocage ciblé ;
9. autoriser au plus trois rerolls.

---

## 11. Corrections prospectives obligatoires

### `cache_state_mode: replay_last`

Après rollback, tokens et cache doivent représenter exactement le même préfixe.

Approche sûre :
- crop du cache jusqu'avant le dernier token retenu ;
- replay du dernier token retenu ;
- reprise de génération du token suivant.

Test obligatoire : logits après reconstruction ≈ logits d'un forward propre sur le même préfixe.

### `changepoint_index_mode: tracked_windows`

Après reroll :
- supprimer toute fenêtre appartenant au suffixe abandonné ;
- reconstruire correctement monitor/n-gram state ;
- les index suivent les vrais tokens retenus/re-générés ;
- aucun changepoint ne référence une fenêtre stale.

---

## 12. Invariants

### No-alarm identity

Pour un même item/seed :

```text
si aucune alarme :
MGT-B token_ids == Vanilla token_ids
```

### Token accounting

Tracer :
- sampled ;
- emitted ;
- deleted ;
- alarms ;
- rerolls ;
- alarm positions ;
- rollback spans ;
- termination reason.

Les tokens abandonnés comptent dans `sampled`.

---

## 13. Scorer et métriques

Conserver le scorer exact-normalized historique.

Sur le test 300, produire :
- accuracy Vanilla ;
- accuracy MGT-B ;
- différence en pp ;
- corrections ;
- regressions ;
- exact two-sided McNemar ;
- paired problem-level bootstrap 95 % CI ;
- extractability ;
- truncation rate ;
- alarm / reroll frequency ;
- sampled / emitted / deleted tokens ;
- alarm positions ;
- rollback lengths ;
- wall-clock latency ;
- peak VRAM si disponible.

Une seule génération par approche et par problème.

---

## 14. Freeze avant MATH-500

Avant toute exécution test, créer un lock contenant :
- hash du manifest ;
- IDs/content hashes test ;
- modèle + revision ;
- quantification ;
- dataset revisions ;
- protocol seed / seed strategy ;
- calibrator hash ;
- threshold development ;
- config MGT-B résolue ;
- scorer hash/version ;
- budget ;
- Git commit + source-tree hash ;
- environnement logiciel ;
- GPU.

Le runner `protocol_role: test` doit refuser de démarrer si le freeze manque ou ne correspond pas.

Créer un freeze pour :
- Vanilla INT4 ;
- MGT-B INT4.

---

## 15. Organisation minimale

```text
configs/science_fast/
  protocol.yaml
  controller_v1.yaml
  collect_reference_int4.yaml
  collect_development_int4.yaml
  test_vanilla_int4.yaml
  test_mgtb_int4.yaml

outputs/science_fast/
  splits/
  features/reference/
  features/development/
  calibration/
  checkpoints/
  freeze/
  test/vanilla/
  test/mgtb/
  analysis/
```

Réutiliser au maximum le code existant.

---

## 16. Tests avant GPU

Unit tests :
- entropy ;
- chosen log-prob ;
- R ;
- D ;
- L+/L- ;
- score ;
- bucket lookup ;
- q empirique ;
- beta mixture ;
- CUSUM recurrence ;
- rollback ;
- n-gram blocking ;
- max rerolls / refractory.

Anti-fuite :
- overlap ID ;
- overlap content hash ;
- dataset/revision mismatch ;
- test avant freeze refusé.

Rollback/KV :
- crop ;
- replay_last ;
- logits reconstructed vs clean forward ;
- state n-gram/monitor après reroll ;
- aucune stale window.

Pairing/reprise :
- même seed par item entre méthodes ;
- ordre des items indépendant ;
- no-alarm exact identity ;
- interruption après quelques items puis reprise ;
- les items terminés ne sont pas recalculés ;
- l'item interrompu est relancé avec le même seed ;
- reprise et run continu produisent les mêmes artifacts finaux.

Smoke synthétique, sans consommer les 300 MATH-500 test :
- Vanilla ;
- MGT-B no alarm ;
- MGT-B avec alarme forcée ;
- interruption/reprise ;
- sauvegarde/reload ;
- analyse paired.

---

## 17. Reproductibilité

Chaque run sauvegarde :
- commande ;
- config résolue ;
- hash config ;
- Git commit ;
- source-tree hash si dirty ;
- modèle + revision ;
- dataset + revision ;
- IDs/content hashes ;
- seed strategy ;
- versions Python/PyTorch/Transformers/bitsandbytes/CUDA ;
- GPU ;
- raw token IDs/text ;
- scorer outputs ;
- monitor traces ;
- token accounting ;
- timing.

Tous les tableaux sont reconstruits depuis les raw artifacts.

---

## 18. Ordre d'exécution

1. Manifest : 300 ref + 100 dev MATH train ; 300 test MATH-500.
2. Reference INT4 : 300 items, une passe chacun, checkpoint après chaque item.
3. Construire l'ECDF.
4. Afficher : completed, correct, extractable, truncated, healthy retained, fenêtres et trajectoires par bucket.
5. Development INT4 : 100 items, une passe chacun, checkpoint après chaque item.
6. Sélectionner `h`.
7. Freeze Vanilla + MGT-B.
8. Test MATH-500 :
   - Vanilla INT4 : 300 items, une passe chacun ;
   - MGT-B INT4 : mêmes 300 items, une passe chacun ;
   - checkpoint après chaque exemple.
9. Analyse paired.
10. Ensuite seulement : previous controller, matched-random, ablations, self-consistency, multi-seed, autres settings.

---

## 19. Commit et push

Après avoir terminé l'implémentation, les tests et les smoke tests :

1. vérifier `git status` ;
2. ne pas versionner les gros outputs, caches, modèles ou artefacts temporaires ;
3. ajouter les fichiers source/config/tests/docs pertinents ;
4. faire un commit explicite, par exemple :

```text
Rebuild resumable MGT-B scientific pipeline
```

5. pousser la branche courante vers son remote/upstream.

Avant le push, afficher :
- branche courante ;
- remote ciblé ;
- fichiers inclus dans le commit ;
- hash du commit.

Si aucun upstream n'existe, configurer l'upstream vers la branche distante correspondante plutôt que créer une autre organisation de branches sans raison.

---

## 20. Critère de réussite

La reconstruction est réussie lorsque :
1. les partitions 300/100/300 sont déterministes ;
2. aucune fuite de contenu n'est possible ;
3. l'ECDF vient uniquement de reference ;
4. `h` vient uniquement de development ;
5. le test refuse de démarrer sans freeze ;
6. Vanilla/MGT-B sont pairés par item seed ;
7. chaque item n'est généré qu'une fois par approche dans chaque phase, hors relance nécessaire d'un item interrompu ;
8. les runs sont reprenables après interruption sans refaire les items terminés ;
9. no-alarm MGT-B est token-identique à Vanilla ;
10. rollback restaure tokens + KV + monitor + n-grams ;
11. les résultats sont reconstruits depuis les raw artifacts ;
12. le code est testé, commité et poussé ;
13. la première comparaison disponible est **INT4 Vanilla vs INT4 MGT-B sur 300 MATH-500**.
