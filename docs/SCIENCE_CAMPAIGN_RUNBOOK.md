# Campagnes scientifiques et ablations MGT-B

## Portée

`scripts/run_ablation_campaign.py` est la pipeline destinée aux nouvelles expériences. Elle ne modifie ni les freezes ni les artefacts de `outputs/science_fast`.

Elle supporte :

- plusieurs seeds appariées par problème ;
- plusieurs modèles, révisions et précisions (`int4`, `fp16`, `bf16`, `fp32`) ;
- calibration positionnelle ou globale propre à chaque monitor ;
- accumulation CUSUM-shaped ou sans reset ;
- ablations entropy-only, repetition-only et entropy+repetition ;
- rollback adaptatif/fixe et ablations de température, pénalité et n-gram blocking ;
- matched-random utilisant le même `BacktrackingController` que MGT-B ;
- self-consistency et sélection best-of-N par log-probabilité moyenne ;
- bootstrap apparié, McNemar exact et correction de Holm ;
- reprise atomique et logs d'avancement par unité `(problème, seed)`.

Le fichier [math500_exploratory_ablations.yaml](../configs/science_campaign/math500_exploratory_ablations.yaml) est immédiatement valide, mais il réutilise le test MATH-500 déjà observé. Ses résultats doivent être étiquetés **exploratoires**. Le fichier [gsm8k_confirmatory_template.yaml](../configs/science_campaign/gsm8k_confirmatory_template.yaml) montre comment construire un nouveau test indépendant.

## Démarrage immédiat sur MATH / MATH-500

La campagne fournie utilise déjà six workers, une seule seed (`[0]`) et le manifest prospectif existant : 300 problèmes MATH train pour `reference`, 100 pour `development` et 300 MATH-500 pour `test`. Le multi-seed reste une extension facultative ; il n'est pas nécessaire pour commencer puisque l'expérience prospective appariée a déjà donné un résultat statistiquement significatif.

Les générations Vanilla de référence et de développement sont terminées. Les réutiliser ne relance pas le modèle :

```bash
python scripts/run_ablation_campaign.py \
  --config configs/science_campaign/math500_exploratory_ablations.yaml \
  --action reuse-science-fast
```

La commande authentifie les 300 + 100 artefacts, recalcule hors ligne les calibrateurs compatibles depuis leurs traces et vérifie que le seuil `full` reste identique au seuil prospectif. Elle est volontairement limitée à `seeds: [0]`.

Le premier calcul GPU à effectuer est uniquement MGT-B sur `development` :

```bash
python scripts/run_ablation_campaign.py \
  --config configs/science_campaign/math500_exploratory_ablations.yaml \
  --action run \
  --role development \
  --variant full_mgtb
```

Les six workers configurés sont utilisés automatiquement. Une fois les 100 items terminés :

```bash
python scripts/run_ablation_campaign.py \
  --config configs/science_campaign/math500_exploratory_ablations.yaml \
  --action build-profile \
  --source-variant full_mgtb
```

Ce run est indispensable à `matched_random`. Le développement existant contient des trajectoires Vanilla pour choisir le seuil, pas les interventions MGT-B nécessaires au profil.

| Variante | Nouvelle génération de calibration ? | Travail restant |
|---|---:|---|
| `full_mgtb` | Non | Aucun avant son run |
| Ablations du repair | Non | Réutilisent le monitor `full` |
| `matched_random` | Non | MGT-B sur `development`, puis profil |
| Ablations monitor 64/32 | Non | Recalibration hors ligne depuis les traces |
| Proxy token-level | Oui | Campagne séparée avec nouvelles traces 1/1 |
| Self-consistency / best-of-5 | Sans objet | Aucune calibration |

Lors de l'import actuel, `full`, `repetition_only` et `no_reset` respectent la cible de fausse alarme. `entropy_only`, `entropy_repetition` et `global` ne l'atteignent pas sur ce développement ; ces trois variantes restent diagnostiques et ne sont pas prioritaires pour le premier lot.

## Garanties et artefacts

Chaque variante écrit sous `output_root/runs/<role>/<variant>/` :

- `items/<sha256>.json` : génération, tokens, score, monitor trace, interventions, coûts et provenance ;
- `run_state.json` : identité exacte et liste des unités terminées ;
- `progress.json` : dernier checkpoint lisible rapidement ;
- `progress.jsonl` : journal append-only de tous les items terminés ;
- `in_progress/` : traces temporaires de l'unité en cours.

Un artefact final contient son propre SHA-256. La reprise vérifie l'item, le contenu, la seed, l'identité du run et ce SHA-256. Le nombre de workers et `--stop-after` ne changent pas les seeds. En revanche, une modification du manifest, de la campagne, d'une variante ou du code refuse la reprise dans le même dossier.

Le test est refusé sans `freeze/campaign.lock.json`. Le freeze authentifie :

- manifest et IDs test ;
- configuration complète de campagne ;
- toutes les variantes ;
- calibrateurs et seuils ;
- profil matched-random ;
- commit Git et hash de l'arbre source ;
- environnement logiciel.

## Préparer une campagne confirmatoire

1. Copier le template confirmatoire sous un nouveau nom.
2. Choisir un `campaign_id` et un `output_root` définitifs.
3. Remplacer chaque révision de dataset/modèle par un commit immuable.
4. Lister dans `exclude_manifests` tous les manifests déjà observés.
5. Fixer les seeds, tailles, métrique principale et variantes avant le test.
6. Construire puis valider le manifest :

```bash
python scripts/run_ablation_campaign.py \
  --config configs/science_campaign/gsm8k_confirmatory_template.yaml \
  --action build-manifest

python scripts/run_ablation_campaign.py \
  --config configs/science_campaign/gsm8k_confirmatory_template.yaml \
  --action validate
```

Une campagne `confirmatory` est rejetée sans manifests d'exclusion. La validation échoue si un contenu test apparaît dans l'un d'eux.

## Calibration des monitors pour une nouvelle campagne

Cette section concerne une campagne sans artefacts `science_fast` réutilisables. Pour la campagne MATH-500 actuelle, utiliser `--action reuse-science-fast` comme indiqué plus haut.

Les monitors partageant `window_size` et `stride` réutilisent les mêmes générations Vanilla. Dans les configs fournies, `entropy_only`, `repetition_only`, `entropy_repetition`, `global` et `no_reset` utilisent `feature_source: full`.

Collecter une seule fois les features 64/32 :

```bash
python scripts/run_ablation_campaign.py --config <campaign.yaml> --action collect --role reference --calibration full
python scripts/run_ablation_campaign.py --config <campaign.yaml> --action collect --role development --calibration full
```

Puis construire chaque calibrateur/seuil sans nouvelle génération :

```bash
python scripts/run_ablation_campaign.py --config <campaign.yaml> --action calibrate --calibration full
python scripts/run_ablation_campaign.py --config <campaign.yaml> --action calibrate --calibration entropy_only
python scripts/run_ablation_campaign.py --config <campaign.yaml> --action calibrate --calibration repetition_only
python scripts/run_ablation_campaign.py --config <campaign.yaml> --action calibrate --calibration entropy_repetition
python scripts/run_ablation_campaign.py --config <campaign.yaml> --action calibrate --calibration global
python scripts/run_ablation_campaign.py --config <campaign.yaml> --action calibrate --calibration no_reset
```

Une géométrie de fenêtre différente exige son propre `feature_source`. Le proxy token-level de la config exploratoire nécessite donc :

```bash
python scripts/run_ablation_campaign.py --config <campaign.yaml> --action collect --role reference --calibration token_level_previous
python scripts/run_ablation_campaign.py --config <campaign.yaml> --action collect --role development --calibration token_level_previous
python scripts/run_ablation_campaign.py --config <campaign.yaml> --action calibrate --calibration token_level_previous
```

Important : l'historique du dépôt ne contient pas la règle de seuil/réparation complète de l'ancien MGT-B v2. `previous_token_controller_proxy` est donc explicitement un proxy token-level calibré prospectivement, pas une prétendue reproduction exacte. Une vraie comparaison historique nécessitera la spécification ou le code original.

## Construire le contrôle matched-random

Après import ou calibration, exécuter MGT-B sur `development`, jamais sur le test et sans utiliser les labels pour construire le profil :

```bash
python scripts/run_ablation_campaign.py --config <campaign.yaml> --action run --role development --variant full_mgtb
python scripts/run_ablation_campaign.py --config <campaign.yaml> --action build-profile --source-variant full_mgtb
```

Le profil conserve la distribution empirique du nombre d'interventions, des tailles de rollback et de la longueur primaire. `matched_random` réassigne ces templates indépendamment des labels, randomise les positions, puis appelle exactement le même `BacktrackingController` et le même repair operator que MGT-B.

## Freeze puis test

Ne lancer le freeze qu'après revue des résumés de référence, seuils et profils :

```bash
python scripts/run_ablation_campaign.py --config <campaign.yaml> --action freeze
```

Après ce point, ne modifier ni config, ni code, ni artefacts calibrés. Exécuter chaque variante séparément permet de répartir les jobs GPU et de reprendre chaque run indépendamment :

```bash
python scripts/run_ablation_campaign.py --config <campaign.yaml> --action run --role test --variant vanilla
python scripts/run_ablation_campaign.py --config <campaign.yaml> --action run --role test --variant full_mgtb
python scripts/run_ablation_campaign.py --config <campaign.yaml> --action run --role test --variant matched_random
python scripts/run_ablation_campaign.py --config <campaign.yaml> --action run --role test --variant monitor_entropy_only
python scripts/run_ablation_campaign.py --config <campaign.yaml> --action run --role test --variant repair_fixed_rollback
python scripts/run_ablation_campaign.py --config <campaign.yaml> --action run --role test --variant self_consistency_5
```

Répéter pour les variantes pré-enregistrées dans la config. La campagne MATH-500 utilise automatiquement six workers. `--workers N` reste un override temporaire qui ne change ni les seeds ni l'identité scientifique. `--stop-after N` permet un test volontaire de reprise sans modifier l'identité scientifique.

## Suivi et analyse

Afficher les compteurs de tous les runs :

```bash
python scripts/run_ablation_campaign.py --config <campaign.yaml> --action status
```

Les lignes de progression émises sur stdout et `progress.jsonl` contiennent le compteur, l'ID, la correction et les tokens échantillonnés. Les items terminés ne sont jamais recalculés lors d'une reprise valide.

Une fois toutes les variantes test complètes :

```bash
python scripts/run_ablation_campaign.py --config <campaign.yaml> --action analyze
```

L'analyse refuse un run incomplet ou non apparié. Elle écrit `analysis/campaign_results.json` depuis les artefacts bruts authentifiés et rapporte, pour chaque méthode : accuracy globale/par seed/par domaine, variabilité entre seeds, extractabilité, troncatures, alarmes, rerolls, sampled/emitted/deleted tokens, latence, VRAM et terminaisons. Toutes les méthodes sont comparées à la baseline déclarée avec bootstrap apparié, McNemar exact descriptif et p-values ajustées par Holm. Avec la configuration actuelle à une seed, l'unité d'appariement est le problème. Si plusieurs seeds sont ajoutées plus tard, le bootstrap clusterisé par problème devient l'inférence principale.

## Interprétation des baselines de compute

`self_consistency_5` choisit la réponse normalisée majoritaire. `best_of_5_logprob` choisit la génération de meilleure log-probabilité moyenne ; il s'agit d'un proxy sans reward model, pas d'un best-of-N oracle. Les artefacts conservent les cinq candidats et leur coût total. Pour un papier, reporter les courbes accuracy versus sampled tokens et ne pas présenter ces méthodes comme ayant un budget identique si la config ne l'impose pas.

## Ordre scientifique recommandé

1. Réutiliser la calibration prospective existante.
2. Construire `matched_random` depuis MGT-B sur `development`.
3. Tester Vanilla, full MGT-B et matched-random avec une seed appariée.
4. Tester les ablations du repair partageant la calibration `full`.
5. Tester les ablations monitor dont la calibration atteint la cible.
6. Ajouter self-consistency/best-of-N avec coût explicite.
7. Traiter le proxy token-level dans une autre configuration, car il exige de nouvelles traces.
8. En extension seulement : multi-seed, deuxième modèle ou précision.
9. Régénérer l'analyse depuis les artefacts bruts et archiver le freeze.

Le test MATH-500 historique ne doit jamais être utilisé pour retuner le contrôleur. La config exploratoire sert au diagnostic et à la préparation du protocole, tandis que la conclusion confirmatoire doit venir d'un nouveau manifest gelé.
