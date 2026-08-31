# Scoring Gemini aveugle et reprenable d’Omni-MATH

Ce pipeline lit les 4 500 générations gelées dans
`outputs/science_campaign/omnimath_confirmatory_v1_judge_batch1` et écrit exclusivement dans
`outputs/gemini_scoring/omnimath_confirmatory_v1`. Il ne modifie ni les générations ni les
jugements Omni-Judge. Les payloads ne contiennent jamais la variante, la seed, l’ancien verdict
ou une métrique MGTB. La table locale `anonymization.json` est utilisée seulement après jugement.

Le juge principal est exactement `gemini-3.5-flash-lite`, température 0, sortie JSON structurée,
raisonnement `high`, limité à 12 RPM, 200 000 TPM et 500 RPD. Le SDK est figé à
`google-genai==2.20.0`. La clé n’est lue que depuis `GEMINI_API_KEY`; elle n’est jamais écrite.
Le compteur journalier suit la journée Pacific (`America/Los_Angeles`), conformément au reset RPD
de minuit du service Gemini.

## Commandes exactes

Setup :

```bash
.venv/bin/python -m pip install 'google-genai==2.20.0'
.venv/bin/python scripts/score_omnimath_gemini.py setup
```

Dry-run (aucun contact API) :

```bash
.venv/bin/python scripts/score_omnimath_gemini.py dry-run --scope pilot
```

Pilote de 200 cas, audit individuel de 50 cas, puis arbitrage des 98 contradictions
numériques, des ABSTAIN, des désaccords groupé/individuel et d’un audit de cinq cas avec
`gemini-3.5-flash`. Avec 20 RPD, cet arbitrage exige au minimum six journées de quota Pacific et se reprend
avec la même commande :

```bash
export GEMINI_API_KEY='...'
.venv/bin/python scripts/score_omnimath_gemini.py pilot --workers 1 --resume
```

Arrêt borné, utile pour vérifier les quotas :

```bash
.venv/bin/python scripts/score_omnimath_gemini.py pilot --stop-after 5 --workers 1 --resume
```

Tant que le jugement groupé n’est pas complet, cette commande n’envoie que ces cinq requêtes
principales : les audits sont volontairement différés. La borne compte toutes les tentatives API,
y compris une éventuelle réponse 429/5xx ou un JSON invalide, afin de borner réellement le quota.

Status :

```bash
.venv/bin/python scripts/score_omnimath_gemini.py status --scope pilot
```

Reprise après Ctrl-C, crash, réseau, quota ou changement de clé :

```bash
export GEMINI_API_KEY='nouvelle-valeur'
.venv/bin/python scripts/score_omnimath_gemini.py resume --scope pilot --workers 1
```

Analyse et rapport :

```bash
.venv/bin/python scripts/score_omnimath_gemini.py analyze
```

Cette commande analyse le pilote et décide mécaniquement `GO` ou `NO-GO`. L'analyse des 4 500
décisions complètes est séparée :

```bash
.venv/bin/python scripts/score_omnimath_gemini.py analyze-full
```

Elle écrit `full/report.json` et `full/REPORT.md`, avec exactitude par variante et seed,
bootstrap apparié clusterisé par problème, McNemar descriptif, correction de Holm et accord avec
Omni-Judge. Tant que le rapport pilote ne recommande pas `GO`, ces sorties portent explicitement
le statut `PROVISIONAL_PENDING_PILOT_GO` et ne permettent aucune revendication confirmatoire.
Après obtention du `GO`, relancer `analyze-full` rattache le rapport complet au hash du nouveau
rapport pilote et lui donne le statut `CONFIRMATORY`.

Une décision explicite d'accepter le juge principal sans terminer la porte secondaire peut être
consignée sans masquer la déviation :

```bash
.venv/bin/python scripts/score_omnimath_gemini.py analyze-full --finalize-without-pilot-go
```

Le rapport prend alors le statut `FINAL_USER_ACCEPTED_JUDGE`, conserve l'état et le hash du pilote,
et indique que la porte d'arbitrage secondaire pré-déclarée n'est pas revendiquée comme satisfaite.

Commande de lancement complet — à ne lancer qu’après un rapport `GO` et accord explicite :

```bash
.venv/bin/python scripts/score_omnimath_gemini.py run --scope full --workers 1 --resume --approved-full
```

Exemple borné montrant les trois options opérationnelles de `run` :

```bash
.venv/bin/python scripts/score_omnimath_gemini.py run --scope full --stop-after 10 --workers 2 --resume --approved-full
```

Puis reprise identique :

```bash
.venv/bin/python scripts/score_omnimath_gemini.py resume --scope full --workers 1 --approved-full
```

## Reprise et intégrité

Chaque réponse API dont le JSON et les identifiants ont été validés est écrite atomiquement dans
un artefact immuable avant l’agrégation. Chaque résultat candidat est ensuite indexé par le hash du
modèle, du prompt, du problème, de la référence et du candidat. Au redémarrage, seuls les artefacts
dont le hash d’authentification est valide sont acceptés. Une réponse manquante, dupliquée ou
incomplète est une erreur temporaire retentée; elle ne devient jamais TRUE ou FALSE.
Le cache primaire est partagé entre le pilote et le lancement complet : un tuple validé
`(modèle, prompt, problème, référence, réponse finale)` du pilote n’est jamais renvoyé au modèle
principal pendant le full run. Les re-jugements individuels sont l’exception volontaire, car ils
mesurent justement le biais potentiel du regroupement.

`ABSTAIN` vaut incorrect pour l’accuracy principale et son taux est publié séparément. Le rapport
restaure les noms réels des variantes uniquement depuis la table locale et inclut les matrices
d’accord, les erreurs certaines, les 52 cas symboliques lisibles, l’audit groupé/individuel et
l’estimation du reste. Tant que les 200 décisions et les 50 paires d’audit ne sont pas complètes,
la recommandation est mécaniquement `NO-GO`.
