# Campagne prospective Omni-MATH v1

Cette campagne est indépendante des campagnes MATH-500. Elle écrit uniquement sous
`outputs/science_campaign/omnimath_confirmatory_v1` et traite les manifests MATH-500
existants comme des exclusions en lecture seule. Elle compare `vanilla`, `full_mgtb`
et `matched_random` pour les seeds 0, 1 et 2 avec le modèle, la quantification, le
prompt mathématique et la limite de 20 000 nouveaux tokens déjà utilisés.

## Sélection et absence de fuite

La source est exclusivement `KbsdJames/Omni-MATH`, fichier `Omni-Math.jsonl`, au
commit Git `23be225c8e268df51990f6c5c1448f34d3b56911`; le blob Git attendu est
`1a9d46a3a2b52992b010152e8e090f5fb7e7cb4a`. Le chargement échoue si le dépôt, le
chemin, la révision, le blob, ou l'un des champs `problem`, `answer`, `domain`,
`difficulty`, `source` ne peut pas être authentifié.

Tous les énoncés sont normalisés NFKC avec espaces canoniques, puis dédupliqués par
SHA-256. Avant la sélection, tout hash présent dans un manifest exclu est retiré du
pool entier. Cela inclut les manifests MATH-500 historique et all-500 déclarés dans
la config. Un manifest exclu absent est une erreur, pas une exclusion ignorée.

Le test de 500 problèmes est construit avant toute génération. Une strate est le
couple `(premier domaine de la liste officielle, difficulté numérique exacte)`.
Les 500 places sont distribuées proportionnellement aux effectifs éligibles par la
méthode de Hamilton (plus grands restes); les égalités sont ordonnées
lexicographiquement. Dans chaque strate, les items sont
ordonnés par `sha256(protocol_seed|content_sha256)`, avec la graine `20260824`. Les
300 items `reference`, puis les 300 items `development`, sont pris dans le reste
selon le même ordre hashé. Les domaines complets, la difficulté, la source, l'ID de
ligne stable, la réponse et le hash normalisé restent dans le manifest.

## Calibration et profil aléatoire apparié

La calibration Omni-MATH est nécessaire parce que la distribution des trajectoires
et des scores de fenêtre diffère de MATH-500. Elle ne modifie aucun poids de feature,
aucune fenêtre, CUSUM, cible de fausses alertes, règle de rollback ou paramètre du
repair operator. Les traces Vanilla `reference` alimentent le calibrateur positionnel
et les traces Vanilla `development` sélectionnent le seuil avec la règle existante.
L'éligibilité est seulement « non tronqué » : le code ne lit jamais le verdict de
correction pour calibrer.

`full_mgtb` est ensuite exécuté sur `development`. Le profil Omni-MATH est construit
uniquement depuis ses artefacts authentifiés et sans verdicts : il conserve, par
trajectoire, le nombre d'interventions, les positions observées et contraintes de
position, la longueur primaire, les tailles de rollback et les tokens supplémentaires.
Il ne référence aucun profil MATH-500.

## Ordre exact des commandes

Depuis la racine du dépôt, définir une fois :

```bash
CFG=configs/science_campaign/omnimath_confirmatory_v1.yaml
```

Puis exécuter dans cet ordre :

```bash
python scripts/run_ablation_campaign.py --config "$CFG" --action build-manifest
python scripts/run_ablation_campaign.py --config "$CFG" --action validate

python scripts/run_ablation_campaign.py --config "$CFG" --action collect --role reference --calibration full
python scripts/run_ablation_campaign.py --config "$CFG" --action collect --role development --calibration full
python scripts/run_ablation_campaign.py --config "$CFG" --action calibrate --calibration full

python scripts/run_ablation_campaign.py --config "$CFG" --action run --role development --variant full_mgtb
python scripts/run_ablation_campaign.py --config "$CFG" --action build-profile --source-variant full_mgtb
python scripts/run_ablation_campaign.py --config "$CFG" --action freeze

python scripts/run_ablation_campaign.py --config "$CFG" --action run --role test --variant vanilla
python scripts/run_ablation_campaign.py --config "$CFG" --action run --role test --variant full_mgtb
python scripts/run_ablation_campaign.py --config "$CFG" --action run --role test --variant matched_random

python scripts/run_ablation_campaign.py --config "$CFG" --action judge --role test --variant vanilla
python scripts/run_ablation_campaign.py --config "$CFG" --action judge --role test --variant full_mgtb
python scripts/run_ablation_campaign.py --config "$CFG" --action judge --role test --variant matched_random

python scripts/run_ablation_campaign.py --config "$CFG" --action analyze
python scripts/run_ablation_campaign.py --config "$CFG" --action status
```

`--workers N` s'applique aux collectes/runs. `--stop-after N` permet un arrêt de
validation et une reprise atomique. Le freeze doit être créé dans l'environnement
GPU définitif : environnement, code, config, trois listes d'IDs/hashs, source,
exclusions, calibrateur, seuil, profil et judge sont authentifiés. Une seconde
tentative de remplacement d'un manifest, profil ou freeze différent est refusée.

## Scoring Omni-MATH

Le scorer textuel MATH-500 n'est pas employé : il ne prouve pas l'équivalence de
réponses générales Omni-MATH. Les générations portent initialement un score
explicitement non scoré. L'étape `judge` utilise le modèle officiel
`KbsdJames/Omni-Judge` au commit
`de5bdca15ff3c366b90718c4b4be555d25c655b0`, le code officiel au commit et blob
indiqués dans la config, le prompt `tokenizer.get_context` de cette révision, une
génération gloutonne et au plus 300 tokens. Chaque verdict brut, justification,
hash de la génération, tokens et latence du judge est conservé séparément. Un rapport
sans verdict TRUE/FALSE et justification conformes fait échouer l'évaluation.

Limites : Omni-Judge est un juge appris, avec un accord annoncé d'environ 95 % avec
GPT-4o par les auteurs; ce n'est donc pas une preuve formelle. Son coût GPU local est
rapporté séparément et aucun coût API n'est inventé. Le chemin GPT-4o du leaderboard
n'est pas utilisé, car il introduirait un service et une version de modèle externes.

## Coût planifié et analyse

Les phases prospectives utilisent uniquement la seed `0` : 300 Vanilla `reference`,
300 Vanilla `development`, puis 300 `full_mgtb` `development`. Les trois seeds sont
réservées au test, qui demande 4 500 unités (500 problèmes × 3 seeds × 3 variantes).
Le protocole demande donc 5 400 unités de génération du modèle évalué. La borne
nominale est 108 millions
de tokens échantillonnés avant arrêts anticipés, avec l'effet exact des rollbacks
mesuré dans les artefacts. Le judge ajoute 4 500 unités distinctes, bornées à
1,35 million de tokens de verdict; son temps GPU n'est jamais mélangé à celui du
modèle évalué.

L'analyse refuse toute variante incomplète ou tout verdict non authentifié. Elle
rapporte exactitude globale/par seed, bootstrap apparié clusterisé par problème
(inférence principale), McNemar descriptif, Holm contre Vanilla,
corrections/régressions, extraction, troncature, alarmes, rerolls, tokens
échantillonnés/émis/supprimés, latence/coût de génération et coût du judge séparé.
Les résultats par domaine et difficulté sont explicitement descriptifs. Toute
trajectoire `full_mgtb` sans alarme doit être identique token par token à Vanilla,
sinon l'analyse échoue.
