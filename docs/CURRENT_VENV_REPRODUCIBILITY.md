# Verrouillage de l'environnement CUDA actuel

L'environnement de référence est le `.venv` existant du dépôt. Il ne doit être ni recréé ni mis à jour pendant la campagne.

## Identité verrouillée

- Python 3.10.20 ;
- PyTorch 2.13.0+cu130 ;
- CUDA runtime 13.0 ;
- bitsandbytes 0.50.1 ;
- transformers 4.57.6 ;
- NVIDIA RTX A5000, pilote 595.84.

`requirements-lock-cu130.txt` enregistre toutes les distributions installées. `configs/environment/current_venv_lock.json` enregistre la pile critique et le matériel. Avant toute génération :

```bash
.venv/bin/python scripts/verify_current_environment.py
```

Une divergence provoque un code de sortie non nul. Le verrou est un instantané destiné à préserver le `.venv` actuel ; aucune installation n'est nécessaire tant que la vérification passe.

## Audit token par token

Les deux configurations d'audit utilisent le même manifeste, les mêmes 50 premières unités, les mêmes graines et la même génération Vanilla. Elles rechargent séparément le modèle dans deux sorties indépendantes :

```bash
.venv/bin/python scripts/run_ablation_campaign.py --config configs/reproducibility/current_venv_a.yaml --action freeze
.venv/bin/python scripts/run_ablation_campaign.py --config configs/reproducibility/current_venv_a.yaml --action run --role test --variant vanilla --stop-after 50
.venv/bin/python scripts/run_ablation_campaign.py --config configs/reproducibility/current_venv_b.yaml --action freeze
.venv/bin/python scripts/run_ablation_campaign.py --config configs/reproducibility/current_venv_b.yaml --action run --role test --variant vanilla --stop-after 50
.venv/bin/python scripts/compare_reproducibility_runs.py \
  --left outputs/reproducibility/current_venv/a/runs/test/vanilla \
  --right outputs/reproducibility/current_venv/b/runs/test/vanilla \
  --expected 50
```

L'audit passe uniquement si les 50 suites de tokens sont identiques. Les sorties sous `outputs/` restent exclues de Git ; le résultat observé doit être ajouté à ce document après exécution.
