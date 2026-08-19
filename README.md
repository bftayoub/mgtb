# MGT-B v3

MGT-B v3 is a clean MVP for inference-time monitoring of autoregressive reasoning traces, aimed at small and quantized models such as INT4 HuggingFace causal LMs.

The rebuilt prospective 300/100/300 protocol, including atomic resume and test freeze, is documented in [`docs/SCIENCE_FAST_RUNBOOK.md`](docs/SCIENCE_FAST_RUNBOOK.md).

New multi-seed ablation and generalization campaigns use the separate, config-driven runner documented in [`docs/SCIENCE_CAMPAIGN_RUNBOOK.md`](docs/SCIENCE_CAMPAIGN_RUNBOOK.md). It preserves the historical `science_fast` result while adding per-unit checkpoints, live progress logs, method-specific calibration, campaign freeze and multi-method paired analysis.

The controller does not fine-tune the language model. It monitors generated tokens, aggregates robust degeneration features over windows, detects suspicious regime changes, and can backtrack to a likely changepoint before re-decoding with targeted anti-degeneration constraints.

Important: this project does not claim an exact Ville guarantee. LLM scores are dependent, windows overlap, and p-values are not conditionally valid by construction. The CUSUM-e / e-detector form is a design motivation; the operational threshold is calibrated empirically on held-out healthy traces.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Structure

```text
mgtb_v3/
  features/       token and window features
  calibration/    ECDF, positional buckets, empirical threshold
  detector/       betting functions and CUSUM-e detector
  control/        cache crop, logits helpers, backtracking controller
  generation/     HuggingFace monitored generation loop
  eval/           metrics and ablation scaffolding
  baselines/      vanilla-adjacent comparison baselines
scripts/          feature extraction, calibration, generation, eval CLIs
docs/             theory and implementation reference
tests/            unit tests for core behavior
```

## Offline Feature Extraction

```bash
python scripts/extract_features.py \
  --input data/healthy_traces.jsonl \
  --output outputs/window_features.jsonl \
  --config configs/mgtb_v3_default.yaml
```

Input traces must provide pre-sampling logits per generated token to compute entropy and chosen-token log-probability.

## Calibration

```bash
python scripts/calibrate_threshold.py \
  --input outputs/window_features.jsonl \
  --calibrator-output outputs/calibrator.json \
  --threshold-output outputs/threshold.json
```

This builds positional ECDF pools and chooses the smallest searched threshold whose observed false-alert rate per held-out healthy trace is below the configured target.

For an autonomous precision-specific calibration run, use:

```bash
python scripts/calibrate_precision.py \
  --config configs/calibration/compare_n100.yaml
```

This runs healthy vanilla generations for each requested precision, writes `window_features.jsonl`, `healthy_results.jsonl`, `calibrator.json`, `threshold.json`, `calibration_summary.json`, and a global `calibration_manifest.json` under `outputs/calibration/compare_n100/`. The reported `mu0` is a configurable quantile of healthy window scores for diagnostics; the current runtime detector consumes the ECDF calibrator and e-process threshold.

## Generation

```bash
python scripts/run_generation.py \
  --model distilgpt2 \
  --prompt "Solve step by step:" \
  --calibrator outputs/calibrator.json \
  --threshold outputs/threshold.json \
  --output outputs/generation.json \
  --trace-log runs/trace.jsonl
```

## Evaluation

```bash
python scripts/run_eval.py --input outputs/results.jsonl --mode mgtb_v3_window
```

Supported comparison modes include `vanilla`, `mgtb_v2_baseline`, `mgtb_v3_window`, `random_trigger`, `direct_score_threshold`, fixed/adaptive backtracking, and feature knockouts.

## Precision Comparison Runs

Create one YAML or JSON file per experiment under `configs/tests/`, then run it:

```bash
python scripts/run_precision_comparison.py \
  --run-config configs/tests/comparison_100_fp16_int4.yaml
```

Useful keys are `base_model`, `input`, `output_dir`, `limit`, `max_new_tokens`, `methods`, `precisions`, `config`, `calibration`, `seed`, `device_map`, and `allow_cpu_fp32_fallback`. The older `model` key is still accepted as a compatibility alias for `base_model`, and the older top-level `calibrator` / `threshold` pair is still accepted for single-calibration runs.

For precision-specific MGT-B calibration, use one artifact pair per requested precision:

```yaml
calibration:
  fp16:
    calibrator: outputs/calibration/compare_n100/fp16/calibrator.json
    threshold: outputs/calibration/compare_n100/fp16/threshold.json
  int4:
    calibrator: outputs/calibration/compare_n100/int4/calibrator.json
    threshold: outputs/calibration/compare_n100/int4/threshold.json
```

For GSM8K, install the optional dataset dependency and use the built-in loader:

```bash
pip install -e ".[eval]"
python scripts/run_precision_comparison.py \
  --run-config configs/tests/gsm8k_qwen_1p5b_fp16_int4.yaml
```

GSM8K configs use `dataset: gsm8k` instead of `input`, with `dataset_name`, `dataset_config`, `split`, and `prompt_style` controlling the HuggingFace dataset source and prompt template.

MATH-500 uses the same config-first path:

```bash
python scripts/calibrate_precision.py \
  --config configs/calibration/math500_n100.yaml

python scripts/run_precision_comparison.py \
  --run-config configs/tests/math500_fp16_int4.yaml
```

MATH-500 configs use `dataset: math500`, defaulting to `dataset_name: HuggingFaceH4/MATH-500`, `dataset_config: default`, `split: test`, and `prompt_style: math500_cot`. The scorer compares normalized final answers after the `####` marker and handles common LaTeX forms such as `\boxed{...}`, `\frac{...}{...}`, `\sqrt{...}`, and `\text{...}`; it is exact-normalized scoring, not a full symbolic-equivalence prover.

CLI arguments override the config file, for example:

```bash
python scripts/run_precision_comparison.py \
  --run-config configs/tests/comparison_100_fp16_int4.yaml \
  --limit 10 \
  --output-dir outputs/debug_10
```

### Compute-matched priority-1 controls

The precision runner also supports six config-selectable methods:

```yaml
methods:
  - vanilla
  - mgtb_v3_window
  - random_backtrack
  - periodic_backtrack
  - restart
  - self_correct
```

The four controls use a frozen budget profile built from paired development runs, not from labels. Build the included MATH-500 INT4 profile once from the existing `log_threshold=10` artifacts:

```bash
python scripts/build_baseline_budget.py \
  --manifest configs/budgets/math500_int4_logthr10_sources.yaml \
  --output outputs/budgets/math500_int4_logthr10.json
```

Then run the all-method example:

```bash
python scripts/run_precision_comparison.py \
  --run-config configs/tests/math500_priority1_int4.yaml
```

`random_backtrack` and `periodic_backtrack` preserve the empirical intervention and rollback budget while changing intervention positions. `restart` performs a deterministic, budget-matched subset of full second attempts. `self_correct` asks every answer for a short revision using the profile's mean extra-token budget. Results record the profile hash, intervention details, extra decode tokens, total decode events, and budget-match error.
