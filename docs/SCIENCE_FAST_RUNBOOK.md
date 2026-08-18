# MGT-B fast prospective protocol

This pipeline implements the pinned 300 reference / 100 development / 300 MATH-500 test protocol from `MGTB_FAST_REIMPLEMENTATION_SPEC_v2.md`. It does not launch a real generation campaign during installation or testing.

## Safety properties

- Content-normalized SHA-256 selection and leakage checks precede tuning.
- Each item has one order-independent seed, shared by Vanilla and MGT-B.
- Every completed item is written atomically and authenticated by its artifact hash.
- Resume accepts only the same manifest, resolved config and freeze identity.
- Reference alone builds the positional ECDF; development alone selects `h`.
- Test execution is rejected without a matching method-specific freeze lock.
- MGT-B uses `replay_last` cache reconstruction and tracked retained-window indices.
- Final statistics are reconstructed exclusively from per-item raw artifacts.

## Workflow

Install the GPU and dataset dependencies, then build the manifest:

```bash
pip install -e ".[dev,eval]"
python scripts/run_scientific_experiment.py --config configs/science_fast/protocol.yaml
```

The runner requires a CUDA-visible GPU. If `import transformers` reports an
incompatible `huggingface-hub` version, repair the local environment before the
run with:

```bash
python -m pip install --upgrade "huggingface-hub>=0.34,<1.0" "transformers>=4.47,<5"
python -c "import torch, transformers; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0), transformers.__version__)"
```

All scientific configs pin `device_map: {"": 0}`. This deliberately fails on
insufficient VRAM instead of silently offloading modules to CPU or disk.

Run only the 300-item reference phase and build its ECDF/summary:

```bash
python scripts/run_scientific_experiment.py --config configs/science_fast/collect_reference_int4.yaml
python scripts/run_scientific_experiment.py --config configs/science_fast/build_calibrator.yaml
```

Running either command again resumes safely. `--stop-after N` is available for an intentional resume check; it limits newly completed items and does not alter their seeds.

After reviewing `outputs/science_fast/calibration/reference_summary.json`, run development:

```bash
python scripts/run_scientific_experiment.py --config configs/science_fast/collect_development_int4.yaml
python scripts/run_scientific_experiment.py --config configs/science_fast/select_threshold.yaml
python scripts/run_scientific_experiment.py --config configs/science_fast/freeze.yaml
```

Only after freeze:

```bash
python scripts/run_scientific_experiment.py --config configs/science_fast/test_vanilla_int4.yaml
python scripts/run_scientific_experiment.py --config configs/science_fast/test_mgtb_int4.yaml
python scripts/analyze_scientific_results.py --config configs/science_fast/analyze.yaml
```

## Pre-GPU validation

```bash
pytest -q
python scripts/smoke_science_fast.py
```

The smoke is synthetic, CPU-only, and consumes zero MATH-500 items. It covers Vanilla, no-alarm identity, forced alarm/rollback, interruption/resume, artifact reload and paired analysis.

## Reference handoff diagnostics

Before development, preserve and report:

- the complete `reference_summary.json`;
- the calibrator SHA-256 and manifest SHA-256;
- `nvidia-smi` output and any CUDA/OOM errors;
- wall time and peak VRAM from a representative artifact;
- the count of final item files and `run_state.json` completed count;
- any scorer extraction failures or truncations, with item IDs;
- the exact Git commit and source-tree hash recorded in an artifact.
