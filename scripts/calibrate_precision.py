#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mgtb_v3.calibration.positional import DEFAULT_BUCKETS, PositionalCalibrator
from mgtb_v3.calibration.threshold import calibrate_threshold
from mgtb_v3.config import load_config
from mgtb_v3.eval.gsm8k import load_gsm8k_items, score_gsm8k
from mgtb_v3.eval.math500 import (
    DEFAULT_DATASET_CONFIG as MATH500_DEFAULT_DATASET_CONFIG,
    DEFAULT_DATASET_NAME as MATH500_DEFAULT_DATASET_NAME,
    DEFAULT_PROMPT_STYLE as MATH500_DEFAULT_PROMPT_STYLE,
    load_math500_items,
    score_math500,
)
from mgtb_v3.features.window_features import TrajectoryMonitor, linear_window_score
from mgtb_v3.generation.hf_loop import _sample_token

from scripts.run_precision_comparison import (
    _load_model,
    _load_prompts,
    _read_run_config,
    _seed_everything,
    _should_print_progress,
    _stable_seed,
)


DEFAULT_CALIBRATION_SETTINGS: dict[str, Any] = {
    "base_model": None,
    "model": None,
    "input": None,
    "dataset": "gsm8k",
    "dataset_name": "gsm8k",
    "dataset_config": "main",
    "split": "test",
    "prompt_style": "gsm8k_cot",
    "limit": 100,
    "seed": 0,
    "max_new_tokens": 2048,
    "precisions": ["fp16", "int4"],
    "config": "configs/mgtb_v3_default.yaml",
    "output_dir": "outputs/calibration/compare_n100",
    "mu0_quantile": 0.90,
    "healthy_filter": {"correct_only": True, "exclude_truncated": True},
    "min_healthy_examples": 1,
    "device_map": "auto",
    "allow_cpu_fp32_fallback": False,
    "progress_interval": 20,
}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Collect healthy vanilla traces and calibrate MGT-B v3 per precision.")
    parser.add_argument("--config", required=True, help="YAML or JSON calibration config.")
    args = parser.parse_args(argv)
    settings = load_calibration_settings(args.config)
    manifest = run_calibration(settings)
    print(f"Wrote {manifest['manifest_path']}")


def load_calibration_settings(path: str | Path) -> dict[str, Any]:
    settings = dict(DEFAULT_CALIBRATION_SETTINGS)
    settings.update(_read_run_config(str(path)))
    settings["base_model"] = settings.get("base_model") or settings.get("model")
    settings["model"] = settings["base_model"]

    healthy_filter = dict(DEFAULT_CALIBRATION_SETTINGS["healthy_filter"])
    healthy_filter.update(settings.get("healthy_filter") or {})
    settings["healthy_filter"] = healthy_filter

    if isinstance(settings["precisions"], str):
        settings["precisions"] = [settings["precisions"]]
    settings["precisions"] = [str(p) for p in settings["precisions"]]
    if settings.get("dataset") == "math500":
        if settings.get("dataset_name") == DEFAULT_CALIBRATION_SETTINGS["dataset_name"]:
            settings["dataset_name"] = MATH500_DEFAULT_DATASET_NAME
        if settings.get("dataset_config") == DEFAULT_CALIBRATION_SETTINGS["dataset_config"]:
            settings["dataset_config"] = MATH500_DEFAULT_DATASET_CONFIG
        if settings.get("prompt_style") == DEFAULT_CALIBRATION_SETTINGS["prompt_style"]:
            settings["prompt_style"] = MATH500_DEFAULT_PROMPT_STYLE

    if not settings.get("base_model"):
        raise SystemExit("Missing required setting: base_model.")
    if settings.get("dataset") and settings["dataset"] not in {"gsm8k", "math500"}:
        raise SystemExit(f"Unsupported dataset {settings['dataset']!r}; expected one of: gsm8k, math500.")
    if not settings.get("dataset") and not settings.get("input"):
        raise SystemExit("Missing data source: provide input or set dataset: gsm8k or math500.")
    if not 0.0 <= float(settings["mu0_quantile"]) <= 1.0:
        raise SystemExit("mu0_quantile must be in [0, 1].")
    return settings


def run_calibration(settings: dict[str, Any]) -> dict[str, Any]:
    cfg = load_config(settings["config"])
    items = _load_calibration_items(settings)
    if not items:
        raise SystemExit("No calibration prompts found.")

    output_dir = Path(settings["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "base_model": settings["base_model"],
        "dataset": settings.get("dataset") or "jsonl",
        "split": settings.get("split"),
        "limit": settings["limit"],
        "seed": settings["seed"],
        "max_new_tokens": settings["max_new_tokens"],
        "config": settings["config"],
        "mu0_quantile": float(settings["mu0_quantile"]),
        "healthy_filter": settings["healthy_filter"],
        "precisions": {},
    }

    for precision in settings["precisions"]:
        precision_summary = _calibrate_precision(settings, cfg, items, output_dir, precision)
        manifest["precisions"][precision] = precision_summary

    manifest_path = output_dir / "calibration_manifest.json"
    manifest["manifest_path"] = str(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def _load_calibration_items(settings: dict[str, Any]) -> list[dict[str, Any]]:
    if settings.get("dataset") == "gsm8k":
        return load_gsm8k_items(
            dataset_name=settings["dataset_name"],
            dataset_config=settings["dataset_config"],
            split=settings["split"],
            limit=settings["limit"],
            seed=settings["seed"],
            prompt_style=settings["prompt_style"],
        )
    if settings.get("dataset") == "math500":
        return load_math500_items(
            dataset_name=settings["dataset_name"],
            dataset_config=settings["dataset_config"],
            split=settings["split"],
            limit=settings["limit"],
            seed=settings["seed"],
            prompt_style=settings["prompt_style"],
        )
    return _load_prompts(settings["input"], settings["limit"])


def _calibrate_precision(settings: dict[str, Any], cfg, items: list[dict[str, Any]], output_dir: Path, precision: str) -> dict[str, Any]:
    precision_dir = output_dir / precision
    precision_dir.mkdir(parents=True, exist_ok=True)
    all_results_path = precision_dir / "all_results.jsonl"
    healthy_results_path = precision_dir / "healthy_results.jsonl"
    window_features_path = precision_dir / "window_features.jsonl"
    calibrator_path = precision_dir / "calibrator.json"
    threshold_path = precision_dir / "threshold.json"
    summary_path = precision_dir / "calibration_summary.json"

    model, tokenizer, effective_precision = _load_model(
        settings["base_model"],
        precision,
        settings["device_map"],
        settings["allow_cpu_fp32_fallback"],
    )

    healthy_count = 0
    score_values: list[float] = []
    num_windows = 0
    start_time = time.time()
    try:
        with all_results_path.open("w", encoding="utf-8") as all_results:
            with healthy_results_path.open("w", encoding="utf-8") as healthy_results:
                with window_features_path.open("w", encoding="utf-8") as window_features:
                    for index, item in enumerate(items):
                        run_seed = _stable_seed(
                            settings["seed"],
                            item.get("dataset", "jsonl"),
                            item.get("split", ""),
                            index,
                        )
                        _seed_everything(run_seed)
                        run_id = f"{effective_precision}-{index:06d}"
                        generated = _run_vanilla_collect_features(
                            model,
                            tokenizer,
                            item["prompt"],
                            cfg,
                            max_new_tokens=settings["max_new_tokens"],
                        )
                        row = _build_result_row(settings, item, generated, precision, effective_precision, index, run_seed, run_id)
                        all_results.write(json.dumps(row, ensure_ascii=False) + "\n")

                        if _is_healthy(row, settings["healthy_filter"]):
                            healthy_count += 1
                            healthy_results.write(json.dumps(row, ensure_ascii=False) + "\n")
                            for window in generated["windows"]:
                                feature_row = {
                                    "run_id": run_id,
                                    "precision": effective_precision,
                                    "requested_precision": precision,
                                    "example_index": index,
                                    "features": window["features"],
                                    "score": window["score"],
                                }
                                window_features.write(json.dumps(feature_row, ensure_ascii=False) + "\n")
                                score_values.append(float(window["score"]))
                                num_windows += 1

                        all_results.flush()
                        healthy_results.flush()
                        window_features.flush()
                        if _should_print_progress(index + 1, len(items), settings["progress_interval"]):
                            print(
                                f"[calibration {effective_precision}] {index + 1}/{len(items)} "
                                f"healthy={healthy_count} windows={num_windows}"
                            )
    finally:
        del model
        del tokenizer
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    if healthy_count < int(settings["min_healthy_examples"]):
        raise SystemExit(
            f"Not enough healthy examples for {precision}: "
            f"{healthy_count} < {settings['min_healthy_examples']}."
        )
    if not score_values:
        raise SystemExit(f"No healthy window scores produced for {precision}; increase max_new_tokens or relax healthy_filter.")

    calibrator, threshold = _calibrate_from_scores(window_features_path, calibrator_path, threshold_path, settings["config"])
    mu0 = _quantile(score_values, float(settings["mu0_quantile"]))
    summary = {
        "precision": effective_precision,
        "requested_precision": precision,
        "base_model": settings["base_model"],
        "num_examples": len(items),
        "num_healthy_examples": healthy_count,
        "num_windows": num_windows,
        "mu0": mu0,
        "mu0_quantile": float(settings["mu0_quantile"]),
        "threshold": float(threshold["threshold"]),
        "observed_false_alert_rate": threshold.get("observed_false_alert_rate"),
        "score_mean": sum(score_values) / len(score_values),
        "score_min": min(score_values),
        "score_max": max(score_values),
        "all_results_path": str(all_results_path),
        "healthy_results_path": str(healthy_results_path),
        "window_features_path": str(window_features_path),
        "calibrator_path": str(calibrator_path),
        "threshold_path": str(threshold_path),
        "summary_path": str(summary_path),
        "calibration_buckets": sorted(calibrator.ecdfs),
        "elapsed_seconds": time.time() - start_time,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def _run_vanilla_collect_features(model, tokenizer, prompt: str, cfg, max_new_tokens: int) -> dict[str, Any]:
    import torch

    started = time.time()
    model.eval()
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"]
    device = getattr(model, "device", getattr(input_ids, "device", None))
    if hasattr(input_ids, "to") and device is not None:
        input_ids = input_ids.to(device)
    prompt_tokens = input_ids[0].tolist()
    tokens = list(prompt_tokens)
    generated = input_ids
    cache = None
    monitor = TrajectoryMonitor(cfg, prompt_tokens=prompt_tokens)
    windows: list[dict[str, Any]] = []
    stopped_on_eos = False

    for _ in range(int(max_new_tokens)):
        with torch.no_grad():
            if cache is None:
                outputs = model(input_ids=generated, use_cache=True)
            else:
                outputs = model(input_ids=generated[:, -1:], past_key_values=cache, use_cache=True)
        logits = outputs.logits[:, -1, :]
        cache = getattr(outputs, "past_key_values", None)
        next_token = _sample_token(logits, temperature=1.0, generated_tokens=tokens)
        token_id = int(next_token.item())
        tokens.append(token_id)
        generated = torch.tensor([tokens], device=device)
        monitor.update_token(token_id, logits[0])

        while monitor.should_emit_window():
            features = monitor.compute_window_features()
            windows.append({"features": features.to_dict(), "score": linear_window_score(features, cfg.score)})

        if token_id == getattr(tokenizer, "eos_token_id", None):
            stopped_on_eos = True
            break

    prompt_len = len(prompt_tokens)
    completion_tokens = tokens[prompt_len:]
    completion_text = tokenizer.decode(completion_tokens, skip_special_tokens=True) if completion_tokens else ""
    return {
        "text": tokenizer.decode(tokens, skip_special_tokens=True),
        "completion_text": completion_text,
        "tokens_generated": len(completion_tokens),
        "truncated": not stopped_on_eos and len(completion_tokens) >= int(max_new_tokens),
        "windows": windows,
        "latency": time.time() - started,
    }


def _build_result_row(
    settings: dict[str, Any],
    item: dict[str, Any],
    generated: dict[str, Any],
    requested_precision: str,
    effective_precision: str,
    index: int,
    seed: int,
    run_id: str,
) -> dict[str, Any]:
    reference = item.get("reference_answer") or item.get("answer")
    if item.get("dataset") == "math500":
        score = score_math500(generated["completion_text"] or generated["text"], reference)
    else:
        score = score_gsm8k(generated["completion_text"] or generated["text"], reference) if reference is not None else {}
    return {
        "run_id": run_id,
        "id": item.get("id", index),
        "index": index,
        "dataset": item.get("dataset", settings.get("dataset") or "jsonl"),
        "split": item.get("split", settings.get("split")),
        "question": item.get("question"),
        "base_model": settings["base_model"],
        "method": "vanilla",
        "requested_precision": requested_precision,
        "precision": effective_precision,
        "seed": seed,
        "prompt": item["prompt"],
        "text": generated["text"],
        "completion_text": generated["completion_text"],
        "tokens_generated": generated["tokens_generated"],
        "latency": generated["latency"],
        "truncated": generated["truncated"],
        "num_windows": len(generated["windows"]),
        **score,
    }


def _is_healthy(row: dict[str, Any], healthy_filter: dict[str, Any]) -> bool:
    if healthy_filter.get("correct_only", True) and float(row.get("correct", 0.0)) != 1.0:
        return False
    if healthy_filter.get("exclude_truncated", True) and row.get("truncated", False):
        return False
    return True


def _calibrate_from_scores(input_path: Path, calibrator_path: Path, threshold_path: Path, config_path: str):
    cfg = load_config(config_path)
    pools = defaultdict(list)
    runs = defaultdict(list)
    provisional = PositionalCalibrator(DEFAULT_BUCKETS, {"0-512": [0.0]}, p_clip=cfg.detector.p_clip)
    rows = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    for row in rows:
        features = row.get("features", {})
        pos = features.get("end_pos", row.get("end_pos", 0))
        bucket = provisional.bucket_for_position(pos)
        pools[bucket].append(float(row["score"]))
    calibrator = PositionalCalibrator(DEFAULT_BUCKETS, dict(pools), p_clip=cfg.detector.p_clip)
    calibrator.save_json(calibrator_path)
    for row in rows:
        features = row.get("features", {})
        pos = features.get("end_pos", row.get("end_pos", 0))
        p_value = calibrator.p_value(float(row["score"]), pos)
        runs[row.get("run_id", "default")].append(p_value)
    threshold = calibrate_threshold(
        list(runs.values()),
        cfg.detector.target_false_alert_rate,
        gammas=cfg.detector.betting_gammas,
        p_clip=cfg.detector.p_clip,
        refractory_windows=cfg.detector.refractory_windows,
    )
    threshold_path.write_text(json.dumps(threshold, indent=2) + "\n", encoding="utf-8")
    return calibrator, threshold


def _quantile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("values must not be empty")
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * float(q)
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return ordered[lower]
    weight = pos - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


if __name__ == "__main__":
    main()
