#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from mgtb_v3.calibration.positional import PositionalCalibrator
from mgtb_v3.config import load_config
from mgtb_v3.eval.gsm8k import load_gsm8k_items, score_gsm8k
from mgtb_v3.generation.hf_loop import generate_with_mgtb_v3


DEFAULT_RUN_SETTINGS: dict[str, Any] = {
    "base_model": None,
    "model": None,
    "input": None,
    "dataset": None,
    "dataset_name": "gsm8k",
    "dataset_config": "main",
    "split": "test",
    "prompt_style": "gsm8k_cot",
    "output_dir": "outputs/comparison",
    "limit": 100,
    "max_new_tokens": 256,
    "methods": ["vanilla", "mgtb_v3_window"],
    "precisions": ["fp16", "int4"],
    "config": "configs/mgtb_v3_default.yaml",
    "calibrator": None,
    "threshold": None,
    "calibration": None,
    "seed": 0,
    "device_map": "auto",
    "allow_cpu_fp32_fallback": False,
    "progress_interval": 20,
}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Run vanilla and MGT-B v3 generations across FP16/INT4 on JSONL prompts or GSM8K."
    )
    parser.add_argument("--run-config", help="YAML or JSON file describing this test run.")
    parser.add_argument("--base-model", help="Underlying HuggingFace model id or local model path used for generation.")
    parser.add_argument("--model", help="Legacy alias for --base-model.")
    parser.add_argument("--input", help="JSONL with at least a 'prompt' field per row.")
    parser.add_argument("--dataset", choices=["gsm8k"], help="Built-in dataset loader to use instead of --input.")
    parser.add_argument("--dataset-name", help="HuggingFace dataset name for --dataset gsm8k.")
    parser.add_argument("--dataset-config", help="HuggingFace dataset config for --dataset gsm8k.")
    parser.add_argument("--split", help="Dataset split for --dataset gsm8k.")
    parser.add_argument("--prompt-style", help="Prompt template name for --dataset gsm8k.")
    parser.add_argument("--output-dir")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--methods", nargs="+", choices=["vanilla", "mgtb_v3_window"])
    parser.add_argument("--precisions", nargs="+", choices=["fp16", "int4"])
    parser.add_argument("--config", help="MGT-B v3 controller config YAML.")
    parser.add_argument("--calibrator", help="Required for mgtb_v3_window.")
    parser.add_argument("--threshold", help="Threshold JSON or numeric threshold. Required for mgtb_v3_window.")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device-map")
    parser.add_argument("--progress-interval", type=int, help="Print cumulative per-variant metrics every N examples.")
    parser.add_argument(
        "--allow-cpu-fp32-fallback",
        action="store_true",
        default=None,
        help="If CUDA is unavailable, run the FP16 condition in FP32 on CPU and mark it in outputs.",
    )
    args = parser.parse_args(argv)
    settings = _load_run_settings(args)
    _validate_mgtb_settings(settings)

    items = _load_items(settings)
    if not items:
        raise SystemExit("No prompts found. Provide a non-empty JSONL input or a supported dataset.")

    output_dir = Path(settings["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"
    summary_path = output_dir / "summary.json"
    trace_dir = output_dir / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)

    use_mgtb = "mgtb_v3_window" in settings["methods"]
    cfg = load_config(settings["config"]) if use_mgtb else None

    import torch

    torch.manual_seed(settings["seed"])
    rows: list[dict[str, Any]] = []
    with results_path.open("w", encoding="utf-8") as out:
        for precision in settings["precisions"]:
            model, tokenizer, effective_precision = _load_model(
                settings["base_model"],
                precision,
                settings["device_map"],
                settings["allow_cpu_fp32_fallback"],
            )
            calibration = _load_calibration_for_precision(settings, precision, effective_precision) if use_mgtb else None
            try:
                for method in settings["methods"]:
                    variant_rows: list[dict[str, Any]] = []
                    for index, item in enumerate(items):
                        run_seed = _stable_seed(
                            settings["seed"],
                            item.get("dataset", "jsonl"),
                            item.get("split", ""),
                            index,
                            precision,
                            method,
                        )
                        _seed_everything(run_seed)
                        started = time.time()
                        if method == "vanilla":
                            payload = _run_vanilla(model, tokenizer, item["prompt"], settings["max_new_tokens"])
                        else:
                            trace_path = trace_dir / f"{effective_precision}_{method}_{index:04d}.jsonl"
                            result = generate_with_mgtb_v3(
                                model,
                                tokenizer,
                                item["prompt"],
                                cfg,
                                calibration["calibrator"],
                                calibration["threshold"],
                                max_new_tokens=settings["max_new_tokens"],
                                trace_log_path=trace_path,
                                do_backtracking=True,
                            )
                            payload = asdict(result)
                            payload["latency"] = getattr(result, "latency", time.time() - started)
                        prompt_token_count = len(tokenizer(item["prompt"])["input_ids"])
                        row = {
                            "id": item.get("id", index),
                            "index": index,
                            "dataset": item.get("dataset", settings.get("dataset") or "jsonl"),
                            "split": item.get("split", settings.get("split")),
                            "question": item.get("question"),
                            "base_model": settings["base_model"],
                            "method": method,
                            "requested_precision": precision,
                            "precision": effective_precision,
                            "calibration_key": calibration["key"] if calibration else None,
                            "calibrator_path": calibration["calibrator_path"] if calibration else None,
                            "threshold_path": calibration["threshold_path"] if calibration else None,
                            "seed": run_seed,
                            "prompt": item["prompt"],
                            **payload,
                        }
                        row.setdefault("latency", time.time() - started)
                        row["tokens_generated"] = max(0, len(row.get("tokens", [])) - prompt_token_count)
                        row["completion_text"] = _completion_text(tokenizer, row.get("tokens", []), prompt_token_count, row.get("text"))
                        row.update(_score_item(item, row))
                        out.write(json.dumps(row, ensure_ascii=False) + "\n")
                        out.flush()
                        rows.append(row)
                        variant_rows.append(row)
                        print(
                            f"[{effective_precision} {method}] {index + 1}/{len(items)} "
                            f"tokens={row['tokens_generated']} alerts={len(row.get('alerts', []))} "
                            f"backtracks={len(row.get('backtracks', []))} correct={row.get('correct', 'n/a')}"
                        )
                        if _should_print_progress(index + 1, len(items), settings["progress_interval"]):
                            print(_format_progress(effective_precision, method, index + 1, len(items), variant_rows))
            finally:
                del model
                del tokenizer
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    summary = _summarize(rows)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {results_path}")
    print(f"Wrote {summary_path}")


def _load_run_settings(args: argparse.Namespace) -> dict[str, Any]:
    settings = dict(DEFAULT_RUN_SETTINGS)
    if args.run_config:
        settings.update(_read_run_config(args.run_config))

    for key in DEFAULT_RUN_SETTINGS:
        value = getattr(args, key, None)
        if value is not None:
            settings[key] = value

    if "mgtb_config" in settings:
        settings["config"] = settings.pop("mgtb_config")
    settings["base_model"] = settings.get("base_model") or settings.get("model")
    settings["model"] = settings["base_model"]
    if isinstance(settings["methods"], str):
        settings["methods"] = [settings["methods"]]
    if isinstance(settings["precisions"], str):
        settings["precisions"] = [settings["precisions"]]

    if not settings.get("base_model"):
        raise SystemExit("Missing required setting: --base-model. Provide base_model in --run-config or use legacy --model.")
    if settings.get("dataset") and settings["dataset"] != "gsm8k":
        raise SystemExit(f"Unsupported dataset {settings['dataset']!r}; expected 'gsm8k'.")
    if not settings.get("dataset") and not settings.get("input"):
        raise SystemExit("Missing data source: provide --input or set dataset: gsm8k in the run config.")
    return settings


def _read_run_config(path: str) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        if config_path.suffix.lower() == ".json":
            data = json.load(handle)
        else:
            try:
                import yaml
            except Exception as exc:
                raise SystemExit("YAML run configs require PyYAML. Install with: pip install pyyaml") from exc
            data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise SystemExit("Run config must be a YAML/JSON object.")
    return data


def _load_prompts(path: str, limit: int) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            data = json.loads(line)
            prompt = data.get("prompt") or data.get("question") or data.get("input")
            if prompt:
                rows.append(
                    {
                        "id": data.get("id", len(rows)),
                        "dataset": data.get("dataset", "jsonl"),
                        "split": data.get("split"),
                        "question": data.get("question"),
                        "answer": data.get("answer"),
                        "reference_answer": data.get("reference_answer"),
                        "prompt": str(prompt),
                    }
                )
            if limit is not None and len(rows) >= limit:
                break
    return rows


def _load_items(settings: dict[str, Any]) -> list[dict[str, Any]]:
    if settings.get("dataset") == "gsm8k":
        return load_gsm8k_items(
            dataset_name=settings["dataset_name"],
            dataset_config=settings["dataset_config"],
            split=settings["split"],
            limit=settings["limit"],
            seed=settings["seed"],
            prompt_style=settings["prompt_style"],
        )
    return _load_prompts(settings["input"], settings["limit"])


def _validate_mgtb_settings(settings: dict[str, Any]) -> None:
    if "mgtb_v3_window" not in settings["methods"]:
        return
    if settings.get("calibration") is not None:
        if not isinstance(settings["calibration"], dict):
            raise SystemExit("calibration must be a mapping keyed by precision, for example calibration.fp16.calibrator.")
        for precision in settings["precisions"]:
            entry = _precision_calibration_entry(settings, precision)
            if not entry:
                raise SystemExit(f"mgtb_v3_window requires calibration.{precision}.calibrator and calibration.{precision}.threshold.")
            _validate_calibration_entry(entry, precision)
        return
    if not settings.get("calibrator") or not settings.get("threshold"):
        raise SystemExit("mgtb_v3_window requires --calibrator and --threshold.")
    _validate_calibration_entry(settings, "global")


def _precision_calibration_entry(settings: dict[str, Any], precision: str) -> dict[str, Any] | None:
    calibration = settings.get("calibration") or {}
    entry = calibration.get(precision)
    if isinstance(entry, dict):
        return entry
    return None


def _validate_calibration_entry(entry: dict[str, Any], precision: str) -> None:
    if not entry.get("calibrator") or not entry.get("threshold"):
        raise SystemExit(f"mgtb_v3_window requires calibration for {precision}: calibrator and threshold.")
    calibrator_path = Path(entry["calibrator"])
    if not calibrator_path.exists():
        raise SystemExit(f"Calibrator file not found: {calibrator_path}")
    threshold = str(entry["threshold"])
    if not _looks_like_number(threshold) and not Path(threshold).exists():
        raise SystemExit(f"Threshold file not found and value is not numeric: {threshold}")


def _load_calibration_for_precision(settings: dict[str, Any], precision: str, effective_precision: str) -> dict[str, Any]:
    entry = _precision_calibration_entry(settings, precision)
    key = precision
    if entry is None:
        entry = {"calibrator": settings["calibrator"], "threshold": settings["threshold"]}
        key = "global"
    calibrator_path = str(entry["calibrator"])
    threshold_value = str(entry["threshold"])
    return {
        "key": key,
        "requested_precision": precision,
        "effective_precision": effective_precision,
        "calibrator_path": calibrator_path,
        "threshold_path": threshold_value if not _looks_like_number(threshold_value) else None,
        "calibrator": PositionalCalibrator.load_json(calibrator_path),
        "threshold": _load_threshold(threshold_value),
    }


def _score_item(item: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    reference = item.get("reference_answer") or item.get("answer")
    if item.get("dataset") == "gsm8k" or reference is not None:
        return score_gsm8k(row.get("completion_text") or row.get("text"), reference)
    return {
        "reference_answer": None,
        "prediction_answer": None,
        "correct": None,
        "answer_extraction_ok": None,
        "reference_extraction_ok": None,
    }


def _completion_text(tokenizer, tokens: list[int], prompt_token_count: int, fallback: str | None = None) -> str:
    if tokens is None:
        return fallback or ""
    try:
        completion_tokens = list(tokens)[int(prompt_token_count) :]
        if not completion_tokens:
            return ""
        return tokenizer.decode(completion_tokens, skip_special_tokens=True)
    except Exception:
        return fallback or ""


def _stable_seed(base_seed: int, *parts: Any) -> int:
    payload = "::".join(str(part) for part in (base_seed, *parts))
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % (2**31)


def _seed_everything(seed: int) -> None:
    seed32 = int(seed) % (2**32)
    random.seed(seed32)
    try:
        import numpy as np

        np.random.seed(seed32)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed32)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed32)
    except Exception:
        pass


def _looks_like_number(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def _should_print_progress(done: int, total: int, interval: int | None) -> bool:
    if not interval or int(interval) <= 0:
        return False
    return done % int(interval) == 0 or done == total


def _format_progress(precision: str, method: str, done: int, total: int, rows: list[dict[str, Any]]) -> str:
    scored = [row for row in rows if row.get("correct") is not None]
    num_correct = sum(1 for row in scored if float(row.get("correct", 0.0)) == 1.0)
    accuracy = num_correct / len(scored) if scored else None
    mean_tokens = sum(row.get("tokens_generated", 0) for row in rows) / len(rows) if rows else 0.0
    mean_latency = sum(row.get("latency", 0.0) for row in rows) / len(rows) if rows else 0.0
    accuracy_text = "n/a" if accuracy is None else f"{accuracy:.3f}"
    return (
        f"[progress {precision} {method}] {done}/{total} "
        f"accuracy={accuracy_text} correct={num_correct}/{len(scored)} "
        f"mean_tokens={mean_tokens:.1f} mean_latency={mean_latency:.2f}s"
    )


def _load_model(model_name: str, precision: str, device_map: str, allow_cpu_fp32_fallback: bool):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    if precision == "fp16":
        if torch.cuda.is_available():
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map=device_map)
            return model, tokenizer, "fp16"
        if not allow_cpu_fp32_fallback:
            raise RuntimeError("CUDA is unavailable; FP16 generation needs a GPU. Use --allow-cpu-fp32-fallback for a CPU smoke run.")
        model = AutoModelForCausalLM.from_pretrained(model_name)
        return model, tokenizer, "fp32_cpu_fallback"

    try:
        from transformers import BitsAndBytesConfig
    except Exception as exc:
        raise RuntimeError("INT4 requires transformers BitsAndBytesConfig and bitsandbytes.") from exc
    try:
        import bitsandbytes  # noqa: F401
    except Exception as exc:
        raise RuntimeError("INT4 requires bitsandbytes. Install it with: pip install bitsandbytes") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("INT4 bitsandbytes generation requires a visible CUDA GPU/driver.")
    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config, device_map=device_map)
    return model, tokenizer, "int4"


def _run_vanilla(model, tokenizer, prompt: str, max_new_tokens: int) -> dict[str, Any]:
    import torch

    started = time.time()
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(model.device)
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    tokens = output[0].tolist()
    return {
        "text": tokenizer.decode(tokens, skip_special_tokens=True),
        "tokens": tokens,
        "alerts": [],
        "backtracks": [],
        "trace_log_path": None,
        "latency": time.time() - started,
    }


def _load_threshold(value: str) -> float:
    path = Path(value)
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        return float(data["threshold"])
    return float(value)


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = f"{row['precision']}::{row['method']}"
        groups.setdefault(key, []).append(row)
    summary = {"num_rows": len(rows), "groups": {}}
    for key, items in groups.items():
        n = len(items)
        scored = [r for r in items if r.get("correct") is not None]
        num_correct = sum(1 for r in scored if float(r.get("correct", 0.0)) == 1.0)
        summary["groups"][key] = {
            "num_instances": n,
            "accuracy": num_correct / len(scored) if scored else None,
            "num_correct": num_correct,
            "num_scored": len(scored),
            "num_answer_extraction_failures": sum(1 for r in scored if not r.get("answer_extraction_ok", False)),
            "mean_tokens_generated": sum(r.get("tokens_generated", 0) for r in items) / n,
            "mean_latency": sum(r.get("latency", 0.0) for r in items) / n,
            "mean_alerts": sum(len(r.get("alerts", [])) for r in items) / n,
            "mean_backtracks": sum(len(r.get("backtracks", [])) for r in items) / n,
        }
    return summary


if __name__ == "__main__":
    main()
