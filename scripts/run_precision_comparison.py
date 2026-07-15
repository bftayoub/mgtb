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

from mgtb_v3.baselines.budget import (
    assigned_template,
    file_sha256,
    load_profile,
    load_per_id_budget,
    per_id_budget_map,
    periodic_schedule,
    profile_sha256,
    random_schedule,
    restart_indices,
    revision_token_budget,
    stable_int_seed,
)
from mgtb_v3.calibration.positional import PositionalCalibrator
from mgtb_v3.config import load_config
from mgtb_v3.eval.gsm8k import load_gsm8k_items, score_gsm8k
from mgtb_v3.eval.math500 import (
    DEFAULT_DATASET_CONFIG as MATH500_DEFAULT_DATASET_CONFIG,
    DEFAULT_DATASET_NAME as MATH500_DEFAULT_DATASET_NAME,
    DEFAULT_PROMPT_STYLE as MATH500_DEFAULT_PROMPT_STYLE,
    load_math500_items,
    score_math500,
)
from mgtb_v3.generation.hf_loop import _sample_token, generate_with_mgtb_v3
from mgtb_v3.generation.baseline_loop import generate_scheduled_backtracking


METHODS = (
    "vanilla",
    "mgtb_v3_window",
    "random_backtrack",
    "periodic_backtrack",
    "restart",
    "self_correct",
)
SCHEDULED_BASELINES = {"random_backtrack", "periodic_backtrack"}
PROFILE_BASELINES = SCHEDULED_BASELINES | {"restart", "self_correct"}


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
    "exclude_ids_from": [],
    "include_ids_from": [],
    "run_seed_from": [],
    "expected_num_excluded": None,
    "expected_excluded_ids_sha256": None,
    "expected_num_items": None,
    "expected_selected_ids_sha256": None,
    "budget_profile": None,
    "decode_budget_table": None,
    "baseline_config": None,
}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Run vanilla and MGT-B v3 generations across FP16/INT4 on JSONL prompts or GSM8K."
    )
    parser.add_argument("--run-config", help="YAML or JSON file describing this test run.")
    parser.add_argument("--base-model", help="Underlying HuggingFace model id or local model path used for generation.")
    parser.add_argument("--model", help="Legacy alias for --base-model.")
    parser.add_argument("--input", help="JSONL with at least a 'prompt' field per row.")
    parser.add_argument("--dataset", choices=["gsm8k", "math500"], help="Built-in dataset loader to use instead of --input.")
    parser.add_argument("--dataset-name", help="HuggingFace dataset name for the built-in dataset loader.")
    parser.add_argument("--dataset-config", help="HuggingFace dataset config for the built-in dataset loader.")
    parser.add_argument("--split", help="Dataset split for the built-in dataset loader.")
    parser.add_argument("--prompt-style", help="Prompt template name for the built-in dataset loader.")
    parser.add_argument("--output-dir")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--methods", nargs="+", choices=METHODS)
    parser.add_argument("--precisions", nargs="+", choices=["fp16", "int4"])
    parser.add_argument("--config", help="MGT-B v3 controller config YAML.")
    parser.add_argument("--calibrator", help="Required for mgtb_v3_window.")
    parser.add_argument("--threshold", help="Threshold JSON or numeric threshold. Required for mgtb_v3_window.")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device-map")
    parser.add_argument("--progress-interval", type=int, help="Print cumulative per-variant metrics every N examples.")
    parser.add_argument("--budget-profile", help="Frozen JSON budget profile required by control baselines.")
    parser.add_argument("--decode-budget-table", help="Optional strict per-ID total decode budget JSON.")
    parser.add_argument("--baseline-config", help="Optional YAML/JSON settings for priority-1 control baselines.")
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
    if settings.get("expected_num_items") is not None and len(items) != int(settings["expected_num_items"]):
        raise SystemExit(
            f"Expected {int(settings['expected_num_items'])} selected prompts, found {len(items)}. "
            "Refusing to run because the confirmatory split has drifted."
        )
    if settings.get("expected_selected_ids_sha256") is not None:
        selected_digest = _ids_sha256(str(item.get("id")) for item in items)
        if selected_digest != settings["expected_selected_ids_sha256"]:
            raise SystemExit(
                f"Selected-ID digest mismatch: expected {settings['expected_selected_ids_sha256']}, "
                f"found {selected_digest}. Refusing to run because the confirmatory split has drifted."
            )

    frozen_run_seeds: dict[str, dict[str, tuple[int, str]]] = {}
    if settings.get("run_seed_from"):
        selected_ids = {str(item.get("id")) for item in items}
        for requested_precision in settings["precisions"]:
            seed_map = _load_run_seed_map(settings["run_seed_from"], requested_precision)
            missing_ids = sorted(selected_ids - set(seed_map))
            if missing_ids:
                raise SystemExit(
                    f"run_seed_from has no {requested_precision} vanilla seed for {len(missing_ids)} selected IDs; "
                    f"first missing ID: {missing_ids[0]}"
                )
            frozen_run_seeds[requested_precision] = seed_map

    output_dir = Path(settings["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"
    summary_path = output_dir / "summary.json"
    trace_dir = output_dir / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)

    needs_controller_config = bool(set(settings["methods"]) & (SCHEDULED_BASELINES | {"mgtb_v3_window"}))
    cfg = load_config(settings["config"]) if needs_controller_config else None
    baseline_settings = _load_baseline_settings(settings.get("baseline_config"))
    budget_profile = load_profile(settings["budget_profile"]) if set(settings["methods"]) & PROFILE_BASELINES else None
    budget_profile_digest = profile_sha256(settings["budget_profile"]) if budget_profile else None
    decode_budget_table = load_per_id_budget(settings["decode_budget_table"]) if settings.get("decode_budget_table") else None
    decode_budgets = per_id_budget_map(decode_budget_table) if decode_budget_table else {}
    decode_budget_digest = file_sha256(settings["decode_budget_table"]) if decode_budget_table else None
    if decode_budget_table:
        missing_budget_ids = sorted({str(item.get("id")) for item in items} - set(decode_budgets))
        if missing_budget_ids:
            raise SystemExit(
                f"decode_budget_table has no entry for {len(missing_budget_ids)} selected IDs; "
                f"first missing ID: {missing_budget_ids[0]}"
            )

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
            if budget_profile:
                condition = budget_profile["condition"]
                if condition["base_model"] != settings["base_model"] or condition["precision"] != effective_precision:
                    raise SystemExit(
                        "Budget profile condition does not match this run: "
                        f"expected {condition['base_model']} / {condition['precision']}, "
                        f"got {settings['base_model']} / {effective_precision}."
                    )
            if decode_budget_table:
                condition = decode_budget_table["condition"]
                if condition["base_model"] != settings["base_model"] or condition["precision"] != effective_precision:
                    raise SystemExit(
                        "Per-ID decode budget condition does not match this run: "
                        f"expected {condition['base_model']} / {condition['precision']}, "
                        f"got {settings['base_model']} / {effective_precision}."
                    )
            calibration = _load_calibration_for_precision(settings, precision, effective_precision) if "mgtb_v3_window" in settings["methods"] else None
            restart_set = restart_indices(
                num_items=len(items), profile=budget_profile, base_seed=settings["seed"], precision=effective_precision
            ) if budget_profile and "restart" in settings["methods"] else set()
            try:
                for method in settings["methods"]:
                    variant_rows: list[dict[str, Any]] = []
                    for index, item in enumerate(items):
                        frozen_seed = frozen_run_seeds.get(precision, {}).get(str(item.get("id")))
                        if frozen_seed is not None:
                            run_seed, run_seed_source = frozen_seed
                        else:
                            run_seed = _stable_seed(
                                settings["seed"],
                                item.get("dataset", "jsonl"),
                                item.get("split", ""),
                                index,
                                precision,
                            )
                            run_seed_source = None
                        _seed_everything(run_seed)
                        started = time.time()
                        if method == "vanilla":
                            payload = _run_vanilla(model, tokenizer, item["prompt"], settings["max_new_tokens"])
                        elif method == "mgtb_v3_window":
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
                        else:
                            trace_path = trace_dir / f"{effective_precision}_{method}_{index:04d}.jsonl"
                            decode_budget = decode_budgets.get(str(item.get("id")))
                            payload = _run_baseline_method(
                                method=method,
                                model=model,
                                tokenizer=tokenizer,
                                prompt=item["prompt"],
                                max_new_tokens=settings["max_new_tokens"],
                                cfg=cfg,
                                profile=budget_profile,
                                baseline_settings=baseline_settings,
                                run_seed=run_seed,
                                index=index,
                                effective_precision=effective_precision,
                                trace_path=trace_path,
                                restart_selected=index in restart_set,
                                max_decode_events=int(decode_budget["control_max_decode_events"]) if decode_budget else None,
                            )
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
                            "seed_source_path": run_seed_source,
                            "prompt": item["prompt"],
                            "budget_profile_path": settings["budget_profile"] if budget_profile else None,
                            "budget_profile_sha256": budget_profile_digest,
                            "budget_target_extra_tokens": budget_profile["summary"]["mean_extra_decode_tokens"] if budget_profile else None,
                            "decode_budget_table_path": settings.get("decode_budget_table"),
                            "decode_budget_table_sha256": decode_budget_digest,
                            "budget_mgtb_decode_events": decode_budgets.get(str(item.get("id")), {}).get("mgtb_decode_events"),
                            "budget_max_decode_events": decode_budgets.get(str(item.get("id")), {}).get("control_max_decode_events"),
                            **payload,
                        }
                        row.setdefault("latency", time.time() - started)
                        decoded_tokens = max(0, len(row.get("tokens", [])) - prompt_token_count)
                        row["tokens_generated"] = int(row.get("final_tokens_generated", decoded_tokens))
                        row["primary_tokens_generated"] = int(row.get("primary_tokens_generated", decoded_tokens))
                        row["final_tokens_generated"] = int(row.get("final_tokens_generated", row["tokens_generated"]))
                        row["token_events_trace"] = int(
                            row.get("decode_token_events_total", _count_trace_token_events(row.get("trace_log_path"), row["tokens_generated"]))
                        )
                        row["extra_sampled"] = int(row.get("extra_decode_tokens", max(0, row["token_events_trace"] - row["tokens_generated"])))
                        row["decode_token_events_total"] = row["token_events_trace"]
                        row["extra_decode_tokens"] = row["extra_sampled"]
                        row["decode_budget_compliant"] = (
                            row["decode_token_events_total"] <= int(row["budget_max_decode_events"])
                            if row.get("budget_max_decode_events") is not None
                            else None
                        )
                        row["completion_text"] = row.get("completion_text_override") or _completion_text(
                            tokenizer, row.get("tokens", []), prompt_token_count, row.get("text")
                        )
                        row["interventions"] = row.get("interventions", row.get("backtracks", []))
                        if row.get("budget_target_extra_tokens"):
                            row["budget_match_relative_error"] = (
                                row["extra_decode_tokens"] - float(row["budget_target_extra_tokens"])
                            ) / float(row["budget_target_extra_tokens"])
                        else:
                            row["budget_match_relative_error"] = None
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
    if budget_profile:
        budget_path = output_dir / "budget_summary.json"
        budget_path.write_text(
            json.dumps(_budget_summary(summary, budget_profile, settings["budget_profile"], budget_profile_digest), indent=2, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {budget_path}")
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
    if settings.get("dataset") == "math500":
        if settings.get("dataset_name") == DEFAULT_RUN_SETTINGS["dataset_name"]:
            settings["dataset_name"] = MATH500_DEFAULT_DATASET_NAME
        if settings.get("dataset_config") == DEFAULT_RUN_SETTINGS["dataset_config"]:
            settings["dataset_config"] = MATH500_DEFAULT_DATASET_CONFIG
        if settings.get("prompt_style") == DEFAULT_RUN_SETTINGS["prompt_style"]:
            settings["prompt_style"] = MATH500_DEFAULT_PROMPT_STYLE

    if not settings.get("base_model"):
        raise SystemExit("Missing required setting: --base-model. Provide base_model in --run-config or use legacy --model.")
    if settings.get("dataset") and settings["dataset"] not in {"gsm8k", "math500"}:
        raise SystemExit(f"Unsupported dataset {settings['dataset']!r}; expected one of: gsm8k, math500.")
    if not settings.get("dataset") and not settings.get("input"):
        raise SystemExit("Missing data source: provide --input or set dataset: gsm8k or math500 in the run config.")
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


def _load_baseline_settings(path: str | None) -> dict[str, Any]:
    settings: dict[str, Any] = {
        "random_backtrack": {"minimum_position": 64},
        "periodic_backtrack": {"minimum_position": 64},
        "self_correct": {
            "instruction": "\nWait. Re-check the previous solution carefully. Return the corrected final answer on a line formatted as #### <answer>.\n",
            "temperature": 0.6,
            "repetition_penalty": 1.1,
        },
    }
    if not path:
        return settings
    loaded = _read_run_config(path)
    if not isinstance(loaded, dict):
        raise SystemExit("baseline_config must be a mapping.")
    for key, value in loaded.items():
        if isinstance(value, dict) and isinstance(settings.get(key), dict):
            settings[key] = {**settings[key], **value}
        else:
            settings[key] = value
    return settings


def _run_baseline_method(
    *,
    method: str,
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    cfg,
    profile: dict[str, Any],
    baseline_settings: dict[str, Any],
    run_seed: int,
    index: int,
    effective_precision: str,
    trace_path: Path,
    restart_selected: bool,
    max_decode_events: int | None = None,
) -> dict[str, Any]:
    # A strict per-ID table supersedes the generic generation cap: a MGT-B
    # trajectory that used 20k events may grant a control up to 22k at +10%.
    total_budget = max(1, int(max_decode_events)) if max_decode_events is not None else int(max_new_tokens)
    if method in SCHEDULED_BASELINES:
        template = assigned_template(profile, base_seed=run_seed, precision=effective_precision, index=index)
        if method == "random_backtrack":
            schedule = random_schedule(
                template,
                seed=stable_int_seed(run_seed, method, "schedule"),
                minimum_position=int(baseline_settings[method]["minimum_position"]),
            )
        else:
            schedule = periodic_schedule(template, minimum_position=int(baseline_settings[method]["minimum_position"]))
        payload = generate_scheduled_backtracking(
            model,
            tokenizer,
            prompt,
            cfg,
            schedule=schedule,
            max_new_tokens=total_budget,
            trace_log_path=trace_path,
        )
        payload["trigger_schedule"] = schedule
        payload["budget_template"] = template
        return payload

    primary = _run_vanilla(model, tokenizer, prompt, total_budget)
    primary_tokens = max(0, len(primary.get("tokens", [])) - len(tokenizer(prompt)["input_ids"]))
    if method == "restart":
        if not restart_selected:
            primary.update(
                {
                    "interventions": [],
                    "restart_applied": False,
                    "primary_tokens_generated": primary_tokens,
                    "final_tokens_generated": primary_tokens,
                    "decode_token_events_total": primary_tokens,
                    "extra_decode_tokens": 0,
                }
            )
            return primary
        remaining_budget = max(0, total_budget - primary_tokens)
        if remaining_budget <= 0:
            primary.update(
                {
                    "interventions": [{"type": "restart", "applied": False, "reason": "decode_budget_exhausted"}],
                    "restart_applied": False,
                    "primary_tokens_generated": primary_tokens,
                    "final_tokens_generated": primary_tokens,
                    "decode_token_events_total": primary_tokens,
                    "extra_decode_tokens": 0,
                }
            )
            return primary
        secondary_seed = stable_int_seed(run_seed, "restart", 1)
        _seed_everything(secondary_seed)
        restarted = _run_vanilla(model, tokenizer, prompt, remaining_budget)
        final_tokens = max(0, len(restarted.get("tokens", [])) - len(tokenizer(prompt)["input_ids"]))
        restarted.update(
            {
                "interventions": [{"type": "restart", "applied": True}],
                "restart_applied": True,
                "secondary_seed": secondary_seed,
                "initial_completion_text": _completion_text(tokenizer, primary.get("tokens", []), len(tokenizer(prompt)["input_ids"]), primary.get("text")),
                "primary_tokens_generated": primary_tokens,
                "final_tokens_generated": final_tokens,
                "decode_token_events_total": primary_tokens + final_tokens,
                "extra_decode_tokens": final_tokens,
                "restart_completion_text": _completion_text(tokenizer, restarted.get("tokens", []), len(tokenizer(prompt)["input_ids"]), restarted.get("text")),
            }
        )
        return restarted

    if method == "self_correct":
        settings = baseline_settings["self_correct"]
        initial_completion = _completion_text(tokenizer, primary.get("tokens", []), len(tokenizer(prompt)["input_ids"]), primary.get("text"))
        remaining_budget = max(0, total_budget - primary_tokens)
        if remaining_budget <= 0:
            primary.update(
                {
                    "interventions": [{"type": "self_correct", "applied": False, "reason": "decode_budget_exhausted"}],
                    "revision_applied": False,
                    "primary_tokens_generated": primary_tokens,
                    "final_tokens_generated": primary_tokens,
                    "decode_token_events_total": primary_tokens,
                    "extra_decode_tokens": 0,
                }
            )
            return primary
        revision_prompt = f"{prompt}{initial_completion}{settings['instruction']}"
        secondary_seed = stable_int_seed(run_seed, "self_correct", 1)
        _seed_everything(secondary_seed)
        revision = _run_plain(
            model,
            tokenizer,
            revision_prompt,
            min(revision_token_budget(profile), remaining_budget),
            temperature=float(settings["temperature"]),
            repetition_penalty=float(settings["repetition_penalty"]),
        )
        revision_prompt_tokens = len(tokenizer(revision_prompt)["input_ids"])
        revision_tokens = max(0, len(revision.get("tokens", [])) - revision_prompt_tokens)
        revision_completion = _completion_text(tokenizer, revision.get("tokens", []), revision_prompt_tokens, revision.get("text"))
        revision.update(
            {
                "interventions": [{"type": "self_correct", "applied": True, "max_tokens": min(revision_token_budget(profile), remaining_budget)}],
                "revision_applied": True,
                "secondary_seed": secondary_seed,
                "initial_completion_text": initial_completion,
                "revision_text": revision_completion,
                "completion_text_override": revision_completion,
                "primary_tokens_generated": primary_tokens,
                "final_tokens_generated": revision_tokens,
                "decode_token_events_total": primary_tokens + revision_tokens,
                "extra_decode_tokens": revision_tokens,
            }
        )
        return revision

    raise ValueError(f"Unsupported baseline method: {method}")


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
    exclusion_paths = settings.get("exclude_ids_from") or []
    inclusion_paths = settings.get("include_ids_from") or []
    excluded_ids = _load_excluded_ids(exclusion_paths)
    included_ids = _load_included_ids(inclusion_paths)
    if settings.get("expected_num_excluded") is not None and len(excluded_ids) != int(settings["expected_num_excluded"]):
        raise SystemExit(
            f"Expected {int(settings['expected_num_excluded'])} excluded IDs, found {len(excluded_ids)}. "
            "Refusing to run because the exclusion set has drifted."
        )
    if settings.get("expected_excluded_ids_sha256") is not None:
        excluded_digest = _ids_sha256(excluded_ids)
        if excluded_digest != settings["expected_excluded_ids_sha256"]:
            raise SystemExit(
                f"Excluded-ID digest mismatch: expected {settings['expected_excluded_ids_sha256']}, "
                f"found {excluded_digest}. Refusing to run because the exclusion set has drifted."
            )

    if settings.get("dataset") == "gsm8k":
        items = load_gsm8k_items(
            dataset_name=settings["dataset_name"],
            dataset_config=settings["dataset_config"],
            split=settings["split"],
            limit=None if excluded_ids or included_ids else settings["limit"],
            seed=settings["seed"],
            prompt_style=settings["prompt_style"],
        )
    elif settings.get("dataset") == "math500":
        items = load_math500_items(
            dataset_name=settings["dataset_name"],
            dataset_config=settings["dataset_config"],
            split=settings["split"],
            limit=None if excluded_ids or included_ids else settings["limit"],
            seed=settings["seed"],
            prompt_style=settings["prompt_style"],
        )
    else:
        items = _load_prompts(settings["input"], settings["limit"])

    if excluded_ids or included_ids:
        if included_ids:
            items = [item for item in items if str(item.get("id")) in included_ids]
        items = [item for item in items if str(item.get("id")) not in excluded_ids]
        if settings["limit"] is not None:
            items = items[: max(int(settings["limit"]), 0)]
    return items


def _load_excluded_ids(paths: str | list[str]) -> set[str]:
    return _load_ids_from_jsonl(paths, "exclude_ids_from")


def _load_included_ids(paths: str | list[str]) -> set[str]:
    return _load_ids_from_jsonl(paths, "include_ids_from")


def _load_run_seed_map(paths: str | list[str], requested_precision: str) -> dict[str, tuple[int, str]]:
    if isinstance(paths, str):
        paths = [paths]
    if not isinstance(paths, list):
        raise SystemExit("run_seed_from must be a JSONL path or a list of JSONL paths.")

    seeds: dict[str, tuple[int, str]] = {}
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_file():
            raise SystemExit(f"Run-seed source does not exist: {path}")
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SystemExit(f"Invalid JSON in run-seed source {path}:{line_number}: {exc}") from exc
                row_precision = row.get("requested_precision", row.get("precision"))
                if row.get("method") != "vanilla" or row_precision != requested_precision:
                    continue
                if row.get("id") is None or row.get("seed") is None:
                    continue
                seeds.setdefault(str(row["id"]), (int(row["seed"]), str(path)))
    return seeds


def _load_ids_from_jsonl(paths: str | list[str], setting_name: str) -> set[str]:
    if isinstance(paths, str):
        paths = [paths]
    if not isinstance(paths, list):
        raise SystemExit(f"{setting_name} must be a JSONL path or a list of JSONL paths.")

    excluded: set[str] = set()
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_file():
            raise SystemExit(f"Exclusion source does not exist: {path}")
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SystemExit(f"Invalid JSON in exclusion source {path}:{line_number}: {exc}") from exc
                if row.get("id") is not None:
                    excluded.add(str(row["id"]))
    return excluded


def _ids_sha256(ids) -> str:
    payload = "\n".join(sorted(set(ids))) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_mgtb_settings(settings: dict[str, Any]) -> None:
    methods = set(settings["methods"])
    unknown = methods - set(METHODS)
    if unknown:
        raise SystemExit(f"Unsupported methods: {sorted(unknown)}")
    if methods & PROFILE_BASELINES:
        profile_path = settings.get("budget_profile")
        if not profile_path or not Path(profile_path).is_file():
            raise SystemExit("random_backtrack, periodic_backtrack, restart, and self_correct require an existing budget_profile JSON.")
    if methods & SCHEDULED_BASELINES and not settings.get("config"):
        raise SystemExit("random_backtrack and periodic_backtrack require config with backtracking settings.")
    if settings.get("decode_budget_table") and not Path(settings["decode_budget_table"]).is_file():
        raise SystemExit(f"Per-ID decode budget table not found: {settings['decode_budget_table']}")
    if "mgtb_v3_window" not in methods:
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
    if item.get("dataset") == "math500":
        return score_math500(row.get("completion_text") or row.get("text"), reference)
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


def _count_trace_token_events(trace_log_path: str | None, fallback: int = 0) -> int:
    if not trace_log_path:
        return int(fallback)
    path = Path(trace_log_path)
    if not path.exists():
        return int(fallback)
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "token":
                count += 1
    return count


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
    _assert_cuda_only_model(model)
    return model, tokenizer, "int4"


def _assert_cuda_only_model(model) -> None:
    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, dict) and device_map:
        bad = {name: device for name, device in device_map.items() if str(device).lower() in {"cpu", "disk"}}
        if bad:
            raise RuntimeError(f"INT4 run requires GPU-only placement, but these modules were offloaded: {bad}")
        has_cuda = any(isinstance(device, int) or str(device).lower().startswith("cuda") for device in device_map.values())
        if has_cuda:
            return
    if str(getattr(model, "device", "")).lower().startswith("cuda"):
        return
    raise RuntimeError("INT4 run requires every model module on CUDA; no CUDA placement was detected.")


def _run_vanilla(model, tokenizer, prompt: str, max_new_tokens: int) -> dict[str, Any]:
    return _run_plain(model, tokenizer, prompt, max_new_tokens)


def _run_plain(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    *,
    temperature: float = 1.0,
    repetition_penalty: float = 1.0,
) -> dict[str, Any]:
    import torch

    model.eval()
    started = time.time()
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(model.device)
    tokens = input_ids[0].tolist()
    generated = input_ids
    cache = None
    for _ in range(max_new_tokens):
        with torch.no_grad():
            if cache is None:
                outputs = model(input_ids=generated, use_cache=True)
            else:
                outputs = model(input_ids=generated[:, -1:], past_key_values=cache, use_cache=True)
        logits = outputs.logits[:, -1, :]
        cache = getattr(outputs, "past_key_values", None)
        next_token = _sample_token(
            logits,
            temperature=temperature,
            generated_tokens=tokens,
            repetition_penalty=repetition_penalty,
        )
        token_id = int(next_token.item())
        tokens.append(token_id)
        generated = torch.tensor([tokens], device=model.device)
        if token_id == getattr(tokenizer, "eos_token_id", None):
            break
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


def _safe_mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


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
            "mean_token_events_trace": sum(r.get("token_events_trace", r.get("tokens_generated", 0)) for r in items) / n,
            "mean_extra_sampled": sum(r.get("extra_sampled", 0) for r in items) / n,
            "max_extra_sampled": max((r.get("extra_sampled", 0) for r in items), default=0),
            "mean_latency": sum(r.get("latency", 0.0) for r in items) / n,
            "mean_alerts": sum(len(r.get("alerts", [])) for r in items) / n,
            "mean_backtracks": sum(len(r.get("backtracks", [])) for r in items) / n,
            "mean_interventions": sum(len(r.get("interventions", r.get("backtracks", []))) for r in items) / n,
            "mean_primary_tokens_generated": sum(r.get("primary_tokens_generated", r.get("tokens_generated", 0)) for r in items) / n,
            "mean_final_tokens_generated": sum(r.get("final_tokens_generated", r.get("tokens_generated", 0)) for r in items) / n,
            "mean_total_decode_tokens": sum(r.get("decode_token_events_total", r.get("token_events_trace", 0)) for r in items) / n,
            "mean_extra_decode_tokens": sum(r.get("extra_decode_tokens", r.get("extra_sampled", 0)) for r in items) / n,
            "mean_budget_match_relative_error": _safe_mean(
                r.get("budget_match_relative_error") for r in items if r.get("budget_match_relative_error") is not None
            ),
        }
    return summary


def _budget_summary(summary: dict[str, Any], profile: dict[str, Any], profile_path: str, profile_digest: str) -> dict[str, Any]:
    target = float(profile["summary"]["mean_extra_decode_tokens"])
    tolerance = 0.05
    groups: dict[str, Any] = {}
    for key, values in summary["groups"].items():
        observed = float(values["mean_extra_decode_tokens"])
        relative_error = (observed - target) / target if target else None
        groups[key] = {
            "mean_extra_decode_tokens": observed,
            "mean_total_decode_tokens": values["mean_total_decode_tokens"],
            "mean_latency": values["mean_latency"],
            "relative_error_to_profile": relative_error,
            "within_five_percent": abs(relative_error) <= tolerance if relative_error is not None else False,
        }
    return {
        "budget_profile_path": profile_path,
        "budget_profile_sha256": profile_digest,
        "target_mean_extra_decode_tokens": target,
        "tolerance_relative": tolerance,
        "groups": groups,
    }


if __name__ == "__main__":
    main()
