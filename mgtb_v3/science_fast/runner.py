from __future__ import annotations

import json
import math
import multiprocessing
import random
import signal
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from mgtb_v3.calibration.positional import PositionalCalibrator
from mgtb_v3.config import load_config
from mgtb_v3.eval.math500 import format_math500_prompt, score_math500
from mgtb_v3.generation.hf_loop import generate_with_mgtb_v3

from .artifacts import RunStore
from .io import load_json, sha256_json
from .provenance import git_commit, software_environment, source_tree_sha256


_PARALLEL_WORKER_SPEC: dict[str, Any] | None = None
_PARALLEL_WORKER_CONTEXT: dict[str, Any] | None = None


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    import torch
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def load_int4_model(settings: dict[str, Any]):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("science_fast INT4 requires a CUDA-visible GPU; refusing CPU/offload execution")
    device_map = settings.get("device_map")
    if device_map not in ({"": 0}, "auto"):
        raise ValueError("science_fast requires device_map={'': 0} or the checked legacy device_map='auto'")
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    model_cfg = settings["model"]
    quant = settings["quantization"]
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=bool(quant["bnb_4bit_use_double_quant"]),
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_storage=torch.uint8,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"], revision=model_cfg["revision"])
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name"], revision=model_cfg["revision"], quantization_config=bnb, device_map=device_map
    )
    assigned = set(getattr(model, "hf_device_map", {}).values())
    if not assigned or any(device not in {0, "cuda:0", torch.device("cuda:0")} for device in assigned):
        raise RuntimeError(f"refusing non-CUDA or unverifiable model placement: {getattr(model, 'hf_device_map', {})}")
    return model, tokenizer


def resolved_settings(config_path: str | Path) -> dict[str, Any]:
    import yaml
    with Path(config_path).open("r", encoding="utf-8") as handle:
        settings = yaml.safe_load(handle) or {}
    # Throughput controls do not affect sampling, scoring, or the scientific
    # identity. Keeping them out of resolved settings allows a serial run to be
    # resumed with more workers without invalidating completed artifacts.
    settings.pop("parallel_workers", None)
    controller_path = Path(settings["controller_config"])
    if not controller_path.is_absolute():
        controller_path = Path.cwd() / controller_path
    controller = load_config(controller_path)
    settings["controller"] = asdict(controller)
    settings["controller_config"] = str(controller_path)
    validate_fast_spec(settings)
    return settings


def validate_fast_spec(settings: dict[str, Any]) -> None:
    expected_model = {"name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "revision": "ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"}
    expected_quant = {
        "scheme": "bitsandbytes_int4", "bnb_4bit_quant_type": "fp4", "bnb_4bit_use_double_quant": False,
        "bnb_4bit_compute_dtype": "float16", "storage_dtype": "uint8",
    }
    if settings.get("model") != expected_model or settings.get("quantization") != expected_quant:
        raise ValueError("science_fast requires the pinned model revision and exact INT4 FP4 quantization")
    if settings.get("prompt_style") != "math500_cot" or int(settings.get("max_new_tokens", -1)) != 20000:
        raise ValueError("science_fast requires math500_cot and max_new_tokens=20000")
    if float(settings.get("vanilla_temperature", -1)) != 1.0:
        raise ValueError("science_fast requires vanilla_temperature=1.0")
    if settings.get("device_map") not in ({"": 0}, "auto"):
        raise ValueError("science_fast requires CUDA-only placement; unsupported device_map")
    controller = settings["controller"]
    required = {
        ("window", "window_size"): 64, ("window", "stride"): 32, ("window", "ngram_min"): 6, ("window", "ngram_max"): 8,
        ("detector", "p_clip"): 1e-6, ("detector", "refractory_windows"): 2,
        ("backtracking", "max_rerolls"): 3, ("backtracking", "margin_tokens"): 64,
        ("backtracking", "redecode_temperature"): 0.6, ("backtracking", "repetition_penalty"): 1.1,
        ("backtracking", "suspect_ngram_blocking"): True, ("backtracking", "prompt_injection"): False,
        ("backtracking", "cache_state_mode"): "replay_last", ("backtracking", "changepoint_index_mode"): "tracked_windows",
        ("score", "w_entropy"): 0.15, ("score", "w_logprob"): 0.10, ("score", "w_repetition"): 0.20,
        ("score", "w_confident_loop"): 0.35, ("score", "w_local_entropy_pos"): 0.18, ("score", "w_local_entropy_neg"): 0.02,
    }
    for (section, key), expected in required.items():
        if controller.get(section, {}).get(key) != expected:
            raise ValueError(f"science_fast controller mismatch: {section}.{key}")
    if tuple(controller["detector"]["betting_gammas"]) != (0.1, 0.3, 0.5, 0.7):
        raise ValueError("science_fast controller mismatch: detector.betting_gammas")


def run_role(
    *, settings: dict[str, Any], manifest: dict[str, Any], role: str, method: str, output_dir: str | Path,
    calibrator_payload: dict[str, Any] | None = None, selected_h: float | None = None, freeze: dict[str, Any] | None = None,
    stop_after: int | None = None, parallel_workers: int = 1,
) -> list[dict[str, Any]]:
    if role == "test" and freeze is None:
        raise ValueError("test execution refused: matching freeze lock is required")
    if method not in {"vanilla", "mgtb"}:
        raise ValueError("method must be vanilla or mgtb")
    if method == "mgtb" and (calibrator_payload is None or selected_h is None):
        raise ValueError("MGT-B requires frozen calibrator and selected h")
    if int(parallel_workers) < 1:
        raise ValueError("parallel_workers must be at least 1")
    if freeze is not None:
        _assert_runtime_matches_freeze(settings, freeze, calibrator_payload, selected_h)
    identity = {
        "manifest_sha256": manifest["manifest_sha256"], "role": role, "method": method,
        "resolved_config_sha256": sha256_json(settings), "freeze_sha256": freeze.get("freeze_sha256") if freeze else None,
    }
    store = RunStore(output_dir, identity)
    source = {
        "git_commit": git_commit(), "source_tree_sha256": source_tree_sha256(), "software_environment": software_environment(),
        "command": " ".join(__import__("sys").argv),
    }
    worker_spec = {
        "settings": settings,
        "role": role,
        "method": method,
        "output_dir": str(output_dir),
        "calibrator_payload": calibrator_payload,
        "selected_h": selected_h,
        "source": source,
        "run_identity_sha256": store.identity_sha256,
        "resolved_config_sha256": identity["resolved_config_sha256"],
    }
    items = manifest["roles"][role]
    if int(parallel_workers) == 1:
        context = _build_generation_context(worker_spec)
        return store.run(items, lambda item: _generate_item(item, context), stop_after=stop_after)
    return _run_parallel_items(
        store, items, worker_spec, parallel_workers=int(parallel_workers), stop_after=stop_after,
    )


def _build_generation_context(worker_spec: dict[str, Any]) -> dict[str, Any]:
    settings = worker_spec["settings"]
    model, tokenizer = load_int4_model(settings)
    controller = load_config(settings["controller_config"])
    payload = worker_spec["calibrator_payload"]
    if payload is None:
        # A harmless pool is enough because Vanilla/reference/development never
        # act on detector output; their window scores are still traced.
        calibrator = PositionalCalibrator(score_pools_by_bucket={"0-512": [0.0]})
    else:
        calibrator = PositionalCalibrator(payload["buckets"], payload["score_pools_by_bucket"], payload["p_clip"])
    selected_h = worker_spec["selected_h"]
    threshold_factor = math.inf if worker_spec["method"] == "vanilla" else math.exp(float(selected_h))
    return {**worker_spec, "model": model, "tokenizer": tokenizer, "controller": controller,
            "calibrator": calibrator, "threshold_factor": threshold_factor}


def _generate_item(item: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    import torch

    settings = context["settings"]
    model, tokenizer = context["model"], context["tokenizer"]
    seed_everything(item["item_seed"])
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    trace_path = Path(context["output_dir"]) / "in_progress" / (RunStore.filename(item["item_id"]) + ".trace.jsonl")
    prompt = format_math500_prompt(item["problem"])
    started = time.perf_counter()
    result = generate_with_mgtb_v3(
        model, tokenizer, prompt, context["controller"], context["calibrator"], context["threshold_factor"],
        max_new_tokens=int(settings["max_new_tokens"]), trace_log_path=trace_path,
        do_backtracking=(context["method"] == "mgtb"),
    )
    wall = time.perf_counter() - started
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0].tolist()
    completion_ids = result.tokens[len(prompt_ids):]
    generation = tokenizer.decode(completion_ids, skip_special_tokens=True)
    scorer = score_math500(generation, item["reference_answer"])
    trace = []
    if trace_path.exists():
        with trace_path.open("r", encoding="utf-8") as handle:
            trace = [json.loads(line) for line in handle if line.strip()]
        trace_path.unlink()
    peak = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
    accounting = {
        "sampled": result.sampled_tokens, "emitted": result.emitted_tokens, "deleted": result.deleted_tokens,
        "alarms": len(result.alerts), "rerolls": len(result.backtracks),
        "alarm_positions": [a.token_pos - len(prompt_ids) for a in result.alerts],
        "rollback_spans": [int(e.get("rollback_span", 0)) for e in result.backtracks],
        "termination_reason": result.termination_reason,
    }
    return {
        "generation": generation, "token_ids": completion_ids, "scorer": scorer,
        "sampled_tokens": result.sampled_tokens, "emitted_tokens": result.emitted_tokens, "deleted_tokens": result.deleted_tokens,
        "token_accounting": accounting,
        "monitor_trace": trace if context["method"] == "mgtb" or context["role"] in {"reference", "development"} else [],
        "monitor_state": getattr(result, "retained_monitor_windows", []),
        "truncated": result.termination_reason == "max_new_tokens", "timing": {"wall_seconds": wall, "peak_vram_bytes": peak},
        "provenance": {**context["source"], "run_identity_sha256": context["run_identity_sha256"],
                       "resolved_config_sha256": context["resolved_config_sha256"]},
    }


def _configure_parallel_worker(worker_spec: dict[str, Any]) -> None:
    global _PARALLEL_WORKER_SPEC, _PARALLEL_WORKER_CONTEXT
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _PARALLEL_WORKER_SPEC = worker_spec
    _PARALLEL_WORKER_CONTEXT = None


def _parallel_generate_item(item: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    global _PARALLEL_WORKER_CONTEXT
    if _PARALLEL_WORKER_SPEC is None:
        raise RuntimeError("parallel generation worker was not configured")
    if _PARALLEL_WORKER_CONTEXT is None:
        _PARALLEL_WORKER_CONTEXT = _build_generation_context(_PARALLEL_WORKER_SPEC)
    return item["item_id"], _generate_item(item, _PARALLEL_WORKER_CONTEXT)


def _multiprocessing_context():
    # CUDA cannot safely be reinitialized in a forked subprocess.
    return multiprocessing.get_context("spawn")


def _run_parallel_items(
    store: RunStore,
    items: list[dict[str, Any]],
    worker_spec: dict[str, Any],
    *,
    parallel_workers: int,
    stop_after: int | None,
) -> list[dict[str, Any]]:
    pending = [item for item in items if store.valid_artifact(item) is None]
    if stop_after is not None:
        pending = pending[:max(0, int(stop_after))]
    if pending:
        item_by_id = {item["item_id"]: item for item in pending}
        pool = _multiprocessing_context().Pool(
            processes=min(parallel_workers, len(pending)),
            initializer=_configure_parallel_worker,
            initargs=(worker_spec,),
        )
        try:
            for item_id, artifact in pool.imap_unordered(_parallel_generate_item, pending, chunksize=1):
                store.save(item_by_id[item_id], artifact)
                completed = int(load_json(store.state_path)["completed_count"])
                print(f"[science-fast] completed {completed}/{len(items)} ({item_id})", file=sys.stderr, flush=True)
        except BaseException:
            pool.terminate()
            pool.join()
            raise
        else:
            pool.close()
            pool.join()
    return [artifact for item in items if (artifact := store.valid_artifact(item)) is not None]


def load_run_artifacts(root: str | Path, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    root = Path(root) / "items"
    output = []
    for item in items:
        path = root / RunStore.filename(item["item_id"])
        if path.exists():
            output.append(load_json(path))
    return output


def _assert_runtime_matches_freeze(settings, freeze, calibrator_payload, selected_h):
    if settings.get("model") != freeze.get("model"):
        raise ValueError("runtime model does not match freeze")
    if settings.get("quantization") != freeze.get("quantization"):
        raise ValueError("runtime quantization does not match freeze")
    if settings.get("device_map") != freeze.get("device_map"):
        raise ValueError("runtime device map does not match freeze")
    if settings.get("controller") != freeze.get("resolved_controller_config"):
        raise ValueError("runtime controller does not match freeze")
    if int(settings.get("max_new_tokens", -1)) != int(freeze.get("max_new_tokens", -2)):
        raise ValueError("runtime token budget does not match freeze")
    if calibrator_payload is not None and calibrator_payload.get("calibrator_sha256") != freeze.get("calibrator_sha256"):
        raise ValueError("runtime calibrator does not match freeze")
    if selected_h is not None and float(selected_h) != float(freeze.get("selected_h")):
        raise ValueError("runtime threshold does not match freeze")
    source = freeze.get("source", {})
    if source.get("git_commit") != git_commit() or source.get("source_tree_sha256") != source_tree_sha256():
        raise ValueError("runtime source tree does not match freeze")
