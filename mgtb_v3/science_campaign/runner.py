from __future__ import annotations

import json
import math
import multiprocessing
import random
import signal
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from mgtb_v3.baselines.budget import assigned_template, random_schedule, stable_int_seed
from mgtb_v3.calibration.positional import PositionalCalibrator
from mgtb_v3.config import config_from_dict
from mgtb_v3.eval.gsm8k import format_gsm8k_prompt, score_gsm8k
from mgtb_v3.eval.math500 import format_math500_prompt, score_math500
from mgtb_v3.generation.hf_loop import generate_with_mgtb_v3
from mgtb_v3.science_fast.artifacts import RunStore
from mgtb_v3.science_fast.io import atomic_write_json, load_json, sha256_json
from mgtb_v3.science_fast.provenance import git_commit, software_environment, source_tree_sha256

from .config import calibration_spec, output_root, resolve_variant

_WORKER_SPEC: dict[str, Any] | None = None
_WORKER_CONTEXT: dict[str, Any] | None = None


def seed_everything(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def campaign_units(manifest: dict[str, Any], role: str, seeds: list[int]) -> list[dict[str, Any]]:
    units = []
    for item in manifest["roles"][role]:
        for replicate_seed in seeds:
            unit_id = f"{item['item_id']}|replicate:{int(replicate_seed)}"
            if int(replicate_seed) == 0 and item.get("item_seed") is not None:
                generation_seed = int(item["item_seed"])
            else:
                generation_seed = stable_int_seed(manifest["protocol_seed"], item["item_id"], int(replicate_seed))
            units.append({
                **item, "source_item_id": item["item_id"], "item_id": unit_id,
                "replicate_seed": int(replicate_seed),
                "item_seed": generation_seed,
            })
    return units


def load_model(settings: dict[str, Any]):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_cfg = settings["model"]
    name, revision = model_cfg["name"], model_cfg.get("revision")
    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
    precision = model_cfg.get("precision", "int4")
    device_map = model_cfg.get("device_map", {"": 0})
    kwargs: dict[str, Any] = {"revision": revision, "device_map": device_map}
    if precision == "int4":
        if not torch.cuda.is_available():
            raise RuntimeError("INT4 campaign runs require CUDA")
        from transformers import BitsAndBytesConfig
        quant = model_cfg.get("quantization", {})
        dtype = getattr(torch, quant.get("compute_dtype", "float16"))
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant.get("quant_type", "fp4"),
            bnb_4bit_use_double_quant=bool(quant.get("double_quant", False)),
            bnb_4bit_compute_dtype=dtype,
        )
    elif precision in {"fp16", "bf16"}:
        if not torch.cuda.is_available():
            raise RuntimeError(f"{precision} campaign runs require CUDA")
        kwargs["torch_dtype"] = torch.float16 if precision == "fp16" else torch.bfloat16
    elif precision != "fp32":
        raise ValueError(f"unsupported precision {precision!r}")
    model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    if precision != "fp32" and not bool(model_cfg.get("allow_cpu_offload", False)):
        assigned = set(getattr(model, "hf_device_map", {}).values())
        allowed = {0, "cuda:0", torch.device("cuda:0")}
        if not assigned or any(device not in allowed for device in assigned):
            raise RuntimeError(f"refusing CPU/disk or unverifiable model placement: {getattr(model, 'hf_device_map', {})}")
    return model, tokenizer


def _prompt(item: dict[str, Any], campaign: dict[str, Any]) -> str:
    style = campaign["generation"]["prompt_style"]
    kind = item.get("dataset_kind", campaign["generation"].get("dataset_kind", "math500"))
    if kind == "gsm8k":
        return format_gsm8k_prompt(item["problem"], style)
    return format_math500_prompt(item["problem"], style)


def _score(text: str, item: dict[str, Any], campaign: dict[str, Any]) -> dict[str, Any]:
    kind = item.get("dataset_kind", campaign["generation"].get("dataset_kind", "math500"))
    if kind == "gsm8k":
        return score_gsm8k(text, item["reference_answer"])
    return score_math500(text, item["reference_answer"])


def _calibration_payload(campaign: dict[str, Any], key: str):
    root = output_root(campaign) / "calibration" / key
    calibrator = load_json(root / "calibrator.json")
    threshold = load_json(root / "threshold.json")
    if calibrator.get("calibrator_sha256") != sha256_json({k: v for k, v in calibrator.items() if k != "calibrator_sha256"}):
        raise ValueError(f"calibrator hash mismatch for {key}")
    if threshold.get("threshold_sha256") != sha256_json({k: v for k, v in threshold.items() if k != "threshold_sha256"}):
        raise ValueError(f"threshold hash mismatch for {key}")
    return calibrator, threshold


def _build_context(spec: dict[str, Any]) -> dict[str, Any]:
    campaign, variant = spec["campaign"], spec["variant"]
    model, tokenizer = load_model(campaign)
    controller = config_from_dict(variant["controller"])
    calibrator = PositionalCalibrator(score_pools_by_bucket={"0-512": [0.0]})
    selected_h = None
    if variant.get("calibration"):
        payload, threshold = _calibration_payload(campaign, variant["calibration"])
        expected_spec = calibration_spec(campaign, variant["calibration"])
        if payload.get("calibration_spec_sha256") != expected_spec["calibration_sha256"]:
            raise ValueError(f"calibration spec mismatch for {variant['name']}")
        for section in ("window", "score", "detector"):
            if variant["controller"][section] != expected_spec["controller"][section]:
                raise ValueError(f"variant {variant['name']} is incompatible with its calibration ({section})")
        if variant.get("accumulation_mode", "cusum_reset") != expected_spec.get("accumulation_mode", "cusum_reset"):
            raise ValueError(f"variant {variant['name']} accumulation mode is incompatible with its calibration")
        calibrator = PositionalCalibrator(payload["buckets"], payload["score_pools_by_bucket"], payload["p_clip"])
        selected_h = float(threshold["selected_h"])
    profile = load_json(variant["profile"]) if variant.get("profile") else None
    if profile is not None and profile.get("profile_sha256") != sha256_json({k: v for k, v in profile.items() if k != "profile_sha256"}):
        raise ValueError(f"profile hash mismatch for {variant['name']}")
    return {**spec, "model": model, "tokenizer": tokenizer, "controller": controller,
            "calibrator": calibrator, "selected_h": selected_h, "profile": profile}


def _read_trace(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    path.unlink()
    return rows


def _single_generation(item: dict[str, Any], context: dict[str, Any], *, seed: int, trace_path: Path,
                       forced_schedule=None, backtracking=None) -> dict[str, Any]:
    import torch
    seed_everything(seed)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    prompt = _prompt(item, context["campaign"])
    prompt_ids = context["tokenizer"](prompt, return_tensors="pt")["input_ids"][0].tolist()
    variant = context["variant"]
    threshold = math.inf if context["selected_h"] is None else math.exp(context["selected_h"])
    started = time.perf_counter()
    result = generate_with_mgtb_v3(
        context["model"], context["tokenizer"], prompt, context["controller"], context["calibrator"], threshold,
        max_new_tokens=int(context["campaign"]["generation"]["max_new_tokens"]), trace_log_path=trace_path,
        do_backtracking=(variant["kind"] in {"controller", "matched_random"}) if backtracking is None else backtracking,
        detector_accumulation=variant.get("accumulation_mode", "cusum_reset"),
        forced_alert_schedule=forced_schedule,
    )
    wall = time.perf_counter() - started
    completion_ids = result.tokens[len(prompt_ids):]
    generation = context["tokenizer"].decode(completion_ids, skip_special_tokens=True)
    trace = _read_trace(trace_path)
    peak = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
    return {
        "generation": generation, "token_ids": completion_ids, "scorer": _score(generation, item, context["campaign"]),
        "sampled_tokens": result.sampled_tokens, "emitted_tokens": result.emitted_tokens,
        "deleted_tokens": result.deleted_tokens, "alerts": [vars(a) for a in result.alerts],
        "backtracks": result.backtracks, "monitor_trace": trace,
        "monitor_state": getattr(result, "retained_monitor_windows", []),
        "truncated": result.termination_reason == "max_new_tokens",
        "termination_reason": result.termination_reason,
        "timing": {"wall_seconds": wall, "peak_vram_bytes": peak},
    }


def _select_candidate(candidates: list[dict[str, Any]], selection: str) -> int:
    if selection == "majority_answer":
        answers = [candidate["scorer"].get("prediction_answer") for candidate in candidates]
        counts = Counter(answer for answer in answers if answer is not None)
        if counts:
            winner = sorted(counts, key=lambda answer: (-counts[answer], answers.index(answer)))[0]
            return answers.index(winner)
        return 0
    if selection == "mean_logprob":
        means = []
        for candidate in candidates:
            values = [event["logprob"] for event in candidate["monitor_trace"]
                      if event.get("type") == "token" and event.get("logprob") is not None]
            means.append(sum(values) / len(values) if values else float("-inf"))
        return max(range(len(means)), key=means.__getitem__)
    if selection == "first":
        return 0
    raise ValueError(f"unsupported candidate selection {selection!r}")


def _generate_item(item: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    variant = context["variant"]
    trace_base = Path(context["output_dir"]) / "in_progress" / RunStore.filename(item["item_id"])
    if variant["kind"] == "sample_aggregate":
        candidates = []
        for index in range(int(variant["num_samples"])):
            candidate_seed = item["item_seed"] if index == 0 else stable_int_seed(item["item_seed"], "candidate", index)
            candidates.append(_single_generation(
                item, context, seed=candidate_seed,
                trace_path=trace_base.with_suffix(f".candidate-{index}.trace.jsonl"), backtracking=False,
            ))
            candidates[-1]["candidate_seed"] = candidate_seed
        selected = _select_candidate(candidates, variant["selection"])
        result = dict(candidates[selected])
        result["candidate_selection"] = variant["selection"]
        result["selected_candidate"] = selected
        result["candidates"] = candidates
        result["sampled_tokens"] = sum(row["sampled_tokens"] for row in candidates)
        result["timing"] = {
            "wall_seconds": sum(row["timing"]["wall_seconds"] for row in candidates),
            "peak_vram_bytes": max((row["timing"]["peak_vram_bytes"] or 0) for row in candidates),
        }
    else:
        schedule = None
        if variant["kind"] == "matched_random":
            template = assigned_template(
                context["profile"], base_seed=item["item_seed"],
                precision=context["campaign"]["model"].get("precision", "unknown"), index=0,
            )
            schedule = random_schedule(
                template, seed=stable_int_seed(item["item_seed"], variant["name"], "schedule"),
                minimum_position=int(variant.get("minimum_position", 64)),
            )
        result = _single_generation(item, context, seed=item["item_seed"], trace_path=trace_base.with_suffix(".trace.jsonl"),
                                    forced_schedule=schedule, backtracking=variant["kind"] != "vanilla")
        if schedule is not None:
            result["trigger_schedule"] = schedule
    accounting = {
        "sampled": int(result["sampled_tokens"]), "emitted": int(result["emitted_tokens"]),
        "deleted": int(result["deleted_tokens"]), "alarms": len(result.get("alerts", [])),
        "rerolls": len(result.get("backtracks", [])),
        "alarm_positions": [int(row.get("token_pos", 0)) for row in result.get("alerts", [])],
        "rollback_spans": [int(row.get("rollback_span", 0)) for row in result.get("backtracks", [])],
        "termination_reason": result["termination_reason"],
    }
    result["token_accounting"] = accounting
    result["source_item_id"] = item["source_item_id"]
    result["replicate_seed"] = item["replicate_seed"]
    result["variant"] = variant["name"]
    result["campaign_id"] = context["campaign"]["campaign_id"]
    result["experimental_status"] = context["campaign"]["experimental_status"]
    result["item_metadata"] = {
        "dataset_name": item.get("dataset_name"), "dataset_kind": item.get("dataset_kind"),
        "split": item.get("split"), "subject": item.get("subject"), "level": item.get("level"),
    }
    result["provenance"] = {**context["source"], "run_identity_sha256": context["run_identity_sha256"],
                            "variant_sha256": variant["variant_sha256"],
                            "resolved_run_sha256": context["resolved_run_sha256"]}
    return result


def _configure_worker(spec):
    global _WORKER_SPEC, _WORKER_CONTEXT
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _WORKER_SPEC, _WORKER_CONTEXT = spec, None


def _worker(item):
    global _WORKER_CONTEXT
    if _WORKER_CONTEXT is None:
        _WORKER_CONTEXT = _build_context(_WORKER_SPEC)
    return item["item_id"], _generate_item(item, _WORKER_CONTEXT)


def _append_event(root: Path, payload: dict[str, Any]) -> None:
    path = root / "progress.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def run_variant(*, campaign: dict[str, Any], manifest: dict[str, Any], role: str, variant_name: str,
                freeze: dict[str, Any] | None, workers: int = 1, stop_after: int | None = None) -> list[dict[str, Any]]:
    variant = resolve_variant(campaign, variant_name)
    if role == "test" and freeze is None:
        raise ValueError("test run requires campaign freeze")
    if freeze is not None:
        assert_freeze(campaign, manifest, freeze, variant_name)
    seeds = [int(seed) for seed in campaign.get("seeds", [0])]
    units = campaign_units(manifest, role, seeds)
    root = output_root(campaign) / "runs" / role / variant_name
    source = {"git_commit": git_commit(), "source_tree_sha256": source_tree_sha256(),
              "software_environment": software_environment(), "command": " ".join(__import__("sys").argv)}
    identity = {
        "campaign_id": campaign["campaign_id"], "manifest_sha256": manifest["manifest_sha256"],
        "role": role, "variant": variant_name, "variant_sha256": variant["variant_sha256"],
        "freeze_sha256": freeze.get("freeze_sha256") if freeze else None,
        "campaign_config_sha256": sha256_json({k: v for k, v in campaign.items() if not k.startswith("_")}),
        "git_commit": source["git_commit"], "source_tree_sha256": source["source_tree_sha256"],
    }
    store = RunStore(root, identity)
    resolved_run = {
        "schema_version": 1, "identity": identity,
        "campaign": {k: v for k, v in campaign.items() if not k.startswith("_")},
        "variant": variant, "role": role, "source": {k: v for k, v in source.items() if k != "command"},
    }
    resolved_run["resolved_run_sha256"] = sha256_json(resolved_run)
    resolved_path = root / "resolved_run.json"
    if resolved_path.exists():
        existing_resolved = load_json(resolved_path)
        if existing_resolved.get("resolved_run_sha256") != resolved_run["resolved_run_sha256"]:
            raise ValueError("resolved run metadata changed")
    atomic_write_json(resolved_path, resolved_run)
    _append_event(root, {"timestamp": time.time(), "event": "run_invoked", "command": source["command"],
                         "run_identity_sha256": store.identity_sha256})
    spec = {"campaign": campaign, "variant": variant, "output_dir": str(root), "source": source,
            "run_identity_sha256": store.identity_sha256, "resolved_run_sha256": resolved_run["resolved_run_sha256"]}
    existing_rows = []
    pending = []
    for item in units:
        artifact = store.valid_artifact(item)
        if artifact is None:
            pending.append(item)
        else:
            existing_rows.append(artifact)
    if stop_after is not None:
        pending = pending[:max(0, int(stop_after))]
    if not pending:
        return existing_rows
    if workers <= 1:
        context = _build_context(spec)
        iterator = ((item["item_id"], _generate_item(item, context)) for item in pending)
        pool = None
    else:
        mp = multiprocessing.get_context("spawn")
        pool = mp.Pool(processes=min(int(workers), len(pending) or 1), initializer=_configure_worker, initargs=(spec,))
        iterator = pool.imap_unordered(_worker, pending, chunksize=1)
    by_id = {item["item_id"]: item for item in pending}
    progress_correct = sum(float(row.get("scorer", {}).get("correct", 0.0)) for row in existing_rows)
    progress_sampled = sum(int(row.get("token_accounting", {}).get("sampled", 0)) for row in existing_rows)
    progress_completed = len(existing_rows)
    try:
        for item_id, artifact in iterator:
            store.save(by_id[item_id], artifact)
            state = load_json(store.state_path)
            progress_completed += 1
            progress_correct += float(artifact["scorer"].get("correct", 0.0))
            progress_sampled += int(artifact["token_accounting"]["sampled"])
            event = {"timestamp": time.time(), "event": "item_completed", "item_id": item_id,
                     "completed": state["completed_count"], "target": len(units),
                     "correct": artifact["scorer"].get("correct"),
                     "sampled_tokens": artifact["token_accounting"]["sampled"],
                     "running_accuracy": progress_correct / progress_completed,
                     "running_mean_sampled_tokens": progress_sampled / progress_completed,
                     "run_identity_sha256": store.identity_sha256}
            _append_event(root, event)
            atomic_write_json(root / "progress.json", event)
            print(f"[campaign {variant_name}] {event['completed']}/{len(units)} {item_id}", flush=True)
    except BaseException:
        if pool is not None:
            pool.terminate(); pool.join()
        raise
    else:
        if pool is not None:
            pool.close(); pool.join()
    return [row for item in units if (row := store.valid_artifact(item)) is not None]


def collect_features(*, campaign: dict[str, Any], manifest: dict[str, Any], role: str, calibration_key: str,
                     workers: int = 1, stop_after: int | None = None) -> list[dict[str, Any]]:
    spec = calibration_spec(campaign, calibration_key)
    source_key = spec.get("feature_source", calibration_key)
    source_spec = calibration_spec(campaign, source_key)
    synthetic_name = f"features__{source_key}"
    local = dict(campaign)
    local["variants"] = dict(campaign["variants"])
    local["variants"][synthetic_name] = {
        "kind": "vanilla", "controller_overrides": source_spec["controller_overrides"],
        "accumulation_mode": source_spec["accumulation_mode"],
    }
    rows = run_variant(campaign=local, manifest=manifest, role=role, variant_name=synthetic_name,
                       freeze=None, workers=workers, stop_after=stop_after)
    return rows


def build_profile(artifacts: list[dict[str, Any]], source: dict[str, Any]) -> dict[str, Any]:
    templates = []
    for artifact in artifacts:
        spans = [int(row.get("rollback_span", 0)) for row in artifact.get("backtracks", []) if int(row.get("rollback_span", 0)) > 0]
        templates.append({"rollback_lengths": spans,
                          "reference_primary_tokens": int(artifact["token_accounting"]["emitted"]),
                          "extra_decode_tokens": int(artifact["token_accounting"]["deleted"])})
    if not templates:
        raise ValueError("profile requires development artifacts")
    profile = {"schema_version": 1, "templates": templates,
               "summary": {"num_examples": len(templates),
                           "activation_rate": sum(bool(row["rollback_lengths"]) for row in templates) / len(templates),
                           "mean_extra_decode_tokens": sum(row["extra_decode_tokens"] for row in templates) / len(templates)},
               "source": source}
    profile["profile_sha256"] = sha256_json(profile)
    return profile


def build_freeze(campaign: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    resolved = {name: resolve_variant(campaign, name) for name in campaign["variants"]}
    for name, variant in resolved.items():
        if variant["kind"] == "matched_random" and not Path(variant["profile"]).is_file():
            raise ValueError(f"matched-random profile missing for {name}: {variant['profile']}")
    variants = {name: variant["variant_sha256"] for name, variant in resolved.items()}
    calibrations = {}
    for key in campaign.get("calibrations", {}):
        root = output_root(campaign) / "calibration" / key
        calibrations[key] = {
            "calibrator_sha256": load_json(root / "calibrator.json")["calibrator_sha256"],
            "threshold_sha256": load_json(root / "threshold.json")["threshold_sha256"],
        }
    payload = {
        "schema_version": 1, "campaign_id": campaign["campaign_id"],
        "experimental_status": campaign["experimental_status"], "manifest_sha256": manifest["manifest_sha256"],
        "test_items": [{"item_id": row["item_id"], "content_sha256": row["content_sha256"]} for row in manifest["roles"]["test"]],
        "campaign_config_sha256": sha256_json({k: v for k, v in campaign.items() if not k.startswith("_")}),
        "variants": variants, "calibrations": calibrations,
        "source": {"git_commit": git_commit(), "source_tree_sha256": source_tree_sha256()},
        "software_environment": software_environment(),
    }
    payload["freeze_sha256"] = sha256_json(payload)
    return payload


def assert_freeze(campaign: dict[str, Any], manifest: dict[str, Any], freeze: dict[str, Any], variant: str) -> None:
    expected = sha256_json({key: value for key, value in freeze.items() if key != "freeze_sha256"})
    if freeze.get("freeze_sha256") != expected:
        raise ValueError("freeze hash mismatch")
    if freeze.get("manifest_sha256") != manifest["manifest_sha256"]:
        raise ValueError("freeze manifest mismatch")
    if freeze.get("variants", {}).get(variant) != resolve_variant(campaign, variant)["variant_sha256"]:
        raise ValueError("freeze variant mismatch")
    config_hash = sha256_json({k: v for k, v in campaign.items() if not k.startswith("_")})
    if freeze.get("campaign_config_sha256") != config_hash:
        raise ValueError("freeze campaign config mismatch")
    for key, expected_hashes in freeze.get("calibrations", {}).items():
        calibrator, threshold = _calibration_payload(campaign, key)
        if calibrator["calibrator_sha256"] != expected_hashes["calibrator_sha256"] or threshold["threshold_sha256"] != expected_hashes["threshold_sha256"]:
            raise ValueError(f"freeze calibration mismatch for {key}")
    if freeze.get("source", {}).get("git_commit") != git_commit() or freeze.get("source", {}).get("source_tree_sha256") != source_tree_sha256():
        raise ValueError("freeze source mismatch")
