from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

from mgtb_v3.eval.omni_math import parse_omni_judge_report
from mgtb_v3.science_fast.io import atomic_write_json, load_json, sha256_json
from mgtb_v3.science_fast.provenance import software_environment

from .config import output_root, role_seeds


def _filename(unit_id: str) -> str:
    return hashlib.sha256(unit_id.encode("utf-8")).hexdigest() + ".json"


def _valid(path: Path, unit: dict[str, Any], generation: dict[str, Any], identity_sha256: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        row = load_json(path)
    except (OSError, ValueError):
        return None
    expected = row.get("judgment_sha256")
    if expected != sha256_json({key: value for key, value in row.items() if key != "judgment_sha256"}):
        return None
    if row.get("item_id") != unit["item_id"] or row.get("content_sha256") != unit["content_sha256"]:
        return None
    if row.get("generation_artifact_sha256") != generation.get("artifact_sha256"):
        return None
    if row.get("judge_identity_sha256") != identity_sha256:
        return None
    return row


def _load_context(judge: dict[str, Any]):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("official Omni-Judge evaluation requires CUDA")
    revision = judge["revision"]
    tokenizer = AutoTokenizer.from_pretrained(
        judge["model"], revision=revision, trust_remote_code=True, use_fast=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        judge["model"], revision=revision, trust_remote_code=True,
        device_map=judge.get("device_map", "auto"), torch_dtype=torch.bfloat16,
    )
    tokenizer.padding_side = "left"
    terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    return model, tokenizer, terminators


def _predict_batch(context, judge: dict[str, Any], payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    import torch

    model, tokenizer, terminators = context
    prompts = [
        tokenizer.get_context(row["problem"], row["reference_answer"], row["generation"])
        for row in payloads
    ]
    encoded = tokenizer(prompts, padding=True, return_tensors="pt")
    input_ids = encoded["input_ids"].to(model.device)
    attention = encoded["attention_mask"].to(model.device)
    started = time.perf_counter()
    with torch.no_grad():
        predicted = model.generate(
            input_ids,
            attention_mask=attention,
            do_sample=False,
            num_return_sequences=1,
            max_new_tokens=int(judge.get("max_new_tokens", 300)),
        ).cpu().tolist()
    wall = time.perf_counter() - started
    results = []
    for prediction, padded_input in zip(predicted, input_ids.cpu().tolist()):
        if prediction[:len(padded_input)] != padded_input:
            raise ValueError("Omni-Judge generation did not preserve its input prefix")
        completion = prediction[len(padded_input):]
        for terminator in terminators:
            if terminator is not None and terminator in completion:
                completion = completion[:completion.index(terminator)]
        report = "## Student Final Answer\n" + tokenizer.decode(completion, skip_special_tokens=True).strip()
        results.append({"raw_report": report, "sampled_tokens": len(completion), "wall_seconds": wall / len(payloads)})
    return results


def run_judging(
    *, campaign: dict[str, Any], manifest: dict[str, Any], role: str, variant: str,
    generation_rows: list[dict[str, Any]], freeze: dict[str, Any], stop_after: int | None = None,
    predictor=None,
) -> list[dict[str, Any]]:
    judge = campaign["evaluation"]["judge"]
    # campaign_units owns the exact per-seed IDs; import lazily to avoid a module cycle.
    from .runner import campaign_units
    units = campaign_units(manifest, role, role_seeds(campaign, role))
    generations = {row["item_id"]: row for row in generation_rows}
    if set(generations) != {unit["item_id"] for unit in units}:
        raise ValueError("judge input is not exactly paired with manifest problem/seed units")

    identity = {
        "campaign_id": campaign["campaign_id"], "manifest_sha256": manifest["manifest_sha256"],
        "freeze_sha256": freeze["freeze_sha256"], "role": role, "variant": variant,
        "judge": judge,
    }
    identity_sha256 = sha256_json(identity)
    root = output_root(campaign) / "judging" / role / variant
    items_dir = root / "items"
    items_dir.mkdir(parents=True, exist_ok=True)
    state_path = root / "judge_state.json"
    if state_path.exists():
        state = load_json(state_path)
        if state.get("identity") != identity or state.get("identity_sha256") != identity_sha256:
            raise ValueError("refusing Omni-Judge resume: frozen judge identity changed")

    completed = []
    pending = []
    for unit in units:
        generation = generations[unit["item_id"]]
        row = _valid(items_dir / _filename(unit["item_id"]), unit, generation, identity_sha256)
        if row is None:
            pending.append((unit, generation))
        else:
            completed.append(row)
    if stop_after is not None:
        pending = pending[:max(0, int(stop_after))]
    context = None
    batch_size = max(1, int(judge.get("batch_size", 4)))
    for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]
        payloads = [{
            "problem": unit["problem"], "reference_answer": unit["reference_answer"],
            "generation": generation["generation"],
        } for unit, generation in batch]
        if predictor is None:
            if context is None:
                context = _load_context(judge)
            predictions = _predict_batch(context, judge, payloads)
        else:
            predictions = predictor(payloads)
        for (unit, generation), prediction in zip(batch, predictions):
            parsed = parse_omni_judge_report(prediction["raw_report"])
            scorer = {
                **generation["scorer"], **parsed,
                "answer_extraction_ok": generation["scorer"].get("answer_extraction_ok", False),
            }
            row = {
                "schema_version": 1,
                "item_id": unit["item_id"], "source_item_id": unit["source_item_id"],
                "replicate_seed": unit["replicate_seed"], "content_sha256": unit["content_sha256"],
                "generation_artifact_sha256": generation["artifact_sha256"],
                "judge_identity_sha256": identity_sha256,
                "raw_report": prediction["raw_report"], "scorer": scorer,
                "judge_cost": {
                    "sampled_tokens": int(prediction.get("sampled_tokens", 0)),
                    "wall_seconds": float(prediction.get("wall_seconds", 0.0)),
                    "api_cost_usd": 0.0,
                    "cost_note": "local pinned Omni-Judge inference; hardware cost is not monetized",
                },
                "environment": software_environment(),
            }
            row["judgment_sha256"] = sha256_json(row)
            atomic_write_json(items_dir / _filename(unit["item_id"]), row)
            completed.append(row)
        state = {
            "schema_version": 1, "identity": identity, "identity_sha256": identity_sha256,
            "completed_count": len(completed), "target_count": len(units),
        }
        atomic_write_json(state_path, state)
    if not state_path.exists():
        atomic_write_json(state_path, {
            "schema_version": 1, "identity": identity, "identity_sha256": identity_sha256,
            "completed_count": len(completed), "target_count": len(units),
        })
    return [
        row for unit in units
        if (row := _valid(items_dir / _filename(unit["item_id"]), unit, generations[unit["item_id"]], identity_sha256)) is not None
    ]


def load_judgments(
    *, campaign: dict[str, Any], manifest: dict[str, Any], role: str, variant: str,
    generation_rows: list[dict[str, Any]], freeze: dict[str, Any],
) -> list[dict[str, Any]]:
    """Read only fully authenticated judge artifacts; never creates or repairs state."""
    from .runner import campaign_units

    units = campaign_units(manifest, role, role_seeds(campaign, role))
    generations = {row["item_id"]: row for row in generation_rows}
    if set(generations) != {unit["item_id"] for unit in units}:
        raise ValueError("judge input is not exactly paired with manifest problem/seed units")
    identity = {
        "campaign_id": campaign["campaign_id"], "manifest_sha256": manifest["manifest_sha256"],
        "freeze_sha256": freeze["freeze_sha256"], "role": role, "variant": variant,
        "judge": campaign["evaluation"]["judge"],
    }
    identity_sha256 = sha256_json(identity)
    root = output_root(campaign) / "judging" / role / variant
    state_path = root / "judge_state.json"
    if not state_path.is_file():
        raise ValueError(f"missing Omni-Judge state for {role}/{variant}")
    state = load_json(state_path)
    if state.get("identity") != identity or state.get("identity_sha256") != identity_sha256:
        raise ValueError("Omni-Judge state identity changed")
    rows = []
    for unit in units:
        row = _valid(
            root / "items" / _filename(unit["item_id"]), unit, generations[unit["item_id"]], identity_sha256,
        )
        if row is None:
            raise ValueError(f"missing or invalid Omni-Judge artifact: {unit['item_id']}")
        rows.append(row)
    if int(state.get("completed_count", -1)) != len(rows) or int(state.get("target_count", -1)) != len(units):
        raise ValueError("Omni-Judge state is incomplete")
    return rows


def merge_judgments(generation_rows: list[dict[str, Any]], judgments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {row["item_id"]: row for row in judgments}
    if set(by_id) != {row["item_id"] for row in generation_rows}:
        raise ValueError("missing or extra authenticated Omni-Judge verdicts")
    merged = []
    for generation in generation_rows:
        judgment = by_id[generation["item_id"]]
        if judgment["generation_artifact_sha256"] != generation.get("artifact_sha256"):
            raise ValueError("Omni-Judge verdict refers to a different raw generation artifact")
        if not judgment.get("scorer", {}).get("scorable"):
            raise ValueError(f"unscorable Omni-Judge verdict: {generation['item_id']}")
        merged.append({**generation, "scorer": judgment["scorer"], "judge": judgment})
    return merged
