from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROFILE_SCHEMA_VERSION = 1
PER_ID_BUDGET_SCHEMA_VERSION = 1


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_int_seed(*parts: Any) -> int:
    payload = "::".join(str(part) for part in parts)
    return int.from_bytes(hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest(), "big") % (2**31)


def load_profile(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        profile = json.load(handle)
    if profile.get("schema_version") != PROFILE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported budget profile schema: {profile.get('schema_version')!r}")
    templates = profile.get("templates")
    if not isinstance(templates, list) or not templates:
        raise ValueError("Budget profile must contain at least one template.")
    return profile


def profile_sha256(path: str | Path) -> str:
    return file_sha256(path)


def load_per_id_budget(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        table = json.load(handle)
    if table.get("schema_version") != PER_ID_BUDGET_SCHEMA_VERSION:
        raise ValueError(f"Unsupported per-ID budget schema: {table.get('schema_version')!r}")
    budgets = table.get("budgets")
    if not isinstance(budgets, list) or not budgets:
        raise ValueError("Per-ID budget table must contain a non-empty budgets list.")
    return table


def per_id_budget_map(table: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["id"]): row for row in table["budgets"]}


def assigned_template(profile: dict[str, Any], *, base_seed: int, precision: str, index: int) -> dict[str, Any]:
    templates = profile["templates"]
    order = list(range(len(templates)))
    random.Random(stable_int_seed(base_seed, precision, "budget-profile")).shuffle(order)
    return dict(templates[order[index % len(order)]])


def random_schedule(template: dict[str, Any], *, seed: int, minimum_position: int = 64) -> list[dict[str, int]]:
    lengths = [max(1, int(value)) for value in template.get("rollback_lengths", [])]
    constraints = template.get("position_constraints") or {}
    constrained_minimum = max(minimum_position, int(constraints.get("minimum_position", minimum_position)))
    reference_length = max(
        constrained_minimum + 1,
        int(constraints.get("maximum_position", template.get("reference_primary_tokens", constrained_minimum + 1))),
    )
    rng = random.Random(seed)
    positions = sorted(rng.randint(constrained_minimum, reference_length) for _ in lengths)
    return [{"trigger_at": position, "rollback_tokens": length} for position, length in zip(positions, lengths)]


def periodic_schedule(template: dict[str, Any], *, minimum_position: int = 64) -> list[dict[str, int]]:
    lengths = [max(1, int(value)) for value in template.get("rollback_lengths", [])]
    reference_length = max(minimum_position + 1, int(template.get("reference_primary_tokens", minimum_position + 1)))
    count = len(lengths)
    return [
        {
            "trigger_at": max(minimum_position, int((position + 1) * reference_length / (count + 1))),
            "rollback_tokens": length,
        }
        for position, length in enumerate(lengths)
    ]


def restart_indices(*, num_items: int, profile: dict[str, Any], base_seed: int, precision: str) -> set[int]:
    target = float(profile["summary"]["mean_extra_decode_tokens"])
    vanilla_length = max(1.0, float(profile["summary"]["mean_vanilla_tokens_generated"]))
    count = min(num_items, max(0, round(num_items * target / vanilla_length)))
    ranked = sorted(range(num_items), key=lambda index: stable_int_seed(base_seed, precision, "restart", index))
    return set(ranked[:count])


def revision_token_budget(profile: dict[str, Any]) -> int:
    return max(1, round(float(profile["summary"]["mean_extra_decode_tokens"])))


@dataclass(frozen=True)
class ProfileCondition:
    base_model: str
    precision: str
    dataset: str
    threshold_path: str | None


def _read_rows(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_profile(manifest: dict[str, Any]) -> dict[str, Any]:
    pairs = manifest.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("Budget manifest must define a non-empty 'pairs' list.")

    templates: list[dict[str, Any]] = []
    sources: list[dict[str, str]] = []
    conditions: set[ProfileCondition] = set()
    all_extra: list[float] = []
    all_vanilla_tokens: list[float] = []
    all_total_events: list[float] = []
    all_backtracks = 0
    all_activated = 0

    for pair in pairs:
        vanilla_path = Path(pair["vanilla_results"])
        mgtb_path = Path(pair["mgtb_results"])
        sources.extend(
            [
                {"path": str(vanilla_path), "sha256": file_sha256(vanilla_path)},
                {"path": str(mgtb_path), "sha256": file_sha256(mgtb_path)},
            ]
        )
        vanilla_rows = {
            str(row["id"]): row
            for row in _read_rows(vanilla_path)
            if row.get("method") == "vanilla" and row.get("precision") == "int4"
        }
        mgtb_rows = {
            str(row["id"]): row
            for row in _read_rows(mgtb_path)
            if row.get("method") == "mgtb_v3_window" and row.get("precision") == "int4"
        }
        if not vanilla_rows or not mgtb_rows or vanilla_rows.keys() != mgtb_rows.keys():
            raise ValueError(f"Unpaired vanilla/MGT-B results in {vanilla_path} and {mgtb_path}.")

        for item_id in sorted(vanilla_rows):
            vanilla = vanilla_rows[item_id]
            mgtb = mgtb_rows[item_id]
            if vanilla.get("seed") != mgtb.get("seed"):
                raise ValueError(f"Seed mismatch for {item_id}.")
            condition = ProfileCondition(
                base_model=str(mgtb.get("base_model")),
                precision=str(mgtb.get("precision")),
                dataset=str(mgtb.get("dataset")),
                threshold_path=mgtb.get("threshold_path"),
            )
            conditions.add(condition)
            if str(vanilla.get("base_model")) != condition.base_model or str(vanilla.get("dataset")) != condition.dataset:
                raise ValueError(f"Condition mismatch for {item_id}.")
            backtracks = list(mgtb.get("backtracks") or [])
            rollback_lengths = []
            for event in backtracks:
                alert = event.get("alert") or {}
                rollback = event.get("rollback_pos", alert.get("rollback_token_pos"))
                token_pos = alert.get("token_pos")
                if rollback is not None and token_pos is not None:
                    rollback_lengths.append(max(1, int(token_pos) - int(rollback)))
            extra = float(mgtb.get("extra_sampled", 0.0))
            templates.append(
                {
                    "extra_decode_tokens": extra,
                    "backtrack_count": len(backtracks),
                    "rollback_lengths": rollback_lengths,
                    "reference_primary_tokens": int(vanilla.get("tokens_generated", 0)),
                }
            )
            all_extra.append(extra)
            all_vanilla_tokens.append(float(vanilla.get("tokens_generated", 0.0)))
            all_total_events.append(float(mgtb.get("token_events_trace", mgtb.get("tokens_generated", 0.0))))
            all_backtracks += len(backtracks)
            all_activated += int(bool(mgtb.get("alerts")))

    if len(conditions) != 1:
        raise ValueError(f"Budget sources do not share one condition: {conditions!r}")
    condition = next(iter(conditions))
    count = len(templates)
    return {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "condition": {
            "base_model": condition.base_model,
            "precision": condition.precision,
            "dataset": condition.dataset,
            "threshold_path": condition.threshold_path,
        },
        "sources": sources,
        "summary": {
            "num_examples": count,
            "mean_extra_decode_tokens": sum(all_extra) / count,
            "mean_vanilla_tokens_generated": sum(all_vanilla_tokens) / count,
            "mean_mgtb_decode_events": sum(all_total_events) / count,
            "activation_rate": all_activated / count,
            "mean_backtracks": all_backtracks / count,
        },
        "templates": templates,
    }


def build_per_id_budget(manifest: dict[str, Any]) -> dict[str, Any]:
    sources_config = manifest.get("sources")
    if not isinstance(sources_config, list) or not sources_config:
        raise ValueError("Per-ID budget manifest must define a non-empty sources list.")
    tolerance = float(manifest.get("tolerance", 0.10))
    if tolerance < 0:
        raise ValueError("Per-ID budget tolerance must be non-negative.")
    expected_num_items = manifest.get("expected_num_items")

    selected: dict[str, dict[str, Any]] = {}
    sources: list[dict[str, str]] = []
    conditions: set[ProfileCondition] = set()
    for source in sources_config:
        path = Path(source["results"])
        sources.append({"path": str(path), "sha256": file_sha256(path)})
        for row in _read_rows(path):
            if row.get("method") != "mgtb_v3_window" or row.get("precision") != "int4":
                continue
            item_id = str(row["id"])
            if item_id in selected:
                continue
            condition = ProfileCondition(
                base_model=str(row.get("base_model")),
                precision=str(row.get("precision")),
                dataset=str(row.get("dataset")),
                threshold_path=row.get("threshold_path"),
            )
            conditions.add(condition)
            decode_events = int(row.get("token_events_trace", row.get("decode_token_events_total", row.get("tokens_generated", 0))))
            if decode_events <= 0:
                raise ValueError(f"Missing positive decode-event count for {item_id} in {path}.")
            selected[item_id] = {
                "id": item_id,
                "mgtb_seed": int(row["seed"]),
                "mgtb_decode_events": decode_events,
                "control_max_decode_events": int(math.floor(decode_events * (1.0 + tolerance))),
                "source_path": str(path),
            }

    if len(conditions) != 1:
        raise ValueError(f"Per-ID budget sources do not share one condition: {conditions!r}")
    if expected_num_items is not None and len(selected) != int(expected_num_items):
        raise ValueError(f"Expected {int(expected_num_items)} unique budget IDs, found {len(selected)}.")
    condition = next(iter(conditions))
    budgets = [selected[item_id] for item_id in sorted(selected)]
    mgtb_total = sum(row["mgtb_decode_events"] for row in budgets)
    control_total = sum(row["control_max_decode_events"] for row in budgets)
    return {
        "schema_version": PER_ID_BUDGET_SCHEMA_VERSION,
        "condition": {
            "base_model": condition.base_model,
            "precision": condition.precision,
            "dataset": condition.dataset,
            "threshold_path": condition.threshold_path,
        },
        "tolerance": tolerance,
        "rounding": "floor",
        "sources": sources,
        "summary": {
            "num_examples": len(budgets),
            "mean_mgtb_decode_events": mgtb_total / len(budgets),
            "mean_control_max_decode_events": control_total / len(budgets),
            "total_mgtb_decode_events": mgtb_total,
            "total_control_max_decode_events": control_total,
        },
        "budgets": budgets,
    }
