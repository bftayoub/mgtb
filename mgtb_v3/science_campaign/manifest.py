from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from mgtb_v3.science_fast.io import atomic_write_json, load_json, sha256_json
from mgtb_v3.science_fast.protocol import content_sha256, item_seed, normalize_problem, selection_key


def _load_source(spec: dict[str, Any]) -> list[dict[str, Any]]:
    if spec.get("jsonl"):
        import json
        rows = []
        with Path(spec["jsonl"]).open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
        return rows
    revision = spec.get("revision")
    if not revision or str(revision).startswith("REPLACE_"):
        raise ValueError(f"dataset {spec.get('name')} requires an immutable revision")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError('manifest construction requires pip install -e ".[eval]"') from exc
    args = [spec["name"]]
    if spec.get("config"):
        args.append(spec["config"])
    dataset = load_dataset(*args, split=spec["split"], revision=revision)
    return [dict(row) for row in dataset]


def _value(row: dict[str, Any], key: str | None, fallback: str = "") -> Any:
    return row.get(key, fallback) if key else fallback


def _source_candidates(role: str, source: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    rows = _load_source(source)
    fields = source.get("fields", {})
    candidates = []
    for index, row in enumerate(rows):
        problem = str(_value(row, fields.get("problem", "problem")))
        if not problem:
            continue
        digest = content_sha256(problem)
        source_id = str(_value(row, fields.get("id"), f"{role}:{index}"))
        item_id = f"{source.get('name', source.get('jsonl', 'jsonl'))}:{source_id}:{digest[:16]}"
        candidate = {
            "role": role, "item_id": item_id, "source_id": source_id,
            "dataset_name": source.get("name", "jsonl"), "dataset_revision": source.get("revision"),
            "split": source.get("split", role),
            "problem": problem,
            "reference_answer": str(_value(row, fields.get("answer", "answer"))),
            "subject": _value(row, fields.get("subject"), None),
            "level": _value(row, fields.get("level"), None),
            "content_sha256": digest,
            "selection_key": selection_key(digest, seed),
            "item_seed": item_seed(seed, item_id),
        }
        if source.get("dataset_kind"):
            candidate["dataset_kind"] = source["dataset_kind"]
        candidates.append(candidate)
    unique = {row["content_sha256"]: row for row in candidates}
    return sorted(unique.values(), key=lambda row: (row["selection_key"], row["item_id"]))


def build_manifest(spec: dict[str, Any]) -> dict[str, Any]:
    seed = int(spec["protocol_seed"])
    roles: dict[str, list[dict[str, Any]]] = {}
    revisions = {}
    seen_content: set[str] = set()
    for role in ("reference", "development", "test"):
        source = spec["roles"][role]
        selected = _source_candidates(role, source, seed)
        count = int(source["count"])
        selected = [row for row in selected if row["content_sha256"] not in seen_content][:count]
        if len(selected) != count:
            raise ValueError(f"role {role} has only {len(selected)}/{count} unique non-overlapping items")
        seen_content.update(row["content_sha256"] for row in selected)
        roles[role] = selected
        revisions[role] = {"name": source.get("name", "jsonl"), "revision": source.get("revision"), "split": source.get("split")}
    manifest = {
        "schema_version": 2, "protocol_seed": seed,
        "selection_strategy": "unique content sorted by sha256(protocol_seed|content_sha256)",
        "dataset_revisions": revisions, "roles": roles,
        "counts": {role: len(items) for role, items in roles.items()},
    }
    manifest["manifest_sha256"] = sha256_json(manifest)
    validate_manifest(manifest)
    return manifest


def derive_manifest(base: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    """Expand one role while proving that the existing selection is unchanged."""
    validate_manifest(base)
    role = str(spec["role"])
    if role not in base["roles"]:
        raise ValueError(f"cannot derive unknown manifest role {role!r}")
    source = spec["source"]
    count = int(source["count"])
    existing = base["roles"][role]
    if count < len(existing):
        raise ValueError(f"derived {role} count cannot shrink {len(existing)} to {count}")

    excluded = {
        item["content_sha256"]
        for other_role, items in base["roles"].items()
        if other_role != role
        for item in items
    }
    selected = [
        row for row in _source_candidates(role, source, int(base["protocol_seed"]))
        if row["content_sha256"] not in excluded
    ][:count]
    if len(selected) != count:
        raise ValueError(f"role {role} has only {len(selected)}/{count} unique non-overlapping items")
    if selected[:len(existing)] != existing:
        raise ValueError(f"derived {role} does not preserve the existing deterministic selection")

    manifest = deepcopy(base)
    manifest["roles"][role] = selected
    manifest["counts"] = {name: len(items) for name, items in manifest["roles"].items()}
    manifest.pop("manifest_sha256", None)
    manifest["manifest_sha256"] = sha256_json(manifest)
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: dict[str, Any]) -> None:
    supplied = manifest.get("manifest_sha256")
    actual = sha256_json({key: value for key, value in manifest.items() if key != "manifest_sha256"})
    if supplied != actual:
        raise ValueError("manifest hash mismatch")
    roles = manifest.get("roles", {})
    if set(roles) != {"reference", "development", "test"}:
        raise ValueError("manifest requires reference, development and test roles")
    if manifest.get("counts") != {role: len(items) for role, items in roles.items()}:
        raise ValueError("manifest counts mismatch")
    required = {"item_id", "content_sha256", "problem", "reference_answer"}
    for role, items in roles.items():
        if any(not required <= item.keys() for item in items):
            raise ValueError(f"manifest role {role} has incomplete items")
        if len({item["item_id"] for item in items}) != len(items):
            raise ValueError(f"manifest role {role} has duplicate item IDs")
        if len({item["content_sha256"] for item in items}) != len(items):
            raise ValueError(f"manifest role {role} has duplicate contents")
    hashes = {role: {item["content_sha256"] for item in items} for role, items in roles.items()}
    for left, right in (("reference", "development"), ("reference", "test"), ("development", "test")):
        overlap = hashes[left] & hashes[right]
        if overlap:
            raise ValueError(f"manifest leakage {left}/{right}: {sorted(overlap)[:3]}")


def assert_independent_test(manifest: dict[str, Any], excluded_paths: list[str | Path]) -> None:
    current = {item["content_sha256"] for item in manifest["roles"]["test"]}
    for path in excluded_paths:
        old = load_json(path)
        old_hashes = {item["content_sha256"] for items in old.get("roles", {}).values() for item in items}
        overlap = current & old_hashes
        if overlap:
            raise ValueError(f"confirmatory test reuses {len(overlap)} contents from {path}")


def save_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    validate_manifest(manifest)
    atomic_write_json(path, manifest)


def load_manifest(path: str | Path) -> dict[str, Any]:
    manifest = load_json(path)
    validate_manifest(manifest)
    return manifest
