from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from mgtb_v3.science_fast.io import atomic_write_json, load_json, sha256_json
from mgtb_v3.science_fast.protocol import content_sha256, normalize_problem, selection_key


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


def build_manifest(spec: dict[str, Any]) -> dict[str, Any]:
    seed = int(spec["protocol_seed"])
    roles: dict[str, list[dict[str, Any]]] = {}
    revisions = {}
    seen_content: set[str] = set()
    for role in ("reference", "development", "test"):
        source = spec["roles"][role]
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
            candidates.append({
                "role": role, "item_id": item_id, "source_id": source_id,
                "dataset_name": source.get("name", "jsonl"), "dataset_revision": source.get("revision"),
                "split": source.get("split", role), "dataset_kind": source.get("dataset_kind", "math500"),
                "problem": problem,
                "reference_answer": str(_value(row, fields.get("answer", "answer"))),
                "subject": _value(row, fields.get("subject"), None),
                "level": _value(row, fields.get("level"), None),
                "content_sha256": digest,
                "selection_key": hashlib.sha256(f"{seed}|{digest}".encode()).hexdigest(),
            })
        unique = {row["content_sha256"]: row for row in candidates}
        selected = sorted(unique.values(), key=lambda row: (row["selection_key"], row["item_id"]))
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
