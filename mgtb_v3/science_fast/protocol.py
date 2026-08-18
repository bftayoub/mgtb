from __future__ import annotations

import hashlib
import unicodedata
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .io import atomic_write_json, load_json, sha256_json

PROTOCOL_SEED = 20260811
MATH_TRAIN_NAME = "EleutherAI/hendrycks_math"
MATH_TRAIN_REVISION = "21a5633873b6a120296cce3e2df9d5550074f4a3"
MATH500_NAME = "HuggingFaceH4/MATH-500"
MATH500_REVISION = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"


def normalize_problem(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text)).replace("\r\n", "\n").replace("\r", "\n")
    return " ".join(text.split())


def content_sha256(text: str) -> str:
    return hashlib.sha256(normalize_problem(text).encode("utf-8")).hexdigest()


def selection_key(content_hash: str, protocol_seed: int = PROTOCOL_SEED) -> str:
    return hashlib.sha256(f"{int(protocol_seed)}|{content_hash}".encode()).hexdigest()


def item_seed(protocol_seed: int, stable_item_id: str) -> int:
    raw = hashlib.sha256(f"{int(protocol_seed)}|{stable_item_id}".encode()).digest()
    return int.from_bytes(raw[:8], "big") & ((1 << 63) - 1)


def _scientific_row(row: dict[str, Any], *, role: str, source: str, revision: str) -> dict[str, Any]:
    problem = str(row.get("problem", row.get("question", "")))
    digest = content_sha256(problem)
    source_id = str(row.get("id", row.get("unique_id", row.get("source_id", ""))))
    stable_id = f"{source}:{source_id}:{digest[:16]}"
    return {
        "role": role,
        "item_id": stable_id,
        "source_id": source_id,
        "dataset_name": source,
        "dataset_revision": revision,
        "split": str(row.get("split", "train" if role != "test" else "test")),
        "problem": problem,
        "reference_answer": str(row.get("answer", row.get("solution", row.get("reference_answer", "")))),
        "subject": row.get("subject", row.get("type")),
        "level": row.get("level"),
        "content_sha256": digest,
        "selection_key": selection_key(digest),
        "item_seed": item_seed(PROTOCOL_SEED, stable_id),
    }


def _unique_sorted(rows: Iterable[dict[str, Any]], *, role: str, source: str, revision: str) -> list[dict[str, Any]]:
    by_content: dict[str, dict[str, Any]] = {}
    for row in rows:
        item = _scientific_row(row, role=role, source=source, revision=revision)
        by_content.setdefault(item["content_sha256"], item)
    return sorted(by_content.values(), key=lambda item: (item["selection_key"], item["item_id"]))


def assert_disjoint_roles(manifest: dict[str, Any]) -> None:
    roles = manifest["roles"]
    hashes = {role: {item["content_sha256"] for item in items} for role, items in roles.items()}
    ids = {role: {item["item_id"] for item in items} for role, items in roles.items()}
    for left, right in (("reference", "development"), ("reference", "test"), ("development", "test")):
        overlap_hashes = hashes[left] & hashes[right]
        overlap_ids = ids[left] & ids[right]
        if overlap_hashes or overlap_ids:
            raise ValueError(f"split leakage {left}/{right}: content={sorted(overlap_hashes)} ids={sorted(overlap_ids)}")


def build_manifest(train_rows: Iterable[dict[str, Any]], test_rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    train = _unique_sorted(train_rows, role="reference", source=MATH_TRAIN_NAME, revision=MATH_TRAIN_REVISION)
    test = _unique_sorted(test_rows, role="test", source=MATH500_NAME, revision=MATH500_REVISION)
    if len(train) < 400:
        raise ValueError(f"need 400 unique MATH train contents, got {len(train)}")
    if len(test) < 300:
        raise ValueError(f"need 300 unique MATH-500 contents, got {len(test)}")
    reference = [{**item, "role": "reference"} for item in train[:300]]
    development = [{**item, "role": "development"} for item in train[300:400]]
    manifest = {
        "schema_version": 1,
        "protocol_seed": PROTOCOL_SEED,
        "seed_strategy": "sha256(protocol_seed|stable_item_id), first 63 bits",
        "selection_strategy": "unique normalized content sorted by sha256(protocol_seed|content_sha256)",
        "dataset_revisions": {"math_train": MATH_TRAIN_REVISION, "math500_test": MATH500_REVISION},
        "roles": {"reference": reference, "development": development, "test": [{**item, "role": "test"} for item in test[:300]]},
    }
    assert_disjoint_roles(manifest)
    manifest["counts"] = {role: len(items) for role, items in manifest["roles"].items()}
    manifest["manifest_sha256"] = sha256_json(manifest)
    return manifest


def validate_manifest(manifest: dict[str, Any]) -> None:
    supplied = manifest.get("manifest_sha256")
    payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if supplied != sha256_json(payload):
        raise ValueError("manifest hash mismatch")
    if manifest.get("counts") != {role: len(items) for role, items in manifest["roles"].items()}:
        raise ValueError("manifest counts mismatch")
    if manifest.get("dataset_revisions") != {"math_train": MATH_TRAIN_REVISION, "math500_test": MATH500_REVISION}:
        raise ValueError("dataset revision mismatch")
    assert_disjoint_roles(manifest)


def save_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    validate_manifest(manifest)
    atomic_write_json(path, manifest)


def load_manifest(path: str | Path) -> dict[str, Any]:
    manifest = load_json(path)
    validate_manifest(manifest)
    return manifest


def load_pinned_datasets() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    try:
        from datasets import get_dataset_config_names, load_dataset
    except ImportError as exc:
        raise RuntimeError('Install dataset support with: pip install -e ".[eval]"') from exc
    train_rows: list[dict[str, Any]] = []
    configs = get_dataset_config_names(MATH_TRAIN_NAME, revision=MATH_TRAIN_REVISION)
    for config_name in sorted(configs):
        split = load_dataset(MATH_TRAIN_NAME, config_name, split="train", revision=MATH_TRAIN_REVISION)
        for index, row in enumerate(split):
            train_rows.append({**dict(row), "id": f"{config_name}:{index}", "subject": config_name, "split": "train"})
    test_split = load_dataset(MATH500_NAME, "default", split="test", revision=MATH500_REVISION)
    test_rows = [{**dict(row), "id": row.get("unique_id", f"test:{index}"), "split": "test"} for index, row in enumerate(test_split)]
    return train_rows, test_rows
