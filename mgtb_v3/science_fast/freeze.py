from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import atomic_write_json, load_json, sha256_file, sha256_json
from .protocol import validate_manifest


def build_freeze(
    *, manifest: dict[str, Any], resolved_config: dict[str, Any], calibrator: dict[str, Any],
    threshold: dict[str, Any], method: str, source: dict[str, Any], environment: dict[str, Any], scorer_path: str | Path,
) -> dict[str, Any]:
    validate_manifest(manifest)
    if method not in {"vanilla", "mgtb"}:
        raise ValueError("freeze method must be vanilla or mgtb")
    payload = {
        "schema_version": 1,
        "method": method,
        "manifest_sha256": manifest["manifest_sha256"],
        "test_items": [{"item_id": i["item_id"], "content_sha256": i["content_sha256"]} for i in manifest["roles"]["test"]],
        "model": resolved_config["model"],
        "quantization": resolved_config["quantization"],
        "device_map": resolved_config["device_map"],
        "dataset_revisions": manifest["dataset_revisions"],
        "protocol_seed": manifest["protocol_seed"],
        "seed_strategy": manifest["seed_strategy"],
        "calibrator_sha256": calibrator["calibrator_sha256"],
        "selected_h": threshold["selected_h"],
        "threshold_sha256": threshold["threshold_sha256"],
        "resolved_controller_config": resolved_config["controller"],
        "scorer": {"name": "math500_exact_normalized", "source_sha256": sha256_file(scorer_path)},
        "max_new_tokens": resolved_config["max_new_tokens"],
        "source": source,
        "software_environment": environment,
    }
    payload["freeze_sha256"] = sha256_json(payload)
    return payload


def validate_freeze(freeze: dict[str, Any], *, manifest: dict[str, Any], method: str) -> None:
    expected = sha256_json({k: v for k, v in freeze.items() if k != "freeze_sha256"})
    if freeze.get("freeze_sha256") != expected:
        raise ValueError("freeze hash mismatch")
    if freeze.get("manifest_sha256") != manifest.get("manifest_sha256") or freeze.get("method") != method:
        raise ValueError("freeze does not match manifest/method")
    expected_test = [{"item_id": i["item_id"], "content_sha256": i["content_sha256"]} for i in manifest["roles"]["test"]]
    if freeze.get("test_items") != expected_test:
        raise ValueError("freeze test items mismatch")


def save_freeze(path: str | Path, freeze: dict[str, Any]) -> None:
    atomic_write_json(path, freeze)


def load_freeze(path: str | Path, *, manifest: dict[str, Any], method: str) -> dict[str, Any]:
    freeze = load_json(path)
    validate_freeze(freeze, manifest=manifest, method=method)
    return freeze
