#!/usr/bin/env python3
"""Import authenticated completed artifacts into a compatible frozen campaign."""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

from mgtb_v3.science_campaign.config import load_campaign, output_root, resolve_variant
from mgtb_v3.science_campaign.manifest import load_manifest
from mgtb_v3.science_campaign.runner import assert_freeze, campaign_units, run_variant
from mgtb_v3.science_fast.artifacts import RunStore
from mgtb_v3.science_fast.io import atomic_write_json, load_json, sha256_json


def _source_rows(source_root: Path, units: list[dict]) -> tuple[dict, dict, list[dict]]:
    state = load_json(source_root / "run_state.json")
    store = RunStore(source_root, state["identity"])
    rows = []
    for unit in units:
        row = store.valid_artifact(unit)
        if row is None:
            raise ValueError(f"source artifact is missing or invalid: {unit['item_id']}")
        rows.append(row)
    return state, load_json(source_root / "resolved_run.json"), rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Import a completed campaign variant without regenerating tokens.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-run", required=True, help="Source runs/<role>/<variant> directory.")
    parser.add_argument("--role", default="test")
    parser.add_argument("--variant", required=True)
    args = parser.parse_args()

    campaign = load_campaign(args.config)
    manifest = load_manifest(campaign["manifest"])
    freeze = load_json(output_root(campaign) / "freeze" / "campaign.lock.json")
    assert_freeze(campaign, manifest, freeze, args.variant)
    units = campaign_units(manifest, args.role, [int(seed) for seed in campaign.get("seeds", [0])])
    source_state, source_resolved, source_rows = _source_rows(Path(args.source_run), units)

    destination_variant = resolve_variant(campaign, args.variant)
    if source_resolved.get("role") != args.role:
        raise ValueError("source role does not match destination role")
    if source_resolved.get("variant") != destination_variant:
        raise ValueError("source variant/controller differs from destination")
    source_campaign = source_resolved.get("campaign", {})
    for key in ("manifest", "model", "generation", "seeds"):
        if source_campaign.get(key) != campaign.get(key):
            raise ValueError(f"source and destination differ for {key}")

    # Initialize the destination identity and resolved-run metadata through the
    # normal freeze-safe runner, while intentionally generating zero items.
    run_variant(campaign=campaign, manifest=manifest, role=args.role, variant_name=args.variant,
                freeze=freeze, workers=1, stop_after=0)
    destination_root = output_root(campaign) / "runs" / args.role / args.variant
    destination_state = load_json(destination_root / "run_state.json")
    destination_store = RunStore(destination_root, destination_state["identity"])
    destination_resolved = load_json(destination_root / "resolved_run.json")

    completed = []
    for unit, source in zip(units, source_rows):
        existing = destination_store.valid_artifact(unit)
        if existing is not None:
            completed.append(destination_store.artifact_path(unit["item_id"]).stem)
            continue
        artifact = deepcopy(source)
        artifact["campaign_id"] = campaign["campaign_id"]
        artifact["experimental_status"] = campaign["experimental_status"]
        artifact["variant"] = args.variant
        artifact["provenance"] = {
            "run_identity_sha256": destination_store.identity_sha256,
            "resolved_run_sha256": destination_resolved["resolved_run_sha256"],
            "imported_from": {
                "run_directory": str(Path(args.source_run).resolve()),
                "run_identity_sha256": source_state["identity_sha256"],
                "resolved_run_sha256": source_resolved["resolved_run_sha256"],
                "artifact_sha256": source["artifact_sha256"],
                "execution_provenance": source["provenance"],
            },
        }
        artifact["item_id"] = unit["item_id"]
        artifact["content_sha256"] = unit["content_sha256"]
        artifact["item_seed"] = int(unit["item_seed"])
        artifact["completed"] = True
        artifact.pop("artifact_sha256", None)
        artifact["artifact_sha256"] = sha256_json(artifact)
        atomic_write_json(destination_store.artifact_path(unit["item_id"]), artifact)
        completed.append(destination_store.artifact_path(unit["item_id"]).stem)
    destination_store._write_state(sorted(completed))

    valid = [destination_store.valid_artifact(unit) for unit in units]
    if any(row is None for row in valid):
        raise RuntimeError("imported destination artifacts failed validation")
    print({"imported": len(valid), "source": str(Path(args.source_run)), "destination": str(destination_root),
           "destination_identity_sha256": destination_store.identity_sha256})


if __name__ == "__main__":
    main()
