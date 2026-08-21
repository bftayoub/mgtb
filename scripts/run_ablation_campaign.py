#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mgtb_v3.science_campaign.analysis import campaign_analysis
from mgtb_v3.science_campaign.calibration import build_calibrator, select_threshold
from mgtb_v3.science_campaign.config import calibration_spec, load_campaign, manifest_path, output_root
from mgtb_v3.science_campaign.manifest import (
    assert_independent_test, build_manifest, derive_manifest, load_manifest, save_manifest,
)
from mgtb_v3.science_campaign.runner import (
    build_freeze, build_profile, campaign_units, collect_features, run_variant,
)
from mgtb_v3.science_fast.artifacts import RunStore
from mgtb_v3.science_fast.io import atomic_write_json, load_json
from mgtb_v3.science_fast.provenance import git_commit, software_environment, source_tree_sha256


def _valid_rows(root: Path, manifest: dict, role: str, seeds: list[int]) -> list[dict]:
    state = load_json(root / "run_state.json")
    store = RunStore(root, state["identity"])
    return [row for item in campaign_units(manifest, role, seeds) if (row := store.valid_artifact(item)) is not None]


def _valid_science_fast_rows(root: Path, manifest: dict, role: str) -> list[dict]:
    """Load authenticated single-seed artifacts produced by science_fast."""
    state = load_json(root / "run_state.json")
    store = RunStore(root, state["identity"])
    return [row for item in manifest["roles"][role] if (row := store.valid_artifact(item)) is not None]


def _write_calibration(
    destination: Path, calibrator: dict, summary: dict, threshold: dict, *, replace: bool = False
) -> None:
    payloads = {
        "calibrator.json": calibrator,
        "reference_summary.json": summary,
        "threshold.json": threshold,
    }
    for filename, payload in payloads.items():
        path = destination / filename
        if path.exists() and load_json(path) != payload and not replace:
            raise ValueError(f"refusing to replace a different calibration artifact: {path}")
        atomic_write_json(path, payload)


def _source():
    return {"git_commit": git_commit(), "source_tree_sha256": source_tree_sha256(),
            "software_environment": software_environment(), "command": " ".join(sys.argv)}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Resumable, freeze-safe scientific ablation campaigns")
    parser.add_argument("--config", required=True)
    parser.add_argument("--action", required=True, choices=[
        "build-manifest", "validate", "reuse-science-fast", "collect", "calibrate", "run",
        "build-profile", "freeze", "analyze", "status"
    ])
    parser.add_argument("--role", choices=["reference", "development", "test"])
    parser.add_argument("--calibration")
    parser.add_argument("--variant")
    parser.add_argument("--source-variant")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--stop-after", type=int)
    args = parser.parse_args(argv)

    campaign = load_campaign(args.config)
    root = output_root(campaign)
    if args.action == "build-manifest":
        if campaign.get("manifest_build"):
            manifest = build_manifest(campaign["manifest_build"])
        elif campaign.get("manifest_derive"):
            derive = campaign["manifest_derive"]
            manifest = derive_manifest(load_manifest(derive["base_manifest"]), derive)
        else:
            raise ValueError("campaign has no manifest_build or manifest_derive section")
        destination = manifest_path(campaign)
        save_manifest(destination, manifest)
        print(json.dumps({"manifest": str(destination), "counts": manifest["counts"],
                          "manifest_sha256": manifest["manifest_sha256"]}, indent=2))
        return

    manifest = load_manifest(manifest_path(campaign))
    excluded = [Path(path) for path in campaign.get("exclude_manifests", [])]
    if (campaign["experimental_status"] == "confirmatory"
            and campaign.get("confirmatory_design", "independent_test") == "independent_test"):
        assert_independent_test(manifest, excluded)

    if args.action == "validate":
        print(json.dumps({"campaign": campaign["campaign_id"], "status": campaign["experimental_status"],
                          "manifest_sha256": manifest["manifest_sha256"], "variants": sorted(campaign["variants"]),
                          "calibrations": sorted(campaign.get("calibrations", {}))}, indent=2))
        return
    workers = args.workers or int(campaign.get("parallel_workers", 1))
    seeds = [int(seed) for seed in campaign.get("seeds", [0])]

    if args.action == "reuse-science-fast":
        reuse = campaign.get("science_fast_reuse")
        if not reuse:
            raise ValueError("campaign has no science_fast_reuse section")
        if seeds != [0]:
            raise ValueError("science_fast reuse is restricted to the single replicate seed [0]")
        reference_root = Path(reuse["reference_run"])
        development_root = Path(reuse["development_run"])
        reference = _valid_science_fast_rows(reference_root, manifest, "reference")
        development = _valid_science_fast_rows(development_root, manifest, "development")
        expected_ref = len(manifest["roles"]["reference"])
        expected_dev = len(manifest["roles"]["development"])
        if len(reference) != expected_ref or len(development) != expected_dev:
            raise ValueError(
                f"science_fast artifacts incomplete: reference={len(reference)}/{expected_ref} "
                f"development={len(development)}/{expected_dev}"
            )
        requested = [args.calibration] if args.calibration else list(reuse.get("calibrations", ["full"]))
        results = {}
        source = {
            "mode": "authenticated_science_fast_reuse",
            "reference_run": str(reference_root.resolve()),
            "reference_identity_sha256": load_json(reference_root / "run_state.json")["identity_sha256"],
            "development_run": str(development_root.resolve()),
            "development_identity_sha256": load_json(development_root / "run_state.json")["identity_sha256"],
        }
        for key in requested:
            spec = calibration_spec(campaign, key)
            source_key = spec.get("feature_source", key)
            source_spec = calibration_spec(campaign, source_key)
            if source_key != reuse.get("feature_source", "full"):
                raise ValueError(f"calibration {key} requires unavailable feature source {source_key}")
            if spec["controller"]["window"] != source_spec["controller"]["window"]:
                raise ValueError(f"calibration {key} has incompatible window geometry")
            calibrator, summary = build_calibrator(reference, spec, source)
            threshold = select_threshold(development, calibrator, spec)
            if key == reuse.get("verify_calibration", "full") and reuse.get("calibration_root"):
                previous = load_json(Path(reuse["calibration_root"]) / "threshold.json")
                if abs(float(previous["selected_h"]) - float(threshold["selected_h"])) > 1e-12:
                    raise ValueError(
                        f"reused full calibration changed selected_h: "
                        f"{previous['selected_h']} != {threshold['selected_h']}"
                    )
            _write_calibration(root / "calibration" / key, calibrator, summary, threshold, replace=True)
            target = float(spec["controller"]["detector"]["target_false_alert_rate"])
            results[key] = {
                "selected_h": threshold["selected_h"],
                "healthy_alarm_rate": threshold["healthy_alarm_rate"],
                "target_false_alert_rate": target,
                "target_met": float(threshold["healthy_alarm_rate"]) <= target,
                "calibrator_sha256": calibrator["calibrator_sha256"],
                "threshold_sha256": threshold["threshold_sha256"],
            }
        print(json.dumps({"reused_without_generation": results}, indent=2))
        return

    if args.action == "collect":
        if args.role not in {"reference", "development"} or not args.calibration:
            raise ValueError("collect requires --role reference|development and --calibration")
        rows = collect_features(campaign=campaign, manifest=manifest, role=args.role,
                                calibration_key=args.calibration, workers=workers, stop_after=args.stop_after)
        print(json.dumps({"completed": len(rows), "target": len(manifest["roles"][args.role]) * len(seeds)}, indent=2))
        return
    if args.action == "calibrate":
        if not args.calibration:
            raise ValueError("calibrate requires --calibration")
        spec = calibration_spec(campaign, args.calibration)
        source_spec = calibration_spec(campaign, spec.get("feature_source", args.calibration))
        if spec["controller"]["window"] != source_spec["controller"]["window"]:
            raise ValueError(f"calibration {args.calibration} cannot reuse features with a different window geometry")
        synthetic = f"features__{spec.get('feature_source', args.calibration)}"
        reference = _valid_rows(root / "runs" / "reference" / synthetic, manifest, "reference", seeds)
        development = _valid_rows(root / "runs" / "development" / synthetic, manifest, "development", seeds)
        expected_ref = len(manifest["roles"]["reference"]) * len(seeds)
        expected_dev = len(manifest["roles"]["development"]) * len(seeds)
        if len(reference) != expected_ref or len(development) != expected_dev:
            raise ValueError(f"feature runs incomplete: reference={len(reference)}/{expected_ref} development={len(development)}/{expected_dev}")
        calibrator, summary = build_calibrator(reference, spec, _source())
        threshold = select_threshold(development, calibrator, spec)
        destination = root / "calibration" / args.calibration
        _write_calibration(destination, calibrator, summary, threshold)
        print(json.dumps({"calibration": args.calibration, **summary,
                          "selected_h": threshold["selected_h"], "healthy_alarm_rate": threshold["healthy_alarm_rate"]}, indent=2))
        return
    if args.action == "run":
        if not args.variant or not args.role:
            raise ValueError("run requires --variant and --role")
        freeze_path = root / "freeze" / "campaign.lock.json"
        freeze = load_json(freeze_path) if args.role == "test" else None
        rows = run_variant(campaign=campaign, manifest=manifest, role=args.role, variant_name=args.variant,
                           freeze=freeze, workers=workers, stop_after=args.stop_after)
        print(json.dumps({"variant": args.variant, "role": args.role, "completed": len(rows),
                          "target": len(manifest["roles"][args.role]) * len(seeds)}, indent=2))
        return
    if args.action == "build-profile":
        source_variant = args.source_variant or "full_mgtb"
        rows = _valid_rows(root / "runs" / "development" / source_variant, manifest, "development", seeds)
        expected = len(manifest["roles"]["development"]) * len(seeds)
        if len(rows) != expected:
            raise ValueError(f"development source incomplete: {len(rows)}/{expected}")
        profile = build_profile(rows, {"variant": source_variant, "manifest_sha256": manifest["manifest_sha256"]})
        destination = root / "profiles" / f"{source_variant}.json"
        atomic_write_json(destination, profile)
        print(json.dumps({"profile": str(destination), **profile["summary"]}, indent=2))
        return
    if args.action == "freeze":
        freeze = build_freeze(campaign, manifest)
        destination = root / "freeze" / "campaign.lock.json"
        atomic_write_json(destination, freeze)
        print(json.dumps({"freeze": str(destination), "freeze_sha256": freeze["freeze_sha256"]}, indent=2))
        return
    if args.action == "analyze":
        runs = {}
        for name in campaign["variants"]:
            path = root / "runs" / "test" / name
            if (path / "run_state.json").exists():
                rows = _valid_rows(path, manifest, "test", seeds)
                expected = len(manifest["roles"]["test"]) * len(seeds)
                if len(rows) != expected:
                    raise ValueError(f"test run {name} incomplete: {len(rows)}/{expected}")
                runs[name] = rows
        required = set(campaign.get("analysis", {}).get("required_variants", campaign["variants"].keys()))
        missing = sorted(required - runs.keys())
        if missing:
            raise ValueError(f"required test variants have not been run: {missing}")
        result = campaign_analysis(runs, baseline=campaign.get("analysis", {}).get("baseline", "vanilla"),
                                   bootstrap_seed=int(campaign.get("analysis", {}).get("bootstrap_seed", 20260811)),
                                   bootstrap_samples=int(campaign.get("analysis", {}).get("bootstrap_samples", 10000)))
        atomic_write_json(root / "analysis" / "campaign_results.json", result)
        print(json.dumps(result, indent=2))
        return
    if args.action == "status":
        statuses = {}
        for role in ("reference", "development", "test"):
            role_root = root / "runs" / role
            if not role_root.exists():
                continue
            for path in sorted(role_root.iterdir()):
                if (path / "run_state.json").exists():
                    state = load_json(path / "run_state.json")
                    progress_path = path / "progress.json"
                    statuses[f"{role}/{path.name}"] = (
                        load_json(progress_path) if progress_path.exists()
                        else {"completed": state["completed_count"]}
                    )
        print(json.dumps(statuses, indent=2))


if __name__ == "__main__":
    main()
