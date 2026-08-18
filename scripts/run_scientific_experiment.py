#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mgtb_v3.eval import math500
from mgtb_v3.science_fast.analysis import paired_analysis
from mgtb_v3.science_fast.calibration import (
    build_reference_calibrator, load_calibrator, load_threshold, save_payload, select_development_threshold,
)
from mgtb_v3.science_fast.freeze import build_freeze, load_freeze, save_freeze
from mgtb_v3.science_fast.io import atomic_write_json, load_json
from mgtb_v3.science_fast.protocol import build_manifest, load_manifest, load_pinned_datasets, save_manifest
from mgtb_v3.science_fast.provenance import git_commit, software_environment, source_tree_sha256
from mgtb_v3.science_fast.runner import load_run_artifacts, resolved_settings, run_role


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Leakage-safe resumable MGT-B scientific pipeline")
    parser.add_argument("--config", required=True)
    parser.add_argument("--stop-after", type=int, help="test/debug interruption after N newly completed items")
    args = parser.parse_args(argv)
    with Path(args.config).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    action = raw["action"]

    if action == "build_manifest":
        train, test = load_pinned_datasets()
        manifest = build_manifest(train, test)
        save_manifest(raw["manifest"], manifest)
        _print({"manifest": raw["manifest"], "counts": manifest["counts"], "manifest_sha256": manifest["manifest_sha256"]})
        return

    manifest = load_manifest(raw["manifest"])
    if action == "run":
        settings = resolved_settings(args.config)
        role, method = raw["protocol_role"], raw["method"]
        calibrator = load_calibrator(raw["calibrator"]) if raw.get("calibrator") else None
        threshold = load_threshold(raw["threshold"]) if raw.get("threshold") else None
        freeze = load_freeze(raw["freeze"], manifest=manifest, method=method) if role == "test" else None
        artifacts = run_role(
            settings=settings, manifest=manifest, role=role, method=method, output_dir=raw["output_dir"],
            calibrator_payload=calibrator, selected_h=threshold.get("selected_h") if threshold else None,
            freeze=freeze, stop_after=args.stop_after,
        )
        _print({"role": role, "method": method, "completed": len(artifacts), "target": len(manifest["roles"][role])})
        return


    if action == "build_calibrator":
        artifacts = load_run_artifacts(raw["reference_run"], manifest["roles"]["reference"])
        if len(artifacts) != 300:
            raise ValueError(f"reference incomplete: {len(artifacts)}/300")
        calibrator, summary = build_reference_calibrator(artifacts, _source())
        save_payload(raw["calibrator"], calibrator)
        atomic_write_json(raw["summary"], summary)
        _print(summary)
        return

    if action == "select_threshold":
        calibrator = load_calibrator(raw["calibrator"])
        artifacts = load_run_artifacts(raw["development_run"], manifest["roles"]["development"])
        if len(artifacts) != 100:
            raise ValueError(f"development incomplete: {len(artifacts)}/100")
        threshold = select_development_threshold(artifacts, calibrator)
        atomic_write_json(raw["threshold"], threshold)
        _print({key: threshold[key] for key in ("healthy_denominator", "selected_h", "healthy_alarm_rate", "warning")})
        return

    if action == "freeze":
        settings = resolved_settings(args.config)
        calibrator = load_calibrator(raw["calibrator"])
        threshold = load_threshold(raw["threshold"])
        for method, path in raw["freeze_outputs"].items():
            freeze = build_freeze(
                manifest=manifest, resolved_config=settings, calibrator=calibrator, threshold=threshold, method=method,
                source=_source(), environment=software_environment(), scorer_path=Path(math500.__file__),
            )
            save_freeze(path, freeze)
        _print({"freeze_outputs": raw["freeze_outputs"]})
        return

    if action == "analyze":
        vanilla = load_run_artifacts(raw["vanilla_run"], manifest["roles"]["test"])
        mgtb = load_run_artifacts(raw["mgtb_run"], manifest["roles"]["test"])
        if len(vanilla) != 300 or len(mgtb) != 300:
            raise ValueError(f"test incomplete: vanilla={len(vanilla)}/300 mgtb={len(mgtb)}/300")
        result = paired_analysis(vanilla, mgtb)
        atomic_write_json(raw["analysis_output"], result)
        _print(result)
        return
    raise ValueError(f"unknown action: {action}")


def _source() -> dict:
    return {"git_commit": git_commit(), "source_tree_sha256": source_tree_sha256(), "software_environment": software_environment()}


def _print(value) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
