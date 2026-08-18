from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from mgtb_v3.calibration.positional import DEFAULT_BUCKETS, PositionalCalibrator
from mgtb_v3.calibration.threshold import calibrate_threshold

from .io import atomic_write_json, load_json, sha256_json


def _windows(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    return [event for event in artifact.get("monitor_trace", []) if event.get("type") == "window"]


def is_healthy(artifact: dict[str, Any]) -> bool:
    scorer = artifact.get("scorer", {})
    return bool(scorer.get("correct")) and bool(scorer.get("answer_extraction_ok")) and not bool(artifact.get("truncated"))


def build_reference_calibrator(artifacts: list[dict[str, Any]], provenance: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    pools: dict[str, list[float]] = defaultdict(list)
    trajectories: dict[str, set[str]] = defaultdict(set)
    healthy_ids: list[dict[str, str]] = []
    for artifact in artifacts:
        if not is_healthy(artifact):
            continue
        healthy_ids.append({"item_id": artifact["item_id"], "content_sha256": artifact["content_sha256"]})
        for window in _windows(artifact):
            end = int(window["end_pos"])
            bucket = PositionalCalibrator(DEFAULT_BUCKETS).bucket_for_position(end)
            pools[bucket].append(float(window["score"]))
            trajectories[bucket].add(artifact["item_id"])
    if not any(pools.values()):
        raise ValueError("reference contains no healthy windows")
    payload = {
        "schema_version": 1,
        "buckets": DEFAULT_BUCKETS,
        "p_clip": 1e-6,
        "score_pools_by_bucket": dict(pools),
        "healthy_items": healthy_ids,
        "windows_per_bucket": {key: len(value) for key, value in pools.items()},
        "distinct_trajectories_per_bucket": {key: len(value) for key, value in trajectories.items()},
        "provenance": provenance,
    }
    payload["calibrator_sha256"] = sha256_json(payload)
    summary = summarize_reference(artifacts, payload)
    return payload, summary


def summarize_reference(artifacts: list[dict[str, Any]], calibrator: dict[str, Any]) -> dict[str, Any]:
    return {
        "completed": len(artifacts),
        "target": 300,
        "correct": sum(bool(a.get("scorer", {}).get("correct")) for a in artifacts),
        "extractable": sum(bool(a.get("scorer", {}).get("answer_extraction_ok")) for a in artifacts),
        "truncated": sum(bool(a.get("truncated")) for a in artifacts),
        "healthy_retained": len(calibrator.get("healthy_items", [])),
        "total_windows": sum(calibrator.get("windows_per_bucket", {}).values()),
        "windows_per_bucket": calibrator.get("windows_per_bucket", {}),
        "distinct_trajectories_per_bucket": calibrator.get("distinct_trajectories_per_bucket", {}),
    }


def apply_calibrator(artifacts: list[dict[str, Any]], calibrator_payload: dict[str, Any]) -> list[dict[str, Any]]:
    calibrator = PositionalCalibrator(
        calibrator_payload["buckets"], calibrator_payload["score_pools_by_bucket"], calibrator_payload["p_clip"]
    )
    runs = []
    for artifact in artifacts:
        if not is_healthy(artifact):
            continue
        p_values = [calibrator.p_value(float(w["score"]), int(w["end_pos"])) for w in _windows(artifact)]
        if p_values:
            runs.append({"item_id": artifact["item_id"], "p_values": p_values})
    return runs


def select_development_threshold(artifacts: list[dict[str, Any]], calibrator_payload: dict[str, Any]) -> dict[str, Any]:
    healthy_runs = apply_calibrator(artifacts, calibrator_payload)
    result = calibrate_threshold(healthy_runs, target_false_alert_rate=0.05, p_clip=1e-6, refractory_windows=0)
    # Historical EDetector takes an e-factor threshold. The scientific lock and
    # reports expose h = log(threshold), as specified by S_j >= h.
    import math
    result["selected_h"] = math.log(float(result.pop("threshold")))
    result["healthy_denominator"] = len(healthy_runs)
    result["healthy_alarm_rate"] = result.pop("observed_false_alert_rate")
    result["warning"] = "low healthy denominator" if len(healthy_runs) < 30 else None
    result["calibrator_sha256"] = calibrator_payload["calibrator_sha256"]
    result["threshold_sha256"] = sha256_json(result)
    return result


def save_payload(path: str | Path, payload: dict[str, Any]) -> None:
    atomic_write_json(path, payload)


def load_calibrator(path: str | Path) -> dict[str, Any]:
    payload = load_json(path)
    expected = payload.get("calibrator_sha256")
    actual = sha256_json({k: v for k, v in payload.items() if k != "calibrator_sha256"})
    if expected != actual:
        raise ValueError("calibrator hash mismatch")
    return payload


def load_threshold(path: str | Path) -> dict[str, Any]:
    payload = load_json(path)
    expected = payload.get("threshold_sha256")
    actual = sha256_json({k: v for k, v in payload.items() if k != "threshold_sha256"})
    if expected != actual:
        raise ValueError("threshold hash mismatch")
    return payload
