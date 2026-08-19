from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from mgtb_v3.calibration.positional import DEFAULT_BUCKETS, PositionalCalibrator, bucket_name
from mgtb_v3.calibration.threshold import calibrate_threshold
from mgtb_v3.config import config_from_dict
from mgtb_v3.features.window_features import linear_window_score
from mgtb_v3.science_fast.calibration import is_healthy
from mgtb_v3.science_fast.io import sha256_json
from mgtb_v3.types import WindowFeatures


def _scored_windows(artifact: dict[str, Any], controller: dict[str, Any]):
    score_cfg = config_from_dict(controller).score
    for event in artifact.get("monitor_trace", []):
        if event.get("type") != "window":
            continue
        features = WindowFeatures(**event["features"])
        yield features, linear_window_score(features, score_cfg)


def build_calibrator(
    artifacts: list[dict[str, Any]], spec: dict[str, Any], provenance: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    mode = spec.get("calibration_mode", "positional")
    buckets = DEFAULT_BUCKETS if mode == "positional" else [(0, None)]
    pools: dict[str, list[float]] = defaultdict(list)
    trajectories: dict[str, set[str]] = defaultdict(set)
    healthy = []
    calibrator_helper = PositionalCalibrator(buckets=buckets)
    for artifact in artifacts:
        if not is_healthy(artifact):
            continue
        healthy.append({"item_id": artifact["item_id"], "content_sha256": artifact["content_sha256"]})
        for features, score in _scored_windows(artifact, spec["controller"]):
            name = calibrator_helper.bucket_for_position(features.end_pos)
            pools[name].append(score)
            trajectories[name].add(artifact["item_id"])
    if not pools:
        raise ValueError("no healthy calibration windows")
    payload = {
        "schema_version": 1,
        "calibration_spec_sha256": spec["calibration_sha256"],
        "calibration_mode": mode,
        "accumulation_mode": spec.get("accumulation_mode", "cusum_reset"),
        "buckets": buckets,
        "p_clip": float(spec["controller"]["detector"]["p_clip"]),
        "score_pools_by_bucket": dict(pools),
        "healthy_items": healthy,
        "windows_per_bucket": {key: len(values) for key, values in pools.items()},
        "distinct_trajectories_per_bucket": {key: len(values) for key, values in trajectories.items()},
        "provenance": provenance,
    }
    payload["calibrator_sha256"] = sha256_json(payload)
    summary = {
        "completed": len(artifacts), "healthy_retained": len(healthy),
        "total_windows": sum(len(values) for values in pools.values()),
        "windows_per_bucket": payload["windows_per_bucket"],
        "distinct_trajectories_per_bucket": payload["distinct_trajectories_per_bucket"],
    }
    return payload, summary


def select_threshold(
    artifacts: list[dict[str, Any]], calibrator_payload: dict[str, Any], spec: dict[str, Any]
) -> dict[str, Any]:
    calibrator = PositionalCalibrator(
        calibrator_payload["buckets"], calibrator_payload["score_pools_by_bucket"], calibrator_payload["p_clip"]
    )
    runs = []
    for artifact in artifacts:
        if not is_healthy(artifact):
            continue
        values = [calibrator.p_value(score, features.end_pos) for features, score in _scored_windows(artifact, spec["controller"])]
        if values:
            runs.append({"item_id": artifact["item_id"], "p_values": values})
    detector = spec["controller"]["detector"]
    result = calibrate_threshold(
        runs,
        target_false_alert_rate=float(detector["target_false_alert_rate"]),
        gammas=tuple(detector["betting_gammas"]), p_clip=float(detector["p_clip"]),
        refractory_windows=0, accumulation_mode=spec.get("accumulation_mode", "cusum_reset"),
    )
    result["selected_h"] = math.log(float(result.pop("threshold")))
    result["healthy_denominator"] = len(runs)
    result["healthy_alarm_rate"] = result.pop("observed_false_alert_rate")
    result["warning"] = "low healthy denominator" if len(runs) < 30 else None
    result["calibrator_sha256"] = calibrator_payload["calibrator_sha256"]
    result["calibration_spec_sha256"] = spec["calibration_sha256"]
    result["threshold_sha256"] = sha256_json(result)
    return result
