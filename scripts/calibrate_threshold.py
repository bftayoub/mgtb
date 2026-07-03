#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from mgtb_v3.calibration.positional import DEFAULT_BUCKETS, PositionalCalibrator
from mgtb_v3.calibration.threshold import calibrate_threshold
from mgtb_v3.config import load_config


def calibrate(input_path: str, calibrator_path: str, threshold_path: str, config_path: str) -> None:
    cfg = load_config(config_path)
    pools = defaultdict(list)
    runs = defaultdict(list)
    provisional = PositionalCalibrator(DEFAULT_BUCKETS, {"0-512": [0.0]}, p_clip=cfg.detector.p_clip)
    rows = []
    with Path(input_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    for row in rows:
        features = row.get("features", {})
        pos = features.get("end_pos", row.get("end_pos", 0))
        bucket = provisional.bucket_for_position(pos)
        pools[bucket].append(float(row["score"]))
    calibrator = PositionalCalibrator(DEFAULT_BUCKETS, dict(pools), p_clip=cfg.detector.p_clip)
    calibrator.save_json(calibrator_path)
    for row in rows:
        features = row.get("features", {})
        pos = features.get("end_pos", row.get("end_pos", 0))
        p_value = calibrator.p_value(float(row["score"]), pos)
        runs[row.get("run_id", "default")].append(p_value)
    threshold = calibrate_threshold(
        list(runs.values()),
        cfg.detector.target_false_alert_rate,
        gammas=cfg.detector.betting_gammas,
        p_clip=cfg.detector.p_clip,
        refractory_windows=cfg.detector.refractory_windows,
    )
    Path(threshold_path).write_text(json.dumps(threshold, indent=2) + "\n", encoding="utf-8")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Build positional ECDFs and empirical detector threshold.")
    parser.add_argument("--input", required=True, help="healthy window scores JSONL")
    parser.add_argument("--calibrator-output", required=True)
    parser.add_argument("--threshold-output", required=True)
    parser.add_argument("--config", default="configs/mgtb_v3_default.yaml")
    args = parser.parse_args(argv)
    calibrate(args.input, args.calibrator_output, args.threshold_output, args.config)


if __name__ == "__main__":
    main()
