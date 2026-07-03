from __future__ import annotations

import math

from mgtb_v3.detector.e_detector import EDetector


def _extract_pvalues(run) -> list[float]:
    if isinstance(run, dict):
        if "p_values" in run:
            return [float(v) for v in run["p_values"]]
        if "windows" in run:
            return [float(w["p_value"]) for w in run["windows"] if w.get("p_value") is not None]
    return [float(v) for v in run]


def calibrate_threshold(
    healthy_runs,
    target_false_alert_rate: float,
    gammas=(0.1, 0.3, 0.5, 0.7),
    p_clip: float = 1e-6,
    refractory_windows: int = 0,
    grid_size: int = 80,
) -> dict:
    runs = [_extract_pvalues(run) for run in healthy_runs]
    runs = [run for run in runs if run]
    if not runs:
        raise ValueError("healthy_runs must contain at least one run with p-values")

    candidates = [math.exp(x) for x in _linspace(math.log(1.01), math.log(1e8), grid_size)]
    selected = candidates[-1]
    selected_rate = 1.0
    diagnostics = []
    for threshold in candidates:
        false_alerts = 0
        for run in runs:
            detector = EDetector(threshold, gammas=gammas, p_clip=p_clip, refractory_windows=refractory_windows)
            alerted = any(detector.update(p)["alert"] for p in run)
            false_alerts += int(alerted)
        rate = false_alerts / len(runs)
        diagnostics.append({"threshold": threshold, "false_alert_rate": rate})
        if rate <= target_false_alert_rate:
            selected = threshold
            selected_rate = rate
            break

    return {
        "threshold": selected,
        "observed_false_alert_rate": selected_rate,
        "diagnostics": {
            "num_runs": len(runs),
            "target_false_alert_rate": target_false_alert_rate,
            "grid": diagnostics,
        },
    }


def _linspace(start: float, stop: float, num: int) -> list[float]:
    if num <= 1:
        return [start]
    step = (stop - start) / (num - 1)
    return [start + i * step for i in range(num)]
