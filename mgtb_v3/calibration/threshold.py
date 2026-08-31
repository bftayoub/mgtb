from __future__ import annotations

import math
import sys

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
    accumulation_mode: str = "cusum_reset",
) -> dict:
    runs = [_extract_pvalues(run) for run in healthy_runs]
    runs = [run for run in runs if run]
    if not runs:
        raise ValueError("healthy_runs must contain at least one run with p-values")
    if not 0.0 <= float(target_false_alert_rate) <= 1.0:
        raise ValueError("target_false_alert_rate must be in [0, 1]")

    # Alerting at threshold h is equivalent to max_j logE_j >= log(h).
    # Select directly from the empirical per-trajectory maxima instead of an
    # arbitrary bounded grid: a fixed 1e8 cap can fail to reach the requested
    # false-alert rate on long generations while silently returning 1.0.
    maxima = []
    for run in runs:
        detector = EDetector(
            math.inf, gammas=gammas, p_clip=p_clip,
            refractory_windows=refractory_windows, accumulation_mode=accumulation_mode,
        )
        maxima.append(max(detector.update(p)["logE"] for p in run))

    allowed_false_alerts = math.floor(float(target_false_alert_rate) * len(runs) + 1e-12)
    if allowed_false_alerts >= len(runs):
        selected = 1.0
    else:
        # With alert iff max_logE >= selected_log_h, stepping one floating
        # point above the boundary also handles tied maxima conservatively.
        boundary = sorted(maxima, reverse=True)[allowed_false_alerts]
        if boundary >= math.log(sys.float_info.max):
            raise ValueError("required empirical threshold exceeds finite float range")
        selected = math.exp(boundary)
        # exp/log rounding can map the immediately following factor back onto
        # the boundary. Advance until the runtime comparison is truly strict.
        while math.log(selected) <= boundary:
            selected = math.nextafter(selected, math.inf)
            if not math.isfinite(selected):
                raise ValueError("required empirical threshold exceeds finite float range")
    selected_log = math.log(selected)
    false_alerts = sum(value >= selected_log for value in maxima)
    selected_rate = false_alerts / len(runs)
    if selected_rate > float(target_false_alert_rate) + 1e-15:
        raise ValueError(
            f"failed to meet target false-alert rate: {selected_rate} > {target_false_alert_rate}"
        )

    return {
        "threshold": selected,
        "observed_false_alert_rate": selected_rate,
        "diagnostics": {
            "num_runs": len(runs),
            "target_false_alert_rate": target_false_alert_rate,
            "accumulation_mode": accumulation_mode,
            "selection_method": "empirical_max_loge_order_statistic",
            "allowed_false_alerts": allowed_false_alerts,
            "observed_false_alerts": false_alerts,
            "max_loge_by_run_sorted": sorted(maxima),
        },
    }
