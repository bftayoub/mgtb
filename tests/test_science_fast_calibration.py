import math
import numpy as np

from mgtb_v3.calibration.positional import PositionalCalibrator
from mgtb_v3.config import ScoreConfig
from mgtb_v3.features.window_features import linear_window_score
from mgtb_v3.features.entropy import chosen_logprob_from_logits, entropy_from_logits
from mgtb_v3.science_fast.calibration import build_reference_calibrator, select_development_threshold
from mgtb_v3.calibration.threshold import calibrate_threshold
from mgtb_v3.types import WindowFeatures


def _artifact(item_id, scores, healthy=True):
    return {
        "item_id": item_id, "content_sha256": item_id * 8,
        "scorer": {"correct": healthy, "answer_extraction_ok": healthy}, "truncated": False,
        "monitor_trace": [{"type": "window", "end_pos": pos, "score": score} for pos, score in scores],
    }


def test_six_feature_score_uses_specification_weights():
    feature = WindowFeatures(0, 0, 64, 1, -2, 3, 4, 0, 5, 6)
    score = linear_window_score(feature, ScoreConfig())
    assert score == 0.15 + 0.10 * 2 + 0.20 * 3 + 0.35 * 4 + 0.18 * 5 + 0.02 * 6


def test_entropy_and_chosen_log_probability_match_softmax_definition():
    logits = np.array([0.0, math.log(3.0)])
    expected_entropy = -(0.25 * math.log(0.25) + 0.75 * math.log(0.75))
    assert math.isclose(entropy_from_logits(logits), expected_entropy)
    assert math.isclose(chosen_logprob_from_logits(logits, 1), math.log(0.75))


def test_bucket_boundaries_empirical_q_and_global_fallback():
    cal = PositionalCalibrator(score_pools_by_bucket={"0-512": [1, 2, 3], "1024-2048": [10]})
    assert cal.bucket_for_position(0) == "0-512"
    assert cal.bucket_for_position(511) == "0-512"
    assert cal.bucket_for_position(512) == "512-1024"
    assert cal.bucket_for_position(4096) == "4096-inf"
    assert cal.p_value(2, 100) == (1 + 2) / (3 + 1)
    # Empty bucket pools all reference values: scores [1,2,3,10].
    assert cal.p_value(3, 700) == (1 + 2) / (4 + 1)


def test_reference_only_healthy_and_development_selects_h():
    ref = [_artifact("a", [(64, 0.1), (600, 0.2)]), _artifact("b", [(64, 99)], healthy=False)]
    calibrator, summary = build_reference_calibrator(ref, {"source": "test"})
    assert summary["healthy_retained"] == 1
    assert summary["total_windows"] == 2
    dev = [_artifact(f"d{i}", [(64, 0.1 + i / 1000)]) for i in range(40)]
    threshold = select_development_threshold(dev, calibrator)
    assert math.isfinite(threshold["selected_h"])
    assert threshold["healthy_denominator"] == 40
    assert threshold["healthy_alarm_rate"] <= 0.05


def test_threshold_is_not_limited_by_the_old_fixed_grid_cap():
    # Identical long anomalous trajectories all exceed the historical 1e8
    # factor cap. A 5% target with 20 tied runs must conservatively select a
    # finite threshold above every observed maximum and achieve zero alerts.
    runs = [{"p_values": [1e-6] * 40} for _ in range(20)]
    result = calibrate_threshold(runs, target_false_alert_rate=0.05, p_clip=1e-6)
    assert math.isfinite(result["threshold"])
    assert result["threshold"] > 1e8
    assert result["observed_false_alert_rate"] == 0.0
    assert result["diagnostics"]["selection_method"] == "empirical_max_loge_order_statistic"
