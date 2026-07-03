import math

import numpy as np

from mgtb_v3.config import MGTBV3Config, WindowConfig
from mgtb_v3.features.window_features import TrajectoryMonitor


def test_window_emission_after_window_size_and_stride():
    cfg = MGTBV3Config(window=WindowConfig(window_size=4, stride=2, ngram_min=2, ngram_max=2))
    monitor = TrajectoryMonitor(cfg)
    logits = np.array([2.0, 0.0, -1.0])
    for _ in range(4):
        monitor.update_token(0, logits)
    assert monitor.should_emit_window()
    first = monitor.compute_window_features()
    assert first.start_pos == 0
    assert first.end_pos == 4
    assert not monitor.should_emit_window()
    for _ in range(2):
        monitor.update_token(0, logits)
    assert monitor.compute_window_features().start_pos == 2


def test_truncate_and_features_are_finite():
    cfg = MGTBV3Config(window=WindowConfig(window_size=4, stride=2, ngram_min=2, ngram_max=2))
    monitor = TrajectoryMonitor(cfg)
    logits = np.array([2.0, 0.0, -1.0])
    for _ in range(6):
        monitor.update_token(0, logits)
    features = monitor.compute_window_features()
    values = [
        features.mean_entropy,
        features.mean_logprob,
        features.repetition_rate,
        features.confident_loop_score,
        features.local_entropy_log_ratio,
    ]
    assert all(math.isfinite(v) for v in values)
    monitor.truncate(3)
    assert len(monitor.tokens) == 3
