from mgtb_v3.detector.e_detector import EDetector


def test_detector_updates_log_space_and_alerts():
    detector = EDetector(threshold=2.0)
    update = detector.update(1e-6)
    assert update["logE"] > 0.0
    assert update["alert"] is True


def test_detector_refractory_skips_betting():
    detector = EDetector(threshold=2.0, refractory_windows=1)
    detector.enter_refractory()
    update = detector.update(1e-6)
    assert update["refractory"] is True
    assert update["alert"] is False
    assert update["e_value"] == 1.0


def test_detector_reset_and_changepoint():
    detector = EDetector(threshold=10.0)
    detector.update(0.99)
    detector.update(1e-6)
    assert detector.changepoint_window() == 0
    detector.reset()
    assert detector.logE == 0.0
    assert detector.logE_history == []
