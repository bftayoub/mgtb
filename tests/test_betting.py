import math

from mgtb_v3.detector.betting import beta_betting, log_mixture_betting, mixture_betting


def test_beta_betting_positive():
    assert beta_betting(0.05, 0.5) > 0.0


def test_mixture_betting_positive_and_finite():
    value = mixture_betting(0.01)
    assert value > 0.0
    assert math.isfinite(value)


def test_log_mixture_handles_tiny_p_with_clipping():
    value = log_mixture_betting(0.0, p_clip=1e-6)
    assert math.isfinite(value)
