import math

from mgtb_v3.calibration.ecdf import ECDF


def test_tail_pvalue_counts_upper_tail():
    ecdf = ECDF([1.0, 2.0, 3.0], p_clip=1e-9)
    assert ecdf.tail_pvalue(2.0) == (1 + 2) / 4


def test_tail_pvalue_clips_extreme_value():
    ecdf = ECDF([1.0, 2.0, 3.0], p_clip=0.2)
    assert ecdf.tail_pvalue(100.0) == 0.25


def test_quantile_score_positive_and_finite():
    ecdf = ECDF([0.0, 1.0, 2.0])
    score = ecdf.quantile_score(2.0)
    assert score > 0.0
    assert math.isfinite(score)
