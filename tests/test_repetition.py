from mgtb_v3.features.repetition import NgramTracker


def test_repeated_ngram_and_rate():
    tracker = NgramTracker(2, 2)
    tracker.update([1, 2, 3, 1, 2], [-1, -1, -1, -0.5, -0.5], 0)
    assert (1, 2) in tracker.faulty_ngrams(0, 5)
    assert tracker.repetition_rate(0, 5) == 1 / 4


def test_prompt_ngram_exclusion():
    tracker = NgramTracker(2, 2, prompt_tokens=[1, 2], exclude_prompt=True)
    tracker.update([1, 2, 3, 4, 3, 4], [-1] * 6, 2)
    assert (1, 2) not in tracker.faulty_ngrams(2, 8)
    assert (3, 4) in tracker.faulty_ngrams(2, 8)


def test_truncate_rebuilds_tables():
    tracker = NgramTracker(2, 2)
    tracker.update([1, 2, 1, 2], [-1] * 4, 0)
    assert tracker.repetition_rate(0, 4) > 0
    tracker.truncate(2)
    assert tracker.repetition_rate(0, 2) == 0.0
