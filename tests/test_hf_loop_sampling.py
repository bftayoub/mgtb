import torch

from mgtb_v3.generation.hf_loop import _mask_bad_ngram_completions


def test_bad_ngram_completion_is_masked():
    logits = torch.zeros((1, 10))
    _mask_bad_ngram_completions(logits, [4, 5], [(4, 5, 6), (7, 8, 9)])
    assert torch.isneginf(logits[0, 6])
    assert logits[0, 9] == 0.0
