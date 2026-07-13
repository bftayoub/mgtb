import torch

from mgtb_v3.config import BacktrackingConfig, DetectorConfig, MGTBV3Config, ScoreConfig, WindowConfig
from mgtb_v3.generation import hf_loop
from mgtb_v3.generation.hf_loop import _encode_injection_tokens, _mask_bad_ngram_completions, generate_with_mgtb_v3


def test_bad_ngram_completion_is_masked():
    logits = torch.zeros((1, 10))
    _mask_bad_ngram_completions(logits, [4, 5], [(4, 5, 6), (7, 8, 9)])
    assert torch.isneginf(logits[0, 6])
    assert logits[0, 9] == 0.0


def test_encode_injection_tokens_uses_no_special_tokens():
    class Tokenizer:
        def __call__(self, text, add_special_tokens=True):
            assert add_special_tokens is False
            return {"input_ids": [7, 8]}

    assert _encode_injection_tokens(Tokenizer(), "Wait.") == [7, 8]


def test_wait_injection_after_backtrack_is_configurable(tmp_path, monkeypatch):
    class Tokenizer:
        eos_token_id = 99

        def __call__(self, text, return_tensors=None, add_special_tokens=True):
            if text == "PROMPT":
                ids = [10, 11]
            elif text == "\nWait.\n":
                ids = [42, 43]
            else:
                ids = [12]
            if return_tensors == "pt":
                return {"input_ids": torch.tensor([ids])}
            return {"input_ids": ids}

        def decode(self, tokens, skip_special_tokens=True):
            table = {10: "P", 11: "R", 42: "Wait", 43: ".", 20: "A", 21: "B", 22: "C"}
            return " ".join(table.get(int(token), str(int(token))) for token in tokens)

    class Model:
        device = torch.device("cpu")

        def eval(self):
            return None

        def __call__(self, input_ids, use_cache=True, past_key_values=None):
            logits = torch.zeros((1, 1, 128))
            return type("Output", (), {"logits": logits, "past_key_values": object()})()

    class Calibrator:
        def p_value(self, score, end_pos):
            return 1e-6

    samples = iter([20, 21, 22])

    def fake_sample_token(logits, **kwargs):
        return torch.tensor(next(samples))

    monkeypatch.setattr(hf_loop, "_sample_token", fake_sample_token)

    cfg = MGTBV3Config(
        window=WindowConfig(window_size=1, stride=1, ngram_min=1, ngram_max=1),
        detector=DetectorConfig(threshold=1.01, betting_gammas=(0.1,), refractory_windows=10),
        backtracking=BacktrackingConfig(
            max_rerolls=1,
            inject_wait_on_backtrack=True,
            wait_injection_text="\nWait.\n",
        ),
        score=ScoreConfig(
            w_entropy=0.0,
            w_logprob=0.0,
            w_repetition=0.0,
            w_confident_loop=0.0,
            w_local_entropy_pos=0.0,
            w_local_entropy_neg=0.0,
        ),
    )

    trace_path = tmp_path / "trace.jsonl"
    result = generate_with_mgtb_v3(
        Model(),
        Tokenizer(),
        "PROMPT",
        cfg,
        Calibrator(),
        threshold=1.01,
        max_new_tokens=3,
        trace_log_path=trace_path,
    )

    assert [42, 43] == result.tokens[2:4]
    assert "Wait" in result.text
    assert result.backtracks[0]["wait_injection_text"] == "\nWait.\n"
    assert result.backtracks[0]["injected_token_count"] == 2
    assert result.backtracks[0]["injected_tokens"] == [42, 43]
