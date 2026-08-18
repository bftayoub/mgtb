import torch

from mgtb_v3.config import BacktrackingConfig, DetectorConfig, MGTBV3Config, ScoreConfig, WindowConfig
from mgtb_v3.generation.hf_loop import generate_with_mgtb_v3


class Tokenizer:
    eos_token_id = 99
    def __call__(self, text, return_tensors=None, add_special_tokens=True):
        ids = [7, 8]
        return {"input_ids": torch.tensor([ids]) if return_tensors == "pt" else ids}
    def decode(self, ids, skip_special_tokens=True):
        return " ".join(map(str, ids))


class Model:
    device = torch.device("cpu")
    def eval(self):
        pass
    def __call__(self, input_ids, past_key_values=None, use_cache=True):
        logits = torch.tensor([[[0.1, 0.2, 0.3, 0.4]]])
        return type("Output", (), {"logits": logits, "past_key_values": None})()


class Calibrator:
    def p_value(self, score, token_pos):
        return 1.0


def test_no_alarm_mgtb_is_exactly_token_identical_to_vanilla():
    cfg = MGTBV3Config(
        window=WindowConfig(window_size=2, stride=1, ngram_min=1, ngram_max=1),
        detector=DetectorConfig(refractory_windows=2), backtracking=BacktrackingConfig(max_rerolls=3),
        score=ScoreConfig(),
    )
    torch.manual_seed(20260811)
    vanilla = generate_with_mgtb_v3(Model(), Tokenizer(), "p", cfg, Calibrator(), float("inf"), max_new_tokens=8, do_backtracking=False)
    torch.manual_seed(20260811)
    mgtb = generate_with_mgtb_v3(Model(), Tokenizer(), "p", cfg, Calibrator(), float("inf"), max_new_tokens=8, do_backtracking=True)
    assert mgtb.alerts == []
    assert mgtb.tokens == vanilla.tokens
