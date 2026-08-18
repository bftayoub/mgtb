from dataclasses import dataclass

import torch

from mgtb_v3.config import BacktrackingConfig
from mgtb_v3.control.backtracking import BacktrackingController
from mgtb_v3.control.cache_utils import crop_hf_cache, replay_last_logits
from mgtb_v3.detector.e_detector import EDetector
from mgtb_v3.features.window_features import TrajectoryMonitor
from mgtb_v3.config import MGTBV3Config, WindowConfig
from mgtb_v3.types import AlertInfo


class SumCacheModel:
    device = torch.device("cpu")
    def __call__(self, input_ids, past_key_values=None, use_cache=True):
        past = torch.tensor(0.0) if past_key_values is None else past_key_values
        total = past + input_ids.float().sum()
        logits = torch.stack([total, -total]).reshape(1, 1, 2)
        return type("Output", (), {"logits": logits, "past_key_values": total})()


def test_replay_last_logits_matches_clean_forward_on_same_prefix():
    model, prefix = SumCacheModel(), [2, 3, 5]
    clean = model(torch.tensor([prefix]))
    cache_before_last = model(torch.tensor([prefix[:-1]])).past_key_values
    reconstructed, cache = replay_last_logits(model, prefix, cache_before_last)
    assert torch.equal(reconstructed, clean.logits[:, -1, :])
    assert cache == clean.past_key_values


def test_tuple_kv_crop_and_controller_removes_stale_windows_with_replay_last():
    key = torch.arange(20).reshape(1, 1, 10, 2)
    cropped = crop_hf_cache(((key, key.clone()),), 4)
    assert cropped[0][0].shape[-2] == 4
    cfg = MGTBV3Config(window=WindowConfig(window_size=2, stride=1, ngram_min=2, ngram_max=2),
                       backtracking=BacktrackingConfig(max_rerolls=1, margin_tokens=0))
    monitor = TrajectoryMonitor(cfg)
    for token in [1, 2, 1, 2]:
        monitor.update_token(token, torch.zeros(8))
    while monitor.should_emit_window():
        monitor.compute_window_features()
    detector = EDetector(2, refractory_windows=2)
    detector.update(1e-6)
    alert = AlertInfo(2, 6, 0, 4, 0, 1e-6, 10)
    cache = ((torch.zeros(1, 1, 6, 2), torch.zeros(1, 1, 6, 2)),)
    event = BacktrackingController(cfg).on_alert(alert, [9, 9, 1, 2, 1, 2], cache, monitor, detector, prompt_len=2)
    assert event["cache"][0][0].shape[-2] == 3  # rollback full pos 4, cache through pos 2
    assert event["cache_state_mode"] == "replay_last"
    assert all(window.end_pos <= 2 for window in monitor.window_features_history)
    assert len(monitor.ngram_tracker.tokens) == 2
    assert detector.logE_history == []
    assert detector._refractory_remaining == 2


def test_max_rerolls_prevents_an_additional_repair():
    controller = BacktrackingController(BacktrackingConfig(max_rerolls=0))
    event = controller.on_alert({}, [1, 2], None, None, None, prompt_len=1)
    assert event["applied"] is False
    assert event["reason"] == "max_rerolls_exhausted"
