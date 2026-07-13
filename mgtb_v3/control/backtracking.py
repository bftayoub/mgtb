from __future__ import annotations

from dataclasses import asdict

from mgtb_v3.config import BacktrackingConfig, MGTBV3Config
from mgtb_v3.control.cache_utils import crop_hf_cache


class BacktrackingController:
    def __init__(self, config: MGTBV3Config | BacktrackingConfig):
        self.config = config.backtracking if hasattr(config, "backtracking") else config
        self.reroll_count = 0

    def on_alert(self, alert_info, tokens, cache, monitor, detector, prompt_len: int):
        if self.reroll_count >= self.config.max_rerolls:
            return {"applied": False, "reason": "max_rerolls_exhausted", "tokens": tokens, "cache": cache}

        if self.config.fixed_rollback_tokens is not None:
            rollback_pos = max(prompt_len, alert_info.token_pos - self.config.fixed_rollback_tokens)
        else:
            rollback_pos = max(prompt_len, alert_info.rollback_token_pos)
        rollback_pos = min(rollback_pos, len(tokens))
        window_start = max(prompt_len, rollback_pos)
        bad_ngrams = monitor.ngram_tracker.faulty_ngrams(window_start, alert_info.token_pos)
        new_tokens = list(tokens[:rollback_pos])
        new_cache = crop_hf_cache(cache, rollback_pos)
        monitor.truncate(rollback_pos)
        detector.reset()
        detector.enter_refractory()
        self.reroll_count += 1
        event = {
            "applied": True,
            "alert": asdict(alert_info) if hasattr(alert_info, "__dataclass_fields__") else dict(alert_info),
            "rollback_pos": rollback_pos,
            "bad_ngrams": bad_ngrams,
            "reroll_index": self.reroll_count,
            "wait_injection_text": self.config.wait_injection_text if self.config.inject_wait_on_backtrack else "",
            "decode_overrides": {
                "temperature": self.config.redecode_temperature,
                "repetition_penalty": self.config.repetition_penalty,
                "use_no_bad_ngrams": self.config.use_no_bad_ngrams,
            },
            "tokens": new_tokens,
            "cache": new_cache,
        }
        return event
