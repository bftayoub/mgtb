from __future__ import annotations

from dataclasses import asdict

from mgtb_v3.config import BacktrackingConfig, MGTBV3Config
from mgtb_v3.control.cache_utils import crop_hf_cache


class BacktrackingController:
    def __init__(self, config: MGTBV3Config | BacktrackingConfig):
        self.config = config.backtracking if hasattr(config, "backtracking") else config
        self.reroll_count = 0

    def on_alert(self, alert_info, tokens, cache, monitor, detector, prompt_len: int):
        if self.config.cache_state_mode != "replay_last":
            raise ValueError("scientific controller requires cache_state_mode=replay_last")
        if self.config.changepoint_index_mode != "tracked_windows":
            raise ValueError("scientific controller requires changepoint_index_mode=tracked_windows")
        if self.reroll_count >= self.config.max_rerolls:
            return {"applied": False, "reason": "max_rerolls_exhausted", "tokens": tokens, "cache": cache}

        if self.config.fixed_rollback_tokens is not None:
            rollback_pos = max(prompt_len, alert_info.token_pos - self.config.fixed_rollback_tokens)
        else:
            rollback_pos = max(prompt_len, alert_info.rollback_token_pos)
        rollback_pos = min(rollback_pos, len(tokens))
        generated_rollback_pos = max(0, rollback_pos - prompt_len)
        generated_alert_pos = max(0, alert_info.token_pos - prompt_len)
        bad_ngrams = monitor.ngram_tracker.faulty_ngrams(generated_rollback_pos, generated_alert_pos)
        new_tokens = list(tokens[:rollback_pos])
        # replay_last invariant: cache represents the prefix immediately before
        # the last retained token; the decode loop replays that token once.
        new_cache = crop_hf_cache(cache, max(0, rollback_pos - 1))
        stale_windows = [w.window_index for w in monitor.window_features_history if w.end_pos > generated_rollback_pos]
        monitor.truncate(generated_rollback_pos)
        detector.reset()
        detector.enter_refractory()
        self.reroll_count += 1
        event = {
            "applied": True,
            "alert": asdict(alert_info) if hasattr(alert_info, "__dataclass_fields__") else dict(alert_info),
            "rollback_pos": rollback_pos,
            "rollback_span": len(tokens) - rollback_pos,
            "cache_state_mode": "replay_last",
            "changepoint_index_mode": "tracked_windows",
            "invalidated_window_indices": stale_windows,
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
