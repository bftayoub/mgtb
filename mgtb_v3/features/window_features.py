from __future__ import annotations

import math

from mgtb_v3.config import MGTBV3Config, ScoreConfig, WindowConfig
from mgtb_v3.features.entropy import chosen_logprob_from_logits, entropy_from_logits
from mgtb_v3.features.repetition import NgramTracker
from mgtb_v3.types import TokenStats, WindowFeatures


class TrajectoryMonitor:
    def __init__(self, config: MGTBV3Config | WindowConfig, prompt_tokens=None):
        self.window_config = config.window if hasattr(config, "window") else config
        self.prompt_tokens = list(prompt_tokens or [])
        self.prompt_len = len(self.prompt_tokens)
        self.tokens: list[int] = []
        self.entropies: list[float] = []
        self.logprobs: list[float] = []
        self.token_stats: list[TokenStats] = []
        self.ngram_tracker = NgramTracker(
            self.window_config.ngram_min,
            self.window_config.ngram_max,
            prompt_tokens=self.prompt_tokens,
            exclude_prompt=self.window_config.exclude_prompt_ngrams,
        )
        self.window_features_history: list[WindowFeatures] = []
        self._next_window_start = 0

    def update_token(self, token_id: int, logits) -> None:
        entropy = entropy_from_logits(logits)
        logprob = chosen_logprob_from_logits(logits, token_id)
        pos = self.prompt_len + len(self.tokens)
        self.tokens.append(int(token_id))
        self.entropies.append(float(entropy))
        self.logprobs.append(float(logprob))
        self.token_stats.append(TokenStats(int(token_id), pos, float(entropy), float(logprob)))
        self.ngram_tracker.update([int(token_id)], [float(logprob)], pos)

    def should_emit_window(self) -> bool:
        return len(self.tokens) - self._next_window_start >= self.window_config.window_size

    def compute_window_features(self) -> WindowFeatures:
        if not self.should_emit_window():
            raise ValueError("not enough tokens to emit a new window")
        start_rel = self._next_window_start
        end_rel = start_rel + self.window_config.window_size
        start_pos = self.prompt_len + start_rel
        end_pos = self.prompt_len + end_rel
        entropy_window = self.entropies[start_rel:end_rel]
        logprob_window = self.logprobs[start_rel:end_rel]
        mean_entropy = sum(entropy_window) / len(entropy_window)
        mean_logprob = sum(logprob_window) / len(logprob_window)
        global_entropy = sum(self.entropies[:end_rel]) / max(1, end_rel)
        eps = self.window_config.entropy_eps
        log_ratio = math.log((mean_entropy + eps) / (global_entropy + eps))
        features = WindowFeatures(
            window_index=len(self.window_features_history),
            start_pos=start_pos,
            end_pos=end_pos,
            mean_entropy=float(mean_entropy),
            mean_logprob=float(mean_logprob),
            repetition_rate=self.ngram_tracker.repetition_rate(start_pos, end_pos),
            confident_loop_score=self.ngram_tracker.confident_loop_score(start_pos, end_pos),
            local_entropy_log_ratio=float(log_ratio),
            local_entropy_pos=max(0.0, float(log_ratio)),
            local_entropy_neg=max(0.0, float(-log_ratio)),
        )
        self.window_features_history.append(features)
        self._next_window_start += self.window_config.stride
        return features

    def truncate(self, pos: int) -> None:
        keep_count = max(0, min(len(self.tokens), int(pos) - self.prompt_len))
        self.tokens = self.tokens[:keep_count]
        self.entropies = self.entropies[:keep_count]
        self.logprobs = self.logprobs[:keep_count]
        self.token_stats = self.token_stats[:keep_count]
        self.ngram_tracker.truncate(pos)
        self.window_features_history = [w for w in self.window_features_history if w.end_pos <= pos]
        if len(self.tokens) < self.window_config.window_size:
            self._next_window_start = 0
        else:
            self._next_window_start = len(self.window_features_history) * self.window_config.stride

    def reset_after_backtrack(self) -> None:
        # State is already made coherent by truncate(); this hook is kept for online controllers.
        pass


def linear_window_score(features: WindowFeatures, score_config: ScoreConfig) -> float:
    return float(
        score_config.w_entropy * features.mean_entropy
        + score_config.w_logprob * (-features.mean_logprob)
        + score_config.w_repetition * features.repetition_rate
        + score_config.w_confident_loop * features.confident_loop_score
        + score_config.w_local_entropy_pos * features.local_entropy_pos
        + score_config.w_local_entropy_neg * features.local_entropy_neg
    )
