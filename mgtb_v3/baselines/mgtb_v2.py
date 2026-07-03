from __future__ import annotations

from mgtb_v3.features.entropy import chosen_logprob_from_logits, entropy_from_logits


class MGTBV2Baseline:
    """Minimal v2 martingale baseline for comparison with v3 only."""

    def __init__(self):
        self.M = 0.0
        self.history: list[dict] = []

    def update(self, logits, token_id: int) -> dict:
        entropy = entropy_from_logits(logits)
        logprob = chosen_logprob_from_logits(logits, token_id)
        increment = logprob + entropy
        self.M += increment
        event = {"entropy": entropy, "logprob": logprob, "increment": increment, "M": self.M}
        self.history.append(event)
        return event

    def reset(self) -> None:
        self.M = 0.0
        self.history = []
