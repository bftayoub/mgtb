from __future__ import annotations

import random


class RandomTrigger:
    def __init__(self, trigger_probability: float, seed: int | None = None):
        self.trigger_probability = max(0.0, min(1.0, float(trigger_probability)))
        self.rng = random.Random(seed)
        self.history: list[bool] = []

    @classmethod
    def from_mean_trigger_rate(cls, mean_alerts_per_window: float, seed: int | None = None) -> "RandomTrigger":
        return cls(mean_alerts_per_window, seed=seed)

    def update(self) -> bool:
        alert = self.rng.random() < self.trigger_probability
        self.history.append(alert)
        return alert
