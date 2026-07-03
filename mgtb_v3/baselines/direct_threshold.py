from __future__ import annotations


class DirectScoreThreshold:
    def __init__(self, threshold: float):
        self.threshold = float(threshold)

    def update(self, score: float) -> dict:
        return {"score": float(score), "alert": float(score) >= self.threshold}
