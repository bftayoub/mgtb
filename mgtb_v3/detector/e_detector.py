from __future__ import annotations

import math

from mgtb_v3.detector.betting import log_mixture_betting


class EDetector:
    def __init__(
        self,
        threshold: float,
        gammas=(0.1, 0.3, 0.5, 0.7),
        p_clip: float = 1e-6,
        refractory_windows: int = 0,
    ):
        if threshold <= 0:
            raise ValueError("threshold must be positive")
        self.threshold = float(threshold)
        self.log_threshold = math.log(self.threshold)
        self.gammas = tuple(float(g) for g in gammas)
        self.p_clip = float(p_clip)
        self.refractory_windows = int(refractory_windows)
        self._refractory_remaining = 0
        self.reset()

    def update(self, p_value: float) -> dict:
        window_index = len(self.p_history)
        if self._refractory_remaining > 0:
            self._refractory_remaining -= 1
            self.p_history.append(float(p_value))
            self.e_history.append(1.0)
            self.logE_history.append(self.logE)
            return {
                "window_index": window_index,
                "p_value": float(p_value),
                "e_value": 1.0,
                "logE": self.logE,
                "alert": False,
                "refractory": True,
            }

        loge = log_mixture_betting(p_value, self.gammas, p_clip=self.p_clip)
        self.logE = max(0.0, self.logE) + loge
        e_value = math.exp(loge)
        alert = self.logE >= self.log_threshold
        self.p_history.append(float(p_value))
        self.e_history.append(float(e_value))
        self.logE_history.append(float(self.logE))
        return {
            "window_index": window_index,
            "p_value": float(p_value),
            "e_value": float(e_value),
            "logE": float(self.logE),
            "alert": bool(alert),
            "refractory": False,
        }

    def reset(self) -> None:
        self.logE = 0.0
        self.logE_history: list[float] = []
        self.p_history: list[float] = []
        self.e_history: list[float] = []

    def changepoint_window(self) -> int:
        for idx in range(len(self.logE_history) - 1, -1, -1):
            if self.logE_history[idx] <= 0.0:
                return idx
        return 0

    def enter_refractory(self) -> None:
        self._refractory_remaining = self.refractory_windows
