from __future__ import annotations

import json
import math
from bisect import bisect_left
from pathlib import Path


class ECDF:
    def __init__(self, values: list[float], p_clip: float = 1e-6):
        cleaned = [float(v) for v in values if math.isfinite(float(v))]
        if not cleaned:
            raise ValueError("ECDF requires at least one finite calibration value")
        self.values = sorted(cleaned)
        self.p_clip = float(p_clip)

    def tail_pvalue(self, x: float) -> float:
        if not math.isfinite(float(x)):
            x = float("inf")
        idx = bisect_left(self.values, float(x))
        count_ge = len(self.values) - idx
        p = (1.0 + count_ge) / (len(self.values) + 1.0)
        return min(1.0, max(self.p_clip, float(p)))

    def quantile_score(self, x: float) -> float:
        return float(-math.log(self.tail_pvalue(x)))

    def to_dict(self) -> dict:
        return {"values": self.values, "p_clip": self.p_clip}

    def save_json(self, path: str | Path) -> None:
        with Path(path).open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> "ECDF":
        return cls(data["values"], p_clip=data.get("p_clip", 1e-6))

    @classmethod
    def load_json(cls, path: str | Path) -> "ECDF":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))
