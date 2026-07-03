from __future__ import annotations

import json
from pathlib import Path

from mgtb_v3.calibration.ecdf import ECDF

DEFAULT_BUCKETS: list[tuple[int, int | None]] = [(0, 512), (512, 1024), (1024, 2048), (2048, 4096), (4096, None)]


def bucket_name(bucket: tuple[int, int | None]) -> str:
    start, end = bucket
    return f"{start}-{end if end is not None else 'inf'}"


class PositionalCalibrator:
    def __init__(self, buckets=None, score_pools_by_bucket=None, p_clip: float = 1e-6):
        self.buckets = list(buckets or DEFAULT_BUCKETS)
        self.p_clip = float(p_clip)
        pools = score_pools_by_bucket or {}
        self.ecdfs: dict[str, ECDF] = {}
        for bucket in self.buckets:
            name = bucket_name(bucket)
            values = pools.get(name, pools.get(str(bucket), []))
            if values:
                self.ecdfs[name] = ECDF(values, p_clip=p_clip)

    def bucket_for_position(self, token_pos: int) -> str:
        for bucket in self.buckets:
            start, end = bucket
            if int(token_pos) >= start and (end is None or int(token_pos) < end):
                return bucket_name(bucket)
        return bucket_name(self.buckets[-1])

    def p_value(self, score: float, token_pos: int) -> float:
        name = self.bucket_for_position(token_pos)
        if name in self.ecdfs:
            return self.ecdfs[name].tail_pvalue(score)
        if self.ecdfs:
            pooled = []
            for ecdf in self.ecdfs.values():
                pooled.extend(ecdf.values)
            return ECDF(pooled, p_clip=self.p_clip).tail_pvalue(score)
        raise ValueError("no calibration pools available")

    def save_json(self, path: str | Path) -> None:
        data = {
            "buckets": self.buckets,
            "p_clip": self.p_clip,
            "score_pools_by_bucket": {name: ecdf.values for name, ecdf in self.ecdfs.items()},
        }
        with Path(path).open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)

    @classmethod
    def load_json(cls, path: str | Path):
        with Path(path).open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        buckets = [(int(start), None if end is None else int(end)) for start, end in data["buckets"]]
        return cls(buckets, data.get("score_pools_by_bucket", {}), p_clip=data.get("p_clip", 1e-6))
