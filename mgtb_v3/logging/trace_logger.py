from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


class TraceLogger:
    def __init__(self, path: str | Path | None):
        self.path = Path(path) if path else None
        self._handle = None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self.path.open("w", encoding="utf-8")

    def log(self, event: dict[str, Any]) -> None:
        if self._handle is None:
            return
        self._handle.write(json.dumps(_jsonable(event), ensure_ascii=False) + "\n")

    def log_token(self, pos: int, token_id: int, entropy: float | None, logprob: float | None) -> None:
        self.log({"type": "token", "pos": pos, "token_id": token_id, "entropy": entropy, "logprob": logprob})

    def log_window(self, window_score) -> None:
        features = window_score.features.to_dict()
        self.log(
            {
                "type": "window",
                "window_index": window_score.features.window_index,
                "start_pos": window_score.features.start_pos,
                "end_pos": window_score.features.end_pos,
                "features": features,
                "score": window_score.raw_score,
                "p_value": window_score.p_value,
                "e_value": window_score.e_value,
                "logE": window_score.logE,
                "alert": window_score.alert,
            }
        )

    def log_backtrack(self, event: dict[str, Any]) -> None:
        self.log({"type": "backtrack", **event})

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def _jsonable(value):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items() if k not in {"cache"}}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value
