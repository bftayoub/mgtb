from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class TokenStats:
    token_id: int
    position: int
    entropy: float
    logprob: float


@dataclass
class NgramOccurrence:
    ngram: tuple[int, ...]
    start_pos: int
    end_pos: int
    mean_logprob: float


@dataclass
class WindowFeatures:
    window_index: int
    start_pos: int
    end_pos: int
    mean_entropy: float
    mean_logprob: float
    repetition_rate: float
    confident_loop_score: float
    local_entropy_log_ratio: float
    local_entropy_pos: float
    local_entropy_neg: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class WindowScore:
    features: WindowFeatures
    raw_score: float
    p_value: Optional[float] = None
    e_value: Optional[float] = None
    logE: Optional[float] = None
    alert: bool = False


@dataclass
class AlertInfo:
    window_index: int
    token_pos: int
    changepoint_window: int
    rollback_token_pos: int
    score: float
    p_value: float
    logE: float


@dataclass
class GenerationResult:
    text: str
    tokens: list[int]
    alerts: list[AlertInfo] = field(default_factory=list)
    backtracks: list[dict[str, Any]] = field(default_factory=list)
    trace_log_path: Optional[str] = None
    sampled_tokens: int = 0
    emitted_tokens: int = 0
    deleted_tokens: int = 0
    termination_reason: str = "unknown"
    latency: float = 0.0
