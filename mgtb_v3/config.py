from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class WindowConfig:
    window_size: int = 64
    stride: int = 32
    entropy_eps: float = 1e-8
    ngram_min: int = 6
    ngram_max: int = 8
    exclude_prompt_ngrams: bool = True


@dataclass
class DetectorConfig:
    target_false_alert_rate: float = 0.05
    threshold: Optional[float] = None
    p_clip: float = 1e-6
    betting_gammas: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7)
    refractory_windows: int = 2


@dataclass
class BacktrackingConfig:
    max_rerolls: int = 3
    margin_tokens: int = 64
    fixed_rollback_tokens: Optional[int] = None
    use_adaptive_changepoint: bool = True
    redecode_temperature: float = 0.6
    repetition_penalty: float = 1.1
    use_no_bad_ngrams: bool = True


@dataclass
class ScoreConfig:
    w_entropy: float = 0.15
    w_logprob: float = 0.10
    w_repetition: float = 0.20
    w_confident_loop: float = 0.35
    w_local_entropy_pos: float = 0.15
    w_local_entropy_neg: float = 0.05


@dataclass
class MGTBV3Config:
    window: WindowConfig = field(default_factory=WindowConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    backtracking: BacktrackingConfig = field(default_factory=BacktrackingConfig)
    score: ScoreConfig = field(default_factory=ScoreConfig)


def _coerce_dataclass(cls: type, data: dict[str, Any]):
    kwargs: dict[str, Any] = {}
    for item in fields(cls):
        if item.name not in data:
            continue
        value = data[item.name]
        default = getattr(cls(), item.name)
        if is_dataclass(default) and isinstance(value, dict):
            kwargs[item.name] = _coerce_dataclass(type(default), value)
        elif item.name == "betting_gammas" and isinstance(value, list):
            kwargs[item.name] = tuple(float(v) for v in value)
        else:
            kwargs[item.name] = value
    return cls(**kwargs)


def load_config(path: str | Path) -> MGTBV3Config:
    import yaml

    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return _coerce_dataclass(MGTBV3Config, data)
