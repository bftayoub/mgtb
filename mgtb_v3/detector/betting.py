from __future__ import annotations

import math


def _clip_p(p: float, p_clip: float) -> float:
    return min(1.0, max(float(p_clip), float(p)))


def beta_betting(p: float, gamma: float, p_clip: float = 1e-6) -> float:
    p = _clip_p(p, p_clip)
    gamma = float(gamma)
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    value = gamma * (p ** (gamma - 1.0))
    return float(value if math.isfinite(value) and value > 0 else 1.0)


def mixture_betting(
    p: float,
    gammas: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7),
    p_clip: float = 1e-6,
) -> float:
    values = [beta_betting(p, gamma, p_clip=p_clip) for gamma in gammas]
    return float(sum(values) / len(values))


def log_mixture_betting(
    p: float,
    gammas: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7),
    p_clip: float = 1e-6,
) -> float:
    logs = [math.log(beta_betting(p, gamma, p_clip=p_clip)) for gamma in gammas]
    m = max(logs)
    return float(m + math.log(sum(math.exp(v - m) for v in logs)) - math.log(len(logs)))
