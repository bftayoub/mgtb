from __future__ import annotations


def occurrence_mean_logprob(logprobs: list[float], start: int, end: int) -> float:
    values = logprobs[start:end]
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def confident_loop_delta(current_mean: float, previous_means: list[float]) -> float:
    if not previous_means:
        return 0.0
    past_mean = sum(previous_means) / len(previous_means)
    return max(0.0, float(current_mean - past_mean))
