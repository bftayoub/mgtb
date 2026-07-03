from __future__ import annotations

import math
from typing import Any


def _is_torch_tensor(x: Any) -> bool:
    return x.__class__.__module__.startswith("torch")


def entropy_from_logits(logits) -> float:
    """Entropy of the model distribution from pre-sampling logits."""
    if _is_torch_tensor(logits):
        import torch

        vector = logits.detach().float().reshape(-1)
        log_probs = torch.nn.functional.log_softmax(vector, dim=-1)
        probs = log_probs.exp()
        value = -(probs * log_probs).sum()
        return float(torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0).item())

    import numpy as np

    vector = np.asarray(logits, dtype=np.float64).reshape(-1)
    max_logit = np.max(vector)
    shifted = vector - max_logit
    exp = np.exp(shifted)
    probs = exp / np.sum(exp)
    log_probs = shifted - math.log(float(np.sum(exp)))
    value = -float(np.sum(probs * log_probs))
    return value if math.isfinite(value) else 0.0


def chosen_logprob_from_logits(logits, token_id: int) -> float:
    """Log-probability of the sampled token under the pre-sampling model distribution."""
    if _is_torch_tensor(logits):
        import torch

        vector = logits.detach().float().reshape(-1)
        log_probs = torch.nn.functional.log_softmax(vector, dim=-1)
        value = log_probs[int(token_id)]
        return float(torch.nan_to_num(value, nan=-1e9, posinf=0.0, neginf=-1e9).item())

    import numpy as np

    vector = np.asarray(logits, dtype=np.float64).reshape(-1)
    max_logit = np.max(vector)
    shifted = vector - max_logit
    log_z = math.log(float(np.sum(np.exp(shifted))))
    value = float(shifted[int(token_id)] - log_z)
    return value if math.isfinite(value) else -1e9
