from __future__ import annotations

import math
from collections import Counter
from typing import Any

import numpy as np


def paired_analysis(vanilla: list[dict[str, Any]], mgtb: list[dict[str, Any]], bootstrap_seed: int = 20260811, samples: int = 10000) -> dict[str, Any]:
    left = {row["item_id"]: row for row in vanilla}
    right = {row["item_id"]: row for row in mgtb}
    if left.keys() != right.keys():
        raise ValueError("paired analysis requires identical item IDs")
    ids = sorted(left)
    if not ids:
        raise ValueError("paired analysis requires artifacts")
    v = np.array([bool(left[i]["scorer"]["correct"]) for i in ids], dtype=np.int8)
    m = np.array([bool(right[i]["scorer"]["correct"]) for i in ids], dtype=np.int8)
    corrections = int(np.sum((v == 0) & (m == 1)))
    regressions = int(np.sum((v == 1) & (m == 0)))
    discordant = corrections + regressions
    mcnemar = _exact_two_sided_binomial(min(corrections, regressions), discordant)
    differences = (m - v).astype(float)
    rng = np.random.default_rng(int(bootstrap_seed))
    draws = rng.choice(differences, size=(int(samples), len(ids)), replace=True).mean(axis=1)
    ci = np.quantile(draws, [0.025, 0.975]).tolist()
    return {
        "paired_items": len(ids),
        "vanilla_accuracy": float(v.mean()),
        "mgtb_accuracy": float(m.mean()),
        "difference": float(differences.mean()),
        "corrections": corrections,
        "regressions": regressions,
        "mcnemar_exact_two_sided_p": mcnemar,
        "paired_bootstrap_95_ci": ci,
        "bootstrap_samples": int(samples),
        "bootstrap_seed": int(bootstrap_seed),
        "vanilla": _method_summary(vanilla),
        "mgtb": _method_summary(mgtb),
    }


def _method_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    accounts = [row.get("token_accounting", {}) for row in rows]
    alarm_positions = [pos for account in accounts for pos in account.get("alarm_positions", [])]
    rollback_lengths = [span for account in accounts for span in account.get("rollback_spans", [])]
    peaks = [row.get("timing", {}).get("peak_vram_bytes") for row in rows]
    peaks = [value for value in peaks if value is not None]
    terminations = Counter(row.get("token_accounting", {}).get("termination_reason", "unknown") for row in rows)
    return {
        "extractability": _mean(bool(row.get("scorer", {}).get("answer_extraction_ok")) for row in rows),
        "truncation_rate": _mean(bool(row.get("truncated")) for row in rows),
        "items_with_alarm_rate": _mean(int(account.get("alarms", 0)) > 0 for account in accounts),
        "items_with_reroll_rate": _mean(int(account.get("rerolls", 0)) > 0 for account in accounts),
        "sampled_tokens": sum(int(account.get("sampled", 0)) for account in accounts),
        "emitted_tokens": sum(int(account.get("emitted", 0)) for account in accounts),
        "deleted_tokens": sum(int(account.get("deleted", 0)) for account in accounts),
        "alarm_positions": alarm_positions,
        "rollback_lengths": rollback_lengths,
        "mean_latency_seconds": _mean(float(row.get("timing", {}).get("wall_seconds", 0.0)) for row in rows),
        "peak_vram_bytes": max(peaks) if peaks else None,
        "termination_reasons": dict(terminations),
        "items": n,
    }


def _mean(values) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else math.nan


def _exact_two_sided_binomial(k: int, n: int) -> float:
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, int(k) + 1)) / (2**n)
    return float(min(1.0, 2.0 * tail))
