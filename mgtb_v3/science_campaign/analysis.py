from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any

import numpy as np


def method_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"items": 0}
    accounts = [row["token_accounting"] for row in rows]
    if any(row.get("scorer", {}).get("correct") is None or not row.get("scorer", {}).get("scorable", True) for row in rows):
        raise ValueError("analysis refuses missing or explicitly unscorable verdicts")
    correct = np.array([float(row["scorer"]["correct"]) for row in rows])
    by_seed: dict[int, list[float]] = defaultdict(list)
    by_subject: dict[str, list[float]] = defaultdict(list)
    by_domain: dict[str, list[float]] = defaultdict(list)
    by_difficulty: dict[str, list[float]] = defaultdict(list)
    for row, value in zip(rows, correct):
        by_seed[int(row.get("replicate_seed", 0))].append(float(value))
        by_subject[str(row.get("item_metadata", {}).get("subject") or "unknown")].append(float(value))
        metadata = row.get("item_metadata", {})
        domains = metadata.get("domains") or [metadata.get("subject") or "unknown"]
        for domain in domains:
            by_domain[str(domain)].append(float(value))
        difficulty = metadata.get("difficulty", metadata.get("level"))
        by_difficulty[str(difficulty if difficulty is not None else "unknown")].append(float(value))
    candidate_rows = [row for row in rows if row.get("candidates")]
    seed_accuracies = [float(np.mean(values)) for values in by_seed.values()]
    return {
        "items": len(rows), "accuracy": float(correct.mean()),
        "accuracy_by_replicate_seed": {str(seed): float(np.mean(values)) for seed, values in sorted(by_seed.items())},
        "accuracy_seed_std": float(np.std(seed_accuracies, ddof=1)) if len(seed_accuracies) > 1 else 0.0,
        "accuracy_by_subject": {subject: {"items": len(values), "accuracy": float(np.mean(values))}
                                for subject, values in sorted(by_subject.items())},
        "accuracy_by_domain_descriptive": {
            domain: {"items": len(values), "accuracy": float(np.mean(values))}
            for domain, values in sorted(by_domain.items())
        },
        "accuracy_by_difficulty_descriptive": {
            difficulty: {"items": len(values), "accuracy": float(np.mean(values))}
            for difficulty, values in sorted(by_difficulty.items())
        },
        "extractability": float(np.mean([bool(row["scorer"].get("answer_extraction_ok")) for row in rows])),
        "truncation_rate": float(np.mean([bool(row.get("truncated")) for row in rows])),
        "items_with_alarm_rate": float(np.mean([int(a.get("alarms", 0)) > 0 for a in accounts])),
        "items_with_reroll_rate": float(np.mean([int(a.get("rerolls", 0)) > 0 for a in accounts])),
        "sampled_tokens": sum(int(a.get("sampled", 0)) for a in accounts),
        "emitted_tokens": sum(int(a.get("emitted", 0)) for a in accounts),
        "deleted_tokens": sum(int(a.get("deleted", 0)) for a in accounts),
        "mean_sampled_tokens": float(np.mean([int(a.get("sampled", 0)) for a in accounts])),
        "mean_latency_seconds": float(np.mean([float(row.get("timing", {}).get("wall_seconds", 0)) for row in rows])),
        "peak_vram_bytes": max([int(row.get("timing", {}).get("peak_vram_bytes") or 0) for row in rows] or [0]),
        "evaluated_generation_cost": {
            "sampled_tokens": sum(int(a.get("sampled", 0)) for a in accounts),
            "wall_seconds": sum(float(row.get("timing", {}).get("wall_seconds", 0.0)) for row in rows),
            "api_cost_usd": 0.0,
            "cost_note": "local model inference; hardware cost is reported as time/tokens and is not monetized",
        },
        "judge_cost": {
            "sampled_tokens": sum(int(row.get("judge", {}).get("judge_cost", {}).get("sampled_tokens", 0)) for row in rows),
            "wall_seconds": sum(float(row.get("judge", {}).get("judge_cost", {}).get("wall_seconds", 0.0)) for row in rows),
            "api_cost_usd": sum(float(row.get("judge", {}).get("judge_cost", {}).get("api_cost_usd", 0.0)) for row in rows),
            "separate_from_evaluated_generation": True,
        },
        "termination_reasons": dict(Counter(a.get("termination_reason", "unknown") for a in accounts)),
        "candidate_oracle_pass_rate": (
            float(np.mean([any(bool(candidate["scorer"].get("correct")) for candidate in row["candidates"])
                           for row in candidate_rows])) if candidate_rows else None
        ),
    }


def _exact_mcnemar(corrections: int, regressions: int) -> float:
    n = corrections + regressions
    if not n:
        return 1.0
    k = min(corrections, regressions)
    return float(min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / (2**n)))


def paired_comparison(baseline: list[dict[str, Any]], method: list[dict[str, Any]], *, seed: int, samples: int) -> dict[str, Any]:
    left = {(row["source_item_id"], int(row.get("replicate_seed", 0))): row for row in baseline}
    right = {(row["source_item_id"], int(row.get("replicate_seed", 0))): row for row in method}
    if left.keys() != right.keys():
        missing_left, missing_right = sorted(right.keys() - left.keys()), sorted(left.keys() - right.keys())
        raise ValueError(f"unpaired runs: missing baseline={missing_left[:3]} missing method={missing_right[:3]}")
    keys = sorted(left)
    v = np.array([bool(left[key]["scorer"]["correct"]) for key in keys], dtype=np.int8)
    m = np.array([bool(right[key]["scorer"]["correct"]) for key in keys], dtype=np.int8)
    diff = (m - v).astype(float)
    corrections = int(np.sum((v == 0) & (m == 1)))
    regressions = int(np.sum((v == 1) & (m == 0)))
    # Replicate seeds from one problem are dependent. Bootstrap problem clusters,
    # then average all seeds within each resampled cluster.
    cluster_values: dict[str, list[float]] = defaultdict(list)
    for key, value in zip(keys, diff):
        cluster_values[key[0]].append(float(value))
    cluster_means = np.array([np.mean(cluster_values[item]) for item in sorted(cluster_values)])
    rng = np.random.default_rng(int(seed))
    draws = rng.choice(cluster_means, size=(int(samples), len(cluster_means)), replace=True).mean(axis=1)
    no_alarm_keys = [key for key in keys if int(right[key]["token_accounting"].get("alarms", 0)) == 0]
    identical_no_alarm = sum(left[key].get("token_ids") == right[key].get("token_ids") for key in no_alarm_keys)
    return {
        "paired_units": len(keys), "baseline_accuracy": float(v.mean()), "method_accuracy": float(m.mean()),
        "problem_clusters": len(cluster_means),
        "difference": float(diff.mean()), "corrections": corrections, "regressions": regressions,
        "mcnemar_exact_two_sided_p": _exact_mcnemar(corrections, regressions),
        "paired_cluster_bootstrap_95_ci": np.quantile(draws, [0.025, 0.975]).tolist(),
        "bootstrap_samples": int(samples), "bootstrap_seed": int(seed),
        "mcnemar_note": "Exact unit-level McNemar; with multiple seeds, units within a problem are dependent. Use the problem-cluster bootstrap as primary inference.",
        "no_alarm_units": len(no_alarm_keys),
        "token_identical_no_alarm_rate": identical_no_alarm / len(no_alarm_keys) if no_alarm_keys else None,
    }


def _holm_adjust(comparisons: dict[str, dict[str, Any]]) -> None:
    ordered = sorted(comparisons, key=lambda name: comparisons[name]["mcnemar_exact_two_sided_p"])
    running = 0.0
    count = len(ordered)
    for rank, name in enumerate(ordered):
        raw = comparisons[name]["mcnemar_exact_two_sided_p"]
        running = max(running, min(1.0, (count - rank) * raw))
        comparisons[name]["mcnemar_holm_adjusted_p"] = running


def campaign_analysis(runs: dict[str, list[dict[str, Any]]], *, baseline: str,
                      bootstrap_seed: int = 20260811, bootstrap_samples: int = 10000,
                      require_no_alarm_identity: list[str] | tuple[str, ...] = ()) -> dict[str, Any]:
    if baseline not in runs:
        raise ValueError(f"analysis baseline {baseline!r} has no run")
    summaries = {name: method_summary(rows) for name, rows in runs.items()}
    comparisons = {
        name: paired_comparison(runs[baseline], rows, seed=bootstrap_seed, samples=bootstrap_samples)
        for name, rows in runs.items() if name != baseline
    }
    for name in require_no_alarm_identity:
        if name not in comparisons:
            raise ValueError(f"required no-alarm identity variant has no baseline comparison: {name}")
        comparison = comparisons[name]
        if comparison["no_alarm_units"] and comparison["token_identical_no_alarm_rate"] != 1.0:
            raise ValueError(f"{name} diverged token-by-token from {baseline} on a no-alarm trajectory")
    _holm_adjust(comparisons)
    return {
        "baseline": baseline, "methods": summaries, "comparisons": comparisons,
        "multiplicity": "Holm adjustment across McNemar comparisons against the declared baseline",
        "primary_inference": "paired bootstrap clustered by source problem; replicate seeds stay within clusters",
        "subgroup_note": "domain and difficulty summaries are descriptive unless separately preregistered",
    }
