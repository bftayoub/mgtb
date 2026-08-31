from __future__ import annotations

import hashlib
import random
from typing import Any, Iterable


FORBIDDEN_PAYLOAD_TERMS = ("vanilla", "full_mgtb", "matched_random", "omni-judge", "mgtb")


def deterministic_order(candidates: Iterable[dict[str, Any]], problem_hash: str, salt: str) -> list[dict[str, Any]]:
    rows = list(candidates)
    seed = int(hashlib.sha256(f"{salt}\0{problem_hash}".encode()).hexdigest(), 16)
    random.Random(seed).shuffle(rows)
    return rows


def anonymize_candidates(
    candidates: Iterable[dict[str, Any]], problem_hash: str, salt: str,
) -> tuple[list[dict[str, str]], dict[str, str]]:
    ordered = deterministic_order(candidates, problem_hash, salt)
    public: list[dict[str, str]] = []
    mapping: dict[str, str] = {}
    for index, candidate in enumerate(ordered):
        anonymous_id = f"C{index + 1:02d}_{hashlib.sha256(f'{salt}:{problem_hash}:{index}'.encode()).hexdigest()[:8]}"
        mapping[anonymous_id] = str(candidate["unit_id"])
        public.append({"candidate_id": anonymous_id, "answer": str(candidate["candidate_answer"])})
    return public, mapping


def assert_blind_payload(payload: dict[str, Any]) -> None:
    # Inspect keys and values. Problem text itself could theoretically contain a forbidden
    # word, so reject only provenance-shaped leakage and exact variant marker occurrences.
    flattened = repr(payload).lower()
    for term in FORBIDDEN_PAYLOAD_TERMS:
        if term in flattened:
            raise ValueError(f"blind payload contains forbidden term: {term}")
    forbidden_keys = {"variant", "seed", "replicate_seed", "old_verdict", "metrics", "scorer"}
    stack: list[Any] = [payload]
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            if forbidden_keys & set(value):
                raise ValueError("blind payload contains forbidden provenance keys")
            stack.extend(value.values())
        elif isinstance(value, list):
            stack.extend(value)
