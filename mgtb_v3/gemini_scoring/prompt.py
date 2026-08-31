from __future__ import annotations

import hashlib
from typing import Any

from mgtb_v3.science_fast.io import canonical_json, sha256_json

from .blinding import assert_blind_payload


SYSTEM_INSTRUCTION = """You are a rigorous mathematical answer-equivalence judge.
Evaluate every candidate independently against the problem and reference answer. Other
candidates are batching only: never vote, rank, infer consensus, or compare candidates.
Return TRUE only when equivalence is mathematically established, FALSE when a material
contradiction or omission is established, and ABSTAIN when the supplied final answers are
insufficient to decide. Keep each reason concise. Do not require identical notation."""
USER_INSTRUCTION = (
    "Judge each candidate independently. Use only its final answer unless the final answer "
    "is explicitly insufficient; in that case return ABSTAIN."
)


def prompt_hash() -> str:
    return sha256_json({
        "system_instruction": SYSTEM_INSTRUCTION,
        "user_instruction": USER_INSTRUCTION,
        "payload_template": ["problem", "reference_answer", "candidates", "instruction"],
    })


def build_payload(problem: str, reference: str, candidates: list[dict[str, str]]) -> dict[str, Any]:
    payload = {
        "problem": str(problem),
        "reference_answer": str(reference),
        "candidates": candidates,
        "instruction": USER_INSTRUCTION,
    }
    assert_blind_payload(payload)
    return payload


def request_hash(model: str, payload: dict[str, Any]) -> str:
    return sha256_json({"model": model, "system_instruction": SYSTEM_INSTRUCTION, "payload": payload})


def candidate_cache_key(model: str, problem: str, reference: str, candidate: str) -> str:
    return sha256_json({
        "model": model, "prompt_hash": prompt_hash(), "problem": problem,
        "reference": reference, "candidate": candidate,
    })


def serialized_contents(payload: dict[str, Any]) -> str:
    return canonical_json(payload)
