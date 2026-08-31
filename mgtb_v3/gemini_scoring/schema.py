from __future__ import annotations

import json
from typing import Any


VERDICTS = {"TRUE", "FALSE", "ABSTAIN"}
RESULT_KEYS = {
    "candidate_id", "verdict", "normalized_candidate", "normalized_reference", "reason",
}
JUDGE_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["results"],
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": sorted(RESULT_KEYS),
                "properties": {
                    "candidate_id": {"type": "string"},
                    "verdict": {"type": "string", "enum": sorted(VERDICTS)},
                    "normalized_candidate": {"type": "string"},
                    "normalized_reference": {"type": "string"},
                    "reason": {"type": "string"},
                },
            },
        }
    },
}


def validate_judge_response(payload: str | dict[str, Any], expected_ids: list[str]) -> list[dict[str, str]]:
    if isinstance(payload, str):
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError("judge response is not valid JSON") from exc
    else:
        parsed = payload
    if not isinstance(parsed, dict) or set(parsed) != {"results"} or not isinstance(parsed["results"], list):
        raise ValueError("judge response must contain only a results array")
    results = parsed["results"]
    if len(results) != len(expected_ids):
        raise ValueError(f"judge response count mismatch: {len(results)} != {len(expected_ids)}")
    seen: list[str] = []
    for index, row in enumerate(results):
        if not isinstance(row, dict) or set(row) != RESULT_KEYS:
            raise ValueError(f"judge result {index} has missing or extra fields")
        if not all(isinstance(row[key], str) for key in RESULT_KEYS):
            raise ValueError(f"judge result {index} contains non-string fields")
        if row["verdict"] not in VERDICTS:
            raise ValueError(f"judge result {index} has invalid verdict")
        if not row["candidate_id"] or not row["reason"].strip():
            raise ValueError(f"judge result {index} has empty identifier or reason")
        seen.append(row["candidate_id"])
    if len(set(seen)) != len(seen):
        raise ValueError("judge response contains duplicate candidate identifiers")
    if set(seen) != set(expected_ids):
        raise ValueError("judge response contains missing or unexpected candidate identifiers")
    by_id = {row["candidate_id"]: row for row in results}
    return [by_id[candidate_id] for candidate_id in expected_ids]
