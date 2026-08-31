from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from mgtb_v3.science_fast.io import atomic_write_json, load_json, sha256_json


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def quota_day() -> str:
    """Gemini RPD resets at midnight in the US Pacific time zone."""
    return datetime.now(ZoneInfo("America/Los_Angeles")).date().isoformat()


class ArtifactStore:
    def __init__(self, root: Path):
        self.root = root
        self.responses = root / "api_responses"
        self.results = root / "results"
        self.decisions = root / "decisions"
        self.errors = root / "errors"
        self.state = root / "state"

    @staticmethod
    def _valid(path: Path, hash_field: str) -> dict[str, Any] | None:
        if not path.is_file():
            return None
        try:
            row = load_json(path)
        except (OSError, ValueError):
            return None
        expected = row.get(hash_field)
        actual = sha256_json({key: value for key, value in row.items() if key != hash_field})
        return row if expected == actual else None

    @staticmethod
    def _immutable_write(path: Path, row: dict[str, Any], hash_field: str) -> dict[str, Any]:
        payload = dict(row)
        payload[hash_field] = sha256_json(payload)
        existing = ArtifactStore._valid(path, hash_field)
        if existing is not None:
            if existing != payload:
                raise ValueError(f"refusing to replace immutable artifact: {path}")
            return existing
        if path.exists():
            raise ValueError(f"refusing to replace invalid artifact: {path}")
        atomic_write_json(path, payload)
        return payload

    def save_response(self, request_hash: str, row: dict[str, Any]) -> dict[str, Any]:
        return self._immutable_write(self.responses / f"{request_hash}.json", row, "response_artifact_sha256")

    def valid_response(self, request_hash: str, identity_sha256: str) -> dict[str, Any] | None:
        row = self._valid(self.responses / f"{request_hash}.json", "response_artifact_sha256")
        if not row:
            return None
        identity = row.get("judge_identity", {})
        return row if identity.get("judge_identity_sha256") == identity_sha256 else None

    def save_result(self, cache_key: str, row: dict[str, Any]) -> dict[str, Any]:
        return self._immutable_write(self.results / f"{cache_key}.json", row, "result_artifact_sha256")

    def valid_result(self, cache_key: str, identity_sha256: str) -> dict[str, Any] | None:
        row = self._valid(self.results / f"{cache_key}.json", "result_artifact_sha256")
        return row if row and row.get("judge_identity_sha256") == identity_sha256 else None

    def save_decision(self, unit_id: str, row: dict[str, Any]) -> dict[str, Any]:
        return self._immutable_write(self.decisions / f"{unit_id}.json", row, "decision_sha256")

    def valid_decision(self, unit_id: str) -> dict[str, Any] | None:
        return self._valid(self.decisions / f"{unit_id}.json", "decision_sha256")

    def save_error(self, request_hash: str, attempt: int, row: dict[str, Any]) -> None:
        # Error records are immutable and deliberately contain no exception repr that could
        # accidentally echo client configuration or credentials.
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        self._immutable_write(
            self.errors / f"{request_hash}.{attempt:03d}.{stamp}.json", row, "error_artifact_sha256",
        )

    def ledger(self, model: str) -> dict[str, Any]:
        day = quota_day()
        path = self.state / f"quota-{day}-{model}.json"
        if not path.is_file():
            return {"day_pacific": day, "model": model, "requests": 0, "input_tokens": 0,
                    "output_tokens": 0, "retries": 0, "temporary_errors": 0}
        return load_json(path)

    def update_ledger(self, model: str, **increments: int) -> dict[str, Any]:
        row = self.ledger(model)
        for key, value in increments.items():
            row[key] = int(row.get(key, 0)) + int(value)
        row["updated_at"] = utc_now()
        atomic_write_json(self.state / f"quota-{quota_day()}-{model}.json", row)
        return row

    def checkpoint(self, row: dict[str, Any]) -> None:
        payload = {**row, "updated_at": utc_now()}
        payload["checkpoint_sha256"] = sha256_json(payload)
        atomic_write_json(self.state / "checkpoint.json", payload)
