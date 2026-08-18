from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable

from .io import atomic_write_json, load_json, sha256_json

REQUIRED_ARTIFACT_FIELDS = {
    "item_id", "content_sha256", "item_seed", "completed", "generation", "token_ids",
    "scorer", "token_accounting", "timing", "provenance",
}


class RunStore:
    def __init__(self, root: str | Path, identity: dict[str, Any]):
        self.root = Path(root)
        self.items_dir = self.root / "items"
        self.state_path = self.root / "run_state.json"
        self.identity = dict(identity)
        self.identity_sha256 = sha256_json(self.identity)
        self.items_dir.mkdir(parents=True, exist_ok=True)
        if self.state_path.exists():
            state = load_json(self.state_path)
            if state.get("identity_sha256") != self.identity_sha256 or state.get("identity") != self.identity:
                raise ValueError("refusing resume: run manifest/config/provenance identity changed")
        else:
            self._write_state([])

    @staticmethod
    def filename(item_id: str) -> str:
        return hashlib.sha256(item_id.encode()).hexdigest() + ".json"

    def artifact_path(self, item_id: str) -> Path:
        return self.items_dir / self.filename(item_id)

    def valid_artifact(self, item: dict[str, Any]) -> dict[str, Any] | None:
        path = self.artifact_path(item["item_id"])
        if not path.exists():
            return None
        try:
            artifact = load_json(path)
        except (OSError, ValueError):
            return None
        if not REQUIRED_ARTIFACT_FIELDS <= artifact.keys() or artifact.get("completed") is not True:
            return None
        if artifact.get("item_id") != item["item_id"] or artifact.get("content_sha256") != item["content_sha256"]:
            return None
        if int(artifact.get("item_seed", -1)) != int(item["item_seed"]):
            return None
        if artifact.get("provenance", {}).get("run_identity_sha256") != self.identity_sha256:
            return None
        expected_hash = artifact.get("artifact_sha256")
        if expected_hash != sha256_json({key: value for key, value in artifact.items() if key != "artifact_sha256"}):
            return None
        return artifact

    def save(self, item: dict[str, Any], artifact: dict[str, Any]) -> None:
        complete = {
            **artifact,
            "item_id": item["item_id"],
            "content_sha256": item["content_sha256"],
            "item_seed": int(item["item_seed"]),
            "completed": True,
        }
        missing = REQUIRED_ARTIFACT_FIELDS - complete.keys()
        if missing:
            raise ValueError(f"incomplete final artifact: {sorted(missing)}")
        complete["artifact_sha256"] = sha256_json(complete)
        atomic_write_json(self.artifact_path(item["item_id"]), complete)
        completed = []
        for path in self.items_dir.glob("*.json"):
            try:
                candidate = load_json(path)
            except (OSError, ValueError):
                continue
            if candidate.get("completed") is True and candidate.get("provenance", {}).get("run_identity_sha256") == self.identity_sha256:
                completed.append(path.stem)
        completed.sort()
        self._write_state(completed)

    def run(self, items: list[dict[str, Any]], worker: Callable[[dict[str, Any]], dict[str, Any]], stop_after: int | None = None) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        newly_completed = 0
        for item in items:
            artifact = self.valid_artifact(item)
            if artifact is None:
                if stop_after is not None and newly_completed >= stop_after:
                    break
                artifact = worker(item)
                self.save(item, artifact)
                artifact = self.valid_artifact(item)
                if artifact is None:
                    raise RuntimeError("atomic artifact failed validation")
                newly_completed += 1
            output.append(artifact)
        return output

    def _write_state(self, completed: list[str]) -> None:
        atomic_write_json(self.state_path, {
            "schema_version": 1,
            "identity": self.identity,
            "identity_sha256": self.identity_sha256,
            "completed_artifact_files": completed,
            "completed_count": len(completed),
        })
