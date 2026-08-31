from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelQuota:
    model: str
    rpm: int
    tpm: int
    rpd: int
    thinking_level: str


@dataclass(frozen=True)
class ScoringConfig:
    path: Path
    source_root: Path
    manifest: Path
    output_root: Path
    variants: tuple[str, ...]
    primary: ModelQuota
    secondary: ModelQuota
    temperature: float
    max_output_tokens: int
    permutation_salt: str
    pilot_counts: dict[str, int]
    individual_audit_count: int
    secondary_audit_count: int


def _quota(raw: dict[str, Any]) -> ModelQuota:
    return ModelQuota(
        model=str(raw["model"]), rpm=int(raw["rpm"]), tpm=int(raw["tpm"]),
        rpd=int(raw["rpd"]), thinking_level=str(raw["thinking_level"]),
    )


def load_scoring_config(path: str | Path) -> ScoringConfig:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    required = {"source_root", "manifest", "output_root", "variants", "primary", "secondary"}
    missing = required - set(raw or {})
    if missing:
        raise ValueError(f"Gemini scoring config lacks fields: {sorted(missing)}")
    primary, secondary = _quota(raw["primary"]), _quota(raw["secondary"])
    if primary.model != "gemini-3.5-flash-lite" or secondary.model != "gemini-3.5-flash":
        raise ValueError("the frozen Gemini scorer requires the exact requested model names")
    if primary.rpm > 12 or primary.tpm > 200_000 or primary.rpd > 500:
        raise ValueError("primary limits exceed conservative project quotas")
    if secondary.rpm > 5 or secondary.tpm > 250_000 or secondary.rpd > 20:
        raise ValueError("secondary limits exceed project quotas")
    counts = {str(k): int(v) for k, v in raw.get("pilot_counts", {}).items()}
    if counts != {"numeric_contradictions": 98, "manifest_equalities": 50, "symbolic": 52}:
        raise ValueError("pilot must contain exactly 98 + 50 + 52 cases")
    return ScoringConfig(
        path=config_path, source_root=Path(raw["source_root"]), manifest=Path(raw["manifest"]),
        output_root=Path(raw["output_root"]), variants=tuple(raw["variants"]),
        primary=primary, secondary=secondary, temperature=float(raw.get("temperature", 0)),
        max_output_tokens=int(raw.get("max_output_tokens", 4096)),
        permutation_salt=str(raw.get("permutation_salt", "omnimath-gemini-v1")),
        pilot_counts=counts, individual_audit_count=int(raw.get("individual_audit_count", 50)),
        secondary_audit_count=int(raw.get("secondary_audit_count", 5)),
    )
