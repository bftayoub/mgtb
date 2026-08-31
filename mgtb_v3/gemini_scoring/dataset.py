from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from mgtb_v3.science_fast.io import load_json, sha256_json

from .config import ScoringConfig
from .controls import deterministic_verdict, simple_number


def unit_id(content_sha256: str, variant: str, seed: int) -> str:
    return hashlib.sha256(f"{content_sha256}\0{variant}\0{seed}".encode()).hexdigest()


def _authenticated_json(path: Path, hash_field: str) -> dict[str, Any]:
    row = load_json(path)
    expected = row.get(hash_field)
    actual = sha256_json({key: value for key, value in row.items() if key != hash_field})
    if expected != actual:
        raise ValueError(f"invalid {hash_field} in {path}")
    return row


def load_candidates(config: ScoringConfig) -> list[dict[str, Any]]:
    manifest = load_json(config.manifest)
    expected_manifest = sha256_json({k: v for k, v in manifest.items() if k != "manifest_sha256"})
    if manifest.get("manifest_sha256") != expected_manifest:
        raise ValueError("Omni-MATH manifest hash mismatch")
    problems = {row["content_sha256"]: row for row in manifest["roles"]["test"]}
    rows: list[dict[str, Any]] = []
    for variant in config.variants:
        generation_dir = config.source_root / "runs" / "test" / variant / "items"
        if not generation_dir.is_dir():
            raise ValueError(f"missing generation directory: {generation_dir}")
        for path in sorted(generation_dir.glob("*.json")):
            generation = _authenticated_json(path, "artifact_sha256")
            content_hash = generation["content_sha256"]
            if content_hash not in problems:
                raise ValueError(f"generation outside frozen test manifest: {path}")
            item = problems[content_hash]
            seed = int(generation["replicate_seed"])
            if seed not in (0, 1, 2):
                raise ValueError(f"unexpected replicate seed in {path}")
            if generation.get("variant") != variant:
                raise ValueError(f"variant/path mismatch in {path}")
            scorer = generation.get("scorer", {})
            candidate_answer = scorer.get("prediction_answer")
            reference = str(item["reference_answer"])
            old = None
            old_path = config.source_root / "judging" / "test" / variant / "items" / path.name
            if old_path.is_file():
                old_row = _authenticated_json(old_path, "judgment_sha256")
                if old_row.get("generation_artifact_sha256") != generation["artifact_sha256"]:
                    raise ValueError(f"Omni-Judge points to another generation: {old_path}")
                old = old_row.get("scorer", {}).get("equivalence_judgement")
            rows.append({
                "unit_id": unit_id(content_hash, variant, seed),
                "item_id": generation["item_id"], "source_item_id": generation["source_item_id"],
                "content_sha256": content_hash, "variant": variant, "replicate_seed": seed,
                "problem": item["problem"], "reference_answer": reference,
                "candidate_answer": candidate_answer, "domains": item.get("domains", []),
                "generation_artifact_sha256": generation["artifact_sha256"],
                "old_omni_verdict": old,
                "control": deterministic_verdict(candidate_answer, reference),
            })
    expected = len(problems) * 3 * len(config.variants)
    if len(rows) != expected:
        raise ValueError(f"generation set incomplete: {len(rows)}/{expected}")
    if len({row["unit_id"] for row in rows}) != expected:
        raise ValueError("duplicate generation units")
    return rows


def select_pilot(candidates: list[dict[str, Any]], counts: dict[str, int]) -> list[dict[str, Any]]:
    def stable(rows: list[dict[str, Any]], label: str) -> list[dict[str, Any]]:
        return sorted(rows, key=lambda row: hashlib.sha256(f"pilot:{label}:{row['unit_id']}".encode()).hexdigest())

    numeric = [
        row for row in candidates
        if row["control"]["rule"] == "different_simple_numbers" and row["old_omni_verdict"] == "TRUE"
    ]
    # The campaign audit identified 98 cases. Select immutably from the authenticated
    # superset so reruns do not depend on filesystem traversal order.
    numeric = stable(numeric, "numeric")[:counts["numeric_contradictions"]]
    equal = [
        row for row in candidates if row["control"]["rule"] == "certain_normalized_equality"
        and row["unit_id"] not in {value["unit_id"] for value in numeric}
    ]
    equal = stable(equal, "equal")[:counts["manifest_equalities"]]
    undecided = [row for row in candidates if row["control"]["verdict"] is None]
    # Round-robin over first-domain strata, with stable ordering inside each stratum.
    strata: dict[str, list[dict[str, Any]]] = {}
    for row in undecided:
        domain = str((row.get("domains") or ["unclassified"])[0])
        strata.setdefault(domain, []).append(row)
    for domain in strata:
        strata[domain] = stable(strata[domain], f"symbolic:{domain}")
    symbolic: list[dict[str, Any]] = []
    domains = sorted(strata)
    while len(symbolic) < counts["symbolic"] and any(strata.values()):
        for domain in domains:
            if strata[domain] and len(symbolic) < counts["symbolic"]:
                symbolic.append(strata[domain].pop(0))
    selected = numeric + equal + symbolic
    expected = sum(counts.values())
    if len(numeric) != counts["numeric_contradictions"] or len(equal) != counts["manifest_equalities"]:
        raise ValueError("authenticated inputs cannot reproduce requested pilot strata")
    if len(selected) != expected or len({row["unit_id"] for row in selected}) != expected:
        raise ValueError("pilot selection is incomplete or duplicated")
    labels = (["numeric_contradictions"] * len(numeric) + ["manifest_equalities"] * len(equal)
              + ["symbolic"] * len(symbolic))
    return [{**row, "pilot_stratum": label} for row, label in zip(selected, labels)]
