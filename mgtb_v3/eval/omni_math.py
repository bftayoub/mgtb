from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from pathlib import Path
from typing import Any

from mgtb_v3.eval.math500 import extract_model_answer, format_math500_prompt
from mgtb_v3.science_fast.io import sha256_json
from mgtb_v3.science_fast.protocol import content_sha256


OFFICIAL_REPOSITORY = "KbsdJames/Omni-MATH"
OFFICIAL_DATA_PATH = "Omni-Math.jsonl"
OFFICIAL_JUDGE_MODEL = "KbsdJames/Omni-Judge"
OMNI_MATH_PROMPT_STYLE = "math500_cot"


def format_omni_math_prompt(problem: str, prompt_style: str = OMNI_MATH_PROMPT_STYLE) -> str:
    """Keep the established campaign math prompt unchanged for Omni-MATH."""
    return format_math500_prompt(problem, prompt_style)


def pending_omni_math_score(text: str | None, reference_answer: str | None) -> dict[str, Any]:
    """Record extraction, but never pretend textual equality is an Omni-MATH verdict."""
    prediction = extract_model_answer(text)
    return {
        "reference_answer": reference_answer,
        "prediction_answer": prediction,
        "correct": None,
        "scorable": False,
        "answer_extraction_ok": prediction is not None,
        "reference_extraction_ok": bool(str(reference_answer or "").strip()),
        "method": "pending_official_omni_judge",
    }


def git_blob_sha1(payload: bytes) -> str:
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload).hexdigest()


def load_official_omni_math(spec: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    repository = str(spec.get("repository", ""))
    revision = str(spec.get("revision", ""))
    source_path = str(spec.get("path", ""))
    expected_blob = str(spec.get("git_blob_sha1", ""))
    if repository != OFFICIAL_REPOSITORY or source_path != OFFICIAL_DATA_PATH:
        raise ValueError("omni_math requires the official KbsdJames/Omni-MATH/Omni-Math.jsonl source")
    if not re.fullmatch(r"[0-9a-f]{40}", revision) or not re.fullmatch(r"[0-9a-f]{40}", expected_blob):
        raise ValueError("omni_math source revision and git_blob_sha1 must be immutable 40-hex identifiers")

    local_path = spec.get("local_path")
    if local_path:
        payload = Path(local_path).read_bytes()
        source_url = None
    else:
        source_url = f"https://raw.githubusercontent.com/{repository}/{revision}/{source_path}"
        with urllib.request.urlopen(source_url) as response:  # nosec: URL is restricted above and revision-pinned
            payload = response.read()
    actual_blob = git_blob_sha1(payload)
    if actual_blob != expected_blob:
        raise ValueError(f"Omni-MATH git blob mismatch: {actual_blob} != {expected_blob}")

    rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(payload.decode("utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        try:
            row = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid Omni-MATH JSON on line {line_number}") from exc
        missing = [key for key in ("problem", "answer", "domain", "difficulty", "source") if key not in row]
        if missing:
            raise ValueError(f"Omni-MATH line {line_number} lacks provenance fields: {missing}")
        domains = row["domain"]
        # The pinned official file contains one otherwise valid row whose
        # domain list is empty (line 2067).  Keep it authenticated here so the
        # manifest builder can exclude and account for it explicitly instead
        # of making the whole immutable source unloadable.
        if not isinstance(domains, list) or not all(str(value).strip() for value in domains):
            raise ValueError(f"Omni-MATH line {line_number} has invalid domain provenance")
        try:
            difficulty = float(row["difficulty"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Omni-MATH line {line_number} has invalid difficulty") from exc
        rows.append({
            **row,
            "domain": [str(value) for value in domains],
            "difficulty": difficulty,
            "_source_line": line_number,
        })
    if not rows:
        raise ValueError("official Omni-MATH source is empty")
    authentication = {
        "kind": "github_jsonl",
        "repository": repository,
        "revision": revision,
        "path": source_path,
        "git_blob_sha1": actual_blob,
        "source_url": source_url,
        "rows": len(rows),
    }
    authentication["authentication_sha256"] = sha256_json(authentication)
    return rows, authentication


def normalized_omni_math_row(row: dict[str, Any], source: dict[str, Any], protocol_seed: int) -> dict[str, Any]:
    from mgtb_v3.science_fast.protocol import item_seed, selection_key

    problem = str(row["problem"])
    digest = content_sha256(problem)
    source_id = str(row.get("id") or f"{source['path']}:line:{int(row['_source_line']):06d}")
    item_id = f"{source['repository']}@{source['revision']}:{source_id}:{digest[:16]}"
    domains = [str(value) for value in row["domain"]]
    return {
        "item_id": item_id,
        "source_id": source_id,
        "dataset_name": source["repository"],
        "dataset_kind": "omni_math",
        "dataset_revision": source["revision"],
        "split": "official",
        "problem": problem,
        "reference_answer": str(row["answer"]),
        "domains": domains,
        "subject": domains[0],
        "difficulty": float(row["difficulty"]),
        "level": float(row["difficulty"]),
        "source_provenance": str(row["source"]),
        "content_sha256": digest,
        "selection_key": selection_key(digest, protocol_seed),
        "item_seed": item_seed(protocol_seed, item_id),
    }


_JUDGEMENT_RE = re.compile(r"^## Equivalence Judgement\s*\n\s*(TRUE|FALSE)\s*$", re.MULTILINE)
_JUSTIFICATION_RE = re.compile(
    r"^## Justification\s*\n(?P<text>.*?)(?:\n\s*=== report over ===|\Z)", re.MULTILINE | re.DOTALL,
)


def parse_omni_judge_report(report: str) -> dict[str, Any]:
    judgement = _JUDGEMENT_RE.search(str(report))
    justification = _JUSTIFICATION_RE.search(str(report))
    if judgement is None or justification is None or not justification.group("text").strip():
        raise ValueError("Omni-Judge report is not scorable with the pinned TRUE/FALSE report format")
    verdict = judgement.group(1) == "TRUE"
    return {
        "correct": 1.0 if verdict else 0.0,
        "scorable": True,
        "method": "official_omni_judge",
        "equivalence_judgement": judgement.group(1),
        "justification": justification.group("text").strip(),
        "raw_report_sha256": hashlib.sha256(str(report).encode("utf-8")).hexdigest(),
    }
