from __future__ import annotations

from copy import deepcopy
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from mgtb_v3.eval.omni_math import load_official_omni_math, normalized_omni_math_row
from mgtb_v3.science_fast.io import atomic_write_json, load_json, sha256_file, sha256_json
from mgtb_v3.science_fast.protocol import content_sha256, item_seed, normalize_problem, selection_key


def _load_source(spec: dict[str, Any]) -> list[dict[str, Any]]:
    if spec.get("jsonl"):
        import json
        rows = []
        with Path(spec["jsonl"]).open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
        return rows
    revision = spec.get("revision")
    if not revision or str(revision).startswith("REPLACE_"):
        raise ValueError(f"dataset {spec.get('name')} requires an immutable revision")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError('manifest construction requires pip install -e ".[eval]"') from exc
    args = [spec["name"]]
    if spec.get("config"):
        args.append(spec["config"])
    dataset = load_dataset(*args, split=spec["split"], revision=revision)
    return [dict(row) for row in dataset]


def _value(row: dict[str, Any], key: str | None, fallback: str = "") -> Any:
    return row.get(key, fallback) if key else fallback


def _source_candidates(role: str, source: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    rows = _load_source(source)
    fields = source.get("fields", {})
    candidates = []
    for index, row in enumerate(rows):
        problem = str(_value(row, fields.get("problem", "problem")))
        if not problem:
            continue
        digest = content_sha256(problem)
        source_id = str(_value(row, fields.get("id"), f"{role}:{index}"))
        item_id = f"{source.get('name', source.get('jsonl', 'jsonl'))}:{source_id}:{digest[:16]}"
        candidate = {
            "role": role, "item_id": item_id, "source_id": source_id,
            "dataset_name": source.get("name", "jsonl"), "dataset_revision": source.get("revision"),
            "split": source.get("split", role),
            "problem": problem,
            "reference_answer": str(_value(row, fields.get("answer", "answer"))),
            "subject": _value(row, fields.get("subject"), None),
            "level": _value(row, fields.get("level"), None),
            "content_sha256": digest,
            "selection_key": selection_key(digest, seed),
            "item_seed": item_seed(seed, item_id),
        }
        if source.get("dataset_kind"):
            candidate["dataset_kind"] = source["dataset_kind"]
        candidates.append(candidate)
    unique = {row["content_sha256"]: row for row in candidates}
    return sorted(unique.values(), key=lambda row: (row["selection_key"], row["item_id"]))


def build_manifest(spec: dict[str, Any]) -> dict[str, Any]:
    if spec.get("strategy") == "stratified_omni_math_v1":
        return _build_omni_math_manifest(spec, [])
    seed = int(spec["protocol_seed"])
    roles: dict[str, list[dict[str, Any]]] = {}
    revisions = {}
    seen_content: set[str] = set()
    for role in ("reference", "development", "test"):
        source = spec["roles"][role]
        selected = _source_candidates(role, source, seed)
        count = int(source["count"])
        selected = [row for row in selected if row["content_sha256"] not in seen_content][:count]
        if len(selected) != count:
            raise ValueError(f"role {role} has only {len(selected)}/{count} unique non-overlapping items")
        seen_content.update(row["content_sha256"] for row in selected)
        roles[role] = selected
        revisions[role] = {"name": source.get("name", "jsonl"), "revision": source.get("revision"), "split": source.get("split")}
    manifest = {
        "schema_version": 2, "protocol_seed": seed,
        "selection_strategy": "unique content sorted by sha256(protocol_seed|content_sha256)",
        "dataset_revisions": revisions, "roles": roles,
        "counts": {role: len(items) for role, items in roles.items()},
    }
    manifest["manifest_sha256"] = sha256_json(manifest)
    validate_manifest(manifest)
    return manifest


def _excluded_contents(paths: list[str | Path]) -> tuple[set[str], list[dict[str, Any]]]:
    hashes: set[str] = set()
    provenance = []
    for raw_path in paths:
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise ValueError(f"excluded manifest does not exist: {path}")
        manifest = load_manifest(path)
        contents = {
            item["content_sha256"]
            for items in manifest["roles"].values()
            for item in items
        }
        hashes.update(contents)
        provenance.append({
            "path": str(path),
            "file_sha256": sha256_file(path),
            "manifest_sha256": manifest["manifest_sha256"],
            "content_count": len(contents),
        })
    return hashes, provenance


def _stratum(item: dict[str, Any]) -> tuple[str, str]:
    domains = item.get("domains")
    if not isinstance(domains, list) or not domains:
        raise ValueError(f"Omni-MATH item lacks domains: {item.get('source_id')}")
    difficulty = float(item["difficulty"])
    return str(domains[0]), format(difficulty, ".12g")


def _stratified_quotas(groups: dict[tuple[str, str], list[dict[str, Any]]], count: int) -> dict[tuple[str, str], int]:
    total = sum(len(items) for items in groups.values())
    if count > total:
        raise ValueError(f"test size {count} exceeds the {total} eligible Omni-MATH items")
    ideals = {key: count * len(items) / total for key, items in groups.items()}
    quotas = {key: int(ideals[key]) for key in groups}
    left = count - sum(quotas.values())
    ranked = sorted(groups, key=lambda key: (-(ideals[key] - quotas[key]), key[0], key[1]))
    for key in ranked[:left]:
        quotas[key] += 1
    return quotas


def _build_omni_math_manifest(spec: dict[str, Any], excluded_paths: list[str | Path]) -> dict[str, Any]:
    seed = int(spec["protocol_seed"])
    requested = {key: int(spec["counts"][key]) for key in ("reference", "development", "test")}
    if requested != {"reference": 300, "development": 300, "test": 500}:
        raise ValueError("stratified_omni_math_v1 requires counts reference=300 development=300 test=500")
    raw_rows, authentication = load_official_omni_math(spec["source"])
    rows_without_domain = [row for row in raw_rows if not row["domain"]]
    rows_without_problem = [row for row in raw_rows if not str(row["problem"]).strip()]
    rows_without_answer = [row for row in raw_rows if not str(row["answer"]).strip()]
    rows_without_source = [row for row in raw_rows if not str(row["source"]).strip()]
    eligible_source_rows = [
        row for row in raw_rows
        if row["domain"]
        and str(row["problem"]).strip()
        and str(row["answer"]).strip()
        and str(row["source"]).strip()
    ]
    candidates = [
        normalized_omni_math_row(row, spec["source"], seed)
        for row in eligible_source_rows
    ]
    by_content: dict[str, dict[str, Any]] = {}
    duplicate_counts: Counter[str] = Counter()
    for item in candidates:
        duplicate_counts[item["content_sha256"]] += 1
        by_content.setdefault(item["content_sha256"], item)

    excluded_hashes, exclusions = _excluded_contents(excluded_paths)
    excluded_overlap = sorted(set(by_content) & excluded_hashes)
    eligible = [item for digest, item in by_content.items() if digest not in excluded_hashes]
    if len(eligible) < sum(requested.values()):
        raise ValueError(
            f"only {len(eligible)} unique provenance-authenticated Omni-MATH items remain after exclusions; "
            f"need {sum(requested.values())}"
        )

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in eligible:
        groups[_stratum(item)].append(item)
    for items in groups.values():
        items.sort(key=lambda item: (item["selection_key"], item["item_id"]))
    quotas = _stratified_quotas(groups, requested["test"])
    test = [item for key in sorted(groups) for item in groups[key][:quotas[key]]]
    test_hashes = {item["content_sha256"] for item in test}
    remainder = sorted(
        (item for item in eligible if item["content_sha256"] not in test_hashes),
        key=lambda item: (item["selection_key"], item["item_id"]),
    )
    reference = remainder[:requested["reference"]]
    development = remainder[requested["reference"]:requested["reference"] + requested["development"]]

    roles = {
        "reference": [{**item, "role": "reference"} for item in reference],
        "development": [{**item, "role": "development"} for item in development],
        "test": [{**item, "role": "test"} for item in test],
    }
    stratum_counts = {
        f"{key[0]} | difficulty={key[1]}": {"available": len(groups[key]), "selected": quotas[key]}
        for key in sorted(groups)
    }
    manifest = {
        "schema_version": 3,
        "protocol_seed": seed,
        "selection_strategy": (
            "exclude source rows lacking a listed official domain or a non-empty problem, answer, or source; "
            "deduplicate normalized statements; "
            "remove every content hash in excluded manifests; "
            "test grouped by first listed official domain and exact numeric difficulty; allocate proportionally "
            "by Hamilton largest remainder; select within strata by "
            "sha256(protocol_seed|content_sha256); select reference then development from the sorted remainder"
        ),
        "seed_strategy": "sha256(protocol_seed|stable_item_id), first 63 bits",
        "source_authentication": authentication,
        "dataset_revisions": {
            role: {
                "name": spec["source"]["repository"],
                "revision": spec["source"]["revision"],
                "path": spec["source"]["path"],
                "git_blob_sha1": spec["source"]["git_blob_sha1"],
            }
            for role in roles
        },
        "deduplication": {
            "source_rows": len(raw_rows),
            "source_rows_ineligible_removed": len(raw_rows) - len(eligible_source_rows),
            "source_rows_without_domain_removed": len(rows_without_domain),
            "source_rows_without_problem_removed": len(rows_without_problem),
            "source_rows_without_answer_removed": len(rows_without_answer),
            "source_rows_without_source_removed": len(rows_without_source),
            "unique_normalized_statements": len(by_content),
            "duplicate_rows_removed": sum(count - 1 for count in duplicate_counts.values()),
            "excluded_overlap_removed": len(excluded_overlap),
        },
        "excluded_manifests": exclusions,
        "test_stratification": {
            "seed": seed,
            "domain_rule": "first element of the official domain list, retained verbatim",
            "difficulty_rule": "exact official numeric difficulty, canonical decimal representation",
            "allocation_rule": "proportional Hamilton largest remainder over eligible stratum counts",
            "strata": stratum_counts,
        },
        "roles": roles,
        "counts": {role: len(items) for role, items in roles.items()},
    }
    manifest["manifest_sha256"] = sha256_json(manifest)
    validate_manifest(manifest)
    assert_no_manifest_overlap(manifest, excluded_paths)
    return manifest


def build_manifest_with_exclusions(spec: dict[str, Any], excluded_paths: list[str | Path]) -> dict[str, Any]:
    if spec.get("strategy") == "stratified_omni_math_v1":
        return _build_omni_math_manifest(spec, excluded_paths)
    manifest = build_manifest(spec)
    assert_no_manifest_overlap(manifest, excluded_paths)
    return manifest


def derive_manifest(base: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    """Expand one role while proving that the existing selection is unchanged."""
    validate_manifest(base)
    role = str(spec["role"])
    if role not in base["roles"]:
        raise ValueError(f"cannot derive unknown manifest role {role!r}")
    source = spec["source"]
    count = int(source["count"])
    existing = base["roles"][role]
    if count < len(existing):
        raise ValueError(f"derived {role} count cannot shrink {len(existing)} to {count}")

    excluded = {
        item["content_sha256"]
        for other_role, items in base["roles"].items()
        if other_role != role
        for item in items
    }
    selected = [
        row for row in _source_candidates(role, source, int(base["protocol_seed"]))
        if row["content_sha256"] not in excluded
    ][:count]
    if len(selected) != count:
        raise ValueError(f"role {role} has only {len(selected)}/{count} unique non-overlapping items")
    if selected[:len(existing)] != existing:
        raise ValueError(f"derived {role} does not preserve the existing deterministic selection")

    manifest = deepcopy(base)
    manifest["roles"][role] = selected
    manifest["counts"] = {name: len(items) for name, items in manifest["roles"].items()}
    manifest.pop("manifest_sha256", None)
    manifest["manifest_sha256"] = sha256_json(manifest)
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: dict[str, Any]) -> None:
    supplied = manifest.get("manifest_sha256")
    actual = sha256_json({key: value for key, value in manifest.items() if key != "manifest_sha256"})
    if supplied != actual:
        raise ValueError("manifest hash mismatch")
    roles = manifest.get("roles", {})
    if set(roles) != {"reference", "development", "test"}:
        raise ValueError("manifest requires reference, development and test roles")
    if manifest.get("counts") != {role: len(items) for role, items in roles.items()}:
        raise ValueError("manifest counts mismatch")
    required = {"item_id", "source_id", "content_sha256", "problem", "reference_answer"}
    for role, items in roles.items():
        if any(not required <= item.keys() for item in items):
            raise ValueError(f"manifest role {role} has incomplete items")
        if len({item["item_id"] for item in items}) != len(items):
            raise ValueError(f"manifest role {role} has duplicate item IDs")
        if len({item["content_sha256"] for item in items}) != len(items):
            raise ValueError(f"manifest role {role} has duplicate contents")
        if any(item["content_sha256"] != content_sha256(item["problem"]) for item in items):
            raise ValueError(f"manifest role {role} has a non-canonical statement hash")
        for item in items:
            if item.get("dataset_kind") == "omni_math":
                if (not item.get("domains") or item.get("difficulty") is None
                        or not item.get("source_provenance") or not item.get("dataset_revision")):
                    raise ValueError(f"manifest role {role} has incomplete Omni-MATH provenance")
    hashes = {role: {item["content_sha256"] for item in items} for role, items in roles.items()}
    for left, right in (("reference", "development"), ("reference", "test"), ("development", "test")):
        overlap = hashes[left] & hashes[right]
        if overlap:
            raise ValueError(f"manifest leakage {left}/{right}: {sorted(overlap)[:3]}")


def assert_independent_test(manifest: dict[str, Any], excluded_paths: list[str | Path]) -> None:
    assert_no_manifest_overlap(manifest, excluded_paths, roles=("test",))


def assert_no_manifest_overlap(
    manifest: dict[str, Any], excluded_paths: list[str | Path], *, roles: tuple[str, ...] | None = None
) -> None:
    selected_roles = roles or tuple(manifest["roles"])
    current = {item["content_sha256"] for role in selected_roles for item in manifest["roles"][role]}
    for path in excluded_paths:
        if not Path(path).is_file():
            raise ValueError(f"excluded manifest does not exist: {Path(path).resolve()}")
        old = load_json(path)
        old_hashes = {item["content_sha256"] for items in old.get("roles", {}).values() for item in items}
        overlap = current & old_hashes
        if overlap:
            raise ValueError(f"campaign reuses {len(overlap)} excluded contents from {path}")


def save_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    validate_manifest(manifest)
    path = Path(path)
    if path.exists():
        existing = load_json(path)
        if existing != manifest:
            raise ValueError(f"refusing to replace immutable manifest: {path}")
        return
    atomic_write_json(path, manifest)


def load_manifest(path: str | Path) -> dict[str, Any]:
    manifest = load_json(path)
    validate_manifest(manifest)
    return manifest
