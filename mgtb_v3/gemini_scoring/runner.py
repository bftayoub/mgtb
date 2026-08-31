from __future__ import annotations

import signal
import threading
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Callable

from mgtb_v3.science_fast.io import atomic_write_json, load_json, sha256_json

from .api import GeminiClient, RateLimiter, retry_delay, status_code
from .blinding import anonymize_candidates
from .config import ModelQuota, ScoringConfig
from .dataset import load_candidates, select_pilot
from .prompt import build_payload, candidate_cache_key, prompt_hash, request_hash
from .schema import JUDGE_RESPONSE_SCHEMA, validate_judge_response
from .store import ArtifactStore, utc_now


def judge_identity(config: ScoringConfig, quota: ModelQuota, *, launched_at: str | None = None) -> dict[str, Any]:
    identity = {
        "model": quota.model,
        "google_genai_version": "2.20.0",
        "prompt_sha256": prompt_hash(),
        "json_schema": JUDGE_RESPONSE_SCHEMA,
        "temperature": config.temperature,
        "max_output_tokens": config.max_output_tokens,
        "thinking_level": quota.thinking_level,
        "launched_at": launched_at or utc_now(),
    }
    identity["judge_identity_sha256"] = sha256_json(identity)
    return identity


def _load_or_create_identity(config: ScoringConfig, quota: ModelQuota) -> dict[str, Any]:
    prospective = judge_identity(config, quota, launched_at="fingerprint")
    fingerprint = sha256_json({
        key: value for key, value in prospective.items()
        if key not in {"launched_at", "judge_identity_sha256"}
    })[:16]
    path = config.output_root / "judge-identities" / f"{quota.model}-{fingerprint}.json"
    if path.is_file():
        identity = load_json(path)
        expected = judge_identity(config, quota, launched_at=identity.get("launched_at"))
        if identity != expected:
            raise ValueError(f"judge configuration changed since launch: {path}")
        return identity
    identity = judge_identity(config, quota)
    atomic_write_json(path, identity)
    return identity


def _selection_path(config: ScoringConfig) -> Path:
    return config.output_root / "pilot" / "selection.json"


def prepare_scope(config: ScoringConfig, scope: str) -> list[dict[str, Any]]:
    full_index_path = config.output_root / "full" / "source_index.json"
    if scope == "full" and full_index_path.is_file():
        index = load_json(full_index_path)
        expected = index.get("index_sha256")
        actual = sha256_json({key: value for key, value in index.items() if key != "index_sha256"})
        if expected != actual or len(index.get("candidates", [])) != 4500:
            raise ValueError("full source index is invalid")
        current_states = {
            variant: sha256_json(load_json(config.source_root / "runs" / "test" / variant / "run_state.json"))
            for variant in config.variants
        }
        if index.get("generation_state_sha256") != current_states:
            raise ValueError("frozen generation states changed since full indexing")
        return index["candidates"]
    details_path = config.output_root / "pilot" / "selection_details.json"
    if scope == "pilot" and details_path.is_file():
        details = load_json(details_path)
        expected = details.get("details_sha256")
        actual = sha256_json({key: value for key, value in details.items() if key != "details_sha256"})
        if expected != actual or len(details.get("candidates", [])) != sum(config.pilot_counts.values()):
            raise ValueError("pilot selection details are invalid")
        return details["candidates"]
    candidates = load_candidates(config)
    if scope == "full":
        index = {
            "schema_version": 1,
            "generation_state_sha256": {
                variant: sha256_json(load_json(
                    config.source_root / "runs" / "test" / variant / "run_state.json"
                )) for variant in config.variants
            },
            "candidates": candidates,
        }
        index["index_sha256"] = sha256_json(index)
        atomic_write_json(full_index_path, index)
        return candidates
    path = _selection_path(config)
    if path.is_file():
        frozen = load_json(path)
        expected = frozen.get("selection_sha256")
        actual = sha256_json({key: value for key, value in frozen.items() if key != "selection_sha256"})
        if expected != actual or len(frozen.get("units", [])) != sum(config.pilot_counts.values()):
            raise ValueError("frozen pilot selection is invalid")
        by_id = {row["unit_id"]: row for row in candidates}
        selected = []
        for unit in frozen["units"]:
            row = by_id.get(unit["unit_id"])
            if row is None or any(row[key] != unit[key] for key in (
                "item_id", "content_sha256", "variant", "replicate_seed", "generation_artifact_sha256",
            )):
                raise ValueError("frozen pilot unit no longer matches authenticated sources")
            selected.append({**row, "pilot_stratum": unit["pilot_stratum"]})
        details = {"schema_version": 1, "candidates": selected}
        details["details_sha256"] = sha256_json(details)
        atomic_write_json(details_path, details)
        return selected
    selected = select_pilot(candidates, config.pilot_counts)
    artifact = {
        "schema_version": 1,
        "counts": dict(Counter(row["pilot_stratum"] for row in selected)),
        "units": [{
            key: row[key] for key in (
                "unit_id", "item_id", "content_sha256", "variant", "replicate_seed",
                "generation_artifact_sha256", "pilot_stratum",
            )
        } for row in selected],
    }
    artifact["selection_sha256"] = sha256_json(artifact)
    if path.is_file() and load_json(path) != artifact:
        raise ValueError("pilot selection changed; refusing to overwrite it")
    if not path.exists():
        atomic_write_json(path, artifact)
    details = {"schema_version": 1, "candidates": selected}
    details["details_sha256"] = sha256_json(details)
    if details_path.is_file() and load_json(details_path) != details:
        raise ValueError("pilot selection details changed")
    if not details_path.exists():
        atomic_write_json(details_path, details)
    return selected


def build_groups(config: ScoringConfig, candidates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    undecided = [row for row in candidates if row["control"]["verdict"] is None]
    by_problem: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in undecided:
        by_problem[row["content_sha256"]].append(row)
    groups, local_mapping = [], {}
    for problem_hash in sorted(by_problem):
        rows = by_problem[problem_hash]
        unique: dict[str, dict[str, Any]] = {}
        for row in rows:
            key = candidate_cache_key(
                config.primary.model, row["problem"], row["reference_answer"], str(row["candidate_answer"]),
            )
            unique.setdefault(key, row)
        public, mapping = anonymize_candidates(unique.values(), problem_hash, config.permutation_salt)
        first = rows[0]
        payload = build_payload(first["problem"], first["reference_answer"], public)
        groups.append({"problem_hash": problem_hash, "payload": payload, "mapping": mapping, "rows": rows})
        local_mapping[problem_hash] = {
            anonymous_id: {
                "unit_id": unit,
                "variant": next(row["variant"] for row in rows if row["unit_id"] == unit),
                "replicate_seed": next(row["replicate_seed"] for row in rows if row["unit_id"] == unit),
            } for anonymous_id, unit in mapping.items()
        }
    return groups, local_mapping


def dry_run(config: ScoringConfig, scope: str = "pilot") -> dict[str, Any]:
    candidates = prepare_scope(config, scope)
    groups, mapping = build_groups(config, candidates)
    mapping_artifact = {
        "schema_version": 1, "scope": scope, "mapping": mapping,
        "note": "local-only restoration table; never included in API payloads",
    }
    mapping_artifact["mapping_sha256"] = sha256_json(mapping_artifact)
    path = config.output_root / scope / "anonymization.json"
    if path.is_file() and load_json(path) != mapping_artifact:
        raise ValueError("anonymization table changed")
    if not path.exists():
        atomic_write_json(path, mapping_artifact)
    return {
        "scope": scope, "total_candidates": len(candidates),
        "deterministic": sum(row["control"]["verdict"] is not None for row in candidates),
        "api_candidates": sum(row["control"]["verdict"] is None for row in candidates),
        "grouped_requests": len(groups), "payloads_validated": len(groups),
        "api_contacted": False,
    }


def _decision_from_control(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1, "unit_id": row["unit_id"], "item_id": row["item_id"],
        "content_sha256": row["content_sha256"],
        "generation_artifact_sha256": row["generation_artifact_sha256"],
        "verdict": row["control"]["verdict"], "source": "deterministic_control",
        "reason": row["control"]["rule"], "control": row["control"],
        "created_at": utc_now(),
    }


def _materialize_group_results(
    store: ArtifactStore, cache_store: ArtifactStore, config: ScoringConfig, identity_hash: str,
    rows: list[dict[str, Any]], pending_cache: dict[str, dict[str, Any]],
    anonymous: dict[str, str], req_hash: str, response_artifact: dict[str, Any],
    parsed: list[dict[str, str]], *, source: str,
) -> None:
    for result in parsed:
        source_unit = anonymous[result["candidate_id"]]
        source_row = next(value for value in pending_cache.values() if value["unit_id"] == source_unit)
        cache_key = candidate_cache_key(
            config.primary.model, source_row["problem"], source_row["reference_answer"],
            str(source_row["candidate_answer"]),
        )
        cache_store.save_result(cache_key, {
            "schema_version": 1, "cache_key": cache_key,
            "judge_identity_sha256": identity_hash, "request_sha256": req_hash,
            "response_artifact_sha256": response_artifact["response_artifact_sha256"],
            "problem_sha256": source_row["content_sha256"], "result": result,
            "created_at": response_artifact["validated_at"],
        })
    # Fan the authenticated cache out to every variant/seed unit sharing an answer.
    for row in rows:
        if store.valid_decision(row["unit_id"]) is not None:
            continue
        cache_key = candidate_cache_key(
            config.primary.model, row["problem"], row["reference_answer"], str(row["candidate_answer"]),
        )
        cached = cache_store.valid_result(cache_key, identity_hash)
        if cached:
            store.save_decision(row["unit_id"], {
                "schema_version": 1, "unit_id": row["unit_id"], "item_id": row["item_id"],
                "content_sha256": row["content_sha256"],
                "generation_artifact_sha256": row["generation_artifact_sha256"],
                "verdict": cached["result"]["verdict"], "source": source,
                "reason": cached["result"]["reason"], "cache_key": cache_key,
                "result_artifact_sha256": cached["result_artifact_sha256"], "created_at": utc_now(),
            })


def run_scope(
    config: ScoringConfig, scope: str, *, stop_after: int | None, workers: int,
    resume: bool, client_factory: Callable[..., Any] = GeminiClient,
) -> dict[str, Any]:
    if workers < 1:
        raise ValueError("--workers must be at least one")
    candidates = prepare_scope(config, scope)
    store = ArtifactStore(config.output_root / scope)
    # Valid grouped primary results are content-addressed globally.  A full run can
    # therefore reuse pilot calls even if the full problem batch has a different order.
    cache_store = ArtifactStore(config.output_root / "cache" / config.primary.model)
    existing = sum(store.valid_decision(row["unit_id"]) is not None for row in candidates)
    if existing and not resume:
        raise ValueError("existing valid decisions found; pass --resume")
    for row in candidates:
        if row["control"]["verdict"] is not None and store.valid_decision(row["unit_id"]) is None:
            store.save_decision(row["unit_id"], _decision_from_control(row))

    identity = _load_or_create_identity(config, config.primary)
    identity_hash = identity["judge_identity_sha256"]
    quota_store = ArtifactStore(config.output_root / "quota")
    ledger = quota_store.ledger(config.primary.model)
    limiter = RateLimiter(config.primary, requests_today=int(ledger.get("requests", 0)))
    client = None
    groups, mapping = build_groups(config, candidates)
    # Persist restoration data before calls, but never place it in a payload.
    dry_run(config, scope)
    sent = 0
    stop = threading.Event()
    previous_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    try:
        for group in groups:
            if stop.is_set() or (stop_after is not None and sent >= max(0, stop_after)):
                break
            rows = group["rows"]
            pending_cache: dict[str, dict[str, Any]] = {}
            for row in rows:
                if store.valid_decision(row["unit_id"]) is not None:
                    continue
                cache_key = candidate_cache_key(
                    config.primary.model, row["problem"], row["reference_answer"], str(row["candidate_answer"]),
                )
                cached = cache_store.valid_result(cache_key, identity_hash)
                if cached is not None:
                    store.save_decision(row["unit_id"], {
                        "schema_version": 1, "unit_id": row["unit_id"], "item_id": row["item_id"],
                        "content_sha256": row["content_sha256"],
                        "generation_artifact_sha256": row["generation_artifact_sha256"],
                        "verdict": cached["result"]["verdict"], "source": "gemini_cache",
                        "reason": cached["result"]["reason"], "cache_key": cache_key,
                        "result_artifact_sha256": cached["result_artifact_sha256"], "created_at": utc_now(),
                    })
                else:
                    pending_cache.setdefault(cache_key, row)
            if not pending_cache:
                continue
            public, anonymous = anonymize_candidates(
                pending_cache.values(), group["problem_hash"], config.permutation_salt,
            )
            payload = build_payload(rows[0]["problem"], rows[0]["reference_answer"], public)
            req_hash = request_hash(config.primary.model, payload)
            expected_ids = [row["candidate_id"] for row in public]
            recovered = store.valid_response(req_hash, identity_hash)
            if recovered is not None:
                if recovered.get("payload_sha256") != sha256_json(payload):
                    raise ValueError("cached API response payload hash mismatch")
                parsed = validate_judge_response(recovered.get("response", {}), expected_ids)
                _materialize_group_results(
                    store, cache_store, config, identity_hash, rows, pending_cache, anonymous, req_hash,
                    recovered, parsed, source="gemini_recovered_response",
                )
                store.checkpoint(_status(config, candidates, store, config.primary))
                continue
            attempt = 0
            while True:
                if stop_after is not None and sent >= max(0, stop_after):
                    stop.set()
                    break
                attempt += 1
                try:
                    if client is None:
                        client = client_factory(
                            config.primary, config.temperature, config.max_output_tokens, limiter,
                        )
                    if int(quota_store.ledger(config.primary.model).get("requests", 0)) >= config.primary.rpd:
                        raise RuntimeError(f"daily request budget exhausted for {config.primary.model}")
                    quota_store.update_ledger(config.primary.model, requests=1)
                    sent += 1
                    response_text, usage, latency = client.generate(payload)
                    parsed = validate_judge_response(response_text, expected_ids)
                    response_artifact = store.save_response(req_hash, {
                        "schema_version": 1, "request_sha256": req_hash,
                        "judge_identity": identity, "payload_sha256": sha256_json(payload),
                        "candidate_ids": expected_ids, "response": {"results": parsed},
                        "usage": usage, "latency_seconds": latency, "validated_at": utc_now(),
                    })
                    quota_store.update_ledger(
                        config.primary.model,
                        input_tokens=usage.get("input_tokens", 0), output_tokens=usage.get("output_tokens", 0),
                    )
                    _materialize_group_results(
                        store, cache_store, config, identity_hash, rows, pending_cache, anonymous, req_hash,
                        response_artifact, parsed, source="gemini_grouped",
                    )
                    store.checkpoint(_status(config, candidates, store, config.primary))
                    break
                except BaseException as exc:
                    code = status_code(exc)
                    retryable = (code == 429 or (code is not None and 500 <= code <= 599)
                                 or isinstance(exc, (ValueError, TimeoutError, ConnectionError, OSError)))
                    store.save_error(req_hash, attempt, {
                        "schema_version": 1, "request_sha256": req_hash, "attempt": attempt,
                        "error_type": type(exc).__name__, "status_code": code,
                        "retryable": retryable, "occurred_at": utc_now(),
                    })
                    quota_store.update_ledger(
                        config.primary.model, temporary_errors=1 if retryable else 0,
                        retries=1 if retryable else 0,
                    )
                    store.checkpoint(_status(config, candidates, store, config.primary))
                    if not retryable or attempt >= 6 or stop.is_set():
                        if not retryable:
                            raise
                        break
                    time.sleep(retry_delay(exc, attempt))
    finally:
        signal.signal(signal.SIGINT, previous_handler)
    return _status(config, candidates, store, config.primary)


def _status(config: ScoringConfig, candidates: list[dict[str, Any]], store: ArtifactStore,
            quota: ModelQuota) -> dict[str, Any]:
    decisions = [store.valid_decision(row["unit_id"]) for row in candidates]
    valid = [row for row in decisions if row is not None]
    counts = Counter(row["verdict"] for row in valid)
    ledger = ArtifactStore(config.output_root / "quota").ledger(quota.model)
    completed = len(valid)
    cutoff = datetime.now(timezone.utc).timestamp() - 600.0
    recent_api = 0
    for row in valid:
        if not str(row.get("source", "")).startswith("gemini"):
            continue
        try:
            if datetime.fromisoformat(row["created_at"]).timestamp() >= cutoff:
                recent_api += 1
        except (KeyError, TypeError, ValueError):
            pass
    elapsed_rate = recent_api / 10.0
    # Conservative ETA is request based: at most 12 grouped requests/minute and 500/day.
    remaining = len(candidates) - completed
    api_remaining = sum(
        store.valid_decision(row["unit_id"]) is None and row["control"]["verdict"] is None
        for row in candidates
    )
    estimated_requests = len({row["content_sha256"] for row in candidates
                              if store.valid_decision(row["unit_id"]) is None
                              and row["control"]["verdict"] is None})
    eta_minutes = estimated_requests / max(1, quota.rpm)
    return {
        "total": len(candidates), "completed": completed, "remaining": remaining,
        "TRUE": counts["TRUE"], "FALSE": counts["FALSE"], "ABSTAIN": counts["ABSTAIN"],
        "temporary_errors": int(ledger.get("temporary_errors", 0)),
        "retries": int(ledger.get("retries", 0)),
        "requests_sent_today": int(ledger.get("requests", 0)),
        "estimated_rpd_remaining": max(0, quota.rpd - int(ledger.get("requests", 0))),
        "input_tokens": int(ledger.get("input_tokens", 0)),
        "output_tokens": int(ledger.get("output_tokens", 0)),
        "recent_candidates_per_minute": elapsed_rate,
        "estimated_requests_remaining": estimated_requests,
        "eta_minutes_rate_only": eta_minutes,
    }


def status(config: ScoringConfig, scope: str) -> dict[str, Any]:
    candidates = prepare_scope(config, scope)
    result = _status(config, candidates, ArtifactStore(config.output_root / scope), config.primary)
    if scope == "pilot":
        from .audit import arbitration_candidates, individual_audit_candidates

        individual = individual_audit_candidates(config)
        arbitration = arbitration_candidates(config)
        individual_store = ArtifactStore(config.output_root / "pilot_individual")
        arbitration_store = ArtifactStore(config.output_root / "pilot_arbitration")
        secondary_ledger = ArtifactStore(config.output_root / "quota").ledger(config.secondary.model)
        arbitration_remaining = sum(
            arbitration_store.valid_decision(row["unit_id"]) is None for row in arbitration
        )
        result["individual_audit"] = {
            "total": len(individual),
            "completed": sum(individual_store.valid_decision(row["unit_id"]) is not None for row in individual),
        }
        result["secondary_arbitration"] = {
            "total": len(arbitration), "remaining": arbitration_remaining,
            "completed": len(arbitration) - arbitration_remaining,
            "requests_sent_today": int(secondary_ledger.get("requests", 0)),
            "estimated_rpd_remaining": max(
                0, config.secondary.rpd - int(secondary_ledger.get("requests", 0)),
            ),
            "temporary_errors": int(secondary_ledger.get("temporary_errors", 0)),
            "retries": int(secondary_ledger.get("retries", 0)),
            "input_tokens": int(secondary_ledger.get("input_tokens", 0)),
            "minimum_quota_days_remaining": (
                (arbitration_remaining + config.secondary.rpd - 1) // config.secondary.rpd
            ),
        }
    return result
