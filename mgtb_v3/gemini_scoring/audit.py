from __future__ import annotations

import hashlib
import time
from typing import Any, Callable

from .api import GeminiClient, RateLimiter, retry_delay, status_code
from .blinding import anonymize_candidates
from .config import ModelQuota, ScoringConfig
from .prompt import build_payload, candidate_cache_key, request_hash
from .runner import _load_or_create_identity, prepare_scope
from .schema import validate_judge_response
from .store import ArtifactStore, utc_now
from mgtb_v3.science_fast.io import sha256_json


def individual_audit_candidates(config: ScoringConfig) -> list[dict[str, Any]]:
    pilot = [row for row in prepare_scope(config, "pilot") if row["control"]["verdict"] is None]
    return sorted(
        pilot, key=lambda row: hashlib.sha256(f"individual-audit:{row['unit_id']}".encode()).hexdigest(),
    )[:config.individual_audit_count]


def run_individual_audit(
    config: ScoringConfig, *, stop_after: int | None, resume: bool,
    client_factory: Callable[..., Any] = GeminiClient,
) -> dict[str, Any]:
    return _run_individual_set(
        config, individual_audit_candidates(config), config.primary, "pilot_individual",
        "gemini_individual", stop_after=stop_after, resume=resume, client_factory=client_factory,
    )


def arbitration_candidates(config: ScoringConfig) -> list[dict[str, Any]]:
    pilot = prepare_scope(config, "pilot")
    grouped = ArtifactStore(config.output_root / "pilot")
    individual = ArtifactStore(config.output_root / "pilot_individual")
    conflicts = [row for row in pilot if row.get("pilot_stratum") == "numeric_contradictions"]
    audit_pool = []
    conflict_ids = {row["unit_id"] for row in conflicts}
    individual_ids = {row["unit_id"] for row in individual_audit_candidates(config)}
    for row in pilot:
        group_decision = grouped.valid_decision(row["unit_id"])
        individual_decision = individual.valid_decision(row["unit_id"]) if row["unit_id"] in individual_ids else None
        if group_decision and (
            group_decision["verdict"] == "ABSTAIN"
            or (individual_decision and group_decision["verdict"] != individual_decision["verdict"])
        ):
            if row["unit_id"] not in conflict_ids:
                conflicts.append(row)
                conflict_ids.add(row["unit_id"])
        elif group_decision and row["control"]["verdict"] is None:
            audit_pool.append(row)
    conflicts = sorted(conflicts, key=lambda row: row["unit_id"])
    audits = sorted(
        (row for row in audit_pool if row["unit_id"] not in conflict_ids),
        key=lambda row: hashlib.sha256(f"secondary-audit:{row['unit_id']}".encode()).hexdigest(),
    )[:config.secondary_audit_count]
    return conflicts + audits


def run_secondary_arbitration(
    config: ScoringConfig, *, stop_after: int | None, resume: bool,
    client_factory: Callable[..., Any] = GeminiClient,
) -> dict[str, Any]:
    return _run_individual_set(
        config, arbitration_candidates(config), config.secondary, "pilot_arbitration",
        "gemini_secondary_arbitration", stop_after=stop_after, resume=resume,
        client_factory=client_factory,
    )


def _run_individual_set(
    config: ScoringConfig, candidates: list[dict[str, Any]], quota: ModelQuota,
    store_name: str, source: str, *, stop_after: int | None, resume: bool,
    client_factory: Callable[..., Any],
) -> dict[str, Any]:
    store = ArtifactStore(config.output_root / store_name)
    existing = sum(store.valid_decision(row["unit_id"]) is not None for row in candidates)
    if existing and not resume:
        raise ValueError("existing individual-audit decisions found; pass --resume")
    identity = _load_or_create_identity(config, quota)
    identity_hash = identity["judge_identity_sha256"]
    quota_store = ArtifactStore(config.output_root / "quota")
    ledger = quota_store.ledger(quota.model)
    limiter = RateLimiter(quota, int(ledger.get("requests", 0)))
    client = None
    sent = 0
    daily_exhausted = False
    for row in candidates:
        if store.valid_decision(row["unit_id"]) is not None:
            continue
        if stop_after is not None and sent >= max(0, stop_after):
            break
        public, mapping = anonymize_candidates(
            [row], row["content_sha256"], store_name + ":" + config.permutation_salt,
        )
        payload = build_payload(row["problem"], row["reference_answer"], public)
        req_hash = request_hash(quota.model, payload)
        candidate_id = public[0]["candidate_id"]
        recovered = store.valid_response(req_hash, identity_hash)
        if recovered is not None:
            if recovered.get("payload_sha256") != sha256_json(payload):
                raise ValueError("cached individual response payload hash mismatch")
            result = validate_judge_response(recovered.get("response", {}), [candidate_id])[0]
            cache_key = candidate_cache_key(
                quota.model, row["problem"], row["reference_answer"], str(row["candidate_answer"]),
            )
            cached = store.save_result(cache_key, {
                "schema_version": 1, "cache_key": cache_key,
                "judge_identity_sha256": identity_hash, "request_sha256": req_hash,
                "response_artifact_sha256": recovered["response_artifact_sha256"],
                "problem_sha256": row["content_sha256"], "result": result,
                "created_at": recovered["validated_at"],
            })
            store.save_decision(row["unit_id"], {
                "schema_version": 1, "unit_id": row["unit_id"], "item_id": row["item_id"],
                "content_sha256": row["content_sha256"],
                "generation_artifact_sha256": row["generation_artifact_sha256"],
                "verdict": result["verdict"], "source": source + "_recovered_response",
                "reason": result["reason"], "cache_key": cache_key,
                "result_artifact_sha256": cached["result_artifact_sha256"], "created_at": utc_now(),
            })
            continue
        attempt = 0
        while True:
            if stop_after is not None and sent >= max(0, stop_after):
                daily_exhausted = False
                break
            attempt += 1
            try:
                if client is None:
                    client = client_factory(quota, config.temperature, config.max_output_tokens, limiter)
                if int(quota_store.ledger(quota.model).get("requests", 0)) >= quota.rpd:
                    daily_exhausted = True
                    break
                quota_store.update_ledger(quota.model, requests=1)
                sent += 1
                text, usage, latency = client.generate(payload)
                result = validate_judge_response(text, [candidate_id])[0]
                response = store.save_response(req_hash, {
                    "schema_version": 1, "request_sha256": req_hash, "judge_identity": identity,
                    "payload_sha256": sha256_json(payload), "candidate_ids": [candidate_id],
                    "response": {"results": [result]}, "usage": usage,
                    "latency_seconds": latency, "validated_at": utc_now(),
                })
                cache_key = candidate_cache_key(
                    quota.model, row["problem"], row["reference_answer"], str(row["candidate_answer"]),
                )
                cached = store.save_result(cache_key, {
                    "schema_version": 1, "cache_key": cache_key,
                    "judge_identity_sha256": identity_hash, "request_sha256": req_hash,
                    "response_artifact_sha256": response["response_artifact_sha256"],
                    "problem_sha256": row["content_sha256"], "result": result,
                    "created_at": response["validated_at"],
                })
                store.save_decision(row["unit_id"], {
                    "schema_version": 1, "unit_id": row["unit_id"], "item_id": row["item_id"],
                    "content_sha256": row["content_sha256"],
                    "generation_artifact_sha256": row["generation_artifact_sha256"],
                    "verdict": result["verdict"], "source": source,
                    "reason": result["reason"], "cache_key": cache_key,
                    "result_artifact_sha256": cached["result_artifact_sha256"], "created_at": utc_now(),
                })
                quota_store.update_ledger(
                    quota.model, input_tokens=usage.get("input_tokens", 0),
                    output_tokens=usage.get("output_tokens", 0),
                )
                completed_now = sum(store.valid_decision(value["unit_id"]) is not None for value in candidates)
                store.checkpoint({
                    "total": len(candidates), "completed": completed_now,
                    "remaining": len(candidates) - completed_now, "model": quota.model,
                })
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
                    quota.model, temporary_errors=1 if retryable else 0,
                    retries=1 if retryable else 0,
                )
                completed_now = sum(store.valid_decision(value["unit_id"]) is not None for value in candidates)
                store.checkpoint({
                    "total": len(candidates), "completed": completed_now,
                    "remaining": len(candidates) - completed_now, "model": quota.model,
                    "last_request_error": True,
                })
                if not retryable or attempt >= 6:
                    if not retryable:
                        raise
                    break
                time.sleep(retry_delay(exc, attempt))
        if daily_exhausted:
            break
    completed = sum(store.valid_decision(row["unit_id"]) is not None for row in candidates)
    return {"total": len(candidates), "completed": completed, "remaining": len(candidates) - completed,
            "requests_sent_this_run": sent, "daily_quota_exhausted": daily_exhausted}
