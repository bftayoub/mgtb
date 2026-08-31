from __future__ import annotations

import json

import pytest

from mgtb_v3.gemini_scoring.analysis import full_statistics
from mgtb_v3.gemini_scoring.blinding import anonymize_candidates, assert_blind_payload
from mgtb_v3.gemini_scoring.controls import deterministic_verdict
from mgtb_v3.gemini_scoring.prompt import build_payload
from mgtb_v3.gemini_scoring.schema import validate_judge_response
from mgtb_v3.gemini_scoring.store import ArtifactStore


def _result(candidate_id: str, verdict: str = "TRUE") -> dict[str, str]:
    return {
        "candidate_id": candidate_id, "verdict": verdict,
        "normalized_candidate": "2", "normalized_reference": "2",
        "reason": "The values are exactly equal.",
    }


def test_anonymization_and_permutation_are_deterministic_and_seed_blind():
    candidates = [
        {"unit_id": "real-vanilla-seed-0", "candidate_answer": "1"},
        {"unit_id": "real-mgtb-seed-2", "candidate_answer": "2"},
        {"unit_id": "real-random-seed-1", "candidate_answer": "3"},
    ]
    first, mapping = anonymize_candidates(candidates, "problem-hash", "fixed-salt")
    second, second_mapping = anonymize_candidates(list(reversed(candidates)), "problem-hash", "fixed-salt")
    repeated, repeated_mapping = anonymize_candidates(candidates, "problem-hash", "fixed-salt")
    assert (first, mapping) == (repeated, repeated_mapping)
    assert all(set(row) == {"candidate_id", "answer"} for row in first)
    assert "seed" not in json.dumps(first).lower()
    assert set(second_mapping.values()) == set(mapping.values())


@pytest.mark.parametrize("left,right,verdict,rule", [
    ("2", "2.0", "TRUE", "certain_normalized_equality"),
    (r"\frac{1}{2}", "1/2", "TRUE", "certain_normalized_equality"),
    ("2", "3", "FALSE", "different_simple_numbers"),
    (None, "3", "FALSE", "empty_or_invalid_extraction"),
    (r"x \equiv 2 \pmod{3}", r"x \equiv 1 \pmod{3}", None, "requires_mathematical_judge"),
    ("[0,1]", "(0,1)", None, "requires_mathematical_judge"),
])
def test_conservative_numeric_controls(left, right, verdict, rule):
    result = deterministic_verdict(left, right)
    assert result["verdict"] == verdict
    assert result["rule"] == rule


def test_json_validation_is_strict_and_preserves_abstain():
    payload = {"results": [_result("C01", "ABSTAIN"), _result("C02", "FALSE")]}
    rows = validate_judge_response(payload, ["C01", "C02"])
    assert rows[0]["verdict"] == "ABSTAIN"
    broken = {"results": [{**_result("C01"), "extra": "forbidden"}, _result("C02")]}
    with pytest.raises(ValueError, match="missing or extra"):
        validate_judge_response(broken, ["C01", "C02"])


def test_missing_and_duplicate_responses_are_rejected_not_scored():
    with pytest.raises(ValueError, match="count mismatch"):
        validate_judge_response({"results": [_result("C01")]}, ["C01", "C02"])
    with pytest.raises(ValueError, match="duplicate"):
        validate_judge_response({"results": [_result("C01"), _result("C01")]}, ["C01", "C02"])


def test_cache_and_resume_accept_only_authenticated_immutable_results(tmp_path):
    store = ArtifactStore(tmp_path)
    saved = store.save_result("cache", {
        "cache_key": "cache", "judge_identity_sha256": "identity",
        "result": _result("C01"),
    })
    assert store.valid_result("cache", "identity") == saved
    assert store.valid_result("cache", "different") is None
    # Restart reconstructs the cache from disk.
    assert ArtifactStore(tmp_path).valid_result("cache", "identity") == saved
    with pytest.raises(ValueError, match="immutable"):
        store.save_result("cache", {
            "cache_key": "cache", "judge_identity_sha256": "identity",
            "result": _result("C01", "FALSE"),
        })


def test_variant_restoration_is_local_and_payload_has_no_provenance():
    candidates = [
        {"unit_id": "u1", "candidate_answer": "x+1", "variant": "full_mgtb", "replicate_seed": 2},
        {"unit_id": "u2", "candidate_answer": "x-1", "variant": "vanilla", "replicate_seed": 0},
    ]
    public, mapping = anonymize_candidates(candidates, "problem", "salt")
    local = {row["unit_id"]: (row["variant"], row["replicate_seed"]) for row in candidates}
    restored = {anonymous: local[unit] for anonymous, unit in mapping.items()}
    assert sorted(restored.values()) == sorted(local.values())
    payload = build_payload("Find x.", "x=1", public)
    serialized = json.dumps(payload).lower()
    for forbidden in ("vanilla", "full_mgtb", "matched_random", "replicate_seed", "omni-judge"):
        assert forbidden not in serialized
    assert_blind_payload(payload)


def test_payload_leak_guard_rejects_variant_names_and_seed_keys():
    with pytest.raises(ValueError, match="forbidden term"):
        assert_blind_payload({"answer": "vanilla"})
    with pytest.raises(ValueError, match="provenance keys"):
        assert_blind_payload({"candidate": {"seed": 1}})


def test_full_statistics_counts_abstain_as_incorrect_and_pairs_variants():
    verdicts = {
        "vanilla": {"p1": "TRUE", "p2": "FALSE"},
        "full_mgtb": {"p1": "TRUE", "p2": "TRUE"},
        "matched_random": {"p1": "ABSTAIN", "p2": "FALSE"},
    }
    rows = [
        {
            "variant": variant, "source_item_id": problem, "replicate_seed": 0,
            "domains": ["algebra"], "verdict": verdict,
            "old_omni_verdict": "TRUE" if problem == "p1" else "FALSE",
        }
        for variant, by_problem in verdicts.items()
        for problem, verdict in by_problem.items()
    ]
    report = full_statistics(
        rows, ["vanilla", "full_mgtb", "matched_random"],
        bootstrap_seed=7, bootstrap_samples=100,
    )
    assert report["methods"]["vanilla"]["accuracy"] == 0.5
    assert report["methods"]["full_mgtb"]["accuracy"] == 1.0
    assert report["methods"]["matched_random"]["counts"]["ABSTAIN"] == 1
    assert report["methods"]["matched_random"]["accuracy"] == 0.0
    mgtb = report["comparisons_against_vanilla"]["full_mgtb"]
    assert (mgtb["corrections"], mgtb["regressions"], mgtb["problem_clusters"]) == (1, 0, 2)
    random = report["comparisons_against_vanilla"]["matched_random"]
    assert (random["corrections"], random["regressions"]) == (0, 1)
    assert "mcnemar_holm_adjusted_p" in random


def test_full_statistics_refuses_unpaired_variants():
    rows = [
        {"variant": "vanilla", "source_item_id": "p1", "replicate_seed": 0,
         "domains": [], "verdict": "TRUE", "old_omni_verdict": "TRUE"},
        {"variant": "full_mgtb", "source_item_id": "p2", "replicate_seed": 0,
         "domains": [], "verdict": "TRUE", "old_omni_verdict": "TRUE"},
    ]
    with pytest.raises(ValueError, match="not paired"):
        full_statistics(rows, ["vanilla", "full_mgtb"], bootstrap_samples=10)
