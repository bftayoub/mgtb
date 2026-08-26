from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from mgtb_v3.eval.omni_math import (
    git_blob_sha1, load_official_omni_math, parse_omni_judge_report, pending_omni_math_score,
)
from mgtb_v3.science_campaign.analysis import campaign_analysis
from mgtb_v3.science_campaign.config import load_campaign
from mgtb_v3.science_campaign.config import calibration_spec, role_seeds
from mgtb_v3.science_campaign.calibration import build_calibrator
from mgtb_v3.science_campaign.judging import merge_judgments, run_judging
from mgtb_v3.science_campaign.manifest import (
    assert_no_manifest_overlap, build_manifest_with_exclusions, save_manifest,
)
from mgtb_v3.science_campaign.runner import assert_freeze, build_freeze, build_profile, campaign_units
from mgtb_v3.science_fast.io import sha256_json
from mgtb_v3.science_fast.protocol import content_sha256


def _official_payload(count: int = 1200) -> bytes:
    rows = []
    for index in range(count):
        rows.append({
            "problem": f"Official problem {index}", "answer": f"a_{index}",
            "domain": [f"Mathematics -> Domain {index % 4}"],
            "difficulty": float(index % 10 + 1), "source": f"competition_{index % 7}",
        })
    return "".join(json.dumps(row) + "\n" for row in rows).encode()


def _source(tmp_path: Path, count: int = 1200) -> dict:
    payload = _official_payload(count)
    path = tmp_path / "Omni-Math.jsonl"
    path.write_bytes(payload)
    return {
        "repository": "KbsdJames/Omni-MATH",
        "revision": "1" * 40,
        "path": "Omni-Math.jsonl",
        "git_blob_sha1": git_blob_sha1(payload),
        "local_path": str(path),
    }


def _excluded_manifest(problem: str) -> dict:
    digest = content_sha256(problem)
    roles = {}
    for role, suffix in (("reference", "r"), ("development", "d"), ("test", "t")):
        roles[role] = [{
            "role": role, "item_id": suffix, "source_id": suffix,
            "problem": problem if role == "test" else f"excluded-{suffix}",
            "reference_answer": "x",
            "content_sha256": digest if role == "test" else content_sha256(f"excluded-{suffix}"),
        }]
    manifest = {"schema_version": 2, "protocol_seed": 1, "roles": roles, "counts": {key: 1 for key in roles}}
    manifest["manifest_sha256"] = sha256_json(manifest)
    return manifest


def test_official_loader_authenticates_and_normalizes_required_metadata(tmp_path):
    source = _source(tmp_path, 3)
    rows, authentication = load_official_omni_math(source)
    assert len(rows) == 3
    assert rows[0]["domain"] == ["Mathematics -> Domain 0"]
    assert authentication["git_blob_sha1"] == source["git_blob_sha1"]
    broken = dict(source, git_blob_sha1="0" * 40)
    with pytest.raises(ValueError, match="blob mismatch"):
        load_official_omni_math(broken)


def test_manifest_accounts_for_official_rows_with_incomplete_fields(tmp_path):
    rows = [json.loads(line) for line in _official_payload(1200).decode().splitlines()]
    rows.append({
        "problem": "Official unclassified problem", "answer": "x", "domain": [],
        "difficulty": 5.0, "source": "competition_unclassified",
    })
    rows.append({
        "problem": "Official unanswered problem", "answer": "", "domain": ["D"],
        "difficulty": 5.0, "source": "competition_unanswered",
    })
    payload = "".join(json.dumps(row) + "\n" for row in rows).encode()
    path = tmp_path / "Omni-Math.jsonl"
    path.write_bytes(payload)
    source = {
        "repository": "KbsdJames/Omni-MATH", "revision": "1" * 40,
        "path": "Omni-Math.jsonl", "git_blob_sha1": git_blob_sha1(payload),
        "local_path": str(path),
    }
    loaded, authentication = load_official_omni_math(source)
    assert len(loaded) == authentication["rows"] == 1202
    manifest = build_manifest_with_exclusions({
        "strategy": "stratified_omni_math_v1", "protocol_seed": 20260824,
        "counts": {"reference": 300, "development": 300, "test": 500},
        "source": source,
    }, [])
    assert manifest["deduplication"]["source_rows"] == 1202
    assert manifest["deduplication"]["source_rows_ineligible_removed"] == 2
    assert manifest["deduplication"]["source_rows_without_domain_removed"] == 1
    assert manifest["deduplication"]["source_rows_without_answer_removed"] == 1
    assert all(
        item["problem"] not in {"Official unclassified problem", "Official unanswered problem"}
        for items in manifest["roles"].values()
        for item in items
    )


def test_omnimath_reserves_three_seeds_for_test_only():
    campaign = load_campaign("configs/science_campaign/omnimath_confirmatory_v1.yaml")
    assert role_seeds(campaign, "reference") == [0]
    assert role_seeds(campaign, "development") == [0]
    assert role_seeds(campaign, "test") == [0, 1, 2]


def test_omnimath_manifest_is_deterministic_stratified_disjoint_and_excludes_math500(tmp_path):
    excluded_path = tmp_path / "math500.json"
    save_manifest(excluded_path, _excluded_manifest("Official problem 0"))
    spec = {
        "strategy": "stratified_omni_math_v1", "protocol_seed": 20260824,
        "counts": {"reference": 300, "development": 300, "test": 500},
        "source": _source(tmp_path),
    }
    first = build_manifest_with_exclusions(spec, [excluded_path])
    second = build_manifest_with_exclusions(spec, [excluded_path])
    assert first == second
    assert first["counts"] == {"reference": 300, "development": 300, "test": 500}
    hashes = {role: {row["content_sha256"] for row in items} for role, items in first["roles"].items()}
    assert not hashes["reference"] & hashes["development"]
    assert not hashes["reference"] & hashes["test"]
    assert not hashes["development"] & hashes["test"]
    assert content_sha256("Official problem 0") not in set().union(*hashes.values())
    assert all(row["domains"] and row["source_provenance"] for items in first["roles"].values() for row in items)
    assert sum(value["selected"] for value in first["test_stratification"]["strata"].values()) == 500

    leaked = copy.deepcopy(first)
    leaked["roles"]["test"][0]["content_sha256"] = content_sha256("Official problem 0")
    with pytest.raises(ValueError, match="reuses"):
        assert_no_manifest_overlap(leaked, [excluded_path])


def _analysis_row(item: str, variant: str, seed: int, correct: int, alarms: int = 0):
    return {
        "item_id": f"{item}|replicate:{seed}", "source_item_id": item, "replicate_seed": seed,
        "variant": variant, "token_ids": [seed, 7],
        "scorer": {"correct": float(correct), "scorable": True, "answer_extraction_ok": True},
        "item_metadata": {"domains": ["D"], "difficulty": 7.0}, "truncated": False,
        "token_accounting": {"sampled": 2, "emitted": 2, "deleted": 0, "alarms": alarms,
                             "rerolls": 0, "termination_reason": "eos"},
        "timing": {"wall_seconds": 0.1, "peak_vram_bytes": 0},
    }


def test_three_variants_are_paired_by_problem_and_seed_and_no_alarm_identity_is_enforced():
    vanilla = [_analysis_row(item, "vanilla", seed, seed % 2) for item in ("a", "b") for seed in (0, 1, 2)]
    full = [dict(row, variant="full_mgtb") for row in vanilla]
    random = [dict(row, variant="matched_random") for row in vanilla]
    result = campaign_analysis(
        {"vanilla": vanilla, "full_mgtb": full, "matched_random": random}, baseline="vanilla",
        bootstrap_samples=100, require_no_alarm_identity=["full_mgtb"],
    )
    assert result["comparisons"]["full_mgtb"]["problem_clusters"] == 2
    assert result["comparisons"]["full_mgtb"]["paired_units"] == 6
    broken = copy.deepcopy(full)
    broken[0]["token_ids"] = [999]
    with pytest.raises(ValueError, match="diverged"):
        campaign_analysis(
            {"vanilla": vanilla, "full_mgtb": broken}, baseline="vanilla", bootstrap_samples=20,
            require_no_alarm_identity=["full_mgtb"],
        )


def test_pending_score_and_mock_judge_keep_raw_verdict_separate(tmp_path):
    pending = pending_omni_math_score("work \\boxed{2}", "2")
    assert pending["correct"] is None and pending["scorable"] is False
    report = (
        "## Student Final Answer\n2\n\n## Equivalence Judgement\nTRUE\n\n"
        "## Justification\nEquivalent exact values.\n\n=== report over ==="
    )
    assert parse_omni_judge_report(report)["correct"] == 1.0

    campaign = load_campaign("configs/science_campaign/omnimath_confirmatory_v1.yaml")
    campaign["output_root"] = str(tmp_path / "campaign")
    campaign["seeds"] = [0]
    item = {
        "item_id": "omni:item", "source_id": "line:1", "content_sha256": content_sha256("p"),
        "problem": "p", "reference_answer": "2", "item_seed": 1,
    }
    manifest = {"protocol_seed": 1, "manifest_sha256": "m", "roles": {"test": [item]}}
    unit = campaign_units(manifest, "test", [0])[0]
    generation = {
        **unit, "generation": "\\boxed{2}", "artifact_sha256": "raw-generation",
        "scorer": pending,
    }
    judgments = run_judging(
        campaign=campaign, manifest=manifest, role="test", variant="vanilla",
        generation_rows=[generation], freeze={"freeze_sha256": "f"},
        predictor=lambda payloads: [{"raw_report": report, "sampled_tokens": 12, "wall_seconds": 0.2}],
    )
    merged = merge_judgments([generation], judgments)
    assert merged[0]["scorer"]["correct"] == 1.0
    assert merged[0]["judge"]["raw_report"] == report


def test_omnimath_calibration_and_profile_do_not_use_correctness_labels():
    campaign = load_campaign("configs/science_campaign/omnimath_confirmatory_v1.yaml")
    spec = calibration_spec(campaign, "full")
    features = {
        "window_index": 0, "start_pos": 0, "end_pos": 64, "mean_entropy": 1.0,
        "mean_logprob": -1.0, "repetition_rate": 0.0, "confident_loop_score": 0.0,
        "local_entropy_log_ratio": 0.0, "local_entropy_pos": 0.0, "local_entropy_neg": 0.0,
    }
    artifacts = [{
        "item_id": f"item-{index}", "content_sha256": str(index), "truncated": False,
        "scorer": {"correct": correctness, "answer_extraction_ok": bool(correctness)},
        "monitor_trace": [{"type": "window", "features": features}],
    } for index, correctness in enumerate((0.0, 1.0))]
    calibrator, summary = build_calibrator(artifacts, spec, {"source": "test"})
    assert summary["healthy_retained"] == 2
    assert calibrator["correctness_labels_used"] is False

    development = [{
        "item_id": "dev|replicate:0", "artifact_sha256": "raw", "prompt_token_count": 100,
        "scorer": {"correct": 1.0}, "token_accounting": {"emitted": 500, "deleted": 96},
        "backtracks": [{"rollback_span": 96, "alert": {"token_pos": 356}}],
    }]
    profile = build_profile(development, {"correctness_labels_used": False})
    assert profile["templates"][0]["backtrack_count"] == 1
    assert profile["templates"][0]["observed_trigger_positions"] == [256]
    assert "scorer" not in json.dumps(profile)


def test_omnimath_freeze_rejects_config_mutation(tmp_path, monkeypatch):
    from mgtb_v3.science_campaign import runner
    from mgtb_v3.science_fast.artifacts import RunStore
    campaign = load_campaign("configs/science_campaign/omnimath_confirmatory_v1.yaml")
    campaign["output_root"] = str(tmp_path / "campaign")
    manifest = {
        "protocol_seed": 7, "manifest_sha256": "manifest", "dataset_revisions": {},
        "source_authentication": {"revision": "x"},
        "roles": {role: [{"item_id": role, "source_id": role, "content_sha256": role}]
                  for role in ("reference", "development", "test")},
    }
    development_root = Path(campaign["output_root"]) / "runs" / "development" / "full_mgtb"
    store = RunStore(development_root, {"test": "development-source"})
    source_hashes = []
    for unit in campaign_units(manifest, "development", role_seeds(campaign, "development")):
        store.save(unit, {
            "generation": "x", "token_ids": [1], "scorer": {"correct": None},
            "token_accounting": {"sampled": 1, "emitted": 1, "deleted": 0},
            "timing": {"wall_seconds": 0.0},
            "provenance": {"run_identity_sha256": store.identity_sha256},
        })
        source_hashes.append(store.valid_artifact(unit)["artifact_sha256"])
    profile_path = tmp_path / "profile.json"
    profile = {"schema_version": 1, "templates": [{"rollback_lengths": [], "reference_primary_tokens": 10}],
               "summary": {"num_examples": 1}, "source": {
                   "campaign_id": campaign["campaign_id"], "manifest_sha256": "manifest",
                   "variant": "full_mgtb", "role": "development", "seeds": [0],
                   "correctness_labels_used": False, "source_artifact_sha256": sorted(source_hashes),
               }}
    profile["profile_sha256"] = sha256_json(profile)
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    campaign["variants"]["matched_random"]["profile"] = str(profile_path)
    calibration_root = Path(campaign["output_root"]) / "calibration" / "full"
    calibration_root.mkdir(parents=True)
    calibrator = {"buckets": [], "score_pools_by_bucket": {}, "p_clip": 0.1}
    calibrator["calibrator_sha256"] = sha256_json(calibrator)
    threshold = {"selected_h": 1.0}
    threshold["threshold_sha256"] = sha256_json(threshold)
    (calibration_root / "calibrator.json").write_text(json.dumps(calibrator), encoding="utf-8")
    (calibration_root / "threshold.json").write_text(json.dumps(threshold), encoding="utf-8")
    (calibration_root / "reference_summary.json").write_text(json.dumps({"completed": 1}), encoding="utf-8")
    monkeypatch.setattr(runner, "git_commit", lambda: "git")
    monkeypatch.setattr(runner, "source_tree_sha256", lambda: "tree")
    monkeypatch.setattr(runner, "software_environment", lambda: {"python": "test"})
    campaign["exclude_manifests"] = []
    freeze = build_freeze(campaign, manifest)
    assert_freeze(campaign, manifest, freeze, "vanilla")
    campaign["generation"]["max_new_tokens"] = 19999
    with pytest.raises(ValueError, match="config mismatch"):
        assert_freeze(campaign, manifest, freeze, "vanilla")
