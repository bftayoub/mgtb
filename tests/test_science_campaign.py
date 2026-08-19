from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from mgtb_v3.config import BacktrackingConfig, DetectorConfig, MGTBV3Config, ScoreConfig, WindowConfig
from mgtb_v3.detector.e_detector import EDetector
from mgtb_v3.generation.hf_loop import generate_with_mgtb_v3
from mgtb_v3.science_campaign.analysis import campaign_analysis
from mgtb_v3.science_campaign.calibration import build_calibrator, select_threshold
from mgtb_v3.science_campaign.config import calibration_spec, load_campaign, resolve_variant
from mgtb_v3.science_campaign.manifest import assert_independent_test, build_manifest, save_manifest
from mgtb_v3.science_campaign import runner


class _Tokenizer:
    eos_token_id = 99
    def __call__(self, text, return_tensors=None, add_special_tokens=True):
        ids = [7, 8]
        return {"input_ids": torch.tensor([ids]) if return_tensors == "pt" else ids}
    def decode(self, ids, skip_special_tokens=True):
        return " ".join(map(str, ids))


class _Model:
    device = torch.device("cpu")
    def eval(self):
        return None
    def __call__(self, input_ids, past_key_values=None, use_cache=True):
        logits = torch.tensor([[[0.1, 0.2, 0.3, 0.4]]])
        return type("Output", (), {"logits": logits, "past_key_values": None})()


class _Calibrator:
    def p_value(self, score, token_pos):
        return 1.0


def _small_config():
    return MGTBV3Config(
        window=WindowConfig(window_size=2, stride=1, ngram_min=1, ngram_max=1),
        detector=DetectorConfig(refractory_windows=0),
        backtracking=BacktrackingConfig(max_rerolls=1, margin_tokens=0), score=ScoreConfig(),
    )


def test_no_reset_detector_does_not_apply_cusum_floor():
    reset = EDetector(1e9, accumulation_mode="cusum_reset")
    no_reset = EDetector(1e9, accumulation_mode="no_reset")
    reset.update(1.0)
    no_reset.update(1.0)
    reset_second = reset.update(1e-6)["logE"]
    no_reset_second = no_reset.update(1e-6)["logE"]
    assert no_reset_second < reset_second


def test_forced_schedule_uses_repair_operator_and_empty_schedule_disables_detector():
    torch.manual_seed(3)
    forced = generate_with_mgtb_v3(
        _Model(), _Tokenizer(), "p", _small_config(), _Calibrator(), float("inf"),
        max_new_tokens=5, forced_alert_schedule=[{"trigger_at": 2, "rollback_tokens": 1}],
    )
    assert len(forced.backtracks) == 1
    assert forced.backtracks[0]["trigger_source"] == "forced_schedule"
    assert forced.backtracks[0]["rollback_span"] == 1

    empty = generate_with_mgtb_v3(
        _Model(), _Tokenizer(), "p", _small_config(), _Calibrator(), 1.000001,
        max_new_tokens=5, forced_alert_schedule=[],
    )
    assert empty.alerts == []
    assert empty.backtracks == []


def test_campaign_config_resolves_feature_reuse_and_repair_overrides():
    campaign = load_campaign("configs/science_campaign/math500_exploratory_ablations.yaml")
    assert calibration_spec(campaign, "entropy_only")["feature_source"] == "full"
    rollback = resolve_variant(campaign, "repair_rollback_only")
    assert rollback["controller"]["backtracking"]["redecode_temperature"] == 1.0
    assert rollback["controller"]["backtracking"]["use_no_bad_ngrams"] is False


def _feature_artifact(index):
    features = {
        "window_index": 0, "start_pos": 0, "end_pos": 64,
        "mean_entropy": 1.0 + index / 100, "mean_logprob": -1.0,
        "repetition_rate": 0.1, "confident_loop_score": 0.0,
        "local_entropy_log_ratio": 0.0, "local_entropy_pos": 0.0, "local_entropy_neg": 0.0,
    }
    return {
        "item_id": f"item-{index}", "content_sha256": f"hash-{index}",
        "scorer": {"correct": 1.0, "answer_extraction_ok": True}, "truncated": False,
        "monitor_trace": [{"type": "window", "features": features}],
    }


def test_campaign_calibration_rescores_shared_features_and_authenticates_outputs():
    campaign = load_campaign("configs/science_campaign/math500_exploratory_ablations.yaml")
    spec = calibration_spec(campaign, "global")
    artifacts = [_feature_artifact(index) for index in range(35)]
    calibrator, summary = build_calibrator(artifacts, spec, {"source": "test"})
    threshold = select_threshold(artifacts, calibrator, spec)
    assert calibrator["calibration_mode"] == "global"
    assert calibrator["windows_per_bucket"] == {"0-inf": 35}
    assert summary["healthy_retained"] == 35
    assert threshold["healthy_denominator"] == 35
    assert threshold["calibrator_sha256"] == calibrator["calibrator_sha256"]


def test_generic_manifest_is_disjoint_and_confirmatory_exclusion_detects_reuse(tmp_path):
    source = tmp_path / "rows.jsonl"
    source.write_text("".join(json.dumps({"id": i, "question": f"q{i}", "answer": str(i)}) + "\n" for i in range(12)), encoding="utf-8")
    common = {"jsonl": str(source), "dataset_kind": "gsm8k", "count": 2,
              "fields": {"problem": "question", "answer": "answer", "id": "id"}}
    manifest = build_manifest({"protocol_seed": 9, "roles": {
        "reference": {**common, "count": 3}, "development": common, "test": common,
    }})
    hashes = [{item["content_sha256"] for item in manifest["roles"][role]} for role in manifest["roles"]]
    assert not (hashes[0] & hashes[1] or hashes[0] & hashes[2] or hashes[1] & hashes[2])
    old = tmp_path / "old.json"
    save_manifest(old, manifest)
    with pytest.raises(ValueError, match="reuses"):
        assert_independent_test(manifest, [old])


def _analysis_row(item, variant, correct, seed=0):
    return {
        "source_item_id": item, "replicate_seed": seed, "variant": variant,
        "scorer": {"correct": correct, "answer_extraction_ok": True}, "truncated": False,
        "token_accounting": {"sampled": 10, "emitted": 10, "deleted": 0, "alarms": 0,
                             "rerolls": 0, "termination_reason": "eos"},
        "timing": {"wall_seconds": 1.0, "peak_vram_bytes": 1},
    }


def test_campaign_analysis_pairs_units_and_adjusts_multiple_comparisons():
    vanilla = [_analysis_row("a", "vanilla", 0), _analysis_row("b", "vanilla", 1)]
    better = [_analysis_row("a", "better", 1), _analysis_row("b", "better", 1)]
    same = [_analysis_row("a", "same", 0), _analysis_row("b", "same", 1)]
    result = campaign_analysis({"vanilla": vanilla, "better": better, "same": same}, baseline="vanilla", bootstrap_samples=100)
    assert result["comparisons"]["better"]["corrections"] == 1
    assert "mcnemar_holm_adjusted_p" in result["comparisons"]["same"]


def test_campaign_run_checkpoints_and_resumes_per_seed_unit(tmp_path, monkeypatch):
    campaign = load_campaign("configs/science_campaign/math500_exploratory_ablations.yaml")
    campaign["output_root"] = str(tmp_path / "campaign")
    campaign["seeds"] = [0, 1]
    campaign["variants"] = {"vanilla": {"kind": "vanilla"}}
    rows = [{"id": i, "problem": f"p{i}", "answer": "1"} for i in range(8)]
    spec = {"protocol_seed": 4, "roles": {role: {"jsonl": str(tmp_path / f"{role}.jsonl"), "count": 2}
                                                for role in ("reference", "development", "test")}}
    for role in spec["roles"]:
        Path(spec["roles"][role]["jsonl"]).write_text(
            "".join(json.dumps({**row, "problem": f"{role}-{row['problem']}"}) + "\n" for row in rows), encoding="utf-8"
        )
    manifest = build_manifest(spec)
    calls = []
    monkeypatch.setattr(runner, "_build_context", lambda worker_spec: worker_spec)

    def fake_generate(item, context):
        calls.append(item["item_id"])
        return {
            "generation": "#### 1", "token_ids": [1], "scorer": {"correct": 1.0, "answer_extraction_ok": True},
            "token_accounting": {"sampled": 1, "emitted": 1, "deleted": 0, "alarms": 0, "rerolls": 0,
                                 "alarm_positions": [], "rollback_spans": [], "termination_reason": "eos"},
            "timing": {"wall_seconds": 0.1, "peak_vram_bytes": None},
            "provenance": {"run_identity_sha256": context["run_identity_sha256"]},
        }
    monkeypatch.setattr(runner, "_generate_item", fake_generate)
    partial = runner.run_variant(campaign=campaign, manifest=manifest, role="development", variant_name="vanilla",
                                 freeze=None, workers=1, stop_after=2)
    assert len(partial) == 2
    complete = runner.run_variant(campaign=campaign, manifest=manifest, role="development", variant_name="vanilla",
                                  freeze=None, workers=1)
    assert len(complete) == 4
    assert len(calls) == 4
    progress = json.loads((tmp_path / "campaign/runs/development/vanilla/progress.json").read_text())
    assert progress["completed"] == 4
