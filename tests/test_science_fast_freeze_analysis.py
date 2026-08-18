import copy
import json

import pytest

from mgtb_v3.science_fast.analysis import paired_analysis
from mgtb_v3.science_fast.freeze import build_freeze, validate_freeze
from mgtb_v3.science_fast.io import sha256_json
from mgtb_v3.science_fast.protocol import build_manifest
from mgtb_v3.science_fast.provenance import git_commit, source_tree_sha256
from mgtb_v3.science_fast.runner import _assert_runtime_matches_freeze, run_role


def _rows(prefix, count):
    return [{"id": f"{prefix}-{i}", "problem": f"{prefix} p {i}", "answer": "1"} for i in range(count)]


def test_test_role_refuses_to_start_without_freeze_before_model_load(tmp_path):
    manifest = build_manifest(_rows("train", 410), _rows("test", 310))
    with pytest.raises(ValueError, match="freeze"):
        run_role(settings={}, manifest=manifest, role="test", method="vanilla", output_dir=tmp_path)


def test_freeze_covers_test_ids_and_detects_mutation(tmp_path):
    manifest = build_manifest(_rows("train", 410), _rows("test", 310))
    scorer = tmp_path / "scorer.py"
    scorer.write_text("VERSION = 1\n", encoding="utf-8")
    threshold = {"selected_h": 3.0, "threshold_sha256": "threshold"}
    calibrator = {"calibrator_sha256": "calibrator"}
    config = {"model": {"name": "m", "revision": "r"}, "quantization": {"scheme": "int4"},
              "device_map": {"": 0}, "controller": {"cache_state_mode": "replay_last"}, "max_new_tokens": 20000}
    freeze = build_freeze(manifest=manifest, resolved_config=config, calibrator=calibrator, threshold=threshold,
                          method="vanilla", source={"git": "x"}, environment={"python": "x"}, scorer_path=scorer)
    validate_freeze(freeze, manifest=manifest, method="vanilla")
    broken = copy.deepcopy(freeze)
    broken["test_items"][0]["content_sha256"] = "bad"
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_freeze(broken, manifest=manifest, method="vanilla")


def test_runtime_controller_matches_json_round_tripped_freeze():
    settings = {
        "model": {"name": "m", "revision": "r"},
        "quantization": {"scheme": "int4"},
        "device_map": {"": 0},
        "controller": {"detector": {"betting_gammas": (0.1, 0.3)}},
        "max_new_tokens": 20000,
    }
    freeze = {
        "model": settings["model"],
        "quantization": settings["quantization"],
        "device_map": settings["device_map"],
        "resolved_controller_config": settings["controller"],
        "max_new_tokens": settings["max_new_tokens"],
        "source": {"git_commit": git_commit(), "source_tree_sha256": source_tree_sha256()},
    }
    reloaded_freeze = json.loads(json.dumps(freeze))

    _assert_runtime_matches_freeze(settings, reloaded_freeze, None, None)


def _result(item_id, correct, alarms=0, sampled=10, emitted=10, deleted=0):
    return {"item_id": item_id, "scorer": {"correct": correct, "answer_extraction_ok": True}, "truncated": False,
            "token_accounting": {"alarms": alarms, "rerolls": alarms, "sampled": sampled, "emitted": emitted,
                                 "deleted": deleted, "alarm_positions": [5] * alarms, "rollback_spans": [2] * alarms,
                                 "termination_reason": "eos"}, "timing": {"wall_seconds": 1.0, "peak_vram_bytes": None}}


def test_paired_analysis_reconstructs_corrections_regressions_and_costs():
    vanilla = [_result("a", True), _result("b", False), _result("c", True)]
    mgtb = [_result("a", True), _result("b", True, 1, 14, 12, 2), _result("c", False)]
    result = paired_analysis(vanilla, mgtb, samples=1000)
    assert result["corrections"] == 1
    assert result["regressions"] == 1
    assert result["mcnemar_exact_two_sided_p"] == 1.0
    assert result["mgtb"]["sampled_tokens"] == 34
    assert result["mgtb"]["deleted_tokens"] == 2
