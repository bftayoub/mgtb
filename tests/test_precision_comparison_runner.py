import json

import pytest

import scripts.run_precision_comparison as runner
from mgtb_v3.types import GenerationResult


class DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, prompt, return_tensors=None):
        return {"input_ids": [10, 11]}

    def decode(self, tokens, skip_special_tokens=True):
        return "#### 4" if tokens else ""


def test_precision_comparison_gsm8k_matrix_with_mocks(tmp_path, monkeypatch, capsys):
    output_dir = tmp_path / "out"
    calibrator_path = tmp_path / "calibrator.json"
    threshold_path = tmp_path / "threshold.json"
    calibrator_path.write_text(
        json.dumps(
            {
                "buckets": [[0, 512]],
                "p_clip": 1e-6,
                "score_pools_by_bucket": {"0-512": [0.0, 1.0]},
            }
        ),
        encoding="utf-8",
    )
    threshold_path.write_text(json.dumps({"threshold": 2.0}), encoding="utf-8")

    def fake_load_gsm8k_items(**kwargs):
        return [
            {
                "id": "gsm8k-test-000000",
                "dataset": "gsm8k",
                "split": "test",
                "question": "What is 2 + 2?",
                "answer": "Reasoning #### 4",
                "reference_answer": "4",
                "prompt": "Question: What is 2 + 2?",
            },
            {
                "id": "gsm8k-test-000001",
                "dataset": "gsm8k",
                "split": "test",
                "question": "What is 1 + 3?",
                "answer": "Reasoning #### 4",
                "reference_answer": "4",
                "prompt": "Question: What is 1 + 3?",
            },
        ]

    def fake_load_model(model_name, precision, device_map, allow_cpu_fp32_fallback):
        return object(), DummyTokenizer(), precision

    def fake_run_vanilla(model, tokenizer, prompt, max_new_tokens):
        return {
            "text": f"{prompt}\n#### 4",
            "tokens": [10, 11, 12, 13],
            "alerts": [],
            "backtracks": [],
            "trace_log_path": None,
            "latency": 0.01,
        }

    def fake_generate_with_mgtb_v3(
        model,
        tokenizer,
        prompt,
        config,
        calibrator,
        threshold,
        max_new_tokens,
        trace_log_path,
        do_backtracking,
    ):
        result = GenerationResult(
            text=f"{prompt}\n#### 4",
            tokens=[10, 11, 12, 13],
            alerts=[],
            backtracks=[],
            trace_log_path=str(trace_log_path),
        )
        result.latency = 0.02
        return result

    monkeypatch.setattr(runner, "load_gsm8k_items", fake_load_gsm8k_items)
    monkeypatch.setattr(runner, "_load_model", fake_load_model)
    monkeypatch.setattr(runner, "_run_vanilla", fake_run_vanilla)
    monkeypatch.setattr(runner, "generate_with_mgtb_v3", fake_generate_with_mgtb_v3)

    runner.main(
        [
            "--base-model",
            "dummy-model",
            "--dataset",
            "gsm8k",
            "--output-dir",
            str(output_dir),
            "--limit",
            "2",
            "--methods",
            "vanilla",
            "mgtb_v3_window",
            "--precisions",
            "fp16",
            "int4",
            "--calibrator",
            str(calibrator_path),
            "--threshold",
            str(threshold_path),
            "--progress-interval",
            "1",
        ]
    )
    captured = capsys.readouterr()
    assert "[progress fp16 vanilla] 1/2 accuracy=1.000" in captured.out
    assert "[progress int4 mgtb_v3_window] 2/2 accuracy=1.000" in captured.out

    rows = [json.loads(line) for line in (output_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 8
    assert {row["method"] for row in rows} == {"vanilla", "mgtb_v3_window"}
    assert {row["precision"] for row in rows} == {"fp16", "int4"}
    assert all(row["dataset"] == "gsm8k" for row in rows)
    assert all(row["base_model"] == "dummy-model" for row in rows)
    assert all(row["calibration_key"] == "global" for row in rows if row["method"] == "mgtb_v3_window")
    assert all(row["reference_answer"] == "4" for row in rows)
    assert all(row["prediction_answer"] == "4" for row in rows)
    assert all(row["correct"] == 1.0 for row in rows)
    assert all(row["answer_extraction_ok"] is True for row in rows)
    assert all(isinstance(row["seed"], int) for row in rows)

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert set(summary["groups"]) == {
        "fp16::vanilla",
        "fp16::mgtb_v3_window",
        "int4::vanilla",
        "int4::mgtb_v3_window",
    }
    for group in summary["groups"].values():
        assert group["num_instances"] == 2
        assert group["accuracy"] == 1.0
        assert group["num_correct"] == 2
        assert group["num_answer_extraction_failures"] == 0


def test_mgtb_window_requires_calibration_artifacts(tmp_path):
    with pytest.raises(SystemExit, match="mgtb_v3_window requires"):
        runner.main(
            [
                "--base-model",
                "dummy-model",
                "--dataset",
                "gsm8k",
                "--output-dir",
                str(tmp_path / "out"),
                "--methods",
                "mgtb_v3_window",
                "--precisions",
                "fp16",
            ]
        )


def test_precision_specific_calibration_is_used(tmp_path, monkeypatch):
    output_dir = tmp_path / "out"
    fp16_calibrator = tmp_path / "calibrator_fp16.json"
    int4_calibrator = tmp_path / "calibrator_int4.json"
    fp16_threshold = tmp_path / "threshold_fp16.json"
    int4_threshold = tmp_path / "threshold_int4.json"
    calibrator_payload = {
        "buckets": [[0, 512]],
        "p_clip": 1e-6,
        "score_pools_by_bucket": {"0-512": [0.0, 1.0]},
    }
    for path in [fp16_calibrator, int4_calibrator]:
        path.write_text(json.dumps(calibrator_payload), encoding="utf-8")
    fp16_threshold.write_text(json.dumps({"threshold": 2.0}), encoding="utf-8")
    int4_threshold.write_text(json.dumps({"threshold": 3.0}), encoding="utf-8")
    run_config = tmp_path / "run.json"
    run_config.write_text(
        json.dumps(
            {
                "base_model": "dummy-model",
                "dataset": "gsm8k",
                "output_dir": str(output_dir),
                "limit": 1,
                "methods": ["mgtb_v3_window"],
                "precisions": ["fp16", "int4"],
                "calibration": {
                    "fp16": {"calibrator": str(fp16_calibrator), "threshold": str(fp16_threshold)},
                    "int4": {"calibrator": str(int4_calibrator), "threshold": str(int4_threshold)},
                },
                "progress_interval": 0,
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        runner,
        "load_gsm8k_items",
        lambda **kwargs: [
            {
                "id": "gsm8k-test-000000",
                "dataset": "gsm8k",
                "split": "test",
                "question": "What is 2 + 2?",
                "answer": "Reasoning #### 4",
                "reference_answer": "4",
                "prompt": "Question: What is 2 + 2?",
            }
        ],
    )
    monkeypatch.setattr(runner, "_load_model", lambda model_name, precision, device_map, allow_cpu_fp32_fallback: (object(), DummyTokenizer(), precision))

    seen_thresholds = []

    def fake_generate_with_mgtb_v3(
        model,
        tokenizer,
        prompt,
        config,
        calibrator,
        threshold,
        max_new_tokens,
        trace_log_path,
        do_backtracking,
    ):
        seen_thresholds.append(threshold)
        result = GenerationResult(text=f"{prompt}\n#### 4", tokens=[10, 11, 12], trace_log_path=str(trace_log_path))
        result.latency = 0.01
        return result

    monkeypatch.setattr(runner, "generate_with_mgtb_v3", fake_generate_with_mgtb_v3)

    runner.main(["--run-config", str(run_config)])

    assert seen_thresholds == [2.0, 3.0]
    rows = [json.loads(line) for line in (output_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[0]["calibration_key"] == "fp16"
    assert rows[0]["calibrator_path"] == str(fp16_calibrator)
    assert rows[0]["threshold_path"] == str(fp16_threshold)
    assert rows[1]["calibration_key"] == "int4"
    assert rows[1]["calibrator_path"] == str(int4_calibrator)
    assert rows[1]["threshold_path"] == str(int4_threshold)
