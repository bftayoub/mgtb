import json
from pathlib import Path
from types import SimpleNamespace

import torch

import scripts.calibrate_precision as calibration


class DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 999

    def __call__(self, prompt, return_tensors=None):
        return {"input_ids": torch.tensor([[10, 11]])}

    def decode(self, tokens, skip_special_tokens=True):
        return "#### 4" if list(tokens) else ""


class DummyModel:
    device = torch.device("cpu")

    def eval(self):
        return None

    def __call__(self, input_ids, use_cache=True, past_key_values=None):
        logits = torch.full((1, 1, 16), -1e9)
        logits[:, :, 4] = 0.0
        return SimpleNamespace(logits=logits, past_key_values=None)


def _write_small_mgtb_config(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "window:",
                "  window_size: 2",
                "  stride: 1",
                "  ngram_min: 1",
                "  ngram_max: 1",
                "detector:",
                "  target_false_alert_rate: 1.0",
                "  threshold: null",
                "  p_clip: 1.0e-6",
                "  betting_gammas: [0.1]",
                "  refractory_windows: 0",
                "score:",
                "  w_entropy: 0.0",
                "  w_logprob: 0.0",
                "  w_repetition: 1.0",
                "  w_confident_loop: 0.0",
                "  w_local_entropy_pos: 0.0",
                "  w_local_entropy_neg: 0.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_load_calibration_settings_reads_required_fields(tmp_path):
    config_path = tmp_path / "calibration.yaml"
    config_path.write_text(
        "\n".join(
            [
                "base_model: dummy-model",
                "dataset: gsm8k",
                "limit: 7",
                "seed: 123",
                "precisions: [fp16]",
                "output_dir: outputs/calibration/test",
                "mu0_quantile: 0.75",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    settings = calibration.load_calibration_settings(config_path)

    assert settings["base_model"] == "dummy-model"
    assert settings["limit"] == 7
    assert settings["seed"] == 123
    assert settings["precisions"] == ["fp16"]
    assert settings["output_dir"] == "outputs/calibration/test"
    assert settings["mu0_quantile"] == 0.75


def test_calibrate_precision_writes_outputs_per_precision(tmp_path, monkeypatch):
    output_dir = tmp_path / "calibration"
    mgtb_config = tmp_path / "mgtb.yaml"
    _write_small_mgtb_config(mgtb_config)
    settings = {
        "base_model": "dummy-model",
        "model": "dummy-model",
        "dataset": "gsm8k",
        "split": "test",
        "limit": 2,
        "seed": 0,
        "max_new_tokens": 4,
        "precisions": ["fp16", "int4"],
        "config": str(mgtb_config),
        "output_dir": str(output_dir),
        "mu0_quantile": 0.5,
        "healthy_filter": {"correct_only": True, "exclude_truncated": False},
        "min_healthy_examples": 1,
        "device_map": "auto",
        "allow_cpu_fp32_fallback": False,
        "progress_interval": 0,
    }
    items = [
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

    monkeypatch.setattr(calibration, "_load_calibration_items", lambda loaded_settings: items)
    monkeypatch.setattr(calibration, "_load_model", lambda model_name, precision, device_map, allow_cpu_fp32_fallback: (DummyModel(), DummyTokenizer(), precision))

    manifest = calibration.run_calibration(settings)

    assert set(manifest["precisions"]) == {"fp16", "int4"}
    manifest_path = output_dir / "calibration_manifest.json"
    assert manifest_path.exists()
    for precision in ["fp16", "int4"]:
        precision_dir = output_dir / precision
        assert (precision_dir / "all_results.jsonl").exists()
        assert (precision_dir / "healthy_results.jsonl").exists()
        assert (precision_dir / "window_features.jsonl").exists()
        assert (precision_dir / "calibrator.json").exists()
        assert (precision_dir / "threshold.json").exists()
        assert (precision_dir / "calibration_summary.json").exists()

        feature_rows = [
            json.loads(line)
            for line in (precision_dir / "window_features.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert feature_rows
        scores = [row["score"] for row in feature_rows]
        summary = json.loads((precision_dir / "calibration_summary.json").read_text(encoding="utf-8"))
        assert summary["num_examples"] == 2
        assert summary["num_healthy_examples"] == 2
        assert summary["mu0"] == calibration._quantile(scores, 0.5)
        assert summary["calibrator_path"] == str(precision_dir / "calibrator.json")
        assert summary["threshold_path"] == str(precision_dir / "threshold.json")


def test_quantile_interpolates():
    assert calibration._quantile([1.0, 2.0, 3.0], 0.5) == 2.0
    assert calibration._quantile([1.0, 3.0], 0.25) == 1.5
