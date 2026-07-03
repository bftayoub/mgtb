#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from mgtb_v3.calibration.positional import PositionalCalibrator
from mgtb_v3.config import load_config
from mgtb_v3.generation.hf_loop import generate_with_mgtb_v3


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Run HuggingFace generation monitored by MGT-B v3.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--config", default="configs/mgtb_v3_default.yaml")
    parser.add_argument("--calibrator", required=True)
    parser.add_argument("--threshold", required=True, help="threshold JSON or numeric threshold")
    parser.add_argument("--method", default="mgtb_v3_window", choices=["vanilla", "mgtb_v3_window"])
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--output", required=True)
    parser.add_argument("--trace-log")
    args = parser.parse_args(argv)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = load_config(args.config)
    calibrator = PositionalCalibrator.load_json(args.calibrator)
    threshold = _load_threshold(args.threshold)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16 if torch.cuda.is_available() else None, device_map="auto")
    result = generate_with_mgtb_v3(
        model,
        tokenizer,
        args.prompt,
        cfg,
        calibrator,
        threshold,
        max_new_tokens=args.max_new_tokens,
        trace_log_path=args.trace_log,
        do_backtracking=args.method == "mgtb_v3_window",
    )
    payload = asdict(result)
    Path(args.output).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _load_threshold(value: str) -> float:
    path = Path(value)
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        return float(data["threshold"])
    return float(value)


if __name__ == "__main__":
    main()
