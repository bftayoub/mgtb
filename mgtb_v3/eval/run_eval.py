from __future__ import annotations

import argparse
import json
from pathlib import Path

from mgtb_v3.eval.ablations import ABLATION_MODES
from mgtb_v3.eval.metrics import compute_metrics


def run_eval(input_path: str | Path, mode: str) -> dict:
    if mode not in ABLATION_MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {ABLATION_MODES}")
    rows = []
    with Path(input_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    metrics = compute_metrics(rows)
    metrics["mode"] = mode
    return metrics


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--mode", required=True, choices=ABLATION_MODES)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    result = run_eval(args.input, args.mode)
    text = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
