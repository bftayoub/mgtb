#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mgtb_v3.config import load_config
from mgtb_v3.features.window_features import TrajectoryMonitor, linear_window_score


def extract_features(input_path: str, output_path: str, config_path: str) -> None:
    cfg = load_config(config_path)
    with Path(output_path).open("w", encoding="utf-8") as out:
        with Path(input_path).open("r", encoding="utf-8") as handle:
            for run_idx, line in enumerate(handle):
                if not line.strip():
                    continue
                trace = json.loads(line)
                prompt_tokens = trace.get("prompt_tokens", [])
                monitor = TrajectoryMonitor(cfg, prompt_tokens=prompt_tokens)
                token_rows = trace.get("tokens", [])
                if token_rows and isinstance(token_rows[0], int):
                    raise ValueError("full extraction needs logits per token; provide rows with token_id and logits")
                for row in token_rows:
                    if "logits" not in row:
                        raise ValueError("logits unavailable: entropy and chosen logprob require pre-sampling logits")
                    monitor.update_token(row["token_id"], row["logits"])
                    while monitor.should_emit_window():
                        features = monitor.compute_window_features()
                        out.write(
                            json.dumps(
                                {
                                    "run_id": trace.get("run_id", run_idx),
                                    "features": features.to_dict(),
                                    "score": linear_window_score(features, cfg.score),
                                }
                            )
                            + "\n"
                        )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Extract MGT-B v3 window features from JSONL traces.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/mgtb_v3_default.yaml")
    args = parser.parse_args(argv)
    extract_features(args.input, args.output, args.config)


if __name__ == "__main__":
    main()
