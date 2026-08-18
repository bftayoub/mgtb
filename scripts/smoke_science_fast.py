#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mgtb_v3.detector.e_detector import EDetector
from mgtb_v3.science_fast.analysis import paired_analysis
from mgtb_v3.science_fast.artifacts import RunStore
from mgtb_v3.science_fast.io import atomic_write_json
from mgtb_v3.science_fast.protocol import content_sha256, item_seed


def main(argv=None):
    parser = argparse.ArgumentParser(description="CPU-only synthetic MGT-B protocol smoke")
    parser.add_argument("--output", default="outputs/science_fast/smoke/result.json")
    args = parser.parse_args(argv)
    items = [{"item_id": f"synthetic-{i}", "content_sha256": content_sha256(f"problem {i}"),
              "item_seed": item_seed(20260811, f"synthetic-{i}")} for i in range(4)]
    identity = {"manifest": "synthetic-not-math500", "config": "science-fast-smoke-v1"}

    def worker(method):
        def run(item):
            rng = random.Random(item["item_seed"])
            tokens = [rng.randrange(32) for _ in range(8)]
            alarms, rerolls, deleted = 0, 0, 0
            if method == "forced" and item["item_id"] == "synthetic-0":
                detector = EDetector(1.01)
                assert detector.update(1e-6)["alert"]
                alarms = rerolls = 1
                deleted = 2
                tokens = tokens[:-2] + [rng.randrange(32), rng.randrange(32)]
            return {"generation": " ".join(map(str, tokens)), "token_ids": tokens,
                    "scorer": {"correct": tokens[-1] % 2 == 0, "answer_extraction_ok": True},
                    "token_accounting": {"sampled": 8 + deleted, "emitted": 8, "deleted": deleted,
                                         "alarms": alarms, "rerolls": rerolls, "alarm_positions": [6] if alarms else [],
                                         "rollback_spans": [2] if alarms else [], "termination_reason": "synthetic_eos"},
                    "timing": {"wall_seconds": 0.0, "peak_vram_bytes": None},
                    "provenance": {"run_identity_sha256": None}}
        return run

    def execute(root, method, stop_after=None):
        store = RunStore(root, {**identity, "method": method})
        base = worker(method)
        def wrapped(item):
            artifact = base(item)
            artifact["provenance"]["run_identity_sha256"] = store.identity_sha256
            return artifact
        return store.run(items, wrapped, stop_after=stop_after)

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        vanilla = execute(root / "vanilla", "vanilla")
        no_alarm = execute(root / "no_alarm", "vanilla")
        assert [a["token_ids"] for a in vanilla] == [a["token_ids"] for a in no_alarm]
        execute(root / "resume", "vanilla", stop_after=2)
        resumed = execute(root / "resume", "vanilla")
        assert vanilla == resumed
        forced = execute(root / "forced", "forced")
        analysis = paired_analysis(vanilla, forced, samples=1000)
    result = {
        "synthetic_only": True, "math500_items_consumed": 0, "vanilla": "passed",
        "mgtb_no_alarm_identity": "passed", "forced_alarm_rollback": "passed",
        "interruption_resume_identical": "passed", "paired_analysis": analysis,
    }
    atomic_write_json(args.output, result)
    print(args.output)


if __name__ == "__main__":
    main()
