from pathlib import Path

import pytest

from mgtb_v3.science_fast.artifacts import RunStore
from mgtb_v3.science_fast.io import load_json
from mgtb_v3.science_fast.protocol import content_sha256, item_seed


def _items(n=5):
    return [{"item_id": f"item-{i}", "content_sha256": content_sha256(f"p{i}"), "item_seed": item_seed(20260811, f"item-{i}")} for i in range(n)]


def _worker(calls):
    def work(item):
        calls.append(item["item_id"])
        token = int(item["item_seed"] % 97)
        return {
            "generation": str(token), "token_ids": [token], "scorer": {"correct": token % 2 == 0},
            "token_accounting": {"sampled": 1, "emitted": 1, "deleted": 0, "alarms": 0, "rerolls": 0,
                                 "alarm_positions": [], "rollback_spans": [], "termination_reason": "eos"},
            "timing": {"wall_seconds": 1.0}, "provenance": {"run_identity_sha256": "filled-by-test"},
        }
    return work


def _run(root, identity, items, calls, stop_after=None):
    store = RunStore(root, identity)
    worker = _worker(calls)
    def wrapped(item):
        artifact = worker(item)
        artifact["provenance"]["run_identity_sha256"] = store.identity_sha256
        return artifact
    return store.run(items, wrapped, stop_after=stop_after)


def test_continuous_equals_interrupted_then_resumed_and_skips_completed(tmp_path):
    items, identity = _items(), {"manifest": "abc", "config": "def"}
    continuous_calls, resumed_calls = [], []
    continuous = _run(tmp_path / "continuous", identity, items, continuous_calls)
    partial = _run(tmp_path / "resumed", identity, items, resumed_calls, stop_after=2)
    assert len(partial) == 2
    resumed = _run(tmp_path / "resumed", identity, items, resumed_calls)
    assert resumed_calls == [item["item_id"] for item in items]
    for left, right in zip(continuous, resumed):
        assert left == right


def test_resume_rejects_identity_change_and_partial_artifact(tmp_path):
    items = _items(1)
    store = RunStore(tmp_path, {"config": "one"})
    path = store.artifact_path(items[0]["item_id"])
    path.write_text('{"completed": false}', encoding="utf-8")
    assert store.valid_artifact(items[0]) is None
    with pytest.raises(ValueError, match="identity changed"):
        RunStore(tmp_path, {"config": "two"})
