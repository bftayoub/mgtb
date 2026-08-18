from __future__ import annotations

from pathlib import Path

import yaml

from mgtb_v3.science_fast.artifacts import RunStore
from mgtb_v3.science_fast.io import sha256_json
from mgtb_v3.science_fast.protocol import content_sha256, item_seed
from mgtb_v3.science_fast import runner


def _items(n=4):
    return [
        {
            "item_id": f"item-{index}",
            "content_sha256": content_sha256(f"problem-{index}"),
            "item_seed": item_seed(20260811, f"item-{index}"),
        }
        for index in range(n)
    ]


def _artifact(store, item):
    return {
        "generation": item["item_id"],
        "token_ids": [1],
        "scorer": {"correct": True},
        "token_accounting": {
            "sampled": 1, "emitted": 1, "deleted": 0, "alarms": 0, "rerolls": 0,
            "alarm_positions": [], "rollback_spans": [], "termination_reason": "eos",
        },
        "timing": {"wall_seconds": 1.0},
        "provenance": {"run_identity_sha256": store.identity_sha256},
    }


class _FakePool:
    def __init__(self, store, processes, initializer, initargs):
        self.store = store
        self.processes = processes
        self.terminated = False

    def imap_unordered(self, function, items, chunksize):
        return iter((item["item_id"], _artifact(self.store, item)) for item in reversed(items))

    def close(self):
        pass

    def terminate(self):
        self.terminated = True

    def join(self):
        pass


class _FakeContext:
    def __init__(self, store):
        self.store = store
        self.pool = None

    def Pool(self, **kwargs):
        self.pool = _FakePool(self.store, **kwargs)
        return self.pool


def test_parallel_parent_saves_unordered_results_and_resume_skips_them(tmp_path, monkeypatch):
    items = _items()
    store = RunStore(tmp_path, {"config": "same"})
    fake_context = _FakeContext(store)
    monkeypatch.setattr(runner, "_multiprocessing_context", lambda: fake_context)

    partial = runner._run_parallel_items(
        store, items, {}, parallel_workers=3, stop_after=2,
    )
    assert [artifact["item_id"] for artifact in partial] == ["item-0", "item-1"]
    assert fake_context.pool.processes == 2

    complete = runner._run_parallel_items(
        store, items, {}, parallel_workers=3, stop_after=None,
    )
    assert [artifact["item_id"] for artifact in complete] == [item["item_id"] for item in items]
    assert fake_context.pool.processes == 2


def test_parallel_workers_does_not_change_resolved_scientific_identity(tmp_path):
    source = Path("configs/science_fast/collect_reference_int4.yaml")
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    with_workers = tmp_path / "with_workers.yaml"
    without_workers = tmp_path / "without_workers.yaml"
    with_workers.write_text(yaml.safe_dump(raw), encoding="utf-8")
    raw.pop("parallel_workers")
    without_workers.write_text(yaml.safe_dump(raw), encoding="utf-8")

    assert sha256_json(runner.resolved_settings(with_workers)) == sha256_json(runner.resolved_settings(without_workers))


def test_reference_config_keeps_legacy_resume_identity():
    settings = runner.resolved_settings("configs/science_fast/collect_reference_int4.yaml")
    assert settings["device_map"] == "auto"
    assert sha256_json(settings) == "e48d201f45b62e179b8156cb18f37ee12126aa0b1174998b37a99abad8805dc7"
