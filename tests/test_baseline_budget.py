import json

import pytest

from mgtb_v3.baselines.budget import (
    build_per_id_budget,
    assigned_template,
    build_profile,
    periodic_schedule,
    random_schedule,
    restart_indices,
    revision_token_budget,
)


def _row(item_id, method, *, tokens=100, extra=0, backtracks=None):
    return {
        "id": item_id,
        "method": method,
        "precision": "int4",
        "base_model": "model",
        "dataset": "math500",
        "threshold_path": "threshold.json" if method == "mgtb_v3_window" else None,
        "seed": 7,
        "tokens_generated": tokens,
        "token_events_trace": tokens + extra,
        "extra_sampled": extra,
        "backtracks": backtracks or [],
        "correct": 1.0,
        "completion_text": "secret label-bearing text",
    }


def _write(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_build_profile_is_paired_and_excludes_result_content(tmp_path):
    vanilla = tmp_path / "vanilla.jsonl"
    mgtb = tmp_path / "mgtb.jsonl"
    _write(vanilla, [_row("a", "vanilla", tokens=100), _row("b", "vanilla", tokens=200)])
    _write(
        mgtb,
        [
            _row(
                "a",
                "mgtb_v3_window",
                tokens=80,
                extra=20,
                backtracks=[{"rollback_pos": 40, "alert": {"token_pos": 60}}],
            ),
            _row("b", "mgtb_v3_window", tokens=180, extra=0),
        ],
    )
    profile = build_profile({"pairs": [{"vanilla_results": str(vanilla), "mgtb_results": str(mgtb)}]})

    assert profile["summary"]["num_examples"] == 2
    assert profile["summary"]["mean_extra_decode_tokens"] == 10
    assert profile["templates"][0]["rollback_lengths"] == [20]
    assert "correct" not in json.dumps(profile)
    assert "secret label-bearing text" not in json.dumps(profile)


def test_budget_schedules_and_restart_selection_are_deterministic(tmp_path):
    profile = {
        "summary": {"mean_extra_decode_tokens": 20.0, "mean_vanilla_tokens_generated": 100.0},
        "templates": [
            {"reference_primary_tokens": 100, "rollback_lengths": [10, 20], "extra_decode_tokens": 30},
            {"reference_primary_tokens": 80, "rollback_lengths": [], "extra_decode_tokens": 0},
        ],
    }
    template = assigned_template(profile, base_seed=3, precision="int4", index=1)
    assert template == assigned_template(profile, base_seed=3, precision="int4", index=1)
    assert periodic_schedule(template) == periodic_schedule(template)
    assert random_schedule(template, seed=9) == random_schedule(template, seed=9)
    assert restart_indices(num_items=10, profile=profile, base_seed=2, precision="int4") == restart_indices(
        num_items=10, profile=profile, base_seed=2, precision="int4"
    )
    assert revision_token_budget(profile) == 20


def test_profile_rejects_unpaired_sources(tmp_path):
    vanilla = tmp_path / "vanilla.jsonl"
    mgtb = tmp_path / "mgtb.jsonl"
    _write(vanilla, [_row("a", "vanilla")])
    _write(mgtb, [_row("b", "mgtb_v3_window")])
    with pytest.raises(ValueError, match="Unpaired"):
        build_profile({"pairs": [{"vanilla_results": str(vanilla), "mgtb_results": str(mgtb)}]})


def test_per_id_budget_deduplicates_in_source_order_and_applies_ten_percent(tmp_path):
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    _write(first, [_row("a", "mgtb_v3_window", tokens=101), _row("b", "mgtb_v3_window", tokens=50)])
    _write(second, [_row("a", "mgtb_v3_window", tokens=999), _row("c", "mgtb_v3_window", tokens=10)])

    table = build_per_id_budget(
        {
            "tolerance": 0.10,
            "expected_num_items": 3,
            "sources": [{"results": str(first)}, {"results": str(second)}],
        }
    )

    budgets = {row["id"]: row for row in table["budgets"]}
    assert budgets["a"]["mgtb_decode_events"] == 101
    assert budgets["a"]["control_max_decode_events"] == 111
    assert budgets["c"]["control_max_decode_events"] == 11
    assert table["summary"]["num_examples"] == 3
    assert "correct" not in json.dumps(table)
    assert "secret label-bearing text" not in json.dumps(table)


def test_per_id_budget_rejects_condition_mismatch(tmp_path):
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    row_a = _row("a", "mgtb_v3_window")
    row_b = _row("b", "mgtb_v3_window")
    row_b["threshold_path"] = "other-threshold.json"
    _write(first, [row_a])
    _write(second, [row_b])

    with pytest.raises(ValueError, match="do not share one condition"):
        build_per_id_budget({"sources": [{"results": str(first)}, {"results": str(second)}]})
