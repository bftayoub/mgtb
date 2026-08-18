import copy

import pytest

from mgtb_v3.science_fast.protocol import (
    MATH500_REVISION, MATH_TRAIN_REVISION, assert_disjoint_roles, build_manifest,
    content_sha256, item_seed, normalize_problem, validate_manifest,
)


def _rows(prefix, count, answer="1"):
    return [{"id": f"{prefix}-{i}", "problem": f"Problem {prefix} {i}", "answer": answer} for i in range(count)]


def test_normalization_hash_and_seed_are_stable_and_content_based():
    assert normalize_problem(" A\r\n  B ") == "A B"
    assert content_sha256("A  B") == content_sha256(" A\nB ")
    assert item_seed(20260811, "x") == item_seed(20260811, "x")
    assert item_seed(20260811, "x") != item_seed(20260811, "y")


def test_manifest_is_deterministic_pinned_and_has_300_100_300():
    train, test = _rows("train", 410), _rows("test", 310)
    first = build_manifest(train, test)
    second = build_manifest(reversed(train), reversed(test))
    assert first == second
    assert first["counts"] == {"reference": 300, "development": 100, "test": 300}
    assert first["dataset_revisions"] == {"math_train": MATH_TRAIN_REVISION, "math500_test": MATH500_REVISION}
    validate_manifest(first)


def test_content_leakage_is_rejected_even_when_ids_differ():
    manifest = build_manifest(_rows("train", 410), _rows("test", 310))
    manifest = copy.deepcopy(manifest)
    manifest["roles"]["test"][0]["content_sha256"] = manifest["roles"]["reference"][0]["content_sha256"]
    with pytest.raises(ValueError, match="split leakage"):
        assert_disjoint_roles(manifest)


def test_revision_or_manifest_mutation_is_rejected():
    manifest = build_manifest(_rows("train", 410), _rows("test", 310))
    manifest["dataset_revisions"]["math500_test"] = "moving-main"
    with pytest.raises(ValueError):
        validate_manifest(manifest)
