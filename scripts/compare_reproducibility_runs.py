#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mgtb_v3.science_fast.io import sha256_json


def _load(directory: Path) -> dict[tuple[str, int], dict]:
    rows = {}
    for path in sorted((directory / "items").glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        supplied = row.get("artifact_sha256")
        actual = sha256_json({key: value for key, value in row.items() if key != "artifact_sha256"})
        if supplied != actual:
            raise ValueError(f"invalid artifact hash: {path}")
        key = (row["source_item_id"], int(row.get("replicate_seed", 0)))
        if key in rows:
            raise ValueError(f"duplicate unit in {directory}: {key}")
        rows[key] = row
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two generation runs token by token.")
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--expected", type=int, required=True)
    args = parser.parse_args()

    left, right = _load(Path(args.left)), _load(Path(args.right))
    if len(left) != args.expected or len(right) != args.expected:
        raise ValueError(f"incomplete audit: left={len(left)} right={len(right)} expected={args.expected}")
    if left.keys() != right.keys():
        raise ValueError("audit runs contain different item/seed units")

    seed_matches = sum(left[key]["item_seed"] == right[key]["item_seed"] for key in left)
    token_matches = sum(left[key]["token_ids"] == right[key]["token_ids"] for key in left)
    score_matches = sum(left[key]["scorer"] == right[key]["scorer"] for key in left)
    accounting_matches = sum(left[key]["token_accounting"] == right[key]["token_accounting"] for key in left)
    mismatched_ids = [key[0] for key in left if left[key]["token_ids"] != right[key]["token_ids"]]
    result = {
        "expected_units": args.expected,
        "seed_identical_units": seed_matches,
        "token_identical_units": token_matches,
        "score_identical_units": score_matches,
        "accounting_identical_units": accounting_matches,
        "first_token_mismatches": mismatched_ids[:10],
        "passed": token_matches == args.expected,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
