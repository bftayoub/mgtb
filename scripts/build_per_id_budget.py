#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mgtb_v3.baselines.budget import build_per_id_budget


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Build a frozen per-ID decode budget table from MGT-B results.")
    parser.add_argument("--manifest", required=True, help="YAML/JSON manifest with ordered MGT-B result sources.")
    parser.add_argument("--output", required=True, help="Destination JSON table.")
    args = parser.parse_args(argv)

    manifest = _read_mapping(args.manifest)
    table = build_per_id_budget(manifest)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(table, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {output} ({table['summary']['num_examples']} IDs, tolerance={table['tolerance']:.1%})")


def _read_mapping(path: str) -> dict:
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    if source.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        import yaml

        data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise SystemExit("Budget manifest must be a YAML/JSON mapping.")
    return data


if __name__ == "__main__":
    main()
