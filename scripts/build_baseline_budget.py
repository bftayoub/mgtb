#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from mgtb_v3.baselines.budget import build_profile


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Build a frozen baseline budget profile from paired MGT-B development runs.")
    parser.add_argument("--manifest", required=True, help="YAML/JSON manifest defining paired vanilla and MGT-B result files.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    manifest_path = Path(args.manifest)
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle) if manifest_path.suffix == ".json" else yaml.safe_load(handle)
    profile = build_profile(manifest or {})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(profile, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
