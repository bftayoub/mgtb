#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _driver_version() -> str | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    values = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return values[0] if values else None


def _locked_requirements(path: Path) -> dict[str, str]:
    locked = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        name, version = line.split("==", 1)
        locked[name] = version
    return locked


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the pinned MGT-B CUDA virtual environment.")
    parser.add_argument(
        "--lock",
        default=str(PROJECT_ROOT / "configs/environment/current_venv_lock.json"),
    )
    args = parser.parse_args()
    lock_path = Path(args.lock)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))

    import torch

    actual = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "driver": _driver_version(),
    }
    mismatches = {
        key: {"expected": lock[key], "actual": actual[key]}
        for key in actual
        if actual[key] != lock[key]
    }

    requirements_path = PROJECT_ROOT / lock["requirements_lock"]
    installed = {dist.metadata["Name"].lower(): dist.version for dist in importlib.metadata.distributions()}
    package_mismatches = {}
    for name, expected in _locked_requirements(requirements_path).items():
        actual_version = installed.get(name.lower())
        if actual_version != expected:
            package_mismatches[name] = {"expected": expected, "actual": actual_version}

    result = {
        "ok": not mismatches and not package_mismatches,
        "lock": str(lock_path),
        "actual": actual,
        "environment_mismatches": mismatches,
        "package_mismatches": package_mismatches,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
