from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def _version(module_name: str) -> str | None:
    try:
        module = __import__(module_name)
        return str(module.__version__)
    except (ImportError, AttributeError):
        return None


def git_commit(root: str | Path = ".") -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, check=False)
    return result.stdout.strip() or "unavailable"


def source_tree_sha256(root: str | Path = ".") -> str:
    root = Path(root).resolve()
    digest = hashlib.sha256()
    ignored = {".git", "outputs", "runs", "data", ".pytest_cache", "__pycache__", ".venv"}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or any(part in ignored for part in path.relative_to(root).parts):
            continue
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def software_environment() -> dict[str, Any]:
    gpu = None
    cuda = None
    try:
        import torch
        cuda = torch.version.cuda
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": _version("torch"),
        "transformers": _version("transformers"),
        "bitsandbytes": _version("bitsandbytes"),
        "datasets": _version("datasets"),
        "cuda": cuda,
        "gpu": gpu,
        "cwd": os.fspath(Path.cwd()),
    }
