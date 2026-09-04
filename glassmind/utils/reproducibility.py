from __future__ import annotations

import os
import platform
import random
import subprocess
from typing import Any

import torch

from glassmind.utils.device import DeviceCapabilities


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def environment_metadata(capabilities: DeviceCapabilities, *, seed: int) -> dict[str, Any]:
    return {
        "seed": seed,
        "git_commit": git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "rocm": torch.version.hip,
        "device": capabilities.to_dict(),
        "process_id": os.getpid(),
    }

