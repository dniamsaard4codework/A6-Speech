"""Shared utilities: reproducible seeding, device selection, logging tee, edit distance / CER."""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path
from datetime import datetime

import numpy as np

# Make stdout/stderr UTF-8 so unicode (—, →, ≫, ␣) prints and tees cleanly on Windows.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

SEED = 42

# Repo root = three levels up from this file (src/a6_speech/utils.py -> repo root)
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "outputs"
FIGURE_DIR = REPO_ROOT / "figures"
LOG_DIR = REPO_ROOT / "logs"

for _d in (DATA_DIR, OUTPUT_DIR, FIGURE_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = SEED) -> None:
    """Seed Python, NumPy and (if available) PyTorch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device(verbose: bool = True):
    """Return the best available torch device and (optionally) print a summary."""
    import torch

    if torch.cuda.is_available():
        device = torch.device("cuda")
        if verbose:
            name = torch.cuda.get_device_name(0)
            cc = ".".join(str(x) for x in torch.cuda.get_device_capability(0))
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"Using device: cuda  |  {name}  |  compute capability {cc}  |  {total:.1f} GB")
            print(f"torch {torch.__version__}  (CUDA {torch.version.cuda})")
    else:
        device = torch.device("cpu")
        if verbose:
            print(f"Using device: cpu  |  torch {torch.__version__}")
    return device


class Tee:
    """Context manager that mirrors stdout to a log file (so training logs are saved)."""

    def __init__(self, log_path: str | os.PathLike, mode: str = "w"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self._file = None
        self._stdout = None

    def __enter__(self):
        self._file = open(self.log_path, self.mode, encoding="utf-8")
        self._stdout = sys.stdout
        header = f"# log started {datetime.now().isoformat(timespec='seconds')}\n"
        self._file.write(header)
        sys.stdout = self
        return self

    def write(self, data):
        self._stdout.write(data)
        self._file.write(data)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def isatty(self):
        return False

    def __getattr__(self, name):
        # Delegate anything we don't implement (encoding, fileno, ...) to real stdout.
        return getattr(self.__dict__.get("_stdout") or sys.__stdout__, name)

    def __exit__(self, *exc):
        sys.stdout = self._stdout
        self._file.close()
        return False


def levenshtein(a: str, b: str) -> int:
    """Edit distance between two strings (used for character error rate)."""
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m]


def char_error_rate(pred: str, target: str) -> float:
    """CER = edit_distance(pred, target) / len(target)."""
    if len(target) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    return levenshtein(pred, target) / len(target)
