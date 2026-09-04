from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_check_no_cache(tmp_path: Path) -> None:
    """When there's no machine.json, check should print autotune info and exit 0."""
    env = {**os.environ, "FLINT_STATE_DIR": str(tmp_path)}
    result = subprocess.run(
        [sys.executable, "-m", "flint", "check"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0
    assert "autotune" in result.stdout.lower()


def test_check_not_enough_memory(tmp_path: Path) -> None:
    """When machine.json has a preset that won't fit, check should exit 2 with 'not enough memory'."""
    cache_path = tmp_path / "machine.json"
    # Write a machine.json with a peak_gb that can't possibly fit (100000 GB)
    cache_path.write_text(
        json.dumps(
            {
                "key": "cpu|bf16|mem|8|100|20|300|16|0.7|180|60",
                "choice": {
                    "device": "cpu",
                    "preset": "S",
                    "params": 240000,
                    "ms_per_step": 95,
                    "peak_gb": 100000.0,
                    "window": 64,
                    "threads": 4,
                    "d_model": 64,
                    "dilations": [1, 2, 4, 8, 16],
                    "n_experts": 3,
                    "n_heads": 4,
                    "budget_ms": 100,
                },
            }
        )
    )
    env = {**os.environ, "FLINT_STATE_DIR": str(tmp_path)}
    result = subprocess.run(
        [sys.executable, "-m", "flint", "check"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 2
    assert "not enough memory" in result.stdout.lower()
