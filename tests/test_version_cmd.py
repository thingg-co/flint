"""flint version subcommand prints a single version line and exits 0."""
import re
import subprocess
import sys


def test_version_command():
    result = subprocess.run(
        [sys.executable, "-m", "flint", "version"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    line = result.stdout.strip()
    assert line == "unknown" or re.fullmatch(r"\d+\.\d+\.\d+", line), f"unexpected output: {line!r}"
