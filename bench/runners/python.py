"""Run a Python program in a subprocess and read the integers it prints."""

import os
import re
import subprocess
import sys
import tempfile


def run(source: str, n: int, timeout: float) -> list[int]:
    """Run `source` as a standalone Python script; return the first n integers
    it prints. Returns fewer than n (or none) on error or timeout."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(source)
        path = f.name
    try:
        result = subprocess.run(
            [sys.executable, path],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return []
    finally:
        os.unlink(path)
    return _parse_ints(result.stdout)[:n]


def _parse_ints(text: str) -> list[int]:
    # lenient: every integer-looking token, any separator
    return [int(t) for t in re.findall(r"-?\d+", text)]
