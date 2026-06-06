"""Run a Befunge program with the project interpreter and read its integers."""

import re
import sys
from io import StringIO
from pathlib import Path

# FIX the interpreter lives in the sibling befunge project, not this package
_BEFUNGE_ROOT = Path.home() / "befunge"
if str(_BEFUNGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_BEFUNGE_ROOT))
import befunge  # noqa: E402

# Befunge loops have no natural halt; bound work by instruction count instead
# of a wall clock. emitters are tight, so this is generous for a few terms.
_MAX_STEPS = 10_000_000


def run(source: str, n: int, timeout: float) -> list[int]:
    """Run `source` on the interpreter; return the first n integers it prints.
    Returns fewer than n (or none) if it errors or hits the step limit before
    emitting n. `timeout` is unused — work is bounded by _MAX_STEPS."""
    buf = StringIO()
    befunge.run(source, max_steps=_MAX_STEPS, out=buf)
    return _parse_ints(buf.getvalue())[:n]


def _parse_ints(text: str) -> list[int]:
    # lenient: every integer-looking token, any separator
    return [int(t) for t in re.findall(r"-?\d+", text)]
