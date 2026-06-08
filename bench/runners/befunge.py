"""
Run a Befunge program with the project interpreter and read its integers.
"""

import re

import befunge as bf

# Befunge loops have no natural halt; bound work by instruction count instead
# of a wall clock. emitters are tight, so this is generous for a few terms.
_MAX_STEPS = 10_000_000


def run(source: str, n: int, timeout: float,
        max_steps: int = None) -> list[int]:
    """
    Run `source` on the interpreter; return the first n integers it prints.
    Returns fewer than n (or none) if it errors or hits the step limit before
    emitting n. `timeout` is unused (work is bounded by instruction count);
    max_steps overrides the default _MAX_STEPS budget (e.g. a tight cap for
    fast training-reward verification).
    """
    output, *_ = bf.run(
        source, max_steps=_MAX_STEPS if max_steps is None else max_steps)
    return _parse_ints(output)[:n]


def _parse_ints(text: str) -> list[int]:
    # lenient: every integer-looking token, any separator
    return [int(t) for t in re.findall(r"-?\d+", text)]
