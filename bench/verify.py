"""Run a proposed program and check whether it reproduces a target sequence."""

from bench.runners import befunge, python

_RUNNERS = {"python": python.run, "befunge": befunge.run}

def run(source: str, language: str, n: int, timeout: float = 5.0) -> list[int]:
    """Run `source` (written in `language`); return the first n integers it
    prints. Returns fewer than n (or none) if it errors, halts early, or
    times out."""
    return _RUNNERS[language](source, n, timeout)

def num_leading(
    source: str, language: str, target: list[int], timeout: float = 5.0
) -> int:
    """The number of leading outputs that match the target sequence."""
    output = run(source, language, len(target), timeout)
    count = 0
    for a, b in zip(output, target):
        if a != b:
            break
        count += 1
    return count
