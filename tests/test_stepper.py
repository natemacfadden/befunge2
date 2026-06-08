"""
Plumbing test: driving the stepper's fill/pause loop with the true op at each
cell reproduces befunge.py's output. Exercises the fill mechanic and the
vocab-id <-> ASCII bridge (the stepper shares the interpreter's op semantics,
so this checks the plumbing, not the semantics).
"""

from pathlib import Path

import befunge as bf
from models.cnn.stepper import Stepper
from models.cnn.tokenization import to_grid

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def _load(name):
    with open(EXAMPLES / name) as f:
        return f.read().rstrip("\n")


def _step_output(source):
    truth = to_grid(source)
    s = Stepper(truth.shape)                 # blank grid, like generation
    s.fill(lambda s: int(truth[s.y, s.x]))  # feed the true op at each new cell
    return s.output


def _befunge_output(source):
    output, *_ = bf.run(source, max_steps=1_000_000)
    return output


def _check_full(name):
    src = _load(name)
    assert _step_output(src) == _befunge_output(src)


def test_count():
    _check_full("count.bf")


def test_factorials():
    _check_full("factorials.bf")


def test_fib_leading():
    # infinite emitter: compare leading output, not full
    src = _load("fib.bf")
    n = 40
    assert _step_output(src)[:n] == _befunge_output(src)[:n]
