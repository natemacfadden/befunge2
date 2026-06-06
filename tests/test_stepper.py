"""Cross-test: the stepper reproduces befunge.py's output."""

import sys
from io import StringIO

import numpy as np

sys.path.insert(0, "/home/nate/befunge")
import befunge

from models.cnn.grid import to_grid
from models.cnn.stepper import Stepper

EXAMPLES = "/home/nate/befunge/examples"


def _load(name):
    with open(f"{EXAMPLES}/{name}") as f:
        return f.read().rstrip("\n")


def _step_output(source):
    truth = to_grid(source)
    s = Stepper(np.zeros_like(truth))            # blank grid, like generation
    s.fill(lambda s: int(truth[s.y, s.x]))  # feed the true op at each new cell
    return "".join(s.output)


def _befunge_output(source):
    buf = StringIO()
    befunge.run(source, max_steps=1_000_000, out=buf)
    return buf.getvalue()


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
