"""
Tests for the random-world generator: every (output, program) pair must be
self-consistent (the program really prints that output), and programs must use
only ops the model can place.
"""

import re

import befunge as bf
from models.common.random_worlds import _OPS, random_worlds

_INT = re.compile(r"-?\d+")


def test_pairs_reproduce_their_output():
    worlds = random_worlds(15, seed=1, density=0.2, min_terms=4, n_terms=8)
    assert len(worlds) == 15
    for out, src in worlds:
        printed = [int(t) for t in _INT.findall(bf.run(src, max_steps=2000)[0])]
        assert len(out) >= 4                  # min_terms honored
        assert len(out) <= 8                  # truncated to n_terms
        assert printed[:len(out)] == out      # program reproduces the output


def test_only_allowed_ops_appear():
    worlds = random_worlds(5, seed=2, density=0.2)
    allowed = set(_OPS) | {" ", "\n"}
    excluded = set('?"&~,@')
    for _out, src in worlds:
        assert set(src) <= allowed
        assert not (excluded & set(src))
