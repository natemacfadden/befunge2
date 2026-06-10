"""
Random "worlds": random Befunge programs and the integer sequences they print,
used as unbiased SFT data (the model reconstructs the world that produced an
observed output). Programs use only ops the model can place, minus a few:
? is nondeterministic (unreproducible output), & ~ read stdin, , emits chars
(we want integer output), @ halts (we want loopers). A high space fraction
(low density) keeps programs sparse, simple, and more likely to run cleanly.
"""

import re

import numpy as np

import befunge as bf
from models.common.tokenization import OP_VOCAB, H

_EXCLUDE = set('?"&~,@ ')
_OPS = [c for c in OP_VOCAB if c not in _EXCLUDE]
_INT = re.compile(r"-?\d+")


def _random_src(rng, size, density):
    # each cell is a random op with probability `density`, else a space (no-op)
    cells = rng.choice(_OPS, size=(size, size))
    blank = rng.random((size, size)) >= density
    grid = np.where(blank, " ", cells)
    return "\n".join("".join(row) for row in grid)


def random_worlds(n, seed=0, density=(0.1, 0.3), min_terms=4, n_terms=8,
                  max_term=10 ** 9, path_per_term=6, max_path=64,
                  max_steps=2000):
    """
    Generate n (output_sequence, program_source) pairs. Each program is a random
    sparse grid run on the torus, kept only if it emits >= min_terms integers
    (each within +/-max_term) before erroring or hitting max_steps. The stored
    output is truncated to the first n_terms.

    Observability filter: a world is kept only if its executed path is short
    relative to what it prints (path <= path_per_term * observed terms, with a
    hard max_path backstop). Short, output-dense paths are the ones the
    observation actually constrains; long wandering paths are mostly invisible
    noise that teacher-forcing cannot pin down.

    density may be a float or a (lo, hi) range sampled per world (spreads world
    complexity). Repeated outputs are canonicalized to the shortest-path
    program seen, so output -> program is close to a function.
    """
    rng = np.random.default_rng(seed)
    lo, hi = density if isinstance(density, tuple) else (density, density)
    best = {}                              # output -> (path, src)
    while len(best) < n:
        src = _random_src(rng, H, rng.uniform(lo, hi))
        out, _status, _stack, visited = bf.run(src, max_steps=max_steps)
        ints = [int(t) for t in _INT.findall(out)]
        if len(ints) < min_terms or any(abs(t) > max_term for t in ints):
            continue
        terms = tuple(ints[:n_terms])
        path = int(visited.sum())
        if path > min(path_per_term * len(terms), max_path):
            continue
        if terms not in best or path < best[terms][0]:
            best[terms] = (path, src)
    return [(list(terms), src) for terms, (_path, src) in best.items()]
