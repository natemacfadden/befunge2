"""Steppable befunge interpreter: pauses on blank cells so we can fill them.

Step-by-step Python mirror of befunge.py's run-to-completion njit core. Same
op semantics, but pauses on blank cells so generation can fill them live.
TODO: cross-test against befunge.py for identical output.
"""

import numpy as np


class Stepper:
    """Walks the IP, pausing whenever it lands on a blank cell to be filled."""

    def __init__(self, grid):
        self.grid = grid               # (W, H) ids, indexed grid[x, y]
        self.filled = np.zeros_like(grid, dtype=bool)
        self.x, self.y = 0, 0          # start top-left, moving right
        self.dx, self.dy = 1, 0
        self.stack = []
        self.regs = {}
        self.output = []

    def step(self):
        """Advance the IP one cell, executing ops along the way. TODO."""
        ...

    def run(self):
        """Step until landing on a blank cell; return there to ask for an op. TODO."""
        ...
