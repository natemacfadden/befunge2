"""
Steppable befunge generator: walks bf._run_core one instruction at a time,
pausing whenever the IP lands on an unfilled cell so a caller can place an op.

No op semantics live here -- every instruction is executed by the real
interpreter (befunge._run_core), the same engine the verifier uses. This class
only adds the pause-on-blank behavior and the vocab-id <-> ASCII bridge between
the model and the interpreter.
"""

import numpy as np

import befunge as bf
from models.common.tokenization import OP_FROM_ID, OP_TO_ID

# (dx, dy) -> heading index, matching the model's {0:^, 1:>, 2:v, 3:<}
_HEADING = {(0, -1): 0, (1, 0): 1, (0, 1): 2, (-1, 0): 3}
_BLANK = ord(" ")   # unfilled cells read as space (a no-op for the interpreter)

# ascii byte -> vocab id, so worldstate converts the grid in one numpy lookup
_BYTE_TO_ID = np.zeros(256, dtype=np.int64)
for _ch, _id in OP_TO_ID.items():
    _BYTE_TO_ID[ord(_ch)] = _id


class Stepper:
    """
    Walks the IP, pausing whenever it lands on an unfilled cell to be filled.
    """

    def __init__(self, shape):
        """
        Start a blank generator over an (H, W) grid.
        """
        self.grid = np.full(shape, _BLANK, dtype=np.int32)
        self.filled = np.zeros(shape, dtype=bool)
        self.stack = np.zeros(bf.STACK_CAP, dtype=np.int64)
        self.out_buf = np.zeros(bf.OUTPUT_CAP, dtype=np.int32)
        self.visited = np.zeros(shape, dtype=np.uint8)
        self.state = bf.new_state()
        self.regs = {}
        self.halted = False
        self.error = None

    @property
    def x(self):
        return int(self.state[bf.S_X])

    @property
    def y(self):
        return int(self.state[bf.S_Y])

    @property
    def output(self):
        n = int(self.state[bf.S_OUT_LEN])
        return "".join(chr(int(c)) for c in self.out_buf[:n])

    def place(self, op):
        """
        Fill the current cell with op (a vocab id) and mark it filled.
        """
        self.grid[self.y, self.x] = ord(OP_FROM_ID[op])
        self.filled[self.y, self.x] = True

    def step(self):
        """
        Execute the current (filled) cell via the real interpreter, advancing
        the IP. Sets halted on a halt (@) or an error.
        """
        with np.errstate(over="ignore"):   # int64 arithmetic wraps; don't warn
            status = bf._run_core(
                self.grid, 1, self.stack, self.out_buf, self.state,
                self.visited, self.regs,
            )
        if status == 0:
            self.halted = True
        elif status == 2:
            self.halted = True
            self.error = "p: stack underflow"

    def run(self, max_steps=100000):
        """
        Walk filled cells until the IP lands on a new cell, halts, or hits the
        cap. Returns 'newcell', 'halt', 'error', or 'limit' ('error' is a
        runtime fault like a p underflow; 'halt' is a clean @).
        """
        for _ in range(max_steps):
            if not self.filled[self.y, self.x]:
                return "newcell"
            self.step()
            if self.halted:
                return "error" if self.error else "halt"
        return "limit"

    def fill(self, choose_op, max_steps=100000):
        """
        Run to halt/limit, calling choose_op(self) at each new cell to place
        an op.
        """
        while True:
            status = self.run(max_steps)
            if status != "newcell":
                return status
            self.place(choose_op(self))

    def worldstate(self):
        """
        The model's view: (vocab_grid, filled, ip, heading), where vocab_grid
        is the (H, W) grid as vocab ids, ip is (x, y), and heading is the IP's
        direction as {0:^, 1:>, 2:v, 3:<}.
        """
        vocab_grid = _BYTE_TO_ID[self.grid]
        heading = _HEADING[(int(self.state[bf.S_DX]), int(self.state[bf.S_DY]))]
        return vocab_grid, self.filled.copy(), (self.x, self.y), heading
