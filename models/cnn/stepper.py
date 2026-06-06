"""Steppable befunge interpreter: pauses on blank cells so we can fill them.

Step-by-step Python mirror of befunge.py's run-to-completion njit core. Same
op semantics, but pauses on blank cells so generation can fill them live.
"""

import numpy as np

from models.cnn.vocab import ID_TO_CHAR


def _cdivmod(b, a):
    """
    Integer division and remainder truncating toward zero, with (0, 0) when
    a == 0.

    Befunge-93 defines / and % via C integer division (truncates toward zero).
    Python's // and % floor toward negative infinity, so they disagree for
    negative operands -- hence this explicit truncating version.
    """
    if a == 0:
        return 0, 0
    q = abs(b) // abs(a)
    if (b < 0) != (a < 0):
        q = -q
    return q, b - a * q


class Stepper:
    """
    Walks the IP, pausing whenever it lands on a blank cell to be filled.
    """

    def __init__(self, grid):
        self.grid = grid               # (H, W) ids, indexed grid[y, x]
        self.filled = np.zeros_like(grid, dtype=bool)
        self.x, self.y = 0, 0          # start top-left, moving right
        self.dx, self.dy = 1, 0
        self.stack = []
        self.regs = {}
        self.output = []
        # self.string_mode = False  # disabled: no string chars in vocab
        self.halted = False

    def _advance(self):
        """
        Move the IP one cell along its direction, wrapping the torus.
        """
        H, W = self.grid.shape
        self.x = (self.x + self.dx) % W
        self.y = (self.y + self.dy) % H

    def _pop(self):
        """
        Pop the stack, returning 0 if empty (matches befunge.py).
        """
        return self.stack.pop() if self.stack else 0

    def _exec(self, op):
        """
        Apply one op's effect on the state. op is a vocab id.
        """
        ch = ID_TO_CHAR[op]
        if ch in "0123456789": # push the digit's value
            self.stack.append(int(ch))
        elif ch == ">":
            self.dx, self.dy = 1, 0
        elif ch == "<":
            self.dx, self.dy = -1, 0
        elif ch == "^":
            self.dx, self.dy = 0, -1
        elif ch == "v":
            self.dx, self.dy = 0, 1
        elif ch == "_": # pop, go right if 0 else left
            self.dx, self.dy = (1, 0) if self._pop() == 0 else (-1, 0)
        elif ch == "|": # pop, go down if 0 else up
            self.dx, self.dy = (0, 1) if self._pop() == 0 else (0, -1)
        elif ch == "+": # pop a, pop b, push b + a
            a, b = self._pop(), self._pop()
            self.stack.append(b + a)
        elif ch == "*": # pop a, pop b, push b * a
            a, b = self._pop(), self._pop()
            self.stack.append(b * a)
        elif ch == "-": # pop a, pop b, push b - a
            a, b = self._pop(), self._pop()
            self.stack.append(b - a)
        elif ch == "/": # pop a, pop b, push trunc(b / a) (0 if a == 0)
            a, b = self._pop(), self._pop()
            self.stack.append(_cdivmod(b, a)[0])
        elif ch == "%": # pop a, pop b, push C-remainder of b / a (0 if a == 0)
            a, b = self._pop(), self._pop()
            self.stack.append(_cdivmod(b, a)[1])
        elif ch == ":": # duplicate top (0 if empty)
            v = self._pop()
            self.stack.append(v)
            self.stack.append(v)
        elif ch == "\\": # swap top two
            a, b = self._pop(), self._pop()
            self.stack.append(a)
            self.stack.append(b)
        elif ch == "$": # pop and discard
            self._pop()
        elif ch == "!": # logical not: pop v, push 1 if v == 0 else 0
            self.stack.append(1 if self._pop() == 0 else 0)
        elif ch == "`": # greater-than: pop a, pop b, push 1 if b > a else 0
            a, b = self._pop(), self._pop()
            self.stack.append(1 if b > a else 0)
        elif ch == ".": # pop v, output str(v) + " "
            self.output.append(str(self._pop()) + " ")
        elif ch == ",": # pop v, output chr(v % 256)
            self.output.append(chr(self._pop() % 256))
        elif ch == "g": # get: pop i, push regs[i] (0 if absent)
            self.stack.append(self.regs.get(self._pop(), 0))
        elif ch == "p": # put: pop i, pop v, regs[i] = v
            i, v = self._pop(), self._pop()
            self.regs[i] = v
        elif ch == "&": # read int from stdin -> push 0 (this variant)
            self.stack.append(0)
        elif ch == "~": # read char from stdin -> push 0 (this variant)
            self.stack.append(0)
        # elif ch == '"': # toggle string mode (disabled: not in vocab)
        #     self.string_mode = not self.string_mode
        elif ch == "?": # random direction -- nondeterministic
            raise NotImplementedError("'?' is nondeterministic; not supported")
        elif ch == "#": # bridge: skip the next cell
            self._advance()
        elif ch == "@": # halt
            self.halted = True

    def place(self, op):
        """
        Fill the current blank cell with op id and mark it filled.
        """
        self.grid[self.y, self.x] = op
        self.filled[self.y, self.x] = True

    def step(self):
        """
        Execute the current (filled) cell, then advance unless halted.
        """
        op = int(self.grid[self.y, self.x])
        # string mode disabled: vocab has no string chars (letters)
        # ch = ID_TO_CHAR[op]
        # if self.string_mode and ch != '"':
        #     self.stack.append(ord(ch))   # string mode: push char
        # else:
        #     self._exec(op)
        self._exec(op)
        if not self.halted:
            self._advance()

    def run(self, max_steps=100000):
        """Walk filled cells until the IP lands on a new cell, halts,
        or hits the cap. Returns 'newcell', 'halt', or 'limit'."""
        for _ in range(max_steps):
            if not self.filled[self.y, self.x]:
                return "newcell"
            self.step()
            if self.halted:
                return "halt"
        return "limit"

    def fill(self, choose_op, max_steps=100000):
        """Run to halt/limit, calling choose_op(self) at each new cell
        to place an op."""
        while True:
            status = self.run(max_steps)
            if status != "newcell":
                return status
            self.place(choose_op(self))
