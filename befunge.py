# =============================================================================
#    Copyright (C) 2026  Nate MacFadden
# =============================================================================
#
# -----------------------------------------------------------------------------
# Description:  MODIFIED Befunge-93 interpreter; runs a .bf file or launches the
#               GUI. Modification: we write to a tape, not the playground.
# -----------------------------------------------------------------------------

import sys

import numpy as np
from numba import njit, types
from numba.typed import Dict

# config
STACK_CAP  = 65536 # max stack depth; pushes past this are silently dropped
OUTPUT_CAP = 8192  # max output bytes per _run_core call; rest is truncated

# =============================================================================
# Language
# =============================================================================
# Befunge-93 variant (see https://esolangs.org/wiki/Befunge). The IP travels an
# 80x25 toroidal playfield executing the char it lands on, with inertia until an
# op redirects it; most ops read/write an int64 stack.
#
# This variant splits code from data: the playfield is immutable code, and
# `g`/`p` access a separate, unbounded register tape (int64) addressed by a
# single integer index, which the IP can never reach. In string mode (toggled by
# `"`) chars push their ASCII value instead of executing. SP is the stack
# pointer (next push lands at stack[SP-1]).
#
W, H = 80, 25 # playfield dimensions (columns, rows)

# INSTRUCTIONS maps each instruction char to its ord value; ALPHABET adds the
# two non-instruction chars that can appear on the grid (space, newline)
INSTRUCTIONS = {
    # digits - push value 0..9
    '0': ord('0'), '1': ord('1'), '2': ord('2'), '3': ord('3'), '4': ord('4'),
    '5': ord('5'), '6': ord('6'), '7': ord('7'), '8': ord('8'), '9': ord('9'),
    # registers/memory
    'g':  ord('g'),  # get: pop i, push regs[i] (separate register space)
    'p':  ord('p'),  # put: pop i, pop v, regs[i] = v
    # stack operations
    '$':  ord('$'),  # pop and discard
    ':':  ord(':'),  # duplicate top of stack
    '\\': ord('\\'), # swap top two
    # unary operations
    '!':  ord('!'),  # logical not: pop v, push 1 if v==0 else 0
    # binary operations
    '+':  ord('+'),  # pop a, pop b, push b+a
    '*':  ord('*'),  # pop a, pop b, push b*a
    '-':  ord('-'),  # pop a, pop b, push b-a
    '/':  ord('/'),  # pop a, pop b, push trunc(b/a)  (C-style, 0 if a==0)
    '%':  ord('%'),  # pop a, pop b, push b - a*trunc(b/a)  (C-style, 0 if a==0)
    '`':  ord('`'),  # greater-than: pop a, pop b, push 1 if b>a else 0
    # printing
    '.':  ord('.'),  # pop v, output str(v) + ' '
    ',':  ord(','),  # pop v, output chr(v % 256)
    # routing
    '^':  ord('^'),  # IP up
    '>':  ord('>'),  # IP right
    'v':  ord('v'),  # IP down
    '<':  ord('<'),  # IP left
    '#':  ord('#'),  # bridge: skip next cell along IP direction
    '?':  ord('?'),  # random of the 4 cardinal directions
    # conditionals
    '_':  ord('_'),  # horizontal if: pop v, go right if v==0 else left
    '|':  ord('|'),  # vertical if:   pop v, go down  if v==0 else up
    # halting
    '@':  ord('@'),  # halt
    # IO/misc (unused)
    '&':  ord('&'),  # read integer from stdin (we push 0)
    '~':  ord('~'),  # read char from stdin    (we push 0)
    '"':  ord('"'),  # toggle stringmode
}

ALPHABET = {
    **INSTRUCTIONS,
    ' ':  ord(' '),   # no-op padding
    '\n': ord('\n'),  # row separator in .bf source files
}

# =============================================================================
# Source parsing
# =============================================================================

def str_to_grid(src, min_h=H, min_w=W):
    """
    Lay out a .bf source string onto an int32 grid sized to the source content.
    Grid dimensions are at least (min_h, min_w) and grow to fit larger programs.
    """
    lines = src.splitlines()
    h = max(min_h, len(lines))
    w = max(min_w, max((len(line) for line in lines), default=0))
    grid = np.full((h, w), SPACE, dtype=np.int32)
    for y, line in enumerate(lines):
        for x, ch in enumerate(line):
            grid[y, x] = ord(ch) & 0xff
    return grid

# =============================================================================
# Runtime state
# =============================================================================
# pausable runtime state lives in a small int64 array, mutated in place by
# _run_core (lets the GUI pause between steps and lets numba compile the loop).
# indexes into that array:
S_SP          = 0  # stack pointer
S_OUT_LEN     = 1  # bytes written to the output buffer
S_X           = 2  # IP column
S_Y           = 3  # IP row
S_DX          = 4  # IP horizontal direction (-1, 0, +1)
S_DY          = 5  # IP vertical direction   (-1, 0, +1)
S_STRING_MODE = 6  # 0 or 1
STATE_SIZE    = 7

def new_state():
    """
    Initial interpreter state: IP at (0,0) heading right.
    """
    s = np.zeros(STATE_SIZE, dtype=np.int64)
    s[S_DX] = 1
    return s

# =============================================================================
# Interpreter
# =============================================================================

# ALPHABET aliases for _run_core's dispatch; reading them from the enclosing
# scope lets numba fold each comparison to a literal int compare. names mirror
# the char they encode
SPACE    = ALPHABET[' ']
# digits
ZERO     = ALPHABET['0']
NINE     = ALPHABET['9']
# registers/memory
GET      = ALPHABET['g']
PUT      = ALPHABET['p']
# stack operations
POP      = ALPHABET['$']
DUPLICATE= ALPHABET[':']
SWAP     = ALPHABET['\\']
# unary operations
NOT      = ALPHABET['!']
# binary operations
ADD      = ALPHABET['+']
MULT     = ALPHABET['*']
MINUS    = ALPHABET['-']
INTDIV   = ALPHABET['/']
MODULO   = ALPHABET['%']
GREATER  = ALPHABET['`']
# printing
PRNTINT  = ALPHABET['.']
PRNTCHAR = ALPHABET[',']
# routing
UP       = ALPHABET['^']
RIGHT    = ALPHABET['>']
DOWN     = ALPHABET['v']
LEFT     = ALPHABET['<']
SKIP     = ALPHABET['#']
RANDOM   = ALPHABET['?']
# conditionals
IF_HORIZ = ALPHABET['_']
IF_VERT  = ALPHABET['|']
# halting
END      = ALPHABET['@']
# IO/misc (unused)
READCHAR = ALPHABET['~']
READINT  = ALPHABET['&']
STRMODE  = ALPHABET['"']

@njit(cache=True, inline='always')
def _pop(stack, sp):
    """
    Pop one value; returns (value, new_sp). Underflow yields (0, 0).
    """
    if sp > 0:
        return stack[sp - 1], sp - 1
    return 0, 0

@njit(cache=True, inline='always')
def _push(stack, sp, v, cap):
    """
    Push one value if there's room; returns new sp.
    """
    if sp < cap:
        stack[sp] = v
        return sp + 1
    return sp

@njit(cache=True, inline='always')
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

def _run_core(grid, max_steps, stack, out_buf, state, visited, regs):
    """
    Shared, numba-friendly dispatch loop (no Python objects)

    State is mutated in place so callers can drive it step-by-step (the GUI
    does). `&`/`~` (stdin input) push 0 since the core can't block on stdin.

    Parameters
    ----------
    grid : ndarray
        The (H, W) int32 playfield; immutable code, never written.
    max_steps : int
        Max instructions to execute before returning step-limit status.
    stack : ndarray
        Preallocated int64 stack buffer; its length is the stack cap.
    out_buf : ndarray
        Preallocated int32 output buffer; its length is the output cap.
    state : ndarray
        Length-STATE_SIZE int64 state vector (sp, out_len, x, y, dx, dy,
        string_mode). Read at entry and written back at exit so runs resume.
    visited : ndarray
        An (H, W) uint8 mask set to 1 for cells the IP lands on; pass a
        throwaway array if you don't need the trace.
    regs : typed dict (int64 -> int64)
        Unbounded register tape `g`/`p` read/write by index; unwritten indices
        read as 0. The IP cannot reach it.

    Returns
    -------
    int
        Status code: 0=halted, 1=step budget exhausted, 2=runtime error.
    """
    sp          = int(state[S_SP])
    out_len     = int(state[S_OUT_LEN])
    x           = int(state[S_X])
    y           = int(state[S_Y])
    dx          = int(state[S_DX])
    dy          = int(state[S_DY])
    string_mode = state[S_STRING_MODE] != 0
    stack_cap   = stack.shape[0]
    out_cap     = out_buf.shape[0]
    gh          = grid.shape[0]
    gw          = grid.shape[1]
    steps       = 0
    halted      = False
    errored     = False

    while steps < max_steps and not halted and not errored:
        steps += 1
        c = grid[y, x]
        visited[y, x] = 1

        if string_mode:
            if c == STRMODE:
                string_mode = False
            else:
                sp = _push(stack, sp, c, stack_cap)
        elif c == SPACE:
            pass
        # digits
        elif ZERO <= c <= NINE:
            sp = _push(stack, sp, c - ZERO, stack_cap)
        # registers/memory
        elif c == GET:
            # get from the register tape by index (the IP never touches it);
            # an unwritten index reads as 0
            idx, sp = _pop(stack, sp)
            val = regs[idx] if idx in regs else 0
            sp = _push(stack, sp, val, stack_cap)
        elif c == PUT:
            # put a value into the register tape by index (grows on demand);
            # underflow (< 2 items) errors instead of popping 0s
            if sp < 2:
                errored = True
            else:
                idx, sp = _pop(stack, sp)
                v, sp = _pop(stack, sp)
                regs[idx] = v
        # stack operations
        elif c == POP:
            _, sp = _pop(stack, sp)
        elif c == DUPLICATE:
            v, sp = _pop(stack, sp)
            sp = _push(stack, sp, v, stack_cap)
            sp = _push(stack, sp, v, stack_cap)
        elif c == SWAP:
            a, sp = _pop(stack, sp)
            b, sp = _pop(stack, sp)
            sp = _push(stack, sp, a, stack_cap)
            sp = _push(stack, sp, b, stack_cap)
        # unary operations
        elif c == NOT:
            v, sp = _pop(stack, sp)
            sp = _push(stack, sp, 0 if v != 0 else 1, stack_cap)
        # binary operations
        elif (c == ADD or c == MULT or c == MINUS or c == INTDIV
                or c == MODULO or c == GREATER):
            # binary ops: pop a, pop b, push f(b, a). work in the int64 domain
            # so overflow wraps (matching the jit path) instead of promoting to
            # an unbounded python int and failing at the int64 stack boundary
            a, sp = _pop(stack, sp)
            b, sp = _pop(stack, sp)
            if c == ADD:      res = np.int64(b) + np.int64(a)
            elif c == MULT:   res = np.int64(b) * np.int64(a)
            elif c == MINUS:  res = np.int64(b) - np.int64(a)
            elif c == INTDIV: res = _cdivmod(b, a)[0]
            elif c == MODULO: res = _cdivmod(b, a)[1]
            else:             res = 1 if b > a else 0 # GREATER: greater-than
            sp = _push(stack, sp, res, stack_cap)
        # printing
        elif c == PRNTINT:
            v, sp = _pop(stack, sp)
            for ch in str(v):
                if out_len < out_cap:
                    out_buf[out_len] = ord(ch)
                    out_len += 1
            if out_len < out_cap:
                out_buf[out_len] = SPACE
                out_len += 1
        elif c == PRNTCHAR:
            v, sp = _pop(stack, sp)
            if out_len < out_cap:
                out_buf[out_len] = v % 256
                out_len += 1
        # routing
        elif c == UP: dx, dy = 0, -1
        elif c == RIGHT: dx, dy = 1, 0
        elif c == DOWN: dx, dy = 0, 1
        elif c == LEFT: dx, dy = -1, 0
        elif c == SKIP:
            x = (x + dx) % gw
            y = (y + dy) % gh
        elif c == RANDOM:
            r = np.random.randint(0, 4)
            if r == 0:
                dx, dy = 1, 0
            elif r == 1:
                dx, dy = -1, 0
            elif r == 2:
                dx, dy = 0, 1
            else:
                dx, dy = 0, -1
        # conditionals
        elif c == IF_HORIZ:
            v, sp = _pop(stack, sp)
            dx, dy = (1, 0) if v == 0 else (-1, 0)
        elif c == IF_VERT:
            v, sp = _pop(stack, sp)
            dx, dy = (0, 1) if v == 0 else (0, -1)
        # halting
        elif c == END:
            halted = True
        # IO/misc (unused inputs; string mode toggle on)
        elif c == READINT or c == READCHAR:
            sp = _push(stack, sp, 0, stack_cap)
        elif c == STRMODE:
            string_mode = True

        if not halted and not errored:
            x = (x + dx) % gw
            y = (y + dy) % gh

    state[S_SP]          = sp
    state[S_OUT_LEN]     = out_len
    state[S_X]           = x
    state[S_Y]           = y
    state[S_DX]          = dx
    state[S_DY]          = dy
    state[S_STRING_MODE] = 1 if string_mode else 0
    if errored:
        return 2
    return 0 if halted else 1

# JIT version; first `run(..., jit=True)` pays the compile cost (~1s, cached)
_run_core_jit = njit(cache=True)(_run_core)

# reusable per-process buffers. not threadsafe; fine under multiprocessing
# (one process per worker) but would need rethinking across threads
_STACK  = np.zeros(STACK_CAP,  dtype=np.int64)
_OUTBUF = np.zeros(OUTPUT_CAP, dtype=np.int32)

# =============================================================================
# Entry points
# =============================================================================

def run(src, max_steps=None, jit=False):
    """
    Run a Befunge program and capture its output and execution trace.

    Parameters
    ----------
    src : str
        The .bf source text.
    max_steps : int, optional
        Max instructions before bailing out. Defaults to an effectively
        unbounded 1 << 62.
    jit : bool, optional
        Use the numba-compiled hot path. The first jit run pays the compile
        cost (~1s, cached). Defaults to False.

    Returns
    -------
    tuple
        (output, status, final_stack, visited), where output is the printed
        text, status is one of 'ok'/'step_limit'/'error', final_stack is the
        ints left on the stack at termination, and visited is the (H, W) uint8
        mask of touched/`g`-read cells.
    """
    if max_steps is None:
        max_steps = 1 << 62

    # build playground/state info
    grid = str_to_grid(src)
    state = new_state()
    visited = np.zeros(grid.shape, dtype=np.uint8)
    regs = Dict.empty(types.int64, types.int64)        # fresh register tape

    # run it!
    core = _run_core_jit if jit else _run_core
    with np.errstate(over='ignore'):   # int64 arithmetic wraps; don't warn
        status = core(grid, max_steps, _STACK, _OUTBUF, state, visited, regs)

    # _STACK/_OUTBUF are reused across calls, but we only read them up to the
    # live lengths in state, so stale data past that is harmless
    n = int(state[S_OUT_LEN])
    sp = int(state[S_SP])
    output = ''.join(chr(int(b)) for b in _OUTBUF[:n])
    final_stack = [int(v) for v in _STACK[:sp]]
    status_str = {0: 'ok', 1: 'step_limit', 2: 'error'}.get(
        int(status), 'error')

    # return
    return output, status_str, final_stack, visited


def prune_program(src, visited):
    """
    Return `src` with every unvisited initial-grid cell replaced by a space;
    only source chars survive, runtime `p` mutations don't show up here.
    Prunes regardless of status, so for step_limit/error runs this is a faithful
    image of the observed execution, not the program's full intent.
    """
    grid = str_to_grid(src)
    gh, gw = grid.shape
    rows = []
    for y in range(gh):
        row = []
        for x in range(gw):
            row.append(chr(int(grid[y, x])) if visited[y, x] else ' ')
        rows.append(''.join(row).rstrip())
    return '\n'.join(rows).rstrip('\n')

# cli / gui entry
if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(
        description='Run a Befunge program, or launch the GUI when no file '
                    'is given.')
    p.add_argument('file', nargs='?',
                   help='.bf source file (omit to open the GUI)')
    p.add_argument('--max-steps', type=int, default=None)
    p.add_argument('--jit', action='store_true')
    args = p.parse_args()
    if args.file is None:
        from viz.gui import App
        App().run()
    else:
        with open(args.file) as f:
            output, status, _, _ = run(
                f.read(), max_steps=args.max_steps, jit=args.jit)
        sys.stdout.write(output)
        if status == 'step_limit':
            sys.stderr.write(f'\n[step limit {args.max_steps} reached]\n')
