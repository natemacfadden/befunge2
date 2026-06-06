"""
Smoke tests for the runner dispatch and the num_leading scorer.
"""

from bench import verify

# example befunge programs
# ------------------------
# counts 0..9 then halts
COUNT_BF = "0v       @\n >:.1+:9`|\n ^       <\n"
# fibonacci, offset by 1 (goes forever)
FIB_BF = "0:0 p1>0 g\\:0 p+:.84*,v\n      ^               <\n"

# tests
# -----
def test_run_python_basic():
    src = "for i in range(5): print(i)"
    assert verify.run(src, "python", 5) == [0, 1, 2, 3, 4]

def test_run_python_truncates_to_n():
    src = "for i in range(100): print(i)"
    assert verify.run(src, "python", 3) == [0, 1, 2]

def test_run_python_error_gives_empty():
    src = "raise ValueError('boom')"
    assert verify.run(src, "python", 5) == []

def test_run_python_timeout_gives_empty():
    src = "while True: pass"
    assert verify.run(src, "python", 5, timeout=0.5) == []

def test_run_befunge_halting():
    assert verify.run(COUNT_BF, "befunge", 10) == list(range(10))

def test_run_befunge_infinite_emitter_truncates():
    out = verify.run(FIB_BF, "befunge", 8)
    assert out == [1, 2, 3, 5, 8, 13, 21, 34]


def test_num_leading_perfect():
    src = "for i in range(5): print(i)"
    assert verify.num_leading(src, "python", [0, 1, 2, 3, 4]) == 5

def test_num_leading_partial():
    # prints 0 1 99 3 ; target 0 1 2 3 -> matches first two
    src = "print(0); print(1); print(99); print(3)"
    assert verify.num_leading(src, "python", [0, 1, 2, 3]) == 2

def test_num_leading_too_short():
    # prints only two of the four wanted
    src = "print(0); print(1)"
    assert verify.num_leading(src, "python", [0, 1, 2, 3]) == 2

def test_num_leading_error_is_zero():
    src = "raise ValueError('boom')"
    assert verify.num_leading(src, "python", [0, 1, 2]) == 0
