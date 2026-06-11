"""
Curriculum of integer-sequence targets for REINFORCE. Every target is a
constant-recursive (C-finite) sequence s_n = c1*s(n-1) + ... + cd*s(n-d) with d
initial terms. This single family subsumes the simple cases (constant is order
1 with c=[1]; geometric is order 1 with c=[r]; arithmetic is order 2 with
c=[2,-1]; Fibonacci is order 2 with c=[1,1]) and scales by order, which is the
real difficulty axis: higher order is more state the program must carry. The
stages ramp the order; the Fibonacci-class order-2+ recurrences are the goal.
"""

MIN_TERMS = 2        # shortest target (still at least order+1, set per sample)
MAX_TERMS = 8        # longest target; reward checks this many leading outputs
MAX_TERM = 10 ** 6   # reject targets that blow past this (keep them printable)


def _recurrence(rng, order, coeff_lo, coeff_hi, ic_hi, extra=0):
    """
    Sample a constant-coefficient recurrence of the given order and roll it out
    to a random number of terms (from order+1 up to MAX_TERMS, so shorter,
    easier targets appear too) plus `extra` continuation terms. Returns the
    sequence, or None if a term goes negative or past MAX_TERM (the caller
    resamples). coeffs[0] multiplies the most recent term, matching
    f_gp_recurrence in data/synth.
    """
    coeffs = [rng.randint(coeff_lo, coeff_hi) for _ in range(order)]
    if all(c == 0 for c in coeffs):
        coeffs[-1] = 1                 # avoid a degenerate all-zero recurrence
    n_terms = rng.randint(max(MIN_TERMS, order + 1), MAX_TERMS) + extra
    seq = [rng.randint(0, ic_hi) for _ in range(order)]
    while len(seq) < n_terms:
        nxt = sum(c * seq[-i - 1] for i, c in enumerate(coeffs))
        if nxt < 0 or nxt > MAX_TERM:
            return None
        seq.append(nxt)
    return seq


# difficulty stages: increasing recurrence order. order 1 needs no memory;
# order 2+ forces the program to carry prior terms (registers or stack).
STAGES = ["order-1", "order-2", "order-3"]
_PARAMS = {
    "order-1": dict(order=1, coeff_lo=1, coeff_hi=3, ic_hi=9),
    "order-2": dict(order=2, coeff_lo=-1, coeff_hi=2, ic_hi=5),
    "order-3": dict(order=3, coeff_lo=-1, coeff_hi=2, ic_hi=5),
}


def sample_target(rng, stage, extra=0):
    """
    Sample one target sequence for the named stage, resampling until the rolled
    recurrence stays in range. With extra > 0, the sequence is rolled extra
    terms past the shown window: callers show the model seq[:-extra] but verify
    against all of seq, so writing the shown terms out literally no longer
    counts as solving (the continuation exposes it).
    """
    params = _PARAMS[stage]
    while True:
        seq = _recurrence(rng, extra=extra, **params)
        if seq is not None:
            return seq
