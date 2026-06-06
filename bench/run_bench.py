"""Evaluate a model: how well its proposed programs reproduce target
sequences."""

from bench import verify
from bench.interface import Interface


def evaluate(
    model: Interface, sequences: list[list[int]], k: int
) -> list[float]:
    """ For every sequence, sample k programs and score them out to
    len(seq). Return the scores in decreasing order, so the 0th is best. """
    scores = []
    for seq in sequences:
        candidates = model.propose(seq, k)
        best = max(
            (verify.num_leading(c, model.language, seq) for c in candidates),
            default=0,
        )
        scores.append(best / len(seq))
    scores.sort(reverse=True)
    return scores
