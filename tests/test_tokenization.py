"""
Roundtrip tests for the program-grid and observation tokenization.
"""

from models.cnn.tokenization import (
    from_grid,
    obs_from_tokens,
    obs_to_tokens,
    to_grid,
)

COUNT_BF = "0v       @\n >:.1+:9`|\n ^       <"


def test_grid_roundtrip_count():
    assert from_grid(to_grid(COUNT_BF)) == COUNT_BF


def test_obs_roundtrip():
    seqs = [[3, 12, -5], [0, 7], [100, 0, -42]]
    assert obs_from_tokens(obs_to_tokens(seqs)) == seqs
