"""
Roundtrip tests for observation tokenization.
"""

from models.cnn.observation_tokenization import obs_from_tokens, obs_to_tokens


def test_roundtrip():
    seqs = [[3, 12, -5], [0, 7], [100, 0, -42]]
    assert obs_from_tokens(obs_to_tokens(seqs)) == seqs
