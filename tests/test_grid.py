"""
Roundtrip tests for the cnn grid encoding.
"""

from models.cnn import grid

COUNT_BF = "0v       @\n >:.1+:9`|\n ^       <"


def test_roundtrip_count():
    assert grid.from_grid(grid.to_grid(COUNT_BF)) == COUNT_BF
