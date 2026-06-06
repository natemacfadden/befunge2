"""Turn a befunge program into a fixed grid of vocab ids and back."""

import numpy as np
from models.cnn.vocab import CHAR_TO_ID, ID_TO_CHAR

H, W = 8, 32
print("[cnn.grid] grid is hardcoded 8x32 -- tune this")


def to_grid(source: str) -> np.ndarray:
    """Map a .bf source to an (H, W) array of vocab ids."""
    grid = np.zeros((H, W), dtype=np.int64)
    for y, line in enumerate(source.split("\n")):
        for x, ch in enumerate(line):
            grid[y, x] = CHAR_TO_ID[ch]
    return grid


def from_grid(grid: np.ndarray) -> str:
    """Map an (H, W) id array back to a .bf source string."""
    lines = []
    for y in range(grid.shape[0]):
        line = "".join(ID_TO_CHAR[int(grid[y, x])] for x in range(grid.shape[1]))
        lines.append(line.rstrip())
    return "\n".join(lines).rstrip("\n")
