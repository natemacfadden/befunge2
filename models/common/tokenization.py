"""
Vocabulary and pure data transforms between the model's two inputs and their
id encodings: a befunge program <-> a grid of vocab ids, and an integer
sequence <-> padded observation token ids.
"""

import numpy as np

# =============================================================================
# Vocabularies
# =============================================================================
# Two separate id spaces, one per model input (never mixed -- see model.py).
# Each is a list of tokens indexed by id, plus matching token<->id maps.

# worldstate ops: the befunge chars placed in grid cells (id 0 is blank)
OP_VOCAB = list(" 0123456789+*-/%!`><^v?_|\":\\$.,#gp&~@")
OP_TO_ID = {t: i for i, t in enumerate(OP_VOCAB)}
OP_FROM_ID = {i: t for i, t in enumerate(OP_VOCAB)}
OP_VOCAB_SIZE = len(OP_VOCAB)

# action space: place any op (id 0..OP_VOCAB_SIZE-1) or signal DONE (stop)
DONE = OP_VOCAB_SIZE
N_ACTIONS = OP_VOCAB_SIZE + 1

# observation tokens: digits 0-9 then specials, encoding an integer sequence
OBS_VOCAB = list("0123456789") + ["PAD", "SEP", "NEG"]
OBS_TO_ID = {t: i for i, t in enumerate(OBS_VOCAB)}
OBS_FROM_ID = {i: t for i, t in enumerate(OBS_VOCAB)}
OBS_VOCAB_SIZE = len(OBS_VOCAB)
PAD, SEP, NEG = OBS_TO_ID["PAD"], OBS_TO_ID["SEP"], OBS_TO_ID["NEG"]


# =============================================================================
# Program grid
# =============================================================================

H, W = 64, 64
print("[cnn.tokenization] grid is hardcoded 64x64 -- tune this")


def to_grid(source: str) -> np.ndarray:
    """
    Map a .bf source to an (H, W) array of vocab ids.
    """
    grid = np.zeros((H, W), dtype=np.int64)
    for y, line in enumerate(source.split("\n")):
        for x, ch in enumerate(line):
            grid[y, x] = OP_TO_ID[ch]
    return grid


def from_grid(grid: np.ndarray) -> str:
    """
    Map an (H, W) id array back to a .bf source string.
    """
    lines = []
    for y in range(grid.shape[0]):
        line = "".join(
            OP_FROM_ID[int(grid[y, x])] for x in range(grid.shape[1])
        )
        lines.append(line.rstrip())
    return "\n".join(lines).rstrip("\n")


# =============================================================================
# Observation tokens
# =============================================================================

def _term_to_tokens(term):
    """
    Token ids for a single integer term: optional '-' then its digits.
    """
    ids = []
    if term < 0:
        ids.append(NEG)
    for ch in str(abs(term)):
        ids.append(OBS_TO_ID[ch])
    return ids


def _term_from_tokens(ids):
    """
    Decode one term's token ids back to an int: optional NEG then digits.
    """
    neg = len(ids) > 0 and ids[0] == NEG
    digits = ids[1:] if neg else ids

    value = int("".join(OBS_FROM_ID[d] for d in digits))
    return -value if neg else value


def obs_to_tokens(seqs):
    """
    Tokenize a batch of integer sequences into padded token ids:
        1) each term to an optional '-' then its decimal digits,
        2) terms joined by separators,
        3) rows padded with PAD to the batch's max length.
    Returns an (B, L) int array, where
        B = number of sequences,
        L = number of tokens (padded to the batch max).
    """
    # build tokenization as list of lists
    rows = []
    for seq in seqs:
        ids = []
        for i, term in enumerate(seq):
            if i > 0:
                ids.append(SEP)
            ids.extend(_term_to_tokens(term))
        rows.append(ids)

    # build a padded array
    L = max(len(r) for r in rows)
    arr = np.full((len(rows), L), PAD, dtype=np.int64)
    for r, ids in enumerate(rows):
        arr[r, : len(ids)] = ids

    return arr


def obs_from_tokens(arr):
    """
    Inverse of obs_to_tokens: decode padded token ids back to integer sequences.
    Returns a list of B integer sequences.
    """
    seqs = []
    for row in arr:
        # drop padding, then split on SEP into per-term groups
        toks = [int(t) for t in row if t != PAD]
        groups, cur = [], []
        for t in toks:
            if t == SEP:
                groups.append(cur)
                cur = []
            else:
                cur.append(t)
        groups.append(cur)

        seqs.append([_term_from_tokens(g) for g in groups])
    return seqs
