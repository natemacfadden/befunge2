"""
Tokenize an integer sequence into padded token ids for the model.
"""

import numpy as np

# token ids: digits 0-9 are ids 0-9, specials follow
PAD = 10
SEP = 11           # separator between terms
NEG = 12           # leading sign for a negative term
OBS_VOCAB_SIZE = 13


def _term_to_tokens(term):
    """
    Token ids for a single integer term: optional '-' then its digits.
    """
    ids = []
    if term < 0:
        ids.append(NEG)
    for ch in str(abs(term)):
        ids.append(int(ch))
    return ids

def _term_from_tokens(ids):
    """
    Decode one term's token ids back to an int: optional NEG then digits.
    """
    neg = len(ids) > 0 and ids[0] == NEG
    digits = ids[1:] if neg else ids

    value = int("".join(str(d) for d in digits))
    return -value if neg else value


def obs_to_tokens(seqs):
    """
    Tokenize a batch of integer sequences into padded token ids:
        1) each term to an optional '-' then its decimal digits,
        2) terms joined by separators,
        3) rows padded with PAD to the batch's max length.
    Returns an (B, L) int array, where
        B = number of sequences,
        L = padded token length.
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
