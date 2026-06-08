"""
Rotary position embedding (RoPE) and self-attention for the observation encoder.
(RoPE: RoFormer, Su et al. 2021, https://arxiv.org/abs/2104.09864.)

RoPE encodes the relative position of a token within a larger sequence (think
"where is this word in a paragraph?"). This is useful for attention
(https://arxiv.org/abs/1706.03762) in which a token is primarily relevant
relative to another. Here the tokens are the observation sequence (the
digit/separator ids from obs_to_tokens), and a token's "position" is just its
index m in that sequence (0, 1, ..., L-1).

Each token i is embedded as a (query/key) vector. RoPE encodes the position of i
by rotating parts of this vector. Explicitly,
1) the vector is assumed/enforced to be even dimensional
2) the vector is split into d/2 'pairs', viewed each as a 2D vector
3) the jth pair of the ith token is rotated by
        phi = i * theta_j
   for theta_j = base ** (-2j / d)
Then attention dots these vectors across tokens. For tokens i and i':
    (R(i*theta_j) q)^T (R(i'*theta_j) k)
        = q^T R(i*theta_j)^T R(i'*theta_j) k
        = q^T R((i'-i)*theta_j) k
using R(a)^T = R(-a) and R(-a) R(b) = R(b-a). So the score depends only on the
gap i' - i, not on the absolute positions of i and i'.

The reason for the non-trivial rotations (split into pairs, rotate each a
different speed) is to give more information regardless of separation. With a
single speed the score's dependence on the gap is one sinusoid in (i'-i)*theta,
which forces a tradeoff: a fast speed varies sharply with the gap but is
periodic, so far-apart gaps can alias to the same value, while a slow speed
avoids aliasing but barely changes across the sequence, carrying little signal.
A spread of speeds sidesteps the tradeoff: fast pairs resolve nearby gaps, slow
pairs distinguish far ones, so together every gap gets a distinct signature.
"""

import torch


def rope_tables(seq_len, head_dim, base=10000.0):
    """
    Precompute the per-position rotation tables for RoPE.

    Parameters
    ----------
    seq_len : int
        Number of positions (sequence length L).
    head_dim : int
        Per-head dimension; must be even (rotates head_dim // 2 pairs).
    base : float, optional
        Geometric base for the pair speeds. Defaults to 10000.0.

    Returns
    -------
    cos, sin : Tensor
        Each (seq_len, head_dim): cosine and sine of (position * speed), with
        each pair's speed repeated across its two slots.
    """
    half = head_dim // 2
    # pair speeds theta_p = base ** (-2p/head_dim), fast (p=0) to slow
    p = torch.arange(half)
    theta = base ** (-2.0 * p / head_dim)          # (half,)
    pos = torch.arange(seq_len)                    # (L,)
    angles = torch.outer(pos, theta)               # (L, half)
    angles = torch.cat([angles, angles], dim=-1)   # (L, head_dim)
    return angles.cos(), angles.sin()


def rotate_half(x):
    """
    Rearrange the last dim into each pair's rotation partner, with a sign:
    [first_half, second_half] -> [-second_half, first_half].

    This supplies the off-diagonal (-sin, +sin) terms so that
    x*cos + rotate_half(x)*sin rotates each pair (see apply_rope).
    """
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_rope(x, cos, sin):
    """
    Rotate the last-dim pairs of x by the RoPE angles.

    Each pair (slot p, slot p+half) at position m is rotated by phi = m*theta_p:

        [ x_p'        ]   [ cos phi   -sin phi ] [ x_p        ]
        [ x_{p+half}' ] = [ sin phi    cos phi ] [ x_{p+half} ]

    computed for all pairs at once as x*cos + rotate_half(x)*sin (cos/sin are
    the duplicated tables from rope_tables, so a pair's two slots share phi).

    Parameters
    ----------
    x : Tensor
        (..., L, head_dim) queries or keys.
    cos, sin : Tensor
        (L, head_dim) tables from rope_tables.

    Returns
    -------
    Tensor
        x with each pair rotated; same shape as x.
    """
    return x * cos + rotate_half(x) * sin
