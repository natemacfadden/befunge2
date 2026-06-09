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
import torch.nn as nn

# =============================================================================
# RoPE
# =============================================================================

def rope_tables(seq_len, head_dim, base=10000.0):
    """
    Precompute the per-position rotation tables for RoPE.

    Parameters
    ----------
    seq_len : int
        Number of positions, i.e. the number of tokens L.
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


# =============================================================================
# Self-attention
# =============================================================================

class SelfAttention(nn.Module):
    """
    Multi-head self-attention with RoPE on the queries and keys.

    Single head, one observation's L tokens stacked as X (L, d):

        Q = X W_q        # (L, d)  row i is token i's query
        K = X W_k        # (L, d)  row i is token i's key
        V = X W_v        # (L, d)  row i is token i's value
        S = Q K^T / sqrt(d)   # (L, L)  S[i,j] = query i . key j
        A = softmax(S)        # (L, L)  row i sums to 1
        O = A V               # (L, d)  row i = weighted average of values

    Every token gets its own query, key, value (the rows of Q, K, V above).
    Multi-head splits these q, k, v into `heads` slices of width head_dim and
    attends within each slice on its own (its own matrix A).

    The projections are full and learned, so the head can specialize on the
    information relevant to it. E.g., one on adjacent digits and another on term
    boundaries.

    The (L, head_dim) head outputs are concatenated back to (L, D) and mixed by
    out_proj, so nothing stays siloed. RoPE rotates each head's Q and K before S
    (see apply_rope).

    The projections are square (D -> D): keeping heads * head_dim = D leaves the
    output at width D, flowing back into the residual stream unchanged.
    """

    def __init__(self, model_dim, heads):
        super().__init__()
        assert model_dim % heads == 0, "model_dim must be divisible by heads"
        self.heads = heads
        self.head_dim = model_dim // heads
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"
        self.q_proj = nn.Linear(model_dim, model_dim)
        self.k_proj = nn.Linear(model_dim, model_dim)
        self.v_proj = nn.Linear(model_dim, model_dim)
        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, obs_feats, cos, sin, pad_mask):
        """
        Self-attend over the observation's L tokens.

        Turns each token's embedding into a richer, context-aware vector by
        letting it gather information from the tokens it relates to. Each token
        queries the others, and its output is a position-aware convex
        combination of their values, so (say) a digit can absorb its number's
        other digits and its surrounding separators. Stacking several such
        layers lets a token attend to neighbors that already gathered their own
        context, building up the structure of each number.

        Parameters
        ----------
        obs_feats : Tensor
            (B, L, model_dim) observation token features.
        cos, sin : Tensor
            (L, head_dim) RoPE tables from rope_tables.
        pad_mask : Tensor
            (B, L) bool, True at real tokens, False at PAD slots.

        Returns
        -------
        Tensor
            (B, L, model_dim) attended features.
        """
        B, L, _ = obs_feats.shape
        # project, then split model_dim into (heads, head_dim)
        q = self.q_proj(obs_feats).view(B, L, self.heads, self.head_dim)
        k = self.k_proj(obs_feats).view(B, L, self.heads, self.head_dim)
        v = self.v_proj(obs_feats).view(B, L, self.heads, self.head_dim)
        # attention's matmul acts on the last two dims (L, head_dim) and treats
        # leading dims as batch. We want one attention per (batch, head), so
        # move heads beside batch: (B, L, heads, hd) -> (B, heads, L, hd)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        # RoPE on q and k (position enters here)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        # (B, heads, L, L)
        scores = q @ k.transpose(-2, -1) / self.head_dim**0.5
        # never attend to PAD keys
        scores = scores.masked_fill(~pad_mask[:, None, None, :], float("-inf"))
        attn = scores.softmax(dim=-1)
        # softmax weights are nonnegative and sum to 1, so each output row is a
        # convex combination of the value vectors (a point in their convex hull)
        out = attn @ v                            # (B, heads, L, head_dim)
        # merge heads back: (B, heads, L, hd) -> (B, L, model_dim)
        out = out.transpose(1, 2).reshape(B, L, -1)
        return self.out_proj(out)


# =============================================================================
# Encoder layer
# =============================================================================

class EncoderLayer(nn.Module):
    """
    One transformer encoder block: self-attention then a per-token
    feed-forward, each wrapped in a residual connection and a LayerNorm
    (pre-norm, so the norm is applied before each sublayer).
    """

    def __init__(self, model_dim, heads, ff_mult=4):
        super().__init__()
        self.attn = SelfAttention(model_dim, heads)
        self.norm1 = nn.LayerNorm(model_dim)
        self.ff = nn.Sequential(
            nn.Linear(model_dim, ff_mult * model_dim),
            nn.GELU(),
            nn.Linear(ff_mult * model_dim, model_dim),
        )
        self.norm2 = nn.LayerNorm(model_dim)

    def forward(self, obs_feats, cos, sin, pad_mask):
        """
        Pre-norm residual update of the (B, L, model_dim) token features.
        """
        obs_feats = obs_feats + self.attn(
            self.norm1(obs_feats), cos, sin, pad_mask)
        obs_feats = obs_feats + self.ff(self.norm2(obs_feats))
        return obs_feats


# =============================================================================
# Cross-attention
# =============================================================================

class CrossAttention(nn.Module):
    """
    Multi-head cross-attention: grid cells (queries) attend to the observation
    memory (keys/values), conditioning the program on the target sequence.

    No RoPE: queries (2-D grid) and keys (1-D token sequence) live in different
    coordinate spaces, so there is no shared relative position to rotate by.
    The obs memory already carries its position from the encoder, and the grid
    carries the IP position via the marker channel.

    Queries (grid) and keys/values (obs) may have different widths; the
    projections map both into a shared attention width attn_dim, where q . k is
    defined, and out_proj maps back to q_dim so the result adds onto the grid.
    """

    def __init__(self, q_dim, kv_dim, attn_dim, heads):
        super().__init__()
        assert attn_dim % heads == 0, "attn_dim must be divisible by heads"
        self.heads = heads
        self.head_dim = attn_dim // heads
        self.q_proj = nn.Linear(q_dim, attn_dim)     # from grid
        self.k_proj = nn.Linear(kv_dim, attn_dim)    # from obs
        self.v_proj = nn.Linear(kv_dim, attn_dim)
        self.out_proj = nn.Linear(attn_dim, q_dim)   # back to grid width

    def forward(self, grid_feats, obs_mem, pad_mask):
        """
        Grid cells attend to the observation memory.

        Parameters
        ----------
        grid_feats : Tensor
            (B, q_dim, H, W) grid features; the queries are projected from
            these.
        obs_mem : Tensor
            (B, L, kv_dim) observation memory; the keys and values are
            projected from it.
        pad_mask : Tensor
            (B, L) bool, True at real obs tokens, False at PAD.

        Returns
        -------
        Tensor
            (B, q_dim, H, W) obs-conditioned features, one per grid cell.
        """
        B, q_dim, H, W = grid_feats.shape
        # flatten the grid into a sequence of H*W cell "tokens" (feature-last)
        q_in = grid_feats.flatten(2).transpose(1, 2)   # (B, H*W, q_dim)
        # queries from the grid, keys/values from the obs memory
        q = self.q_proj(q_in).view(B, H * W, self.heads, self.head_dim)
        k = self.k_proj(obs_mem).view(B, -1, self.heads, self.head_dim)
        v = self.v_proj(obs_mem).view(B, -1, self.heads, self.head_dim)
        # heads beside batch: (B, heads, seq, head_dim)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        # each of the H*W cells scores over the L obs keys
        scores = q @ k.transpose(-2, -1) / self.head_dim**0.5
        # never attend to PAD obs keys
        scores = scores.masked_fill(~pad_mask[:, None, None, :], float("-inf"))
        attn = scores.softmax(dim=-1)            # over the L keys
        out = attn @ v                           # (B, heads, H*W, head_dim)
        # merge heads (-> attn_dim), then fold the sequence back into the grid
        out = out.transpose(1, 2).reshape(B, H * W, self.heads * self.head_dim)
        return self.out_proj(out).transpose(1, 2).view(B, q_dim, H, W)
