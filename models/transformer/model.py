"""
Transformer program writer: the program is a set of tokens (one per filled
cell, plus the IP query cell), each placed in the IP-centered, heading-canonical
frame (see frame.py). Cells attend to each other with a toroidal relative-
position bias and cross-attend to the encoded target sequence; the op is read
out from the IP token. Batch size 1 (the rollout harness is per-example).
"""

import torch
import torch.nn as nn

from models.common.attention import EncoderLayer, rope_tables
from models.common.tokenization import (
    N_ACTIONS,
    OBS_VOCAB_SIZE,
    OP_VOCAB_SIZE,
    PAD,
    H,
)
from models.transformer.frame import canonical_offsets, canonical_offsets_batch

MODEL_DIM = 64           # token / attention width (D)
NUM_HEADS = 4
NUM_OBS_LAYERS = 2       # RoPE self-attention over the target sequence
NUM_GRID_LAYERS = 4      # self-attn (+ cross-attn) over the cell tokens
GRID_SIZE = H            # square torus side
print(f"[transformer.model] MODEL_DIM={MODEL_DIM}, NUM_HEADS={NUM_HEADS}, "
      f"NUM_GRID_LAYERS={NUM_GRID_LAYERS}, GRID_SIZE={GRID_SIZE} -- tune these")


class RelBiasSelfAttention(nn.Module):
    """
    Multi-head self-attention over the cell tokens with a learned toroidal
    relative-position bias: score(i, j) += bias[head, dx, dy], where (dx, dy) is
    the wrapped offset between the cells' canonical positions.
    """

    def __init__(self, dim, heads, size):
        super().__init__()
        self.heads, self.head_dim, self.size = heads, dim // heads, size
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.bias = nn.Parameter(torch.zeros(heads, size, size))

    def forward(self, x, pos):
        # x: (B, n, dim) tokens; pos: (B, n, 2) canonical positions
        B, n, _ = x.shape
        qkv = self.qkv(x).reshape(B, n, 3, self.heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)      # each (B, heads, n, head_dim)
        scores = (q @ k.transpose(-1, -2)) / self.head_dim ** 0.5
        rel = (pos[:, :, None, :] - pos[:, None, :, :]) % self.size
        bias = self.bias[:, rel[..., 0], rel[..., 1]]      # (heads, B, n, n)
        scores = scores + bias.permute(1, 0, 2, 3)
        out = scores.softmax(-1) @ v              # (B, heads, n, head_dim)
        out = out.transpose(1, 2).reshape(B, n, self.heads * self.head_dim)
        return self.out(out)


class TokenCrossAttention(nn.Module):
    """
    Cell tokens (queries) attend to the encoded target sequence (keys/values),
    with PAD positions masked out.
    """

    def __init__(self, dim, heads):
        super().__init__()
        self.heads, self.head_dim = heads, dim // heads
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2 * dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x, mem, pad_mask):
        # x: (B, n, dim) queries; mem: (B, L, dim); pad_mask: (B, L) bool
        B, n, _ = x.shape
        length = mem.shape[1]
        q = self.q(x).reshape(B, n, self.heads, self.head_dim).transpose(1, 2)
        kv = self.kv(mem).reshape(B, length, 2, self.heads, self.head_dim)
        k, v = kv.permute(2, 0, 3, 1, 4)          # each (B, heads, L, head_dim)
        scores = (q @ k.transpose(-1, -2)) / self.head_dim ** 0.5
        scores = scores.masked_fill(~pad_mask[:, None, None, :], float("-inf"))
        out = scores.softmax(-1) @ v
        out = out.transpose(1, 2).reshape(B, n, self.heads * self.head_dim)
        return self.out(out)


class GridLayer(nn.Module):
    """
    Pre-norm block: relative-bias self-attention, then cross-attention to the
    target memory, then a feed-forward.
    """

    def __init__(self, dim, heads, size, ff_mult=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = RelBiasSelfAttention(dim, heads, size)
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = TokenCrossAttention(dim, heads)
        self.norm3 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_mult * dim), nn.GELU(),
            nn.Linear(ff_mult * dim, dim))

    def forward(self, x, pos, mem, pad_mask):
        x = x + self.self_attn(self.norm1(x), pos)
        x = x + self.cross_attn(self.norm2(x), mem, pad_mask)
        x = x + self.ff(self.norm3(x))
        return x


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.op_embed = nn.Embedding(OP_VOCAB_SIZE, MODEL_DIM)
        self.query_embed = nn.Parameter(torch.zeros(MODEL_DIM))  # IP token
        self.obs_embed = nn.Embedding(
            OBS_VOCAB_SIZE, MODEL_DIM, padding_idx=PAD)
        self.obs_encoder = nn.ModuleList(
            [EncoderLayer(MODEL_DIM, NUM_HEADS) for _ in range(NUM_OBS_LAYERS)])
        self.layers = nn.ModuleList(
            [GridLayer(MODEL_DIM, NUM_HEADS, GRID_SIZE)
             for _ in range(NUM_GRID_LAYERS)])
        self.op_head = nn.Linear(MODEL_DIM, N_ACTIONS)

    def encode_observations(self, tokens):
        """
        Encode the tokenized target sequence into memory (same RoPE encoder as
        the CNN). tokens: (B, L). Returns (B, L, MODEL_DIM).
        """
        x = self.obs_embed(tokens)
        cos, sin = rope_tables(tokens.shape[1], MODEL_DIM // NUM_HEADS)
        cos, sin = cos.to(x.device), sin.to(x.device)
        pad_mask = tokens != PAD
        for layer in self.obs_encoder:
            x = layer(x, cos, sin, pad_mask)
        return x

    def encode_worldstate(self, grid, filled, ip, heading):
        """
        Build the cell tokens for batch size 1: one per filled cell (its op
        embedded) plus the IP query token, each with its canonical position.
        Returns (tokens, positions, ip_index).
        """
        grid, filled = grid[0], filled[0]
        ip_xy = ip[0].to(torch.long)
        ys, xs = filled.nonzero(as_tuple=True)
        coords = torch.stack([xs, ys], dim=1)             # (m, 2) as (x, y)
        tokens = self.op_embed(grid[ys, xs])              # (m, MODEL_DIM)
        # append the IP cell (blank) as the query token
        coords = torch.cat([coords, ip_xy[None]], dim=0)
        tokens = torch.cat([tokens, self.query_embed[None]], dim=0)
        pos = canonical_offsets(coords, ip_xy, int(heading[0]), GRID_SIZE)
        return tokens, pos, tokens.shape[0] - 1

    def forward(self, worldstate, observation_features, pad_mask, ip):
        """
        Run the grid layers and read out the IP token. Returns (1, N_ACTIONS).
        """
        tokens, pos, ip_index = worldstate
        logits = self.forward_batch(tokens[None], pos[None],
                                    observation_features, pad_mask)
        return logits   # ip token is last, which forward_batch reads out

    def encode_worldstate_batch(self, grids, filled, ips, headings):
        """
        Batched encode_worldstate for lockstep teacher-forcing: every program
        in the batch must have the same number of filled cells (true in
        lockstep, where each round places exactly one op per active program).

        grids (B, H, W) vocab ids, filled (B, H, W) bool, ips (B, 2) long,
        headings (B,) long. Returns (tokens (B, k+1, D), pos (B, k+1, 2)),
        with each program's IP query token last.
        """
        B, _, Ww = grids.shape
        flat = filled.reshape(B, -1)
        k = int(flat[0].sum())
        cells = flat.nonzero(as_tuple=False)[:, 1].view(B, k)
        ys, xs = cells // Ww, cells % Ww
        ops = grids.reshape(B, -1).gather(1, cells)        # (B, k)
        tokens = self.op_embed(ops)                        # (B, k, D)
        tokens = torch.cat(
            [tokens, self.query_embed.expand(B, 1, -1)], dim=1)
        coords = torch.stack([xs, ys], dim=-1)             # (B, k, 2)
        coords = torch.cat([coords, ips[:, None, :]], dim=1)
        pos = canonical_offsets_batch(coords, ips, headings, GRID_SIZE)
        return tokens, pos

    def forward_batch(self, tokens, pos, observation_features, pad_mask):
        """
        Batched forward: run the grid layers and read out each program's IP
        token (the last token). Returns (B, N_ACTIONS).
        """
        for layer in self.layers:
            tokens = layer(tokens, pos, observation_features, pad_mask)
        return self.op_head(tokens[:, -1])
