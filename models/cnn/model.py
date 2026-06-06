"""CNN that predicts the op at the IP cell from the partial grid."""

import torch
import torch.nn as nn

from models.cnn.vocab import VOCAB

EMBED_DIM = 16   # per-cell op embedding width -- tune


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.op_embed = nn.Embedding(len(VOCAB), EMBED_DIM)

    def encode_input(self, grid, filled):
        """Build the (B, EMBED_DIM + 1, H, W) input: op-embedding + filled mask."""
        # grid, filled: (B, H, W)
        emb = self.op_embed(grid)              # (B, H, W, EMBED_DIM)
        emb = emb.permute(0, 3, 1, 2)          # (B, EMBED_DIM, H, W)
        # not masking out data... just an extra channel recording which cells are placed
        mask = filled.unsqueeze(1).float()     # (B, 1, H, W)
        return torch.cat([emb, mask], dim=1)   # (B, EMBED_DIM + 1, H, W)
