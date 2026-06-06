"""CNN that predicts the op at the IP cell from the partial grid."""

import torch
import torch.nn as nn

from models.cnn.vocab import VOCAB

EMBED_DIM = 16   # per-cell op embedding width -- tune


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.op_embed = nn.Embedding(len(VOCAB), EMBED_DIM)

    def encode_input(self, grid, filled, ip, heading):
        """Build the model input: op-embedding + filled flag + IP marker (4-way heading), as (B, EMBED_DIM + 5, H, W).
        ip is (B, 2) of (x, y); heading is (B,) in {0:^, 1:>, 2:v, 3:<}."""
        B, H, W = grid.shape
        
        # data embedding
        emb = self.op_embed(grid)     # (B, H, W, EMBED_DIM)
        emb = emb.permute(0, 3, 1, 2) # (B, EMBED_DIM, H, W)

        # indicator variable as to whether cells have been placed
        filled_flag = filled.unsqueeze(1).float() # (B, 1, H, W)

        # positional/directional heading
        # (heading is ^, >, v, or <; stack 4x arrays and place a single 1 in one of them)
        # (the one for IP's direction of travel, in the location of the IP)
        marker = torch.zeros(B, 4, H, W, device=grid.device)
        marker[torch.arange(B), heading, ip[:, 1], ip[:, 0]] = 1.0

        # return
        return torch.cat([emb, filled_flag, marker], dim=1)    # (B, EMBED_DIM + 5, H, W)
