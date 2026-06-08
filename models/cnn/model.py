"""
CNN that predicts the op at the IP cell from the partial grid.
"""

import torch
import torch.nn as nn

from models.cnn.attention import EncoderLayer, rope_tables
from models.cnn.tokenization import OBS_VOCAB_SIZE, OP_VOCAB_SIZE, PAD

EMBED_DIM = 16           # per-cell op embedding width
MODEL_DIM = 64           # observation feature / attention width (D)
NUM_HEADS = 4            # attention heads (head_dim = MODEL_DIM / NUM_HEADS)
NUM_ENCODER_LAYERS = 2   # stacked self-attention + feed-forward blocks
WORLDSTATE_CHANNELS = EMBED_DIM + 5   # op-embed + filled flag + 4-way heading
CONV_DIM = MODEL_DIM     # conv width; = obs width is convenient for cross-attn
NUM_CONV_LAYERS = 3      # toroidal conv layers over the worldstate grid
print(f"[cnn.model] EMBED_DIM={EMBED_DIM}, MODEL_DIM={MODEL_DIM}, "
      f"NUM_HEADS={NUM_HEADS}, NUM_ENCODER_LAYERS={NUM_ENCODER_LAYERS}, "
      f"NUM_CONV_LAYERS={NUM_CONV_LAYERS} -- tune these")


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # at this stage these are just matrices...
        # they'll be used for lookup (id -> row); their rows are parameters,
        # learned during training
        self.op_embed = nn.Embedding(OP_VOCAB_SIZE, EMBED_DIM)
        self.obs_embed = nn.Embedding(
            OBS_VOCAB_SIZE, MODEL_DIM, padding_idx=PAD)
        self.encoder = nn.ModuleList(
            [EncoderLayer(MODEL_DIM, NUM_HEADS)
             for _ in range(NUM_ENCODER_LAYERS)])

        # toroidal conv body over the worldstate grid (circular padding wraps
        # the edges, matching the playfield torus)
        convs = [nn.Conv2d(WORLDSTATE_CHANNELS, CONV_DIM, 3,
                           padding=1, padding_mode="circular"),
                 nn.ReLU()]
        for _ in range(NUM_CONV_LAYERS - 1):
            convs += [nn.Conv2d(CONV_DIM, CONV_DIM, 3,
                               padding=1, padding_mode="circular"),
                      nn.ReLU()]
        self.conv = nn.Sequential(*convs)

    def encode_worldstate(self, grid, filled, ip, heading):
        """
        Build the model input (concatenated):
            1) embedding of placed characters,
            2) indicator variable of what cells have been placed,
            3) IP placement/direction (4-way heading).
        Shape is (B, EMBED_DIM + 1(indicator)+4(IP pos/dir), H, W).

        IP position effectivelygoes like
        [up,right,down,left] = torch.zeros(B, 4, H, W, device=grid.device)
        if IP going up:
            up[y,x] = 1
        elif IP going right:
            right[y,x] = 1
        elif ...
        """
        B, H, W = grid.shape
        
        # data embedding
        # Conv2d needs the feature dim at axis 1 (channels-first), so permute
        emb = self.op_embed(grid)     # (B, H, W, EMBED_DIM)
        emb = emb.permute(0, 3, 1, 2) # (B, EMBED_DIM, H, W)

        # indicator variable as to whether cells have been placed
        filled_flag = filled.unsqueeze(1).float() # (B, 1, H, W)

        # positional/directional heading
        # (heading is ^, >, v, or <; stack 4x arrays and place a single 1 in one
        #  of them)
        # (The one for IP's direction of travel, in the location of the IP)
        marker = torch.zeros(B, 4, H, W, device=grid.device)
        marker[torch.arange(B), heading, ip[:, 1], ip[:, 0]] = 1.0

        # return
        return torch.cat([emb, filled_flag, marker], dim=1)

    def encode_observations(self, tokens):
        """
        Encode the tokenized target sequence into memory for the decoder to
        attend to (embed the tokens, then self-attend over the sequence).

        Parameters
        ----------
        tokens : LongTensor
            (B, L) token ids from obs_to_tokens.

        Returns
        -------
        Tensor
            (B, L, D) per-token features, where B = batch size, L = number of
            tokens, and D = feature width per token.
        """
        # embed -- attention needs the feature dim last (B, L, MODEL_DIM)
        x = self.obs_embed(tokens)

        # encode: run the token features through the self-attention stack
        L = tokens.shape[1]
        cos, sin = rope_tables(L, MODEL_DIM // NUM_HEADS)
        cos, sin = cos.to(x.device), sin.to(x.device)
        pad_mask = tokens != PAD          # (B, L) True at real tokens
        for layer in self.encoder:
            x = layer(x, cos, sin, pad_mask)
        return x

    def forward(self, worldstate_features, observation_features, ip):
        """
        Predict the op at the IP cell:
            1) run the conv layers, conditioning on the observations,
            2) read the op logits at the IP cell.
        worldstate_features and observation_features are precomputed by
        encode_worldstate / encode_observations.
        ip is (B, 2) of (x, y), used to index which cell's logits to return.
        Returns shape (B, V), where
            B = batch size,
            V = vocab size (the op distribution at the IP cell).
        """
        # conv layers (conditioned on observation_features)
        ...

        # read logits at the IP cell
        ...
