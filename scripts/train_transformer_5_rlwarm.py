"""
Recreate transformer_5_rlwarm (transformer experiment 5): RL on the order-1
curriculum, warm started from transformer_3. Result: solves constants, fails
geometrics (answers them with constant/periodic programs).

Produces scratch/rl/cnn_step*.pt; transformer_5_rlwarm.pt was step 900.
"""
# ruff: noqa: E402 (path shim precedes repo imports)

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.set_num_threads(2)

from models.common.train import train
from models.transformer.model import Transformer

m = Transformer()
m.load_state_dict(torch.load("checkpoints/transformer_3.pt"))
train(m, steps=1000, k=8, lr=1e-4, entropy_coef=0.0, max_places=48,
      print_every=10, ckpt_every=100, ckpt_dir="scratch/rl")
