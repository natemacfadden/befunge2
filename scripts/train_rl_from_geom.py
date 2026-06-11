"""
Recreate the experiment-7 RL run: curriculum RL warm started from the
geometric SFT model (transformer_6_geom). First run to unlock order-2
(step ~775), but order-2 "solves" included literal-printing of short shown
targets; superseded by experiment 8's verified reward.

Writes per-run checkpoints to scratch/rl7/. Requires
checkpoints/transformer_6_geom.pt from train_geom_sft.py.
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
m.load_state_dict(torch.load("checkpoints/transformer_6_geom.pt"))
train(m, steps=2000, k=8, lr=1e-4, entropy_coef=0.0, max_places=48,
      print_every=10, ckpt_every=100, ckpt_dir="scratch/rl7")
