"""
Recreate transformer_8_rlextra (experiment 8): same RL as experiment 7 but
with verify_extra=4, so targets are rolled 4 terms past the shown window and
scored on the full sequence. Literal printing of the shown terms no longer
counts as a solve; the order-2 unlock (step ~825) is earned by extrapolation.

Writes per-run checkpoints to scratch/rl8/; transformer_8_rlextra.pt was the
last saved step (1900). Requires checkpoints/transformer_6_geom.pt from
train_geom_sft.py.
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
      verify_extra=4, print_every=10, ckpt_every=100, ckpt_dir="scratch/rl8")
