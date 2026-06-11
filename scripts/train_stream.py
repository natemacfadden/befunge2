"""
Recreate the streaming run (transformer experiment 4): SFT from scratch on a
continuous stream of fresh random worlds, no fixed dataset. Statistically
reproducible only; producer arrival order is nondeterministic.

Produces scratch/stream/stream_final.pt. Requires the disjoint held set
from train_transformer_3.py for the periodic eval.
"""
# ruff: noqa: E402 (path shim precedes repo imports)

import os
import pickle
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.set_num_threads(4)

from models.common.train import sft_stream
from models.transformer.model import Transformer

with open("scratch/held_disjoint.pkl", "rb") as f:
    held = pickle.load(f)

sft_stream(Transformer(), steps=3000, batch=32, producers=4,
           eval_pairs=held, eval_every=100, eval_k=8,
           print_every=20, ckpt_every=500)
