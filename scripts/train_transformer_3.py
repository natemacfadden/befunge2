"""
Recreate transformer_3 (transformer experiment 3): SFT from scratch on 5000
filtered random worlds. Also regenerates the worlds and the output-disjoint
held set deterministically (same seeds). Deterministic up to torch version and
hardware.

Produces scratch/cnn_sft.pt; copy to checkpoints/transformer_3.pt.
"""
# ruff: noqa: E402 (path shim precedes repo imports)

import os
import pickle
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.set_num_threads(4)

from models.common.random_worlds import random_worlds
from models.common.train import eval_reconstruction, sft
from models.transformer.model import Transformer

t = time.time()
train_w = random_worlds(5000, seed=0)
held = random_worlds(50, seed=999)
print(f"generated {len(train_w)}+{len(held)} worlds in {time.time() - t:.0f}s",
      flush=True)
os.makedirs("scratch", exist_ok=True)
with open("scratch/worlds_v2.pkl", "wb") as f:
    pickle.dump((train_w, held), f)

# output-disjoint held set: fresh-seed worlds whose outputs never occur in train
train_outputs = {tuple(o) for o, _ in train_w}
candidates = random_worlds(400, seed=777)
disjoint = [(o, s) for o, s in candidates if tuple(o) not in train_outputs][:50]
with open("scratch/held_disjoint.pkl", "wb") as f:
    pickle.dump(disjoint, f)

m = sft(Transformer(), train_w, steps=1500, batch=32, lr=1e-3, print_every=25,
        ckpt_every=100, ckpt_dir="scratch")

print("held-out (output-overlapping), best-of-32:", flush=True)
eval_reconstruction(m, held, k=32)
print("held-out (output-disjoint), best-of-32:", flush=True)
eval_reconstruction(m, disjoint, k=32)
