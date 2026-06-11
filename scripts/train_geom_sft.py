"""
Recreate the geometric-idiom SFT (transformer experiment 6): warm start from
transformer_3, train on the 5000 random worlds (replay) plus the known-optimal
geometric family a0 > :.r* with 12 of 72 (a0, r) combos held out, geometrics
upweighted 30x. Evals: held-out combos (template generalization), sampled
train combos, and the disjoint worlds (forgetting check), all best-of-8.

Produces scratch/geom/cnn_sft.pt. Requires worlds_v2.pkl, held_disjoint.pkl
and checkpoints/transformer_3.pt from train_transformer_3.py.
"""
# ruff: noqa: E402 (path shim precedes repo imports)

import os
import pickle
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.set_num_threads(2)

from models.common.train import eval_reconstruction, sft
from models.transformer.model import Transformer


def geometric_program(a0, r):
    row0 = f"{a0}>:.{r}*v"
    row1 = " ^" + " " * 4 + "<"
    return row0 + "\n" + row1


def geometric_target(a0, r):
    seq = [a0]
    while len(seq) < 8:
        seq.append(seq[-1] * r)
    return seq


combos = [(a0, r) for a0 in range(1, 10) for r in range(2, 10)]
held_combos = combos[::6]
train_combos = [c for c in combos if c not in held_combos]
geo_train = [(geometric_target(a, r), geometric_program(a, r))
             for a, r in train_combos]
geo_held = [(geometric_target(a, r), geometric_program(a, r))
            for a, r in held_combos]

with open("scratch/worlds_v2.pkl", "rb") as f:
    worlds, _ = pickle.load(f)
with open("scratch/held_disjoint.pkl", "rb") as f:
    disjoint = pickle.load(f)

pairs = worlds + geo_train * 30
print(f"{len(geo_train)} train geos x30 + {len(worlds)} worlds; "
      f"{len(geo_held)} held combos", flush=True)

m = Transformer()
m.load_state_dict(torch.load("checkpoints/transformer_3.pt"))
m = sft(m, pairs, steps=1500, lr=3e-4, batch=32, print_every=25,
        ckpt_every=200, ckpt_dir="scratch/geom")

print("held-out geometric combos, best-of-8:", flush=True)
eval_reconstruction(m, geo_held, k=8)
print("sampled train geometric combos, best-of-8:", flush=True)
eval_reconstruction(m, geo_train[:12], k=8)
print("forgetting check, disjoint worlds, best-of-8:", flush=True)
eval_reconstruction(m, disjoint, k=8)
