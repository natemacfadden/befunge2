"""
Equivalence test for the batched transformer path: lockstep batched
teacher-forcing must produce the same logits as the single-program path,
including when programs leave the batch at different rounds.
"""

import numpy as np
import torch

from models.common.stepper import Stepper
from models.common.tokenization import PAD, H, W, obs_to_tokens, to_grid
from models.transformer.model import Transformer

MAX_PLACES = 12

# different content; the @ one halts early, exercising the done mask
PROGRAMS = [([5, 5, 5], "5."), ([7, 7, 7], "73+."), ([1, 1, 1], "1.@")]


def _sequential_logits(model, target, ref, max_places):
    tokens = torch.tensor(obs_to_tokens([target]))
    pad_mask = tokens != PAD
    mem = model.encode_observations(tokens)
    s = Stepper((H, W))
    out = []
    while len(out) < max_places:
        if s.run(2000) != "newcell":
            break
        vocab_grid, filled, (x, y), heading = s.worldstate()
        ws = model.encode_worldstate(
            torch.tensor(vocab_grid)[None], torch.tensor(filled)[None],
            torch.tensor([[x, y]]), torch.tensor([heading]))
        out.append(model(ws, mem, pad_mask, torch.tensor([[x, y]]))[0])
        s.place(int(ref[y, x]))
    return out


def _lockstep_logits(model, pairs, max_places):
    tokens = torch.tensor(obs_to_tokens([t for t, _ in pairs]))
    pad_mask = tokens != PAD
    mem = model.encode_observations(tokens)
    steppers = [Stepper((H, W)) for _ in pairs]
    active = list(range(len(pairs)))
    out = [[] for _ in pairs]
    for _ in range(max_places):
        active = [i for i in active if steppers[i].run(2000) == "newcell"]
        if not active:
            break
        states = [steppers[i].worldstate() for i in active]
        grids = torch.from_numpy(np.stack([s[0] for s in states]))
        fills = torch.from_numpy(np.stack([s[1] for s in states]))
        ips = torch.tensor([s[2] for s in states])
        heads = torch.tensor([s[3] for s in states])
        toks, pos = model.encode_worldstate_batch(grids, fills, ips, heads)
        idx = torch.tensor(active)
        logits = model.forward_batch(toks, pos, mem[idx], pad_mask[idx])
        for row, (i, s) in enumerate(zip(active, states)):
            out[i].append(logits[row])
            steppers[i].place(int(pairs[i][1][s[2][1], s[2][0]]))
    return out


def test_batched_matches_sequential():
    torch.manual_seed(0)
    model = Transformer().eval()
    pairs = [(t, to_grid(src)) for t, src in PROGRAMS]

    with torch.no_grad():
        batched = _lockstep_logits(model, pairs, MAX_PLACES)
        for i, (target, ref) in enumerate(pairs):
            seq = _sequential_logits(model, target, ref, MAX_PLACES)
            assert len(seq) == len(batched[i])
            for a, b in zip(seq, batched[i]):
                assert torch.allclose(a, b, atol=1e-5), (i, (a - b).abs().max())
