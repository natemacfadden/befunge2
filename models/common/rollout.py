"""
Autoregressive rollout: drive the Stepper with a model to generate a befunge
program for a target sequence, choosing each op in execution order. rollout is
for inference (sampling, no grad); train_rollout is the differentiable version
used by training.
"""

import numpy as np
import torch

from models.common.stepper import Stepper
from models.common.tokenization import (
    DONE,
    N_ACTIONS,
    OP_FROM_ID,
    OP_TO_ID,
    PAD,
    H,
    W,
    from_grid,
    obs_to_tokens,
)

# ops we never place: ? (nondeterministic, breaks verification), " (string
# mode), & ~ (stdin, no-ops here)
BANNED_OPS = [OP_TO_ID[c] for c in '?"&~']
BANNED_MASK = torch.zeros(N_ACTIONS, dtype=torch.bool)
BANNED_MASK[BANNED_OPS] = True


@torch.no_grad()
def rollout(model, target, seed=0, max_places=H * W, max_steps=2000):
    """
    Generate a program for `target` by sampling the model op by op.

    Encodes the target once, then drives the Stepper: at each new cell the IP
    lands on, runs the model conditioned on the obs memory and samples an op.

    Returns (stepper, status, trace): the stepper holds the final grid/output;
    status is 'newcell'/'halt'/'error'/'limit'/'done'; trace is a list of
    per-step dicts (x, y, heading, op, program).
    """
    # seed op sampling (torch) and the '?' op's direction (numpy)
    torch.manual_seed(seed)
    np.random.seed(seed)

    tokens = torch.tensor(obs_to_tokens([target]))
    pad_mask = tokens != PAD
    mem = model.encode_observations(tokens)

    s = Stepper((H, W))
    trace, status = [], "newcell"
    while len(trace) < max_places:
        status = s.run(max_steps)            # walk to the next blank cell
        if status != "newcell":
            break

        vocab_grid, filled, (x, y), heading = s.worldstate()
        ws = model.encode_worldstate(
            torch.tensor(vocab_grid)[None],
            torch.tensor(filled)[None],
            torch.tensor([[x, y]]),
            torch.tensor([heading]),
        )
        logits = model(ws, mem, pad_mask, torch.tensor([[x, y]]))
        logits[:, BANNED_OPS] = float("-inf")
        action = int(torch.multinomial(logits.softmax(-1), 1))
        if action == DONE:
            status = "done"
            break
        s.place(action)
        trace.append({"x": x, "y": y, "heading": "^>v<"[heading],
                      "op": OP_FROM_ID[action],
                      "program": from_grid(s.worldstate()[0])})
    return s, status, trace


def train_rollout(model, target, max_places=H * W, max_steps=2000):
    """
    Differentiable rollout for training: like rollout but runs with gradients
    and returns per-step data instead of a trace, so the caller can score and
    credit each placement on its own.

    Returns (stepper, status, logps, programs, placed, entropy):
        logps     list of per-step log-prob tensors (carry grad),
        programs  list of the program string after each step,
        placed    list of 0/1, 1 if that step placed a non-blank character,
        entropy   summed entropy (scalar tensor).
    """
    tokens = torch.tensor(obs_to_tokens([target]))
    pad_mask = tokens != PAD
    mem = model.encode_observations(tokens)
    s = Stepper((H, W))
    logps, programs, placed, ents, status = [], [], [], [], "newcell"
    while len(logps) < max_places:
        status = s.run(max_steps)
        if status != "newcell":
            break
        vocab_grid, filled, (x, y), heading = s.worldstate()
        ws = model.encode_worldstate(
            torch.tensor(vocab_grid)[None], torch.tensor(filled)[None],
            torch.tensor([[x, y]]), torch.tensor([heading]))
        logits = model(ws, mem, pad_mask, torch.tensor([[x, y]]))
        logp = logits.masked_fill(BANNED_MASK, float("-inf")).log_softmax(-1)
        prob = logp.exp()                                 # 0 at banned ops
        # entropy over allowed ops; drop the banned slots' -inf logp so the
        # product is a finite 0*0, not a 0*-inf nan that poisons the gradient
        ent = -(prob * logp.masked_fill(BANNED_MASK, 0.0)).sum()
        action = int(torch.multinomial(prob, 1))          # sampling is detached
        logps.append(logp[0, action])                     # carries grad
        ents.append(ent)
        if action == DONE:
            status = "done"
            programs.append(from_grid(s.worldstate()[0]))  # grid unchanged
            placed.append(0)
            break
        s.place(action)
        programs.append(from_grid(s.worldstate()[0]))
        placed.append(1 if action != OP_TO_ID[" "] else 0)
    entropy = torch.stack(ents).sum() if ents else torch.zeros(())
    return s, status, logps, programs, placed, entropy
