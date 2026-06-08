"""
Autoregressive rollout: drive the Stepper with the CNN to generate a befunge
program for a target sequence, choosing each op in execution order.
"""

import numpy as np
import torch

from models.cnn.model import DONE, N_ACTIONS
from models.cnn.stepper import Stepper
from models.cnn.tokenization import (
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

# boolean mask over actions, True at banned ops (out-of-place ban for the
# differentiable rollout, which can't mutate logits in place)
BANNED_MASK = torch.zeros(N_ACTIONS, dtype=torch.bool)
BANNED_MASK[BANNED_OPS] = True


@torch.no_grad()
def rollout(model, target, seed=0, max_places=H * W, max_steps=2000):
    """
    Generate a program for `target` by sampling the model op by op.

    Encodes the target once, then drives the Stepper: at each new cell the IP
    lands on, runs the model conditioned on the obs memory and samples an op.

    Parameters
    ----------
    model : CNN
    target : list of int
        The sequence we want the program to print.
    seed : int
        Seeds op sampling (torch) and the '?' op's randomness (numpy).
    max_places : int
        Stop after this many op placements.
    max_steps : int
        Step cap per Stepper.run (bounds execution between placements).

    Returns
    -------
    stepper, status, trace
        stepper holds the final grid/output; status is 'halt'/'limit'/
        'newcell'; trace is a list of per-step dicts (x, y, heading, op,
        program).
    """
    # seed op sampling (torch) and the '?' op's direction (numpy)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # tokenize the target and encode it once; the memory is reused every step
    tokens = torch.tensor(obs_to_tokens([target]))
    pad_mask = tokens != PAD
    mem = model.encode_observations(tokens)

    s = Stepper((H, W))                      # blank grid to fill
    trace, status = [], "newcell"
    while len(trace) < max_places:
        status = s.run(max_steps)            # walk to the next blank cell
        if status != "newcell":
            break                            # halted or hit the step cap

        # the current partial program + IP, as the model sees it
        vocab_grid, filled, (x, y), heading = s.worldstate()
        ws = model.encode_worldstate(
            torch.tensor(vocab_grid)[None],
            torch.tensor(filled)[None],
            torch.tensor([[x, y]]),
            torch.tensor([heading]),
        )
        # action distribution at the IP cell, then sample one
        logits = model(ws, mem, pad_mask, torch.tensor([[x, y]]))
        logits[:, BANNED_OPS] = float("-inf")   # never place banned ops
        action = int(torch.multinomial(logits.softmax(-1), 1))
        if action == DONE:
            status = "done"                     # model says the program is done
            break
        op = action
        s.place(op)

        trace.append({"x": x, "y": y, "heading": "^>v<"[heading],
                      "op": OP_FROM_ID[op],
                      "program": from_grid(s.worldstate()[0])})
    return s, status, trace


def train_rollout(model, target, max_places=H * W, max_steps=2000):
    """
    Like rollout but differentiable: runs with gradients and returns the summed
    log-probability and summed entropy of the sampled actions (for REINFORCE
    with an entropy bonus).

    Returns (stepper, status, logprob, entropy), both scalar tensors.
    """
    tokens = torch.tensor(obs_to_tokens([target]))
    pad_mask = tokens != PAD
    mem = model.encode_observations(tokens)
    s = Stepper((H, W))
    logps, ents, status = [], [], "newcell"
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
            break
        s.place(action)
    logprob = torch.stack(logps).sum() if logps else torch.zeros(())
    entropy = torch.stack(ents).sum() if ents else torch.zeros(())
    return s, status, logprob, entropy
