"""
Structural-contract test for the transformer: token set = filled cells + the IP
query token, the IP token is at canonical origin, and the readout is per-cell.
"""

import torch

from models.common.stepper import Stepper
from models.common.tokenization import (
    N_ACTIONS,
    OP_TO_ID,
    PAD,
    H,
    W,
    obs_to_tokens,
)
from models.transformer.model import Transformer


def test_worldstate_tokens_and_forward_shape():
    torch.manual_seed(0)
    model = Transformer()
    tokens = torch.tensor(obs_to_tokens([[3, 5, 7]]))
    pad_mask = tokens != PAD
    mem = model.encode_observations(tokens)

    # place one op so there's a filled cell plus the (blank) IP cell
    s = Stepper((H, W))
    s.run(2000)
    s.place(OP_TO_ID["5"])
    s.run(2000)
    vocab_grid, filled, (x, y), heading = s.worldstate()
    ws = model.encode_worldstate(
        torch.tensor(vocab_grid)[None], torch.tensor(filled)[None],
        torch.tensor([[x, y]]), torch.tensor([heading]))

    toks, pos, ip_index = ws
    n_filled = int(torch.tensor(filled).sum())
    assert toks.shape[0] == n_filled + 1          # filled cells + IP token
    assert ip_index == toks.shape[0] - 1          # IP token is last
    assert pos[ip_index].tolist() == [0, 0]       # IP sits at the origin

    logits = model(ws, mem, pad_mask, torch.tensor([[x, y]]))
    assert logits.shape == (1, N_ACTIONS)
