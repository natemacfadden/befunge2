"""
Checkpoint smoke test: the saved transformer_3 checkpoint loads into the
current Transformer and can still generate programs end to end (rollout runs,
a program comes out, and at least one sample reproduces a leading term of a
training-distribution target). Skipped when the checkpoint is absent (it is
gitignored), e.g. in CI.
"""

from pathlib import Path

import pytest
import torch

from bench.verify import num_leading
from models.common.rollout import rollout
from models.common.tokenization import from_grid

CKPT = Path(__file__).resolve().parent.parent / "checkpoints/transformer_3.pt"


@pytest.mark.skipif(not CKPT.exists(), reason="checkpoint not on this machine")
def test_transformer_3_generates():
    from models.transformer.model import Transformer

    model = Transformer()
    model.load_state_dict(torch.load(CKPT))
    model.eval()

    target = [5, 5, 5, 5, 5, 5, 5, 5]
    best = 0
    for seed in range(8):
        stepper, status, _trace = rollout(model, target, seed=seed,
                                          max_places=64)
        assert status in {"done", "halt", "error", "limit", "newcell"}
        program = from_grid(stepper.worldstate()[0])
        assert program.strip()                  # it placed something
        best = max(best, num_leading(program, "befunge", target,
                                     max_steps=5000))
        if best:
            break
    assert best >= 1                            # reproduces a leading term
