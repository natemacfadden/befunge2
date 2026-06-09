"""
Tests for the IP-centered, heading-canonical coordinate frame.
"""

import torch

from models.transformer.frame import canonical_offsets

# stepper heading -> direction vector {0:^, 1:>, 2:v, 3:<}
DIRS = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}
SIZE = 64


def test_ip_at_origin_and_forward_is_plus_x():
    ip = torch.tensor([10, 20])
    for h, (vx, vy) in DIRS.items():
        fwd = ip + torch.tensor([vx, vy])
        off = canonical_offsets(torch.stack([ip, fwd]), ip, h, SIZE)
        assert off[0].tolist() == [0, 0]      # the IP itself
        assert off[1].tolist() == [1, 0]      # one step forward -> (1, 0)


def test_jump_is_two_forward():
    ip = torch.tensor([5, 5])
    for h, (vx, vy) in DIRS.items():
        jump = ip + 2 * torch.tensor([vx, vy])
        off = canonical_offsets(jump[None], ip, h, SIZE)
        assert off[0].tolist() == [2, 0]


def test_behind_wraps_toroidally():
    ip = torch.tensor([0, 0])
    for h, (vx, vy) in DIRS.items():
        back = ip - torch.tensor([vx, vy])    # one cell behind the IP
        off = canonical_offsets(back[None], ip, h, SIZE)
        assert off[0].tolist() == [SIZE - 1, 0]   # -1 forward wraps to 63
