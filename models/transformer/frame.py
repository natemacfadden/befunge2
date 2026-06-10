"""
IP-centered, heading-canonical coordinate frame.

On the toroidal playfield every position is equivalent under translation; the
only distinguished point is the IP. So a cell's position is expressed as its
toroidal offset from the IP, rotated so the IP's heading points along +x
(forward). The IP itself is at (0, 0); a straight step lands the next cell at
(1, 0), a jump (#) at (2, 0). This makes the model translation- and
rotation-equivariant on the torus, with the IP as the moving origin.
"""

import torch


def canonical_offsets(coords_xy, ip_xy, heading, size):
    """
    Offset of each cell from the IP, rotated so the heading points along +x.

    coords_xy : (n, 2) int tensor of (x, y) cell coordinates.
    ip_xy     : (2,) int tensor, the IP's (x, y).
    heading   : int in {0:^, 1:>, 2:v, 3:<} (matching the Stepper).
    size      : torus side length (square).

    Returns an (n, 2) int tensor of (forward, lateral) offsets, wrapped to
    [0, size).
    """
    d = coords_xy - ip_xy
    dx, dy = d[:, 0], d[:, 1]
    if heading == 1:        # > right: forward is already +x
        ox, oy = dx, dy
    elif heading == 2:      # v down: rotate (0,1) -> (1,0)
        ox, oy = dy, -dx
    elif heading == 3:      # < left: rotate (-1,0) -> (1,0)
        ox, oy = -dx, -dy
    else:                   # ^ up: rotate (0,-1) -> (1,0)
        ox, oy = -dy, dx
    return torch.stack([ox % size, oy % size], dim=1)


def canonical_offsets_batch(coords_xy, ip_xy, heading, size):
    """
    Batched canonical_offsets: coords_xy (B, n, 2), ip_xy (B, 2), heading (B,)
    int tensor. Returns (B, n, 2) offsets, each program rotated by its own
    heading.
    """
    d = coords_xy - ip_xy[:, None, :]
    dx, dy = d[..., 0], d[..., 1]
    h = heading[:, None]
    ox = torch.where(h == 1, dx, torch.where(h == 2, dy,
                     torch.where(h == 3, -dx, -dy)))
    oy = torch.where(h == 1, dy, torch.where(h == 2, -dx,
                     torch.where(h == 3, -dy, dx)))
    return torch.stack([ox % size, oy % size], dim=-1)
