"""Grid geometry shared by the ROS models and the propagation step.

Direction convention: a raster offset is `(dy, dx)` where increasing `y` (row)
is north and increasing `x` (column) is east. An angle is measured with
`atan2(north, east)`, so a wind vector `(east, north)` and a neighbour offset
share one convention with no conversion.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

# The eight Moore neighbours, in a fixed order that every `[B, 8, H, W]`
# direction tensor in the package follows.
NEIGHBOR_OFFSETS: tuple[tuple[int, int], ...] = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)

N_DIRS = len(NEIGHBOR_OFFSETS)


def neighbor_angles(device=None, dtype=torch.float32) -> Tensor:
    """`[8]` bearing of each neighbour, radians, `atan2(north, east)`."""
    return torch.tensor(
        [math.atan2(dy, dx) for dy, dx in NEIGHBOR_OFFSETS], device=device, dtype=dtype
    )


def neighbor_distances(cell_size: float, device=None, dtype=torch.float32) -> Tensor:
    """`[8]` centre-to-centre distance to each neighbour, metres."""
    return torch.tensor(
        [math.hypot(dy, dx) * cell_size for dy, dx in NEIGHBOR_OFFSETS], device=device, dtype=dtype
    )


def shift(x: Tensor, dy: int, dx: int) -> Tensor:
    """Translate a raster so that `out[..., i, j] == x[..., i - dy, j - dx]`.

    Off-grid reads come back as zero, which makes the domain edge behave as
    non-fuel rather than wrapping the fire around the map.
    """
    h, w = x.shape[-2:]
    pad = (max(dx, 0), max(-dx, 0), max(dy, 0), max(-dy, 0))  # l, r, t, b
    xp = F.pad(x, pad)
    y0, x0 = max(-dy, 0), max(-dx, 0)
    return xp[..., y0 : y0 + h, x0 : x0 + w]


def slope_vector(elevation: Tensor, cell_size: float) -> Tensor:
    """Terrain gradient as `[B, 2, H, W]` = (d z / d east, d z / d north).

    Central differences, one-sided at the border. Dimensionless rise/run, so
    the magnitude is `tan(slope angle)`.
    """
    dz_dx = (shift(elevation, 0, -1) - shift(elevation, 0, 1)) / (2 * cell_size)
    dz_dy = (shift(elevation, -1, 0) - shift(elevation, 1, 0)) / (2 * cell_size)
    return torch.stack([dz_dx, dz_dy], dim=1)
