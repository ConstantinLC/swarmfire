"""The rate-of-spread seam.

This is the interface that lets the project start with toy dynamics and end
with real ones. A ROS model turns the layer stack into a `[B, 8, H, W]` tensor:
for every cell, how fast fire leaves it toward each of its eight neighbours, in
metres per second. Nothing downstream knows or cares how that number was
produced.

Swapping `SimpleROS` for `RothermelROS` is the whole "get realistic" step.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor

from ..grid import N_DIRS, neighbor_distances
from ..state import FireState


class ROSModel(ABC):
    """Directional rate of spread on the grid."""

    @abstractmethod
    def __call__(self, state: FireState) -> Tensor:
        """Return `[B, 8, H, W]` spread rates in m/s, direction order per
        `grid.NEIGHBOR_OFFSETS`."""

    def max_stable_dt(self, state: FireState, courant: float = 0.4) -> float:
        """Largest timestep that keeps the front from jumping a cell per step.

        The explicit propagation step is only conditionally stable: a front
        must not cross a cell in a single timestep. Call this when you change
        wind, fuel, or resolution.
        """
        ros = self(state)
        dist = neighbor_distances(state.cell_size, device=state.device).view(1, N_DIRS, 1, 1)
        peak = (ros / dist).amax()
        return float(courant / peak.clamp(min=1e-9))


class ConstantROS(ROSModel):
    """Isotropic constant spread. Only useful as a test fixture."""

    def __init__(self, speed: float = 0.1):
        self.speed = speed

    def __call__(self, state: FireState) -> Tensor:
        b, h, w = state.batch, *state.grid
        return torch.full((b, N_DIRS, h, w), self.speed, device=state.device)
