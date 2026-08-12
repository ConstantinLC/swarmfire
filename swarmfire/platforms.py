"""Delivery platforms: the seam between a policy's action and the grid.

Every platform, however different it looks operationally, ends up doing the
same thing to the world - adding a `[B, H, W]` mass field to the water or
retardant layer. What differs is only:

* the shape of the footprint (a tanker lays a long line, a drone wets a spot),
* how much can be delivered at once,
* how long until the next delivery,
* how fast the delivery point can move.

Keeping the deposition primitive common and varying only that mapping is what
makes "swap a helicopter for a swarm of drones" a change of one object rather
than a change to the physics. Add a new platform by adding a spec, not by
touching anything upstream.

Footprints are built from continuous coordinates with soft edges, so they are
differentiable with respect to drop position and heading. A policy can receive
a gradient telling it to move a drop fifteen metres north.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import Tensor

from .state import FireState

# 1 litre per square metre is 1 mm of depth - the unit the water layer uses.
LITRES_PER_MM_PER_M2 = 1.0

# Retardant depth, in mm, that counts as full coverage. Roughly a US "level 3"
# aerial coverage level.
RETARDANT_FULL_COVERAGE_MM = 1.0


# --------------------------------------------------------------- footprints


def soft_disc(
    centers: Tensor, radius_cells: float, grid: tuple[int, int], softness: float = 0.7
) -> Tensor:
    """`[B, N, H, W]` unit-mass discs at `centers` (`[B, N, 2]`, (y, x) cells)."""
    h, w = grid
    dev = centers.device
    yy = torch.arange(h, device=dev, dtype=torch.float32).view(1, 1, h, 1)
    xx = torch.arange(w, device=dev, dtype=torch.float32).view(1, 1, 1, w)
    cy = centers[..., 0].unsqueeze(-1).unsqueeze(-1)
    cx = centers[..., 1].unsqueeze(-1).unsqueeze(-1)

    dist = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2 + 1e-8)
    f = torch.sigmoid((radius_cells - dist) / softness)
    return f / f.sum(dim=(-2, -1), keepdim=True).clamp(min=1e-6)


def soft_capsule(
    centers: Tensor,
    heading: Tensor,
    length_cells: float,
    width_cells: float,
    grid: tuple[int, int],
    softness: float = 0.7,
) -> Tensor:
    """`[B, N, H, W]` unit-mass line drops.

    A capsule of the given length along `heading` (radians, `atan2(north,
    east)`) and the given width across it - the shape a fixed-wing tanker
    actually paints on the ground.
    """
    h, w = grid
    dev = centers.device
    yy = torch.arange(h, device=dev, dtype=torch.float32).view(1, 1, h, 1)
    xx = torch.arange(w, device=dev, dtype=torch.float32).view(1, 1, 1, w)
    cy = centers[..., 0].unsqueeze(-1).unsqueeze(-1)
    cx = centers[..., 1].unsqueeze(-1).unsqueeze(-1)
    ux = torch.cos(heading).unsqueeze(-1).unsqueeze(-1)
    uy = torch.sin(heading).unsqueeze(-1).unsqueeze(-1)

    px, py = xx - cx, yy - cy
    along = (px * ux + py * uy).clamp(-length_cells / 2, length_cells / 2)
    perp = torch.sqrt((px - along * ux) ** 2 + (py - along * uy) ** 2 + 1e-8)

    f = torch.sigmoid((width_cells / 2 - perp) / softness)
    return f / f.sum(dim=(-2, -1), keepdim=True).clamp(min=1e-6)


# -------------------------------------------------------------------- specs


@dataclass(frozen=True)
class PlatformSpec:
    """Operational envelope of one class of delivery platform.

    capacity_l : litres carried, per vehicle.
    reload_s : seconds from empty to full, including transit to the source.
        This is the dominant constraint in practice and the reason placement
        timing matters at all.
    cruise_ms : how fast the delivery point can move between drops, m/s.
    agent : "water" or "retardant".
    """

    name: str
    capacity_l: float
    reload_s: float
    cruise_ms: float
    agent: str
    n_agents: int = 1


TANKER = PlatformSpec("tanker", capacity_l=12_000, reload_s=1800, cruise_ms=130, agent="retardant")
HELICOPTER = PlatformSpec("helicopter", capacity_l=3_000, reload_s=300, cruise_ms=50, agent="water")
DRONE_SWARM = PlatformSpec(
    "drone_swarm", capacity_l=20, reload_s=120, cruise_ms=20, agent="water", n_agents=32
)


# ---------------------------------------------------------------- platforms


class Platform(ABC):
    """A fleet of `n_agents` identical vehicles, batched over `B` worlds.

    Actions are `[B, n_agents, action_dim]` in [-1, 1]. Per vehicle:
      * `[0], [1]` target position, mapped to the grid extent
      * `[2]` release, gated through a sigmoid
      * subclasses may append more (a tanker also steers its drop line)
    """

    def __init__(self, spec: PlatformSpec):
        self.spec = spec
        self.position: Tensor
        self.load: Tensor
        self.cooldown: Tensor

    # ---- lifecycle

    def reset(self, batch: int, grid: tuple[int, int], device) -> None:
        n = self.spec.n_agents
        h, w = grid
        centre = torch.tensor([h / 2, w / 2], device=device)
        self.position = centre.view(1, 1, 2).expand(batch, n, 2).clone()
        self.load = torch.full((batch, n), self.spec.capacity_l, device=device)
        self.cooldown = torch.zeros(batch, n, device=device)

    @property
    def action_dim(self) -> int:
        return 3

    def detach(self) -> None:
        """Cut the autograd graph on the fleet's own state.

        Vehicle position carries across steps, so without this it accumulates a
        graph for the whole episode even when the world is being detached.
        """
        self.position = self.position.detach()
        self.load = self.load.detach()
        self.cooldown = self.cooldown.detach()

    # ---- footprint, supplied by subclasses

    @abstractmethod
    def footprint(self, action: Tensor, state: FireState) -> Tensor:
        """`[B, N, H, W]` unit-mass deposition shapes for this step."""

    # ---- the step

    def step(self, action: Tensor, state: FireState, dt: float) -> tuple[Tensor, Tensor]:
        """Advance the fleet and return `(water, retardant)` fields `[B, H, W]`.

        Water is returned in mm of depth; retardant as a coverage fraction.
        """
        h, w = state.grid
        action = action.clamp(-1.0, 1.0)

        # Fly toward the commanded point, limited by cruise speed.
        target = torch.stack(
            [(action[..., 0] + 1) / 2 * (h - 1), (action[..., 1] + 1) / 2 * (w - 1)], dim=-1
        )
        delta = target - self.position
        max_cells = self.spec.cruise_ms * dt / state.cell_size
        travel = delta.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        self.position = self.position + delta * (max_cells / travel).clamp(max=1.0)

        # Release, gated by intent, remaining load, and reload status.
        intent = torch.sigmoid(4.0 * action[..., 2])
        ready = 1.0 - (self.cooldown / max(dt, 1e-6)).clamp(0.0, 1.0)
        volume = self.spec.capacity_l * intent * ready
        volume = torch.minimum(volume, self.load)

        self.load = self.load - volume
        # The release gate is soft, so a full-commit drop leaves a sliver in the
        # tank. Treat a nearly-empty vehicle as empty, or it never goes to reload.
        empty = self.load <= 0.02 * self.spec.capacity_l
        self.load = torch.where(empty, torch.zeros_like(self.load), self.load)
        self.cooldown = torch.where(
            empty & (self.cooldown <= 0), torch.full_like(self.cooldown, self.spec.reload_s), self.cooldown
        )
        self.cooldown = (self.cooldown - dt).clamp(min=0.0)
        refilled = empty & (self.cooldown <= 0)
        self.load = torch.where(refilled, torch.full_like(self.load, self.spec.capacity_l), self.load)

        # Turn litres into a depth field: volume * shape / cell area.
        shape = self.footprint(action, state)
        depth_mm = (volume.unsqueeze(-1).unsqueeze(-1) * shape).sum(dim=1) / state.cell_area

        zero = torch.zeros_like(depth_mm)
        if self.spec.agent == "retardant":
            return zero, (depth_mm / RETARDANT_FULL_COVERAGE_MM).clamp(max=1.0)
        return depth_mm, zero


class DiscPlatform(Platform):
    """Point delivery - helicopter buckets, drone payloads."""

    def __init__(self, spec: PlatformSpec, radius_m: float):
        super().__init__(spec)
        self.radius_m = radius_m

    def footprint(self, action: Tensor, state: FireState) -> Tensor:
        return soft_disc(self.position, self.radius_m / state.cell_size, state.grid)


class LineDropPlatform(Platform):
    """Fixed-wing delivery - a long, narrow line painted along the flight path."""

    def __init__(self, spec: PlatformSpec, length_m: float, width_m: float):
        super().__init__(spec)
        self.length_m = length_m
        self.width_m = width_m

    @property
    def action_dim(self) -> int:
        return 4  # ..., plus the bearing of the drop line

    def footprint(self, action: Tensor, state: FireState) -> Tensor:
        heading = action[..., 3] * math.pi
        return soft_capsule(
            self.position,
            heading,
            self.length_m / state.cell_size,
            self.width_m / state.cell_size,
            state.grid,
        )


def make_platform(kind: str, **overrides) -> Platform:
    """Factory for the three stock platforms.

    >>> make_platform("drone_swarm", n_agents=64)
    """
    if kind == "tanker":
        spec = _respec(TANKER, overrides)
        return LineDropPlatform(spec, length_m=overrides.get("length_m", 300.0),
                                width_m=overrides.get("width_m", 30.0))
    if kind == "helicopter":
        spec = _respec(HELICOPTER, overrides)
        return DiscPlatform(spec, radius_m=overrides.get("radius_m", 25.0))
    if kind == "drone_swarm":
        spec = _respec(DRONE_SWARM, overrides)
        return DiscPlatform(spec, radius_m=overrides.get("radius_m", 6.0))
    raise ValueError(f"unknown platform {kind!r}; expected tanker, helicopter or drone_swarm")


def _respec(spec: PlatformSpec, overrides: dict) -> PlatformSpec:
    fields = {k: v for k, v in overrides.items() if k in spec.__dataclass_fields__}
    return PlatformSpec(**{**spec.__dict__, **fields})
