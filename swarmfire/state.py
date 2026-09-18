"""The layer stack.

Everything about the world - terrain, fuel, weather, fire, suppressant - lives
in this one object as a set of `[B, H, W]` rasters sharing a grid. Adding new
physics means adding a channel here, not a new code path elsewhere.

Conventions
-----------
* `B` is the batch of independent fires. Nothing in the package ever loops over
  it; the batch dimension is what fills a GPU.
* All fields are `float32` tensors and all dynamics are differentiable with
  respect to them. There are no boolean fire states anywhere in the physics -
  see `propagate.py` for why that matters.
* Units are SI: metres, seconds, kilograms. Moisture is a dry-weight fraction
  (0.08 = 8% moisture), the convention Rothermel uses.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Iterable

import torch
from torch import Tensor

# Layers that evolve. These are what a policy watches change.
DYNAMIC_LAYERS = ("fuel", "moisture", "intensity", "burned", "ignition", "water", "retardant")

# Layers fixed for an episode: terrain and the fuel bed's intrinsic properties.
# `fuel_model` is a standard fire behaviour fuel model number - the thing a
# LANDFIRE raster contains - and is what `RothermelROS` reads the whole fuel bed
# from. `ros0`, `moisture_ext` and `burn_rate` are the summary the simpler
# models and the propagator use; `fuels.rothermel_world` fills them from
# Rothermel so the two descriptions agree.
STATIC_LAYERS = ("elevation", "fuel_model", "ros0", "moisture_ext", "burn_rate")

OBS_LAYERS = DYNAMIC_LAYERS + STATIC_LAYERS


@dataclass
class FireState:
    """A batch of fire worlds on a shared grid.

    Attributes
    ----------
    fuel : remaining fuel, as a fraction of the original load (1 -> 0).
    moisture : fuel moisture content, dry-weight fraction. The physical hook
        that suppression acts through - see `suppression.effective_moisture`.
    intensity : continuous combustion rate in [0, 1]. Deliberately *not* a
        discrete burning/not-burning flag. Derived each step from `ignition`,
        remaining fuel, and flammability - never integrated directly.
    burned : cumulative fuel consumed. This is the quantity a controller is
        ultimately trying to minimise.
    ignition : progress of the arriving fire front, in [0, ~1]. It accumulates
        at exactly the rate at which the front closes on the cell, so it
        crosses 1 after precisely the physical travel time. This is what keeps
        spread paced by rate of spread rather than by a feedback loop - see
        `propagate.py`.
    water : free water on the fuel bed, in mm of equivalent depth. Evaporates.
    retardant : coverage level in [0, 1]. Persistent; decays over hours.
    elevation : metres above datum, used for the slope term in spread.
    fuel_model : standard fire behaviour fuel model number per cell, as a float
        so it stacks with everything else. See `fuel_models.FUEL_MODELS`; 91-99
        are the non-burnable codes.
    ros0 : no-wind, no-slope rate of spread, m/s. Per cell, so heterogeneous
        fuel beds work out of the box.
    moisture_ext : moisture of extinction. Above this, the fuel will not carry
        fire - this is what makes a water drop stop a front.
    burn_rate : fuel consumed per second at unit intensity, 1/s. Sets residence
        time.
    wind : `[B, 2]` wind vector (east, north) in m/s at midflame height.
    cell_size : grid resolution in metres. 30 m matches LANDFIRE rasters.
    t : `[B]` elapsed simulated seconds.
    """

    fuel: Tensor
    moisture: Tensor
    intensity: Tensor
    burned: Tensor
    ignition: Tensor
    water: Tensor
    retardant: Tensor

    elevation: Tensor
    fuel_model: Tensor
    ros0: Tensor
    moisture_ext: Tensor
    burn_rate: Tensor

    wind: Tensor
    cell_size: float
    t: Tensor

    # ---------------------------------------------------------------- shapes

    @property
    def batch(self) -> int:
        return self.fuel.shape[0]

    @property
    def grid(self) -> tuple[int, int]:
        return tuple(self.fuel.shape[-2:])

    @property
    def device(self) -> torch.device:
        return self.fuel.device

    @property
    def cell_area(self) -> float:
        """Square metres per cell, for turning burned fraction into hectares."""
        return self.cell_size**2

    # ------------------------------------------------------------- plumbing

    def _raster_names(self) -> Iterable[str]:
        return OBS_LAYERS

    def stack(self, layers: Iterable[str] = OBS_LAYERS) -> Tensor:
        """Pack layers into a `[B, C, H, W]` tensor for a conv net."""
        return torch.stack([getattr(self, name) for name in layers], dim=1)

    def replace(self, **changes) -> "FireState":
        """Functional update. Keeps the whole step graph differentiable."""
        return replace(self, **changes)

    def to(self, device) -> "FireState":
        moved = {
            f.name: (v.to(device) if isinstance(v := getattr(self, f.name), Tensor) else v)
            for f in fields(self)
        }
        return FireState(**moved)

    def detach(self) -> "FireState":
        """Cut the autograd graph - use between truncated-BPTT windows."""
        moved = {
            f.name: (v.detach() if isinstance(v := getattr(self, f.name), Tensor) else v)
            for f in fields(self)
        }
        return FireState(**moved)

    # --------------------------------------------------------------- summary

    def burned_area(self) -> Tensor:
        """`[B]` burned area in hectares."""
        return self.burned.sum(dim=(-2, -1)) * self.cell_area / 1e4

    def active(self) -> Tensor:
        """`[B]` total combustion intensity - near zero means the fire is out."""
        return self.intensity.sum(dim=(-2, -1))


def ignite(state: FireState, y: int, x: int, radius: float = 1.5) -> FireState:
    """Light a soft circular ignition at a grid point.

    The blob is soft rather than a single hot cell so that gradients (and the
    initial fire shape) do not depend on grid alignment. It seeds `ignition`
    past the arrival threshold; `intensity` is seeded to match so that the very
    first propagation step already has something to spread from.
    """
    b, h, w = state.batch, *state.grid
    yy = torch.arange(h, device=state.device, dtype=torch.float32).view(-1, 1)
    xx = torch.arange(w, device=state.device, dtype=torch.float32).view(1, -1)
    d2 = (yy - y) ** 2 + (xx - x) ** 2
    blob = torch.exp(-d2 / (2 * radius**2)).expand(b, h, w)
    return state.replace(
        ignition=state.ignition + 1.5 * blob,
        intensity=(state.intensity + blob).clamp(max=1.0),
    )
