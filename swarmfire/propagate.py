"""The propagation step: a stencil plus elementwise algebra, and nothing else.

Why spread is paced by an ignition-progress variable
----------------------------------------------------
The tempting formulation - integrate `intensity` directly, with a term that
lets combustion feed itself - is wrong, and wrong in a way that is easy to miss
because the fires it produces still look like fires. Self-feeding combustion is
autocatalytic: any cell that picks up a trace of intensity bootstraps itself to
fully alight on a timescale set by the feedback gain, no matter how weak the
signal that reached it. Spread rate then has almost nothing to do with the rate
of spread you computed, and the anisotropy of the ellipse washes out. Measured
on a 6 m/s wind, that formulation ran the head four times too fast and the
flanks seventy times too fast.

So instead each cell carries `ignition`, a progress variable that accumulates
at exactly the rate the front is closing on it: a neighbour at distance `L`
burning toward this cell at `R` metres per second contributes `R/L` per second,
so `ignition` reaches 1 after precisely `L/R` seconds - the physical travel
time. Combustion begins when it crosses 1. Front speed is then the rate of
spread by construction, with no free gain parameter to tune.

Contributions from the eight neighbours are combined with a smooth p-norm
rather than a sum. Summing lets a cell with three lit neighbours ignite three
times too fast, which shows up as corners of the fire running ahead of the
flanks; the p-norm approximates "the front arrives along the fastest path"
while staying differentiable everywhere.

Why the fire state is continuous
--------------------------------
`intensity` is a real number in [0, 1], not a burning/not-burning flag, and
every gate in this file is a smooth function. That is a deliberate constraint,
and it is expensive to retrofit, which is why it is here from the first commit.

It buys two things:

* Gradients survive the simulator. You can backpropagate a loss like "hectares
  burned" through hundreds of fire steps into the parameters of a network that
  decided where to drop water. That is a far more direct - and far more
  sample-efficient - training signal than RL's reward.
* Fire shape stops depending on grid alignment. Hard-threshold cellular
  automata produce visibly octagonal fires; smooth ones do not.

If you later replace this with a level-set or minimum-arrival-time scheme,
keep the same property.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor

from .grid import NEIGHBOR_OFFSETS, N_DIRS, neighbor_distances, shift
from .ros.base import ROSModel
from .state import FireState
from .suppression import SuppressionModel


class Propagator(ABC):
    """Advances the fire one timestep. Swappable, like the ROS model."""

    @abstractmethod
    def step(self, state: FireState, dt: float) -> FireState: ...


class CAPropagator(Propagator):
    """Smooth cellular automaton driven by directional rate of spread.

    Parameters
    ----------
    ros_model : supplies `[B, 8, H, W]` spread rates.
    suppression : supplies the flammability gate.
    pnorm : how sharply neighbour contributions combine. 1 is a plain sum
        (over-fast at corners), large values approach a max. 6 is a good
        compromise between fidelity and gradient flow.
    arrival_softness : width of the ignition threshold, in progress units.
    fuel_floor, fuel_softness : where combustion fades out as fuel runs down.
        Together with `burn_rate` these set the residence time.
    """

    def __init__(
        self,
        ros_model: ROSModel,
        suppression: SuppressionModel | None = None,
        pnorm: float = 6.0,
        arrival_softness: float = 0.08,
        fuel_floor: float = 0.03,
        fuel_softness: float = 0.03,
    ):
        self.ros_model = ros_model
        self.suppression = suppression or SuppressionModel()
        self.pnorm = pnorm
        self.arrival_softness = arrival_softness
        self.fuel_floor = fuel_floor
        self.fuel_softness = fuel_softness

    def ignition_drive(self, state: FireState, ros: Tensor) -> Tensor:
        """`[B, H, W]` rate at which the front is closing on each cell, 1/s.

        A neighbour burning at intensity `I`, spreading toward this cell at `R`
        m/s across a gap of `L` m, contributes `I * R / L`. At `I = 1` that is
        the reciprocal of the travel time, which is the whole point.
        """
        dist = neighbor_distances(state.cell_size, device=state.device)
        source = state.intensity

        contributions = torch.stack(
            [
                shift(source * ros[:, d], dy, dx) / dist[d]
                for d, (dy, dx) in enumerate(NEIGHBOR_OFFSETS)
            ],
            dim=1,
        )
        # Smooth approximation to a max over directions.
        return contributions.clamp(min=0).pow(self.pnorm).sum(dim=1).pow(1.0 / self.pnorm)

    def combustion(self, state: FireState, flammability: Tensor) -> Tensor:
        """`[B, H, W]` intensity: alight, still fuelled, and dry enough.

        Note the flammability factor. It means water landing on an already
        burning cell puts it out, rather than only protecting unburnt fuel -
        and that the cell relights once the water evaporates, if fuel remains.
        """
        arrived = torch.sigmoid((state.ignition - 1.0) / self.arrival_softness)
        fuelled = torch.sigmoid((state.fuel - self.fuel_floor) / self.fuel_softness)
        return arrived * fuelled * flammability

    def step(self, state: FireState, dt: float) -> FireState:
        flammability = self.suppression.flammability(state)
        ros = self.ros_model(state)

        # Wet or retardant-covered fuel resists the approaching front.
        drive = self.ignition_drive(state, ros) * flammability
        ignition = state.ignition + dt * drive

        state = state.replace(ignition=ignition)
        intensity = self.combustion(state, flammability)

        # Combustion consumes fuel; what is consumed is what has burned.
        consumed = torch.minimum(dt * intensity * state.burn_rate, state.fuel)
        state = state.replace(
            intensity=intensity,
            fuel=state.fuel - consumed,
            burned=state.burned + consumed,
            t=state.t + dt,
        )

        return self.suppression.update(state, dt)

    def max_stable_dt(self, state: FireState, courant: float = 0.4) -> float:
        """Convenience passthrough - see `ROSModel.max_stable_dt`."""
        return self.ros_model.max_stable_dt(state, courant=courant)
