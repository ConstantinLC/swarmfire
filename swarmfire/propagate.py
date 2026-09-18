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

from .grid import NEIGHBOR_OFFSETS, neighbor_distances, shift
from .ros.base import ROSModel
from .state import FireState
from .suppression import SuppressionModel


def soft_gate(x: Tensor, threshold: float, softness: float, floor_at: float = 0.0) -> Tensor:
    """A sigmoid that is *exactly* zero at `floor_at` and still differentiable there.

    A plain `sigmoid((x - threshold) / softness)` never reaches zero. With the
    defaults in `CAPropagator` that leaves every cell in the domain - lit or
    not, fuelled or not - burning at a small constant rate:

        sigmoid((0 - 1) / 0.08)    = 3.7e-6   for a cell the front never reached
        sigmoid((0 - 0.03) / 0.03) = 0.269    for a cell whose fuel is gone

    Neither is a rounding error. The first makes every unburnt cell accumulate
    `burned` forever, so `burned_area()` carries a bias that grows with grid
    size and elapsed time. The second is worse: a cell that has burnt out keeps
    reporting a quarter of full intensity, so it keeps driving its neighbours
    and keeps counting toward `active()`. Between them `active()` on a 192x192
    grid never fell below 0.13 against a `done` threshold of 1e-3, which meant
    `info["extinguished"]` could not become true on any grid larger than about
    32x32 and no rollout ever stopped early.

    Subtracting the value at `floor_at` and renormalising fixes both while
    keeping every property the smooth gate was chosen for: monotone,
    infinitely differentiable, and unchanged to within a part in 1e5 wherever
    the gate is actually open. The gradient at `floor_at` is small but nonzero,
    so a cell sitting at zero progress still has a defined - if weak - path
    back to whatever might light it.
    """
    gate = torch.sigmoid((x - threshold) / softness)
    base = torch.sigmoid(torch.tensor((floor_at - threshold) / softness, device=x.device))
    return ((gate - base) / (1.0 - base)).clamp(min=0.0)


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

    def front_source(self, state: FireState) -> Tensor:
        """`[B, H, W]` how strongly each cell can ignite its neighbours, in [0, 1].

        Deliberately *not* `state.intensity`. Intensity is the rate fuel is
        being consumed, which collapses on the flame residence timescale - and
        that timescale has nothing to do with the grid. The front needs
        `cell_size / R` seconds to cross a cell: 100 s for grass on a 30 m cell
        against a flame residence of 13 s. Sourcing the front from intensity
        therefore let the source go dark long before the neighbour it was
        pushing had arrived, and the front slowed down, stopped, or - once the
        compensating floors were removed from `combustion` - never moved at
        all. Front speed came out dependent on `burn_rate` and on `cell_size`,
        neither of which it may depend on.

        What propagates a front is that the cell has *ignited*, which `ignition`
        already records and which does not un-happen. Gated by flammability, so
        that a cell the fleet has doused stops pushing its neighbours.

        A latch means a burnt-out cell keeps radiating, so a fire can eventually
        cross a barrier whose suppressant has evaporated even with nothing left
        burning against it. That is the wrong reason for a barrier to fail and a
        minimum-arrival-time propagator would remove it - but it is not new and
        not a cost of this change: the previous formulation had the same
        artefact via a fuel gate that bottomed out at 0.269 instead of zero. On
        a 2 mm water line across a 5 m/s grass fire, the barrier failed at
        12720 s before and 11696 s after.
        """
        return soft_gate(state.ignition, 1.0, self.arrival_softness) * self.suppression.flammability(
            state
        )

    def ignition_drive(self, state: FireState, ros: Tensor) -> Tensor:
        """`[B, H, W]` rate at which the front is closing on each cell, 1/s.

        A neighbour that has ignited, spreading toward this cell at `R` m/s
        across a gap of `L` m, contributes `R / L` - the reciprocal of the
        travel time, which is the whole point.
        """
        dist = neighbor_distances(state.cell_size, device=state.device)
        source = self.front_source(state)

        contributions = torch.stack(
            [
                shift(source * ros[:, d], dy, dx) / dist[d]
                for d, (dy, dx) in enumerate(NEIGHBOR_OFFSETS)
            ],
            dim=1,
        )
        # Smooth approximation to a max over directions.
        #
        # Spelled as `vector_norm` rather than as `.pow(p).sum().pow(1/p)`: the
        # two agree to float precision in the forward pass, but the hand-rolled
        # version differentiates to `(1/p) * s**(1/p - 1)`, which is infinite at
        # `s = 0` - and `s` is exactly zero at every cell with no lit neighbour,
        # which is most of the grid. That put a NaN into the backward pass of
        # every rollout. `vector_norm` defines the subgradient at the origin as
        # zero, which is both finite and right: a cell nothing is burning
        # toward has no sensitivity to anything.
        return torch.linalg.vector_norm(contributions.clamp(min=0), ord=self.pnorm, dim=1)

    def combustion(self, state: FireState, flammability: Tensor) -> Tensor:
        """`[B, H, W]` intensity: alight, still fuelled, and dry enough.

        Note the flammability factor. It means water landing on an already
        burning cell puts it out, rather than only protecting unburnt fuel -
        and that the cell relights once the water evaporates, if fuel remains.
        """
        arrived = soft_gate(state.ignition, 1.0, self.arrival_softness, floor_at=0.0)
        fuelled = soft_gate(state.fuel, self.fuel_floor, self.fuel_softness, floor_at=0.0)
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
