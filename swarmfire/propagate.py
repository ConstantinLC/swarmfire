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

Two propagators
---------------
`LevelSetPropagator` is the default. It moves the front along its own normal at
the rate the fire ellipse gives in that direction, which is what `pyretechnics`
does, and it keeps every property above: `ignition` is its level-set function
and every operation in it is piecewise smooth.

`CAPropagator` is the original cellular automaton. It can only spread along the
eight neighbour bearings, and an elongated ellipse whose head falls between two
of them is under-sampled: at 5.4 m/s midflame wind on grass with the head 22
degrees off the grid axis, the front ran at 0.285 m/s against Rothermel's
0.654. It is kept because it is cheaper and because the validation history in
`VALIDATION.md` was measured on it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from torch import Tensor

from .grid import NEIGHBOR_OFFSETS, neighbor_distances, shift, slope_vector
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

    @abstractmethod
    def max_stable_dt(self, state: FireState, courant: float = 0.4) -> float:
        """Largest timestep that keeps the front under `courant` cells per step."""


class _Combustion:
    """What happens behind the front, shared by both propagators.

    A cell burns once `ignition` has crossed 1, for as long as it has fuel, is
    dry enough, and has heat nearby. How the front gets there is the
    propagator's business; this is only the consumption that follows, and the
    leftover heat it leaves. Subclasses set `suppression`, `arrival_softness`,
    `fuel_floor`, `fuel_softness`, `front_memory`, `holdover` and
    `liveness_softness`.
    """

    def liveness(self, state: FireState, crossing_s: Tensor) -> Tensor:
        """`[B, H, W]` in [0, 1]: is there still a fire here to push the front
        or relight the fuel?

        Neither `ignition` nor fuel knows whether anything is still burning.
        `ignition` is where the front has got to, and it does not un-happen; so
        a front held at a wet line used to sit there with the fire behind it
        long since burnt out, and walk on as soon as the water evaporated. Fuel
        doused inside the line relit the same way. Both restarted a fire that
        was out, with no heat source anywhere on the grid.

        This gate asks how long it has been since there was heat within
        `heat_reach` cells, reading it back from `heat`, which decays as `exp(-t / smoulder)`:
        a quiet time of `-smoulder * log(heat)`. The fire is alive while that
        is shorter than a window, and the window has two parts:

        * `holdover * smoulder`: leftover heat can relight fuel for a few
          smoulder times - minutes in grass, hours in timber. This is the
          rekindle mechanism, and water shortens it by cooling `heat`.
        * `front_memory` times the time this cell's front takes to cross it
          at its unsuppressed rate, while the front is within about a cell.
          Flames in grass last ~13 s but a backing front takes ~25 minutes to
          cross a 30 m cell, so an unsuppressed front routinely has nothing
          burning right behind it. Without this term the gate would put out
          slow backing and flanking fire in a free burn. A front stalled by
          suppression has the same window as the free front would have had,
          so it dies a few crossing times after the fire behind it goes out.

        Both are ratios, so the gate is scale-free: `liveness_softness` is the
        relative width of the transition.
        """
        r = self.heat_reach
        heat_near = F.max_pool2d(state.heat.unsqueeze(1), 2 * r + 1, stride=1, padding=r).squeeze(1)
        tau = state.smoulder.clamp(min=1.0)
        quiet_s = -tau * torch.log(heat_near.clamp(min=1e-30))
        near_front = torch.sigmoid((2.0 - state.ignition) / 0.2)
        window_s = self.holdover * tau + near_front * self.front_memory * crossing_s
        return torch.sigmoid((1.0 - quiet_s / window_s) / self.liveness_softness)

    def combustion(self, state: FireState, flammability: Tensor) -> Tensor:
        """`[B, H, W]` intensity: alight, still fuelled, and dry enough.

        Note the flammability factor. It means water landing on an already
        burning cell puts it out, rather than only protecting unburnt fuel.
        The propagators pass flammability already gated by `liveness`, so
        the cell relights once the water evaporates only if heat remains.
        """
        arrived = soft_gate(state.ignition, 1.0, self.arrival_softness, floor_at=0.0)
        fuelled = soft_gate(state.fuel, self.fuel_floor, self.fuel_softness, floor_at=0.0)
        return arrived * fuelled * flammability

    def burn(self, state: FireState, flammability: Tensor, dt: float) -> FireState:
        """Combust, consume fuel, leave heat, advance the clock, and age the suppressant."""
        intensity = self.combustion(state, flammability)

        # Combustion consumes fuel; what is consumed is what has burned.
        consumed = torch.minimum(dt * intensity * state.burn_rate, state.fuel)

        # Leftover heat: decays with the fuel's smoulder time, is cooled by
        # water, and is topped back up by whatever is burning now. The
        # probabilistic OR keeps it in [0, 1] and at least `intensity`.
        decay = torch.exp(
            -dt / state.smoulder.clamp(min=1.0) - dt * self.suppression.cooling_rate(state)
        )
        heat = 1.0 - (1.0 - state.heat * decay) * (1.0 - intensity)

        state = state.replace(
            intensity=intensity,
            fuel=state.fuel - consumed,
            burned=state.burned + consumed,
            heat=heat,
            t=state.t + dt,
        )
        return self.suppression.update(state, dt)


class CAPropagator(_Combustion, Propagator):
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
    front_memory : how many crossing times a front outlives the last heat
        behind it. See `_Combustion.liveness`.
    holdover : how many smoulder times leftover heat can relight fuel.
    liveness_softness : relative width of the liveness transition.
    heat_reach : how far, in cells, heat can reach to keep a front alive. The
        level set lowers `phi` a cell or two ahead of the burning cell, and
        freezing those cells slows a free head by several percent.
    """

    def __init__(
        self,
        ros_model: ROSModel,
        suppression: SuppressionModel | None = None,
        pnorm: float = 6.0,
        arrival_softness: float = 0.08,
        fuel_floor: float = 0.03,
        fuel_softness: float = 0.03,
        front_memory: float = 3.0,
        holdover: float = 3.0,
        liveness_softness: float = 0.1,
        heat_reach: int = 2,
    ):
        self.ros_model = ros_model
        self.suppression = suppression or SuppressionModel()
        self.pnorm = pnorm
        self.arrival_softness = arrival_softness
        self.fuel_floor = fuel_floor
        self.fuel_softness = fuel_softness
        self.front_memory = front_memory
        self.holdover = holdover
        self.liveness_softness = liveness_softness
        self.heat_reach = heat_reach

    def front_source(self, state: FireState, live: Tensor | float = 1.0) -> Tensor:
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

        A bare latch means a burnt-out cell keeps radiating, so a fire could
        cross a barrier whose suppressant had evaporated with nothing left
        burning against it. `live` - `_Combustion.liveness` - closes that: the
        latch only holds while there has been heat nearby recently enough.
        """
        return (
            soft_gate(state.ignition, 1.0, self.arrival_softness)
            * self.suppression.flammability(state)
            * live
        )

    def ignition_drive(self, state: FireState, ros: Tensor, live: Tensor | float = 1.0) -> Tensor:
        """`[B, H, W]` rate at which the front is closing on each cell, 1/s.

        A neighbour that has ignited, spreading toward this cell at `R` m/s
        across a gap of `L` m, contributes `R / L` - the reciprocal of the
        travel time, which is the whole point.
        """
        dist = neighbor_distances(state.cell_size, device=state.device)
        source = self.front_source(state, live)

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

    def step(self, state: FireState, dt: float) -> FireState:
        flammability = self.suppression.flammability(state)
        ros = self.ros_model(state)

        # This propagator has no front normal, so the crossing time is taken at
        # the slowest bearing - the backing rate - which errs toward keeping a
        # front alive.
        crossing_s = state.cell_size / ros.amin(dim=1).clamp(min=1e-9)
        live = self.liveness(state, crossing_s)

        # Wet or retardant-covered fuel resists the approaching front, and a
        # front with no fire behind it does not move at all.
        drive = self.ignition_drive(state, ros, live) * flammability * live
        state = state.replace(ignition=state.ignition + dt * drive)
        return self.burn(state, flammability * live, dt)

    def max_stable_dt(self, state: FireState, courant: float = 0.4) -> float:
        """Convenience passthrough - see `ROSModel.max_stable_dt`."""
        return self.ros_model.max_stable_dt(state, courant=courant)


# ------------------------------------------------------------------ level set


def _neighbour(phi: Tensor, dy: int, dx: int) -> Tensor:
    """`phi` read at `(i + dy, j + dx)`, with off-grid cells unburnt (phi = 1).

    Padding with 1 rather than 0 keeps the domain edge from looking like a
    burnt region the front could be drawn toward.
    """
    h, w = phi.shape[-2:]
    r = max(abs(dy), abs(dx))
    padded = F.pad(phi, (r, r, r, r), value=1.0)
    return padded[..., r + dy : r + dy + h, r + dx : r + dx + w]


def _half_superbee(up: Tensor, loc: Tensor) -> Tensor:
    """Superbee flux limiter times `0.5 * loc`, as in `pyretechnics`."""
    sign = torch.where(loc >= 0.0, torch.ones_like(loc), -torch.ones_like(loc))
    a_up, a_loc = up.abs(), loc.abs()
    limited = sign * torch.maximum(torch.minimum(a_up / 2.0, a_loc), torch.minimum(a_up, a_loc / 2.0))
    return torch.where(sign * up <= 0.0, torch.zeros_like(limited), limited)


def _flux_limited_difference(p00: Tensor, m2: Tensor, m1: Tensor, p1: Tensor, p2: Tensor) -> Tensor:
    """Flux-limited `phi(+1/2) - phi(-1/2)` along one axis.

    `m2, m1, p1, p2` are `phi` two and one cells back, one and two cells ahead.
    A line-for-line vectorisation of `pyretechnics`' `calc_dphi_flim_x`: the
    upwind side is whichever neighbour is higher, since phi falls toward the
    fire.
    """
    ahead_is_up = p1 >= m1
    loc = p1 - p00
    face_ahead = torch.where(
        ahead_is_up, p00 + _half_superbee(p00 - m1, loc), p1 - _half_superbee(p2 - p1, loc)
    )
    loc = m1 - p00
    face_behind = torch.where(
        ahead_is_up, m1 - _half_superbee(m2 - m1, loc), p00 + _half_superbee(p00 - p1, loc)
    )
    return face_ahead - face_behind


class LevelSetPropagator(_Combustion, Propagator):
    """Eulerian level-set front tracking, after `pyretechnics`.

    The front is the zero contour of `phi = 1 - state.ignition`: 1 on unburnt
    ground, falling through 0 as the front arrives, negative behind it. So a
    cell has been reached when `ignition` crosses 1, exactly as in
    `CAPropagator`, and nothing that reads `ignition` needs to know which
    propagator produced it.

    Each step solves `d phi / dt = -U(n) |grad phi|`, where `U(n)` is the rate
    of spread of the cell's fire ellipse along the front normal `n`. That rate
    comes from Richards' elliptical wavelet in closed form, so the front runs
    at the ellipse's rate in every direction rather than only along the eight
    neighbour bearings. Numerics are `pyretechnics`' own: central differences
    for the normal, a superbee flux limiter for the upwind gradient, and a
    two-stage (Heun) Runge-Kutta step. The ellipse is evaluated once per step
    and held across both stages, as `pyretechnics` holds its inputs.

    `pyretechnics` updates only a narrow band around the front; this updates
    the whole grid, which costs arithmetic on a GPU but no time, and gives the
    same answer - where `phi` is flat its gradient is zero and nothing moves.

    Suppression enters through the spread rate. A cell's flammability scales
    its ellipse, so a soaked cell ahead of the front stops it, and a burnt-out
    cell behind the front has no say in where it goes next.

    The level set on its own has no notion of whether the fire behind the
    front is still burning, so the ellipse is also scaled by
    `_Combustion.liveness`: a front held at a wet line for long enough after
    the fire behind it has gone out stops for good, rather than resuming when
    the water evaporates.

    Parameters
    ----------
    ros_model : supplies the per-cell `SpreadEllipse`.
    suppression : supplies the flammability gate.
    arrival_softness, fuel_floor, fuel_softness : as for `CAPropagator`.
    front_memory, holdover, liveness_softness, heat_reach : as for `CAPropagator`.
    """

    def __init__(
        self,
        ros_model: ROSModel,
        suppression: SuppressionModel | None = None,
        arrival_softness: float = 0.08,
        fuel_floor: float = 0.03,
        fuel_softness: float = 0.03,
        front_memory: float = 3.0,
        holdover: float = 3.0,
        liveness_softness: float = 0.1,
        heat_reach: int = 2,
    ):
        self.ros_model = ros_model
        self.suppression = suppression or SuppressionModel()
        self.arrival_softness = arrival_softness
        self.fuel_floor = fuel_floor
        self.fuel_softness = fuel_softness
        self.front_memory = front_memory
        self.holdover = holdover
        self.liveness_softness = liveness_softness
        self.heat_reach = heat_reach

    def _wavelet(self, state: FireState, flammability: Tensor) -> dict[str, Tensor]:
        """Per-cell coefficients of the elliptical wavelet, fixed for the step.

        With head rate `H`, eccentricity `e` and length/breadth `L`, the
        backing rate is `H (1 - e) / (1 + e)` and the flanking rate is
        `(head + back) / 2L`. Substituting those into `pyretechnics`'
        `prepare_partialed_wavelet` leaves coefficients that depend on the
        shape alone, with `H` factored out.
        """
        ellipse = self.ros_model.ellipse(state)
        lb = ellipse.length_to_breadth.clamp(min=1.0)
        ecc = torch.sqrt((1.0 - 1.0 / (lb * lb)).clamp(min=0.0))
        grad = slope_vector(state.elevation, state.cell_size)
        return {
            "head": ellipse.head_ros * flammability,
            "head_dry": ellipse.head_ros,
            "hx": ellipse.heading[:, 0],
            "hy": ellipse.heading[:, 1],
            "a": -ecc / (1.0 + ecc),
            "b": -1.0 / ((1.0 + ecc) * lb),
            "c": lb * lb - 1.0,
            "gx": grad[:, 0],
            "gy": grad[:, 1],
            "g2": 1.0 + grad[:, 0] ** 2 + grad[:, 1] ** 2,
        }

    @staticmethod
    def _wavelet_shape(phi: Tensor, wave: dict[str, Tensor], cell: float):
        """Central gradient of `phi` and the wavelet's `d phi / dt` per unit head rate.

        Returns `(px, py, norm2, mag2, shape)`, where `head * shape` is the
        uncorrected `d phi / dt`.
        """
        w, e = _neighbour(phi, 0, -1), _neighbour(phi, 0, 1)
        s, n = _neighbour(phi, -1, 0), _neighbour(phi, 1, 0)
        # Central gradient: the front normal.
        px = (e - w) / (2.0 * cell)
        py = (n - s) / (2.0 * cell)
        norm2 = px * px + py * py

        # Squared gradient magnitude on the slope-tangential plane.
        along_slope = px * wave["gx"] + py * wave["gy"]
        mag2 = (norm2 - along_slope * along_slope / wave["g2"]).clamp(min=0.0)

        # Richards' wavelet, `pyretechnics`' `dphi_dt_from_partialed_wavelet`
        # with the head rate factored out. The root is masked where the
        # gradient vanishes: sqrt has an infinite derivative at zero, and zero
        # is what `phi` is almost everywhere.
        delta = wave["hx"] * px + wave["hy"] * py
        radicand = mag2 + wave["c"] * delta * delta
        root = torch.where(
            radicand > 0.0, torch.sqrt(radicand.clamp(min=1e-30)), torch.zeros_like(radicand)
        )
        return px, py, norm2, mag2, wave["a"] * delta + wave["b"] * root

    def _crossing_time(self, phi: Tensor, wave: dict[str, Tensor], cell: float) -> Tensor:
        """`[B, H, W]` seconds the front takes to cross a cell along its own
        normal, at the unsuppressed rate.

        Zero where no front is advancing, not infinite: a cell with no front
        moving through it has no front to keep alive, and an infinite window
        would keep it alive forever - which is how a doused fire used to
        relight an hour after the last heat on the grid had gone.
        """
        _, _, _, mag2, shape = self._wavelet_shape(phi, wave, cell)
        safe = torch.where(mag2 > 0.0, mag2, torch.ones_like(mag2))
        speed = torch.where(
            mag2 > 0.0, -wave["head_dry"] * shape / torch.sqrt(safe), torch.zeros_like(mag2)
        )
        advancing = speed > 1e-6
        return torch.where(
            advancing, cell / torch.where(advancing, speed, torch.ones_like(speed)), torch.zeros_like(speed)
        )

    def _dphi_dt(self, phi: Tensor, wave: dict[str, Tensor], cell: float) -> Tensor:
        """Flux-limited `d phi / dt`, `[B, H, W]`, <= 0 where the front moves."""
        w, e = _neighbour(phi, 0, -1), _neighbour(phi, 0, 1)
        s, n = _neighbour(phi, -1, 0), _neighbour(phi, 1, 0)
        px, py, norm2, _, shape = self._wavelet_shape(phi, wave, cell)
        moving = norm2 > 0.0
        dphi_dt = wave["head"] * shape

        # Rescale by how much of the central gradient the upwind, flux-limited
        # gradient agrees with - `pyretechnics`' `dphi_dt_correction`.
        fx = _flux_limited_difference(phi, _neighbour(phi, 0, -2), w, e, _neighbour(phi, 0, 2)) / cell
        fy = _flux_limited_difference(phi, _neighbour(phi, -2, 0), s, n, _neighbour(phi, 2, 0)) / cell
        safe_norm2 = torch.where(moving, norm2, torch.ones_like(norm2))
        correction = torch.where(moving, (px * fx + py * fy) / safe_norm2, torch.zeros_like(norm2))
        return dphi_dt * correction

    def step(self, state: FireState, dt: float) -> FireState:
        flammability = self.suppression.flammability(state)
        wave = self._wavelet(state, flammability)

        phi = 1.0 - state.ignition
        # A front with no fire behind it does not move. Held, like the rest of
        # the wavelet, across both stages.
        live = self.liveness(state, self._crossing_time(phi, wave, state.cell_size))
        wave["head"] = wave["head"] * live

        k1 = self._dphi_dt(phi, wave, state.cell_size)
        k2 = self._dphi_dt(phi + dt * k1, wave, state.cell_size)
        phi = phi + 0.5 * dt * (k1 + k2)

        state = state.replace(ignition=1.0 - phi)
        return self.burn(state, flammability * live, dt)

    def max_stable_dt(self, state: FireState, courant: float = 0.4) -> float:
        """Largest timestep that keeps the front under `courant` cells per step.

        The front-normal speed never exceeds the head rate, so bounding that
        bounds every direction. 0.4 is `pyretechnics`' `max_cells_per_timestep`.
        """
        peak = self.ros_model.ellipse(state).head_ros.amax()
        return float(courant * state.cell_size / peak.clamp(min=1e-9))
