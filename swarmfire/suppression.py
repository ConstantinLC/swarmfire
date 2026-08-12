"""How water and retardant change fire behaviour.

The design decision worth defending: suppressant is not a special case in the
fire model. It acts through fuel moisture, which is an input the spread physics
already depends on. Water raises a cell's moisture; when moisture passes the
fuel's moisture of extinction, the fuel stops carrying fire. That is a real
mechanism, it is the one Rothermel already encodes, and it means the toy model
and the eventual realistic model respond to a drop through the same pathway.

The alternative - flipping cells to "non-burnable" - is a dead end. It has no
notion of dose, no evaporation, no partial effect, and nothing to calibrate
against reality.

Water and retardant differ in exactly the way they do operationally:

* water works immediately and hard, but the fire boils it off in minutes, so
  it must land on or just ahead of the front to matter.
* retardant is weaker per unit mass but persists for hours and is not consumed
  by the fire, so it is laid down as a line ahead of the fire.

That asymmetry is what makes the control problem interesting, and it is why
both layers exist from the start.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .state import FireState


@dataclass
class SuppressionModel:
    """Suppressant dynamics and their coupling into flammability.

    Parameters
    ----------
    water_to_moisture : moisture fraction added per mm of water depth. ~0.8
        puts a "level 3" aerial coverage (roughly 1.2 mm) far above any fuel's
        moisture of extinction, which is the intended behaviour: coverage
        level, not depth, is the operational constraint.
    water_tau_ambient : e-folding time of evaporation with no fire, seconds.
    water_tau_fire : e-folding time directly in the flame front. Minutes. This
        is why a drop placed too early is wasted.
    retardant_tau : e-folding time of retardant effectiveness, seconds. Hours.
    retardant_efficacy : flammability multiplier at full coverage. Below 1.0
        rather than 0, because retardant lines do get crossed.
    softness : width of the extinction transition, in moisture units. Keeps
        the model differentiable near the threshold; set it small but never
        zero, or gradients through the suppression decision vanish.
    """

    water_to_moisture: float = 0.8
    water_tau_ambient: float = 3600.0
    water_tau_fire: float = 60.0
    retardant_tau: float = 12 * 3600.0
    retardant_efficacy: float = 0.95
    softness: float = 0.02

    # ------------------------------------------------------------ coupling

    def effective_moisture(self, state: FireState) -> Tensor:
        """Fuel moisture including any free water sitting on the fuel bed."""
        return state.moisture + self.water_to_moisture * state.water

    def flammability(self, state: FireState) -> Tensor:
        """`[B, H, W]` in [0, 1]: how readily this cell carries fire now.

        A smooth gate around the moisture of extinction, further reduced by
        retardant coverage. This single number is the only channel through
        which suppression touches the spread physics.
        """
        margin = (state.moisture_ext - self.effective_moisture(state)) / self.softness
        wet_gate = torch.sigmoid(margin)
        retardant_gate = 1.0 - self.retardant_efficacy * state.retardant.clamp(0, 1)
        return wet_gate * retardant_gate

    # ------------------------------------------------------------- dynamics

    def update(self, state: FireState, dt: float) -> FireState:
        """Evaporate water and age retardant by one timestep."""
        evap_rate = 1.0 / self.water_tau_ambient + state.intensity / self.water_tau_fire
        water = state.water * torch.exp(-dt * evap_rate)
        retardant = state.retardant * torch.exp(
            torch.tensor(-dt / self.retardant_tau, device=state.device)
        )
        return state.replace(water=water, retardant=retardant)

    # ------------------------------------------------------------- deposits

    @staticmethod
    def deposit(state: FireState, water: Tensor | None = None, retardant: Tensor | None = None) -> FireState:
        """Add suppressant fields (`[B, H, W]`) produced by platforms.

        Kept separate from `update` so that a policy's action enters the state
        at one identifiable point.
        """
        changes = {}
        if water is not None:
            changes["water"] = state.water + water
        if retardant is not None:
            changes["retardant"] = (state.retardant + retardant).clamp(max=1.0)
        return state.replace(**changes) if changes else state
