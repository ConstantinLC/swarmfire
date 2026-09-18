"""A cheap but structurally honest rate-of-spread model.

It is not Rothermel, but it has the same *shape* as Rothermel, which is what
makes it a safe thing to build the pipeline on:

1. A no-wind no-slope baseline rate, per cell.
2. Multiplicative wind and slope factors, combined as vectors into a single
   effective spread direction and magnitude.
3. An elliptical spread template around that direction, so heading, flanking,
   and backing rates are all derived from one head rate.

Step 3 is the part people skip and regret. A fire spreading at its head rate in
all eight directions produces square fires, and a controller trained on square
fires learns nothing transferable.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..grid import N_DIRS, neighbor_angles, slope_vector
from ..state import FireState
from .base import ROSModel


def length_to_breadth(wind_speed: Tensor) -> Tensor:
    """Fire ellipse length/breadth ratio from effective midflame wind (m/s).

    Alexander (1985), which is calibrated so that L/B == 1 exactly at zero
    wind - a circular fire, as it should be.
    """
    return (
        0.936 * torch.exp(0.2566 * wind_speed)
        + 0.461 * torch.exp(-0.1548 * wind_speed)
        - 0.397
    ).clamp(min=1.0)


class SimpleROS(ROSModel):
    """Wind- and slope-driven elliptical spread.

    Parameters
    ----------
    wind_a, wind_b : coefficients of the wind factor `a * U**b`. Defaults put a
        grass fire near 0.6 m/s at 5 m/s midflame wind and 1.6 m/s at 10 m/s,
        which is the right order of magnitude for a running grass fire.
    slope_c : coefficient of the slope factor `c * tan(phi)**2`, mirroring
        Rothermel's form. 5.5 is mid-range for typical packing ratios.
    max_eccentricity : caps how cigar-shaped the ellipse can get, purely for
        numerical sanity at extreme wind.
    """

    def __init__(
        self,
        wind_a: float = 2.5,
        wind_b: float = 1.5,
        slope_c: float = 5.5,
        max_eccentricity: float = 0.99,
    ):
        self.wind_a = wind_a
        self.wind_b = wind_b
        self.slope_c = slope_c
        self.max_eccentricity = max_eccentricity

    def _drive(self, state: FireState) -> tuple[Tensor, Tensor]:
        """Combined wind+slope forcing as (magnitude `[B,H,W]`, heading `[B,H,W]`).

        Wind and slope are added as vectors rather than as scalars, so a fire
        pushed north by wind and east by terrain runs northeast. This is the
        same construction FARSITE uses.
        """
        b, h, w = state.batch, *state.grid

        wind = state.wind.view(b, 2, 1, 1)
        speed = wind.norm(dim=1).clamp(min=1e-6)
        phi_w = self.wind_a * speed.pow(self.wind_b)
        wind_dir = wind / speed.unsqueeze(1)

        grad = slope_vector(state.elevation, state.cell_size)
        # Fire runs *up* slope, so the driving direction is the positive gradient.
        tan_phi = grad.norm(dim=1).clamp(min=1e-6)
        phi_s = self.slope_c * tan_phi.pow(2)
        slope_dir = grad / tan_phi.unsqueeze(1)

        drive = phi_w.unsqueeze(1) * wind_dir + phi_s.unsqueeze(1) * slope_dir
        magnitude = drive.norm(dim=1)
        heading = torch.atan2(drive[:, 1], drive[:, 0])
        return magnitude, heading

    def __call__(self, state: FireState) -> Tensor:
        magnitude, heading = self._drive(state)

        head_ros = state.ros0 * (1.0 + magnitude)

        # Effective wind speed: the wind that alone would produce this forcing.
        # Lets slope-driven runs inherit the correct fire shape.
        u_eff = (magnitude / self.wind_a).clamp(min=0).pow(1.0 / self.wind_b)
        lb = length_to_breadth(u_eff)
        # Algebraically `sqrt(lb**2 - 1) / lb`, but written so that a very large
        # `lb` cannot produce `inf / inf`. `lb` grows exponentially in the
        # effective wind, so it overflows float32 at around 350 m/s - which no
        # weather produces, but a bad elevation gradient once did, and the NaN
        # covered the whole grid in a single step. This form saturates at 1.
        ecc = torch.sqrt((1.0 - 1.0 / (lb * lb)).clamp(min=0)).clamp(max=self.max_eccentricity)

        angles = neighbor_angles(device=state.device).view(1, N_DIRS, 1, 1)
        cos_off = torch.cos(angles - heading.unsqueeze(1))

        # Polar form of an ellipse with the ignition point at a focus.
        ecc = ecc.unsqueeze(1)
        return head_ros.unsqueeze(1) * (1.0 - ecc) / (1.0 - ecc * cos_off).clamp(min=1e-6)
