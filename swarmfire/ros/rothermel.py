"""The realistic ROS slot. Not implemented yet - deliberately.

This file exists to fix the interface and to record exactly what filling it in
requires, so that the switch from toy to real physics is a one-line change in
the env constructor:

    env = FireEnv(ros_model=RothermelROS(), ...)

What goes here
--------------
Rothermel (1972) surface fire spread. It is closed-form algebra - no solver, no
iteration - so the whole thing is elementwise tensor math over `[B, H, W]` and
is differentiable as written:

    R = I_R * xi * (1 + phi_w + phi_s) / (rho_b * eps * Q_ig)

with reaction intensity `I_R`, propagating flux ratio `xi`, wind and slope
factors `phi_w` / `phi_s`, bulk density `rho_b`, effective heating number
`eps`, and heat of preignition `Q_ig`. Every term is a function of the fuel bed
descriptors and moisture.

New layers this needs in `FireState` (add to `STATIC_LAYERS`)
-------------------------------------------------------------
* oven-dry fuel load per size class (1-h, 10-h, 100-h, live herb, live woody)
* surface-area-to-volume ratio per size class
* fuel bed depth
* particle density, heat content, mineral content
* separate dead and live moisture, plus live moisture of extinction

All forty Scott & Burgan fuel models are just a lookup table of those numbers,
which is why the `landfire` package plus a fuel-model raster gets you a real
landscape immediately.

Two things worth doing rather than skipping
-------------------------------------------
* Validate against `pyretechnics` (`pip install pyretechnics`) cell by cell on
  identical fuel, moisture, wind and slope inputs. It implements the same model
  and is actively maintained; agreeing with it to a few percent is a much
  stronger check than any hand-rolled unit test.
* Keep the elliptical template from `SimpleROS`. Rothermel gives you the head
  rate only; the ellipse turning that into eight directional rates is a
  separate, reusable piece.

Everything else in this package - propagation, suppression, platforms, the env
- stays exactly as it is.
"""

from __future__ import annotations

from torch import Tensor

from ..state import FireState
from .base import ROSModel


class RothermelROS(ROSModel):
    """Rothermel (1972) surface spread. See the module docstring."""

    def __call__(self, state: FireState) -> Tensor:  # pragma: no cover
        raise NotImplementedError(
            "RothermelROS is an intentionally empty slot - see the module "
            "docstring in swarmfire/ros/rothermel.py for what it needs. Use "
            "SimpleROS until the fuel-model layers exist in FireState."
        )
