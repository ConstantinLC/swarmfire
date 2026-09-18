"""Agreement with `pyretechnics`, locked in as tests.

Two kinds of assertion here, and the distinction matters:

* **Identities** - things that should hold to float precision, forever. The
  ellipse template and the matched-ROS bridge are both exact algebra shared
  with the reference; if either drifts, something has been broken.
* **Tolerances** - the current, measured, *imperfect* agreement. These are
  regression guards, not targets. A number that moves is a change in the
  physics; whether it moved the right way is for a human to say. The margins
  are deliberately loose enough that the tests do not fail on float noise and
  tight enough that a real change trips them.

The whole module skips without the reference installed:
`pip install "swarmfire[real]"`.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from swarmfire import PRESETS, CAPropagator, SimpleROS, uniform_world, validation as V
from swarmfire.ros.simple import length_to_breadth

pytestmark = pytest.mark.skipif(
    not V.PYRETECHNICS_AVAILABLE, reason='pyretechnics not installed (pip install "swarmfire[real]")'
)

GRASS = 102  # Scott & Burgan GR2, the stand-in for the "grass" preset

# The propagation tests are a few hundred CA steps on a couple of hundred cells
# squared; that is seconds on a GPU and a minute or two on a loaded CPU.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------------ identities


@pytest.mark.parametrize("wind", [0.0, 1.0, 2.0, 3.0, 5.0])
def test_ellipse_template_is_the_reference_ellipse(wind):
    """`length_to_breadth` is Anderson (1983) in m/s; pyretechnics is the same in mph."""
    ours = float(length_to_breadth(torch.tensor(wind)))
    theirs = V.rothermel_point(GRASS, wind, 0.0).length_to_breadth
    assert ours == pytest.approx(theirs, rel=1e-4)


def test_ellipse_template_diverges_only_at_the_wind_cap():
    """Above Rothermel's effective-wind limit the shapes part company by design."""
    capped = V.rothermel_coefficients(GRASS).max_effective_wind
    beyond = capped + 3.0
    assert V.rothermel_point(GRASS, beyond, 0.0).wind_limited
    assert float(length_to_breadth(torch.tensor(beyond))) > V.rothermel_point(
        GRASS, beyond, 0.0
    ).length_to_breadth


@pytest.mark.parametrize("wind", [1.0, 3.0, 5.0])
def test_matched_ros_reproduces_rothermel_exactly(wind):
    """The bridge the propagator test stands on: same rates in all eight directions.

    If this fails, `scripts/validate_pyretechnics.py front` is no longer
    measuring the propagator in isolation.
    """
    state, ros_model, _ = V.matched_world(GRASS, (16, 16), 30.0, (wind, 0.0))
    ours = V.swarmfire_directional(ros_model, state)
    theirs = V.rothermel_directional(GRASS, wind, 0.0)
    assert np.allclose(ours, theirs, rtol=2e-4)


# ------------------------------------------------- measured, imperfect agreement


def test_simple_ros_grass_tracks_rothermel_in_its_calibrated_band():
    """`SimpleROS` on grass is within 10% of Rothermel from 1 to 5 m/s midflame.

    Not by design - `ros0` is 2.6x too high and the wind factor 3x too weak, and
    over this band the two errors cancel. The band is narrow and the agreement
    outside it is poor, which the next two tests pin down.
    """
    for wind in (1.0, 2.0, 3.0, 4.0, 5.0):
        state = uniform_world(batch=1, grid=(8, 8), preset="grass", wind=(wind, 0.0))
        ours = V.swarmfire_head_ros(SimpleROS(), state)
        theirs = V.rothermel_point(GRASS, wind, 0.0).head_ros
        assert ours == pytest.approx(theirs, rel=0.10), f"at {wind} m/s"


def test_no_wind_spread_is_too_fast_on_grass():
    """The known offset: `PRESETS['grass'].ros0` is 2.5-2.7x Rothermel's GR2."""
    state = uniform_world(batch=1, grid=(8, 8), preset="grass", wind=(0.0, 0.0))
    ratio = V.swarmfire_head_ros(SimpleROS(), state) / V.rothermel_point(GRASS, 0.0, 0.0).head_ros
    assert 2.4 < ratio < 2.8


def test_slope_response_is_too_weak():
    """`slope_c = 5.5` against Rothermel's fuel-dependent 20-37; steep ground under-spreads."""
    state = uniform_world(batch=1, grid=(8, 8), preset="grass", wind=(0.0, 0.0), slope=(0.5, 0.0))
    ratio = V.swarmfire_head_ros(SimpleROS(), state) / V.rothermel_point(GRASS, 0.0, 0.5).head_ros
    assert ratio < 0.7


# ------------------------------------------------------------------ propagation


@pytest.fixture(scope="module")
def flat_grass_fronts():
    """Arrival maps from both codes on one scenario, with identical ROS fields."""
    grid, cell, wind, duration, dt = (160, 160), 20.0, 3.0, 4200.0, 5.0
    reference = V.pyretechnics_arrival_map(GRASS, grid, cell, wind, duration)
    fronts = {}
    for label, residence in (("preset", PRESETS["grass"].residence), ("no_burnout", 1e9)):
        state, ros_model, _ = V.matched_world(
            GRASS, grid, cell, (wind, 0.0), residence_s=residence, device=DEVICE
        )
        fronts[label] = V.swarmfire_arrival_map(state, CAPropagator(ros_model), duration, dt)
    return reference, fronts, cell, duration, V.rothermel_point(GRASS, wind, 0.0)


def test_reference_front_runs_at_its_own_prescribed_rate(flat_grass_fronts):
    """Sanity on the yardstick before measuring anything against it."""
    reference, _, cell, _, point = flat_grass_fronts
    axes = V.axis_spread_rates(reference, cell_size=cell)
    assert axes["head"] == pytest.approx(point.head_ros, rel=0.05)
    # The polar ellipse rate at 90 degrees off the head.
    flank = point.head_ros * (1.0 - point.eccentricity)
    assert axes["flank"] == pytest.approx(flank, rel=0.05)


def test_ca_front_speed_is_independent_of_residence_time(flat_grass_fronts):
    """Burnout time must not change the front speed. It used to, badly.

    The front is driven by `CAPropagator.front_source` - has this cell ignited -
    rather than by its combustion rate, which collapses on the flame residence
    timescale and has nothing to do with how long the front needs to cross a
    cell. Sourcing from combustion made the head run at 45% of its own rate at
    30 m resolution, and stop dead once the compensating floors came out of
    `combustion`.
    """
    _, fronts, cell, _, point = flat_grass_fronts
    preset = V.axis_spread_rates(fronts["preset"], cell_size=cell)["head"]
    unlimited = V.axis_spread_rates(fronts["no_burnout"], cell_size=cell)["head"]
    assert preset == pytest.approx(unlimited, rel=1e-3)
    assert preset == pytest.approx(point.head_ros, rel=0.10)


def test_ca_front_matches_the_reference_when_fuel_never_runs_out(flat_grass_fronts):
    """With burnout removed the head and back are right; the flanks are not."""
    reference, fronts, cell, duration, _ = flat_grass_fronts
    ours = V.axis_spread_rates(fronts["no_burnout"], cell_size=cell)
    theirs = V.axis_spread_rates(reference, cell_size=cell)

    assert ours["head"] == pytest.approx(theirs["head"], rel=0.10)
    assert ours["back"] == pytest.approx(theirs["back"], rel=0.20)
    # Known and unfixed: the p-norm over eight neighbours overstates lateral
    # spread by about a third.
    assert 1.2 < ours["flank"] / theirs["flank"] < 1.5
    assert V.iou(fronts["no_burnout"], reference, duration) > 0.85


def test_shipped_configuration_tracks_the_reference_on_grass(flat_grass_fronts):
    """End to end, as the package is actually configured.

    Grass only. `SimpleROS` happens to agree with Rothermel on grass between 1
    and 5 m/s midflame (two errors cancelling - see
    `test_simple_ros_grass_tracks_rothermel_in_its_calibrated_band`), so this is
    a test of the propagator wearing the shipped preset, not evidence that the
    fuel models are right. Shrub and timber are still out by more than an order
    of magnitude and that is a `fuels.py` problem.
    """
    reference, fronts, _, duration, _ = flat_grass_fronts
    assert V.iou(fronts["preset"], reference, duration) > 0.75


def test_front_moves_when_residence_is_shorter_than_a_cell_crossing():
    """The case that used to stop the fire outright.

    Rothermel's flame residence for grass is 12.7 s against the 68 s the front
    needs to cross a 20 m cell. With the front sourced from combustion the
    nearest unlit cell climbed to 0.94 of the arrival threshold and asymptoted
    there - it never ignited, and the fire went out. It must now spread at very
    nearly the rate it was given.
    """
    state, ros_model, coeffs = V.matched_world(GRASS, (96, 96), 20.0, (3.0, 0.0), device=DEVICE)
    assert coeffs.residence_s < 20.0, "fixture assumes a residence well under one crossing"

    arrival = V.swarmfire_arrival_map(state, CAPropagator(ros_model), 3600.0, 5.0)
    head = V.axis_spread_rates(arrival, cell_size=20.0)["head"]
    reference = V.rothermel_point(GRASS, 3.0, 0.0).head_ros
    assert head == pytest.approx(reference, rel=0.10)
