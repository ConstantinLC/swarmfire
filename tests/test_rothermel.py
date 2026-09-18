"""`RothermelROS` against `pyretechnics`, cell by cell.

This is the check the ROS seam was built for. Rothermel (1972) is closed-form,
so there is no excuse for agreeing only approximately: the assertions here are
float32 tolerances, not physics tolerances, and they cover every burnable fuel
model rather than a convenient few.

Tests that need the reference skip without it; the ones that only need internal
consistency - non-burnable fuel, gradients, the moisture-of-extinction cliff -
run always, because those are the properties the rest of the package leans on.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from swarmfire import RothermelROS, rothermel_world, uniform_world
from swarmfire.fuel_models import FUEL_MODEL_NAMES, FUEL_MODELS, build_table, is_burnable
from swarmfire.grid import NEIGHBOR_OFFSETS
from swarmfire.validation import PYRETECHNICS_AVAILABLE

needs_reference = pytest.mark.skipif(
    not PYRETECHNICS_AVAILABLE, reason='pyretechnics not installed (pip install "swarmfire[real]")'
)

BURNABLE = [n for n in sorted(FUEL_MODELS) if is_burnable(n)]
EAST = NEIGHBOR_OFFSETS.index((0, 1))

# One moisture scenario held fixed, so a disagreement is the model and not the
# weather. Dead 1-hour is varied per test; these are the other four classes.
M10, M100, MHERB, MWOODY = 0.07, 0.08, 0.60, 0.90


def make_ros(**kwargs) -> RothermelROS:
    return RothermelROS(
        moisture_10h=M10, moisture_100h=M100,
        moisture_live_herb=MHERB, moisture_live_woody=MWOODY, **kwargs
    )


def world(fuel_model, moisture=0.06, wind=0.0, slope=0.0, grid=(5, 5)):
    state = uniform_world(
        batch=1, grid=grid, preset="grass", cell_size=30.0, wind=(wind, 0.0), slope=(slope, 0.0)
    )
    models = torch.as_tensor(fuel_model, dtype=torch.float32)
    return state.replace(
        fuel_model=models.expand_as(state.ros0).contiguous(),
        moisture=torch.full_like(state.moisture, moisture),
    )


def reference_directions(number, wind, slope, moisture):
    """`[8]` reference spread rates toward each Moore neighbour, m/s."""
    import pyretechnics.fuel_models as fm
    import pyretechnics.surface_fire as sf

    model = fm.moisturize(fm.get_fuel_model(number), (moisture, M10, M100, 0.0, MHERB, MWOODY))
    base = sf.calc_surface_fire_behavior_no_wind_no_slope(model, 1.0)
    # Wind from the west and terrain falling west: both drive the fire east.
    peak = sf.calc_surface_fire_behavior_max(base, wind * 60.0, 270.0, slope, 270.0, True, "behave")

    rates = []
    for dy, dx in NEIGHBOR_OFFSETS:
        norm = math.hypot(dy, dx)
        x, y = dx / norm, dy / norm
        z = slope * x  # the plane rises to the east
        length = math.sqrt(x * x + y * y + z * z)
        out = sf.calc_surface_fire_behavior_in_direction(peak, (x / length, y / length, z / length))
        rates.append(out["spread_rate"] / 60.0)
    return np.array(rates)


# ----------------------------------------------------------------- fuel models


@needs_reference
def test_fuel_model_table_matches_the_reference():
    """Every number in `fuel_models.FUEL_MODELS`, against pyretechnics' copy.

    The table is transcribed published data, so this is a transcription check -
    the kind of thing that is silently wrong for months otherwise.
    """
    import pyretechnics.fuel_models as fm

    table = build_table()
    for number in FUEL_MODELS:
        theirs = fm.get_fuel_model(number)
        slot = table.slots_for(torch.tensor([float(number)])).item()
        assert table.delta[slot] == pytest.approx(theirs["delta"], rel=1e-6), number
        assert table.w_o[slot].tolist() == pytest.approx(list(theirs["w_o"]), rel=1e-5), number
        assert table.sigma[slot].tolist() == pytest.approx(list(theirs["sigma"]), rel=1e-5), number
        assert table.h[slot].tolist() == pytest.approx(list(theirs["h"]), rel=1e-5), number
        assert table.m_x[slot].tolist() == pytest.approx(list(theirs["M_x"]), rel=1e-5), number
        assert bool(table.dynamic[slot]) == bool(theirs["dynamic"]), number
        assert bool(table.burnable[slot]) == bool(theirs["burnable"]), number


def test_unknown_fuel_model_numbers_are_non_burnable():
    """A code the table does not know must behave as bare ground, not raise.

    Real rasters carry codes this table has never heard of, and a rollout that
    dies partway through is worse than one that treats a stray cell as rock.
    """
    table = build_table()
    for number in (7777.0, 50.0, -3.0, 0.0):
        slot = table.slots_for(torch.tensor([number])).item()
        assert not bool(table.burnable[slot]), number


# ------------------------------------------------------------- against the reference


@needs_reference
@pytest.mark.parametrize("moisture", [0.03, 0.06, 0.12])
def test_head_spread_rate_matches_every_fuel_model(moisture):
    """Head rate of spread, all 53 burnable models at once, flat ground."""
    ros = make_ros()
    state = world(
        torch.tensor(BURNABLE, dtype=torch.float32).view(1, 1, -1),
        moisture=moisture, grid=(1, len(BURNABLE)),
    )
    for wind in (0.0, 1.0, 3.0, 6.0, 12.0):
        mine = ros(state.replace(wind=torch.tensor([[wind, 0.0]])))[0, EAST, 0, :].numpy()
        theirs = np.array(
            [reference_directions(n, wind, 0.0, moisture)[EAST] for n in BURNABLE]
        )
        # Rates below a micrometre per second are fuel past its moisture of
        # extinction; both codes call it out and the ratio is meaningless.
        alive = theirs > 1e-6
        assert np.allclose(mine[alive], theirs[alive], rtol=1e-3), (
            f"worst: {FUEL_MODEL_NAMES[BURNABLE[int(np.argmax(np.abs(mine - theirs)))]]}"
        )
        assert np.allclose(mine[~alive], theirs[~alive], atol=1e-6)


@needs_reference
@pytest.mark.parametrize("number", [102, 142, 183, 124, 165, 1, 10, 145, 204])
@pytest.mark.parametrize("wind,slope", [(0, 0.0), (3, 0.0), (0, 0.3), (3, 0.3), (6, 0.5), (15, 0.3)])
def test_all_eight_directions_match(number, wind, slope):
    """The full directional template, on flat and on sloped ground.

    The offset angle is taken on the slope-tangential plane, which is where the
    reference takes it; doing it in map projection instead is within a percent
    on the flat and out by more than ten on a 50% grade.
    """
    mine = make_ros()(world(number, wind=float(wind), slope=slope))[0, :, 2, 2].numpy()
    theirs = reference_directions(number, wind, slope, 0.06)
    if theirs.max() < 1e-6:
        pytest.skip("fuel past its moisture of extinction in this scenario")
    assert np.allclose(mine, theirs, rtol=2e-3)


@needs_reference
@pytest.mark.parametrize(
    "wind,slope", [((3.0, 2.0), (0.0, 0.0)), ((0.0, 4.0), (0.3, 0.0)), ((2.0, -3.0), (0.1, 0.25))]
)
def test_wind_and_slope_in_different_directions(wind, slope):
    """Wind and slope combine as vectors on the slope plane, not as scalars."""
    import pyretechnics.fuel_models as fm
    import pyretechnics.surface_fire as sf

    state = uniform_world(batch=1, grid=(5, 5), preset="grass", cell_size=30.0,
                          wind=wind, slope=slope)
    state = state.replace(
        fuel_model=torch.full_like(state.ros0, 102.0),
        moisture=torch.full_like(state.moisture, 0.06),
    )
    mine = make_ros()(state)[0, :, 2, 2].numpy()

    model = fm.moisturize(fm.get_fuel_model(102), (0.06, M10, M100, 0.0, MHERB, MWOODY))
    base = sf.calc_surface_fire_behavior_no_wind_no_slope(model, 1.0)
    downwind = math.degrees(math.atan2(wind[0], wind[1])) % 360
    upslope = math.degrees(math.atan2(slope[0], slope[1])) % 360 if any(slope) else 0.0
    peak = sf.calc_surface_fire_behavior_max(
        base, math.hypot(*wind) * 60.0, (downwind + 180) % 360,
        math.hypot(*slope), (upslope + 180) % 360, True, "behave",
    )
    theirs = []
    for dy, dx in NEIGHBOR_OFFSETS:
        norm = math.hypot(dy, dx)
        x, y = dx / norm, dy / norm
        z = slope[0] * x + slope[1] * y
        length = math.sqrt(x * x + y * y + z * z)
        theirs.append(
            sf.calc_surface_fire_behavior_in_direction(
                peak, (x / length, y / length, z / length)
            )["spread_rate"] / 60.0
        )
    assert np.allclose(mine, np.array(theirs), rtol=2e-3)


@needs_reference
def test_effective_wind_limit_is_applied():
    """Rothermel stops believing its wind factor past 0.9 * I_R, and so must this.

    Without the cap a grass fire accelerates without bound; `SimpleROS` does
    exactly that, which is one of the things this model exists to fix.
    """
    capped = make_ros(use_wind_limit=True)(world(102, wind=20.0))[0, EAST, 2, 2]
    uncapped = make_ros(use_wind_limit=False)(world(102, wind=20.0))[0, EAST, 2, 2]
    assert capped < uncapped
    assert capped == pytest.approx(reference_directions(102, 20.0, 0.0, 0.06)[EAST], rel=2e-3)


# ---------------------------------------------------- properties, reference or not


def test_non_burnable_fuel_does_not_spread():
    state = world(torch.tensor([[[102.0, 91.0, 93.0, 99.0, 7777.0]]]), grid=(1, 5), wind=5.0)
    rates = make_ros()(state)[0, EAST, 0, :]
    assert rates[0] > 0.1
    assert torch.all(rates[1:] == 0.0)


def test_spread_stops_above_the_moisture_of_extinction():
    """The mechanism suppression acts through, inside the spread model itself.

    GR2's dead moisture of extinction is 15%. Water raises `state.moisture`;
    past that threshold Rothermel's own damping term takes the rate to zero, so
    a drop stops a front through the physics rather than through a special case.
    """
    ros = make_ros()
    below = ros(world(102, moisture=0.10, wind=3.0))[0, EAST, 2, 2]
    above = ros(world(102, moisture=0.16, wind=3.0))[0, EAST, 2, 2]
    assert below > 0.2
    # Past extinction the damping polynomial `1 - 2.59r + 5.11r^2 - 3.52r^3` is
    # zero at r = 1 by construction, but only to float32 cancellation - a few
    # nanometres per second survive, which is centimetres per year. Assert that
    # rather than an exact zero, or the test is about float32 and not physics.
    assert above < 1e-6


def test_gradients_flow_to_moisture_including_through_non_burnable_cells():
    """The property the whole package is built around, in the new ROS model.

    Rothermel is full of ratios that are genuinely zero over non-burnable
    ground; a plain division there gives NaN gradients that no `torch.where`
    downstream can repair.
    """
    state = world(torch.tensor([[[102.0, 91.0, 142.0]]]), grid=(1, 3), wind=3.0)
    moisture = state.moisture.clone().requires_grad_(True)
    make_ros()(state.replace(moisture=moisture)).sum().backward()

    assert torch.isfinite(moisture.grad).all()
    assert moisture.grad[0, 0, 0] < 0, "wetter fuel must spread slower"
    assert moisture.grad[0, 0, 1] == 0.0, "non-burnable ground has no sensitivity"


def test_rothermel_world_is_self_consistent():
    """`ros0`, `moisture_ext` and `burn_rate` must agree with the spread model.

    The propagator reads those summary layers; if they disagree with the ROS
    model driving it, the fuel accounting and the front describe different fires.
    """
    ros = make_ros()
    state = rothermel_world(grid=(5, 5), fuel_model=102, moisture=0.06, ros_model=ros)

    no_wind = ros(state)[0, EAST, 2, 2]
    assert state.ros0[0, 2, 2] == pytest.approx(float(no_wind), rel=1e-4)
    assert state.moisture_ext[0, 2, 2] == pytest.approx(0.15, rel=1e-3)
    # Anderson's flame residence time for GR2 is about 13 s.
    assert 1.0 / state.burn_rate[0, 2, 2] == pytest.approx(12.7, rel=0.05)


def test_rothermel_world_accepts_a_raster_of_fuel_models():
    """The LANDFIRE shape: a per-cell map of model numbers."""
    raster = torch.tensor([[102.0, 142.0, 91.0], [183.0, 204.0, 99.0]])
    state = rothermel_world(grid=(2, 3), fuel_model=raster, ros_model=make_ros())
    assert state.ros0[0, 0, 0] > state.ros0[0, 0, 1] > 0.0  # GR2 beats SH2
    assert state.ros0[0, 0, 2] == 0.0 and state.ros0[0, 1, 2] == 0.0  # non-burnable
    assert state.burn_rate[0, 0, 2] == 0.0
