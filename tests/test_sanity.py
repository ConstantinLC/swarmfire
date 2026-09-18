"""Physics sanity checks.

These are the properties that catch the mistakes that actually happen: sign
errors in the shift, fire shape that follows the grid instead of the wind,
suppression that does nothing, batch elements leaking into each other, and
gradients that silently die.
"""

from __future__ import annotations

import math

import pytest
import torch

from swarmfire import CAPropagator, EnvConfig, FireEnv, SimpleROS, SuppressionModel, ignite, uniform_world
from swarmfire.grid import shift
from swarmfire.ros import length_to_breadth

DEVICE = "cpu"


def burn(state, steps=120, dt=4.0):
    prop = CAPropagator(SimpleROS(), SuppressionModel())
    for _ in range(steps):
        state = prop.step(state, dt)
    return state


def extent(state, index=0, thresh=0.5):
    """(height, width) of the burn scar in cells."""
    scar = (state.burned[index] > thresh).float()
    ys, xs = torch.nonzero(scar, as_tuple=True)
    if len(ys) == 0:
        return 0.0, 0.0
    return float(ys.max() - ys.min() + 1), float(xs.max() - xs.min() + 1)


def reach(state, origin, index=0, thresh=0.5):
    """Cells the scar extends east / west / north / south of the ignition."""
    y0, x0 = origin
    ys, xs = torch.nonzero(state.burned[index] > thresh, as_tuple=True)
    return (
        float(xs.max() - x0), float(x0 - xs.min()),
        float(ys.max() - y0), float(y0 - ys.min()),
    )


# ------------------------------------------------------------------- grid


def test_shift_moves_the_right_way():
    x = torch.zeros(1, 5, 5)
    x[0, 2, 2] = 1.0
    # out[i, j] == x[i - dy, j - dx]
    assert shift(x, 1, 0)[0, 3, 2] == 1.0
    assert shift(x, -1, 0)[0, 1, 2] == 1.0
    assert shift(x, 0, 1)[0, 2, 3] == 1.0
    assert shift(x, 0, -1)[0, 2, 1] == 1.0


def test_shift_does_not_wrap():
    x = torch.ones(1, 4, 4)
    assert shift(x, 1, 0)[0, 0].sum() == 0.0


def test_length_to_breadth_is_unity_at_zero_wind():
    assert length_to_breadth(torch.zeros(1)).item() == pytest.approx(1.0, abs=1e-3)
    assert length_to_breadth(torch.tensor([8.0])).item() > 3.0


# ------------------------------------------------------------ fire shape


def test_front_speed_matches_rate_of_spread():
    """The property the ignition-progress design exists to guarantee.

    The head of the fire must advance at the rate the ROS model reports - not
    merely in the right direction, but at the right speed. This is what the
    earlier self-sustaining formulation got wrong by a factor of four.
    """
    steps, dt, cell = 300, 4.0, 30.0
    state = uniform_world(grid=(160, 160), wind=(6.0, 0.0), cell_size=cell, device=DEVICE)

    ros = SimpleROS()(state)[0, 4, 80, 80].item()  # index 4 == due east
    expected = ros * steps * dt / cell

    state = burn(ignite(state, 80, 80), steps=steps, dt=dt)
    east, _, _, _ = reach(state, (80, 80))

    assert expected * 0.75 < east < expected * 1.25, (
        f"head advanced {east:.0f} cells, rate of spread implies {expected:.0f}"
    )


def test_no_wind_fire_is_round():
    # Slow spread with no wind, so use a fine grid to get a resolved scar.
    state = uniform_world(grid=(96, 96), wind=(0.0, 0.0), cell_size=2.0, device=DEVICE)
    state = ignite(state, 48, 48)
    state = burn(state, steps=300)

    h, w = extent(state)
    assert h > 8 and w > 8, f"fire failed to spread ({h}x{w})"
    assert abs(h - w) / max(h, w) < 0.15, f"expected a round fire, got {h}x{w}"


def test_wind_makes_the_fire_elliptical_and_downwind():
    state = uniform_world(grid=(160, 160), wind=(6.0, 0.0), device=DEVICE)
    state = ignite(state, 80, 80)
    state = burn(state, steps=300)

    h, w = extent(state)
    assert w > 3 * h, f"wind-driven fire should be elongated downwind, got {h}x{w}"

    east, west, _, _ = reach(state, (80, 80))
    assert east > 5 * west, f"head should far outrun the back: {east} vs {west}"


def test_slope_drives_fire_uphill():
    # Ground rising steeply to the east, no wind at all.
    state = uniform_world(
        grid=(128, 128), wind=(0.0, 0.0), slope=(0.8, 0.0), cell_size=5.0, device=DEVICE
    )
    state = ignite(state, 64, 64)
    state = burn(state, steps=300)

    east, west, north, _ = reach(state, (64, 64))
    assert east > 2 * west, f"fire did not run upslope: east {east}, west {west}"
    assert east > north, "upslope run should beat the flanks"


def test_faster_wind_burns_more():
    areas = []
    for u in (1.0, 4.0, 8.0):
        state = uniform_world(grid=(192, 192), wind=(u, 0.0), device=DEVICE)
        state = burn(ignite(state, 96, 48), steps=300)
        areas.append(state.burned_area().item())
    assert areas[0] < areas[1] < areas[2], f"burned area not monotone in wind: {areas}"


# ------------------------------------------------------------ suppression


def test_water_stops_the_front():
    """A pre-wetted barrier should hold; the same run without it should not."""
    def run(wet: bool):
        state = uniform_world(grid=(128, 128), wind=(5.0, 0.0), device=DEVICE)
        if wet:
            water = state.water.clone()
            water[:, :, 60:66] = 2.0  # mm, a soaked north-south strip
            state = state.replace(water=water)
        state = ignite(state, 64, 50)
        return burn(state, steps=400)

    dry_state, wet_state = run(False), run(True)
    beyond = lambda s: (s.burned[0, :, 70:] > 0.5).sum().item()

    assert beyond(dry_state) > 50, "control run never reached the barrier"
    assert beyond(wet_state) == 0, "fire crossed a soaked barrier"


def test_retardant_persists_and_water_evaporates():
    state = uniform_world(grid=(32, 32), device=DEVICE)
    state = state.replace(
        water=torch.full_like(state.water, 1.0),
        retardant=torch.full_like(state.retardant, 1.0),
    )
    model = SuppressionModel()
    for _ in range(300):  # 20 minutes at dt=4
        state = model.update(state, 4.0)

    assert state.water.mean() < 0.9, "water did not evaporate at all"
    assert state.retardant.mean() > 0.95, "retardant decayed far too fast"


def test_wet_fuel_is_not_flammable():
    state = uniform_world(grid=(16, 16), preset="grass", device=DEVICE)
    model = SuppressionModel()
    dry = model.flammability(state).mean()
    wet = model.flammability(state.replace(water=torch.full_like(state.water, 1.0))).mean()
    assert dry > 0.9 and wet < 0.01


# -------------------------------------------------------------- machinery


def test_nothing_burns_where_nothing_is_lit():
    """The smooth gates must reach zero, not merely approach it.

    A plain `sigmoid((0 - 1) / 0.08)` is 3.7e-6, which sounds like nothing and
    is not: it applies to every cell in the domain, on every step, forever.
    """
    state = uniform_world(batch=1, grid=(96, 96), wind=(4.0, 0.0), device=DEVICE)
    state = burn(state, steps=20)
    assert state.intensity.sum().item() == 0.0
    assert state.burned.sum().item() == 0.0
    assert state.fuel.min().item() == 1.0


def test_burnt_out_fuel_stops_burning():
    """A cell with no fuel left must report no combustion.

    It used to report 27% of full intensity, and because `ignition_drive`
    sources the front from a neighbour's intensity, that phantom kept driving
    the fire outwards for the rest of the episode.
    """
    prop = CAPropagator(SimpleROS(), SuppressionModel())
    state = uniform_world(batch=1, grid=(16, 16), wind=(0.0, 0.0), device=DEVICE)
    state = state.replace(
        fuel=torch.zeros_like(state.fuel),          # nothing left to burn
        ignition=torch.full_like(state.ignition, 5.0),  # but long since arrived
    )
    state = prop.step(state, 1.0)
    assert state.intensity.max().item() == 0.0
    assert state.active().item() == 0.0


def test_batch_elements_are_independent():
    state = uniform_world(batch=3, grid=(64, 64), wind=(4.0, 0.0), device=DEVICE)
    # Light world 1 only. Seed `ignition`, not `intensity`: the front is driven
    # by whether a cell has arrived, and `intensity` is the consumption rate it
    # produces afterwards - see `CAPropagator.front_source`.
    ignition = state.ignition.clone()
    ignition[1, 32, 32] = 1.5
    state = burn(state.replace(ignition=ignition), steps=60)

    assert state.burned[0].sum() == 0.0
    assert state.burned[2].sum() == 0.0
    assert state.burned[1].sum() > 0.0


def test_fuel_is_conserved():
    state = uniform_world(grid=(64, 64), wind=(5.0, 0.0), device=DEVICE)
    state = burn(ignite(state, 32, 32), steps=200)
    total = state.fuel + state.burned
    assert torch.allclose(total, torch.ones_like(total), atol=1e-4)
    assert state.fuel.min() >= 0.0


def test_gradient_reaches_the_drop_position():
    """The property the whole continuous-state design exists to protect."""
    env = FireEnv("helicopter", config=EnvConfig(batch=1, grid=(64, 64), horizon=40,
                                                 wind=(5.0, 0.0), device=DEVICE, dt=4.0))
    env.reset()
    action = torch.zeros(1, 1, 3, requires_grad=True)
    for _ in range(40):
        env.step(action, detach=False)
    env.state.burned_area().sum().backward()

    assert action.grad is not None
    assert torch.isfinite(action.grad).all()
    assert action.grad.abs().sum() > 0, "no gradient reached the action"


def test_platform_runs_out_and_reloads():
    """Empty -> cooldown -> full, on the reload clock and not before.

    Watch the whole cycle rather than sampling the end of it. A vehicle held at
    full release intent re-empties on the very step after it refills, so the
    tank is non-empty for exactly one step in every `reload_s / dt`; a single
    check placed a few steps late sees an empty tank and reads it as "never
    refilled".
    """
    env = FireEnv("helicopter", config=EnvConfig(batch=1, grid=(48, 48), device=DEVICE, dt=4.0))
    env.reset()
    always_drop = torch.zeros(1, 1, 3)
    always_drop[..., 2] = 1.0

    env.step(always_drop)
    assert env.platform.load.item() == 0.0, "capacity was not consumed"
    assert env.platform.cooldown.item() > 0, "reload did not start"

    cycle = int(env.platform.spec.reload_s / env.cfg.dt)
    refills = []
    for step in range(1, cycle + 3):
        env.step(always_drop)
        if env.platform.load.max().item() > 0:
            refills.append(step)

    assert refills, "platform never refilled"
    # The tank comes back only once the reload clock has actually run down.
    assert refills[0] >= cycle - 2, f"refilled early, at step {refills[0]} of {cycle}"
    assert env.platform.load.max().item() == 0.0, "a full-intent drop should empty it again"


def test_nothing_is_dropped_before_dispatch():
    dispatch = 600.0
    env = FireEnv(
        "helicopter",
        config=EnvConfig(batch=1, grid=(48, 48), device=DEVICE, dt=4.0, dispatch_s=dispatch),
    )
    env.reset()
    always_drop = torch.zeros(1, 1, 3)
    always_drop[..., 2] = 1.0

    for _ in range(int(dispatch / env.cfg.dt)):
        env.step(always_drop)
    assert env.state.water.sum() == 0.0, "water was delivered before dispatch"
    assert env.platform.load.item() == env.platform.spec.capacity_l, "tank drained while waiting"

    env.step(always_drop)
    assert env.state.water.sum() > 0.0, "platform never became available"


def test_stability_check_flags_a_bad_timestep():
    env = FireEnv(config=EnvConfig(batch=1, grid=(32, 32), wind=(12.0, 0.0), dt=600.0, device=DEVICE))
    env.reset()
    limit, ok = env.check_stability()
    assert not ok and limit < 600.0


@pytest.mark.parametrize("kind", ["helicopter", "tanker", "drone_swarm"])
def test_every_platform_deposits_somewhere(kind):
    env = FireEnv(kind, config=EnvConfig(batch=2, grid=(64, 64), device=DEVICE, dt=4.0))
    env.reset()
    action = torch.zeros(2, env.n_agents, env.action_dim)
    action[..., 2] = 1.0
    env.step(action)

    deposited = (env.state.water + env.state.retardant).sum()
    assert deposited > 0, f"{kind} delivered nothing"
