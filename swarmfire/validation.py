"""Validation of the simulator against `pyretechnics`, the reference implementation.

`pyretechnics` (Spatial Informatics Group / Pyregence) implements Rothermel
(1972) surface spread with the Albini/Anderson elliptical template and an
Eulerian level-set front tracker. It is actively maintained, it is the same
model BehavePlus and FARSITE are built on, and - unlike them - it is a library
that can be called cell by cell, which makes it the right thing to measure this
package against.

The comparison is deliberately split in two, because the two questions have
different answers:

1. **Point physics** - does `SimpleROS` return the rate of spread Rothermel
   returns, for the same fuel, moisture, midflame wind and slope? This is a
   question about `ros/`, and nothing else.
2. **Front geometry** - given *identical* directional rate-of-spread fields,
   does `CAPropagator` move the front at that rate, and into the right shape?
   This is a question about `propagate.py`, and it is the one that survives
   swapping `SimpleROS` for a real Rothermel model.

Mixing them hides the answer: an under-propagating front and an over-fast ROS
look like a correct simulator on burned-area plots alone.

`matched_world` is what separates them. Rothermel's wind factor is
`phi_w = C (beta/beta_op)^-E * U^B` and its slope factor is `phi_s = G tan^2(phi)`
- exactly the algebraic form `SimpleROS` already uses. So for one fuel and one
moisture state, `SimpleROS` can be given Rothermel's own coefficients and then
agrees with `pyretechnics` to float precision. Propagate that, and any
disagreement left is the propagator's.

Everything here is import-guarded: `pip install "swarmfire[real]"` to use it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from .fuels import uniform_world
from .grid import NEIGHBOR_OFFSETS
from .propagate import Propagator
from .ros import ROSModel, SimpleROS
from .state import FireState, ignite

__all__ = [
    "PYRETECHNICS_AVAILABLE",
    "FUEL_EQUIVALENTS",
    "STANDARD_MOISTURE",
    "RothermelCoefficients",
    "rothermel_coefficients",
    "rothermel_point",
    "rothermel_directional",
    "matched_world",
    "midflame_to_wind_10m",
    "swarmfire_directional",
    "swarmfire_head_ros",
    "swarmfire_arrival_map",
    "pyretechnics_arrival_map",
    "axis_spread_rates",
    "iou",
]

try:  # pragma: no cover - exercised by whether the extra is installed
    import pyretechnics.conversion as _conv
    import pyretechnics.eulerian_level_set as _els
    import pyretechnics.fuel_models as _fm
    import pyretechnics.surface_fire as _sf
    from pyretechnics.space_time_cube import SpaceTimeCube as _SpaceTimeCube

    PYRETECHNICS_AVAILABLE = True
except ImportError:  # pragma: no cover
    PYRETECHNICS_AVAILABLE = False


def _require() -> None:
    if not PYRETECHNICS_AVAILABLE:  # pragma: no cover
        raise ImportError(
            'pyretechnics is not installed. Run: pip install "swarmfire[real]"'
        )


# The Scott & Burgan fuel model each `fuels.PRESETS` entry is standing in for.
# GR2 is a low-load dry-climate grass, SH2 a moderate-load shrub, TL3 a moderate
# conifer litter - the three beds the presets describe in prose.
FUEL_EQUIVALENTS: dict[str, int] = {"grass": 102, "shrub": 142, "timber": 163}

# A single dry-but-not-extreme moisture scenario, held fixed across the whole
# comparison so that differences are attributable to the models and not to the
# weather. Dry-weight fractions, as everywhere in this package.
STANDARD_MOISTURE: dict[str, float] = {
    "dead_1hr": 0.06,
    "dead_10hr": 0.07,
    "dead_100hr": 0.08,
    "live_herbaceous": 0.60,
    "live_woody": 0.90,
}


def _moisture_tuple(moisture: dict[str, float]) -> tuple[float, ...]:
    """pyretechnics' 6-slot fuel-class moisture vector."""
    return (
        moisture["dead_1hr"],
        moisture["dead_10hr"],
        moisture["dead_100hr"],
        0.0,  # dead herbaceous, filled in by the dynamic-load transfer
        moisture["live_herbaceous"],
        moisture["live_woody"],
    )


def _fire_min(fuel_model: int, moisture: dict[str, float]) -> dict:
    _require()
    model = _fm.moisturize(_fm.get_fuel_model(fuel_model), _moisture_tuple(moisture))
    return _sf.calc_surface_fire_behavior_no_wind_no_slope(model, 1.0)


# --------------------------------------------------------------- point physics


@dataclass(frozen=True)
class RothermelCoefficients:
    """Rothermel's own parameters, in this package's units and naming.

    Every field lines up with something `SimpleROS` already has, which is what
    makes the two models comparable term by term rather than only end to end.

    ros0 : no-wind no-slope rate of spread, m/s - `FireState.ros0`.
    wind_a, wind_b : `phi_w = wind_a * U**wind_b`, U in m/s at midflame height -
        `SimpleROS.wind_a`, `SimpleROS.wind_b`.
    slope_c : `phi_s = slope_c * tan(phi)**2` - `SimpleROS.slope_c`.
    max_effective_wind : the midflame wind above which Rothermel stops
        believing its own wind factor and caps it, m/s. `SimpleROS` has no
        equivalent and keeps accelerating past it.
    residence_s : Rothermel flame residence time, seconds - `FuelPreset.residence`.
    moisture_ext : dead-fuel moisture of extinction - `FireState.moisture_ext`.
    """

    fuel_model: int
    ros0: float
    wind_a: float
    wind_b: float
    slope_c: float
    max_effective_wind: float
    residence_s: float
    moisture_ext: float


def rothermel_coefficients(
    fuel_model: int, moisture: dict[str, float] | None = None
) -> RothermelCoefficients:
    """Pull Rothermel's wind/slope coefficients out of `pyretechnics` in SI units.

    `pyretechnics` carries them internally per minute and per foot; this
    converts the wind factor to metres per second so it can be read straight
    against `SimpleROS(wind_a=..., wind_b=...)`.
    """
    fire_min = _fire_min(fuel_model, moisture or STANDARD_MOISTURE)
    fuel = _fm.get_fuel_model(fuel_model)

    # phi_w is scale-covariant in the wind unit: U[m/min] = 60 * U[m/s], so the
    # scalar picks up 60**exponent and the exponent itself is unchanged.
    exponent = float(fire_min["_phiW_expnt"])
    scalar = float(fire_min["_phiW_scalr"]) * 60.0**exponent

    ros0 = float(fire_min["base_spread_rate"]) / 60.0

    return RothermelCoefficients(
        fuel_model=fuel_model,
        ros0=ros0,
        wind_a=scalar,
        wind_b=exponent,
        slope_c=float(fire_min["_phiS_G"]),
        max_effective_wind=float(fire_min["max_effective_wind_speed"]) / 60.0,
        residence_s=_residence_time(fuel_model, moisture or STANDARD_MOISTURE),
        moisture_ext=float(fuel["M_x"][0]),
    )


def _residence_time(fuel_model: int, moisture: dict[str, float]) -> float:
    """Anderson (1969) flame residence time, seconds.

    `sigma_prime`, the fuel-bed weighted surface-area-to-volume ratio, is not
    exported by `pyretechnics`, so it is rebuilt here from the fuel model's own
    weighting factors - the same two lines `calc_surface_area_to_volume_ratio`
    runs.
    """
    model = _fm.moisturize(_fm.get_fuel_model(fuel_model), _moisture_tuple(moisture))
    f_ij, f_i, sigma = model["f_ij"], model["f_i"], model["sigma"]
    dead = sum(f_ij[i] * sigma[i] for i in range(4))
    live = sum(f_ij[i] * sigma[i] for i in range(4, 6))
    sigma_prime = f_i[0] * dead + f_i[1] * live
    return 384.0 / sigma_prime * 60.0


@dataclass(frozen=True)
class RothermelPoint:
    """One evaluation of the reference model. Rates are m/s."""

    ros0: float
    head_ros: float
    length_to_breadth: float
    eccentricity: float
    wind_limited: bool


def rothermel_point(
    fuel_model: int,
    midflame_wind: float = 0.0,
    slope: float = 0.0,
    moisture: dict[str, float] | None = None,
    use_wind_limit: bool = True,
) -> RothermelPoint:
    """Reference head rate and fire shape, for wind and slope both due east.

    `midflame_wind` is m/s at midflame height - the same quantity
    `FireState.wind` holds - and `slope` is rise/run, so this is directly
    comparable with `swarmfire_head_ros` on the same numbers.
    """
    fire_min = _fire_min(fuel_model, moisture or STANDARD_MOISTURE)
    # Wind from the west and terrain falling to the west: both drive the fire east.
    fire_max = _sf.calc_surface_fire_behavior_max(
        fire_min, midflame_wind * 60.0, 270.0, slope, 270.0, use_wind_limit, "behave"
    )
    unlimited = _sf.calc_surface_fire_behavior_max(
        fire_min, midflame_wind * 60.0, 270.0, slope, 270.0, False, "behave"
    )
    return RothermelPoint(
        ros0=float(fire_min["base_spread_rate"]) / 60.0,
        head_ros=float(fire_max["max_spread_rate"]) / 60.0,
        length_to_breadth=float(fire_max["length_to_width_ratio"]),
        eccentricity=float(fire_max["eccentricity"]),
        wind_limited=float(fire_max["max_spread_rate"]) < float(unlimited["max_spread_rate"]) - 1e-9,
    )


def rothermel_directional(
    fuel_model: int,
    midflame_wind: float = 0.0,
    slope: float = 0.0,
    moisture: dict[str, float] | None = None,
) -> np.ndarray:
    """`[8]` reference spread rates toward each Moore neighbour, m/s.

    Same direction order as `grid.NEIGHBOR_OFFSETS`, so this lines up element
    for element with what a `ROSModel` returns.
    """
    fire_min = _fire_min(fuel_model, moisture or STANDARD_MOISTURE)
    fire_max = _sf.calc_surface_fire_behavior_max(
        fire_min, midflame_wind * 60.0, 270.0, slope, 270.0, True, "behave"
    )
    rates = []
    for dy, dx in NEIGHBOR_OFFSETS:
        # pyretechnics takes the query direction as a 3D unit vector lying on
        # the slope-tangential plane, so on sloped ground the compass direction
        # has to be lifted onto the plane before it is asked about.
        norm = math.hypot(dy, dx)
        vec = _tangential_unit(dx / norm, dy / norm, slope)
        out = _sf.calc_surface_fire_behavior_in_direction(fire_max, vec)
        rates.append(float(out["spread_rate"]) / 60.0)
    return np.array(rates)


def _tangential_unit(x: float, y: float, slope: float) -> tuple[float, float, float]:
    """Unit vector in the (x, y) compass direction, lying on a plane rising east."""
    z = slope * x
    norm = math.sqrt(x * x + y * y + z * z)
    return (x / norm, y / norm, z / norm)


# ---------------------------------------------------------- matched-ROS bridge


def matched_world(
    fuel_model: int,
    grid: tuple[int, int],
    cell_size: float,
    wind: tuple[float, float],
    slope: tuple[float, float] = (0.0, 0.0),
    moisture: dict[str, float] | None = None,
    residence_s: float | None = None,
    batch: int = 1,
    device: str | torch.device = "cpu",
) -> tuple[FireState, SimpleROS, RothermelCoefficients]:
    """A `FireState` and a `SimpleROS` that together reproduce Rothermel exactly.

    This is the instrument for the second half of the comparison. The returned
    ROS model is `SimpleROS` with Rothermel's own coefficients substituted for
    the hand-picked ones, and the returned state carries Rothermel's base rate
    in `ros0`. Directional rates then agree with `rothermel_directional` to
    float precision (`tests/test_validation.py` asserts it), so a fire spread
    with this pair differs from a `pyretechnics` fire only through the
    propagator.

    Two caveats on "exactly". Rothermel's effective-wind limit has no
    `SimpleROS` counterpart, so keep `wind` below
    `coefficients.max_effective_wind`. And `pyretechnics` works on the
    slope-tangential plane where this package works in map projection, so on
    sloped ground the two differ by the projection factor - about 2% at a 20%
    grade, growing as the square of the slope. Flat ground is exact.

    `residence_s` overrides the burnout time. It defaults to Rothermel's own
    flame residence time, which for fine fuels is only ten-odd seconds; the
    presets in `fuels.py` use values several times longer. It is exposed
    because the propagator turns out to be sensitive to it - see the
    residence-time sweep in `scripts/validate_pyretechnics.py`.
    """
    coeffs = rothermel_coefficients(fuel_model, moisture)
    state = uniform_world(
        batch=batch, grid=grid, cell_size=cell_size, wind=wind, slope=slope, device=device
    )
    state = state.replace(
        ros0=torch.full_like(state.ros0, coeffs.ros0),
        moisture_ext=torch.full_like(state.moisture_ext, coeffs.moisture_ext),
        burn_rate=torch.full_like(state.burn_rate, 1.0 / (residence_s or coeffs.residence_s)),
    )
    ros_model = SimpleROS(
        wind_a=coeffs.wind_a, wind_b=coeffs.wind_b, slope_c=coeffs.slope_c
    )
    return state, ros_model, coeffs


def midflame_to_wind_10m(midflame_wind: float, fuel_model: int) -> float:
    """Invert the wind profile: midflame m/s -> 10 m open wind, km/h.

    `pyretechnics`' gridded entry points take a 10 m wind and apply the Andrews
    (2012) adjustment down to midflame height; this package's `FireState.wind`
    is already at midflame height. The adjustment is linear in wind speed, so
    one evaluation gives the factor.
    """
    _require()
    depth_ft = float(_fm.get_fuel_model(fuel_model)["delta"])
    unit_20ft = _conv.km_hr_to_m_min(_conv.wind_speed_10m_to_wind_speed_20ft(1.0))
    per_kmh = _sf.calc_midflame_wind_speed(unit_20ft, depth_ft, 0.0, 0.0)
    return midflame_wind * 60.0 / per_kmh


# ------------------------------------------------------------ swarmfire probes


def swarmfire_directional(ros_model: ROSModel, state: FireState) -> np.ndarray:
    """`[8]` spread rates at the grid centre, m/s, in `NEIGHBOR_OFFSETS` order."""
    h, w = state.grid
    return ros_model(state)[0, :, h // 2, w // 2].detach().cpu().numpy()


def swarmfire_head_ros(ros_model: ROSModel, state: FireState) -> float:
    """Head rate of spread at the grid centre, m/s.

    Wind and slope are due east in every scenario here, so the head direction
    lands exactly on the eastward neighbour and no interpolation is needed.
    """
    east = NEIGHBOR_OFFSETS.index((0, 1))
    return float(swarmfire_directional(ros_model, state)[east])


def swarmfire_arrival_map(
    state: FireState,
    propagator: Propagator,
    duration_s: float,
    dt: float,
    ignition: tuple[int, int] | None = None,
) -> np.ndarray:
    """Run the CA and return time of arrival in seconds, NaN where unburnt.

    `[H, W]` for a batch of one, `[B, H, W]` otherwise - so a parameter sweep
    is one batched run rather than a Python loop over runs.

    Arrival is the moment `ignition` crosses 1, which is the propagator's own
    definition of the front reaching a cell - the same event `time_of_arrival`
    records in `pyretechnics`.
    """
    b, (h, w) = state.batch, state.grid
    y, x = ignition or (h // 2, w // 2)
    state = ignite(state, y, x)

    arrival = torch.full((b, h, w), float("nan"), device=state.device)
    with torch.no_grad():
        for i in range(int(round(duration_s / dt))):
            state = propagator.step(state, dt)
            reached = (state.ignition >= 1.0) & torch.isnan(arrival)
            arrival = torch.where(reached, torch.full_like(arrival, (i + 1) * dt), arrival)
    out = arrival.cpu().numpy()
    return out[0] if b == 1 else out


def pyretechnics_arrival_map(
    fuel_model: int,
    grid: tuple[int, int],
    cell_size: float,
    midflame_wind: float,
    duration_s: float,
    slope: float = 0.0,
    moisture: dict[str, float] | None = None,
    ignition: tuple[int, int] | None = None,
) -> np.ndarray:
    """Same map from the reference level-set solver. Seconds, NaN where unburnt.

    Wind blows east and terrain rises to the east, matching the sign convention
    the rest of this module uses.
    """
    _require()
    moisture = moisture or STANDARD_MOISTURE
    h, w = grid
    y, x = ignition or (h // 2, w // 2)
    shape = (1, h, w)

    def cube(value):
        return _SpaceTimeCube(shape, value)

    cubes = {
        "slope": cube(slope),
        # Aspect is the downslope compass bearing; terrain rising east faces west.
        "aspect": cube(270.0),
        "fuel_model": cube(fuel_model),
        "canopy_cover": cube(0.0),
        "canopy_height": cube(0.0),
        "canopy_base_height": cube(0.0),
        "canopy_bulk_density": cube(0.0),
        "wind_speed_10m": cube(midflame_to_wind_10m(midflame_wind, fuel_model)),
        # Wind *from* the west, i.e. blowing east.
        "upwind_direction": cube(270.0),
        "fuel_moisture_dead_1hr": cube(moisture["dead_1hr"]),
        "fuel_moisture_dead_10hr": cube(moisture["dead_10hr"]),
        "fuel_moisture_dead_100hr": cube(moisture["dead_100hr"]),
        "fuel_moisture_live_herbaceous": cube(moisture["live_herbaceous"]),
        "fuel_moisture_live_woody": cube(moisture["live_woody"]),
        "foliar_moisture": cube(0.90),
    }

    spread_state = _els.SpreadState(shape).ignite_cell((y, x))
    result = _els.spread_fire_with_phi_field(
        cubes,
        spread_state,
        (duration_s / 60.0 + 1.0, cell_size, cell_size),  # one band, long enough
        0.0,
        max_duration=duration_s / 60.0,
    )
    matrices = result["spread_state"].get_full_matrices()
    toa = np.asarray(matrices["time_of_arrival"], dtype=float) * 60.0
    return np.where(toa < 0.0, np.nan, toa)


# ---------------------------------------------------------------- comparisons


def axis_spread_rates(
    arrival: np.ndarray, ignition: tuple[int, int] | None = None, cell_size: float = 30.0
) -> dict[str, float]:
    """Head / flank / back rate of spread read off an arrival map, m/s.

    Fits distance against arrival time along each axis through the ignition
    point and inverts the slope. Fitting rather than dividing the final extent
    by the final time removes the head start the ignition blob gives - the two
    codes seed the fire differently - and is insensitive to where exactly the
    front happens to sit when the clock stops.
    """
    h, w = arrival.shape
    y, x = ignition or (h // 2, w // 2)
    rays = {
        "head": arrival[y, x:],
        "back": arrival[y, : x + 1][::-1],
        "flank_north": arrival[y:, x],
        "flank_south": arrival[: y + 1, x][::-1],
    }

    out: dict[str, float] = {}
    for name, ray in rays.items():
        distance = np.arange(len(ray)) * cell_size
        ok = np.isfinite(ray)
        # Drop the ignition blob (first two cells) and the final partially
        # resolved cell; fit what is left.
        ok[:2] = False
        if ok.sum() < 4:
            out[name] = float("nan")
            continue
        slope = np.polyfit(distance[ok], ray[ok], 1)[0]
        out[name] = float(1.0 / slope) if slope > 0 else float("nan")
    out["flank"] = 0.5 * (out["flank_north"] + out["flank_south"])
    return out


def iou(a: np.ndarray, b: np.ndarray, time_s: float) -> float:
    """Intersection over union of the two burned footprints at `time_s`."""
    mask_a = np.isfinite(a) & (a <= time_s)
    mask_b = np.isfinite(b) & (b <= time_s)
    union = (mask_a | mask_b).sum()
    return float((mask_a & mask_b).sum() / union) if union else float("nan")
