"""Fuel bed presets and world construction.

The three presets are stand-ins with plausible orders of magnitude, not
calibrated fuel models. When you move to real physics, this module is where
LANDFIRE's 40 Scott & Burgan fuel models get loaded and turned into per-cell
parameter rasters; the rest of the package does not change.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .state import FireState


@dataclass(frozen=True)
class FuelPreset:
    """Intrinsic properties of a homogeneous fuel bed.

    ros0 : no-wind no-slope rate of spread, m/s.
    moisture_ext : moisture of extinction, dry-weight fraction.
    residence : seconds a cell burns at full intensity before its fuel is gone.
    moisture : default live/dead moisture the bed starts at.
    """

    name: str
    ros0: float
    moisture_ext: float
    residence: float
    moisture: float

    @property
    def burn_rate(self) -> float:
        return 1.0 / self.residence


# The standard fuel model each preset is a stand-in for, so that a world built
# from a preset can also be handed to `RothermelROS`. The presets themselves are
# not calibrated against these - see VALIDATION.md for how far apart they are.
FUEL_MODEL_FOR_PRESET: dict[str, int] = {"grass": 102, "shrub": 142, "timber": 183}


# Grass carries fire fast and dries out early; timber litter is the opposite.
PRESETS: dict[str, FuelPreset] = {
    "grass": FuelPreset("grass", ros0=0.020, moisture_ext=0.12, residence=60.0, moisture=0.06),
    "shrub": FuelPreset("shrub", ros0=0.012, moisture_ext=0.20, residence=150.0, moisture=0.09),
    "timber": FuelPreset("timber", ros0=0.004, moisture_ext=0.25, residence=400.0, moisture=0.12),
}


def uniform_world(
    batch: int = 1,
    grid: tuple[int, int] = (128, 128),
    preset: str = "grass",
    cell_size: float = 30.0,
    wind: tuple[float, float] = (0.0, 0.0),
    slope: tuple[float, float] = (0.0, 0.0),
    device: str | torch.device = "cpu",
) -> FireState:
    """A homogeneous fuel bed on a planar slope.

    Parameters
    ----------
    wind : (east, north) m/s at midflame height.
    slope : (d elevation / d east, d elevation / d north), dimensionless. So
        (0.3, 0.0) is a 30% grade rising to the east.
    """
    fp = PRESETS[preset]
    dev = torch.device(device)
    h, w = grid
    ones = torch.ones(batch, h, w, device=dev)

    yy = torch.arange(h, device=dev, dtype=torch.float32).view(-1, 1) * cell_size
    xx = torch.arange(w, device=dev, dtype=torch.float32).view(1, -1) * cell_size
    elevation = (slope[0] * xx + slope[1] * yy).expand(batch, h, w).contiguous()

    return FireState(
        fuel=ones.clone(),
        moisture=ones * fp.moisture,
        intensity=torch.zeros_like(ones),
        burned=torch.zeros_like(ones),
        ignition=torch.zeros_like(ones),
        water=torch.zeros_like(ones),
        retardant=torch.zeros_like(ones),
        elevation=elevation,
        fuel_model=ones * FUEL_MODEL_FOR_PRESET[preset],
        ros0=ones * fp.ros0,
        moisture_ext=ones * fp.moisture_ext,
        burn_rate=ones * fp.burn_rate,
        wind=torch.tensor(wind, device=dev, dtype=torch.float32).expand(batch, 2).contiguous(),
        cell_size=cell_size,
        t=torch.zeros(batch, device=dev),
    )


def patchy_world(
    batch: int = 1,
    grid: tuple[int, int] = (128, 128),
    presets: tuple[str, ...] = ("grass", "shrub", "timber"),
    scale: float = 16.0,
    seed: int | None = None,
    **kwargs,
) -> FireState:
    """A heterogeneous bed: smooth random blobs of different fuel types.

    Useful early, because a controller trained only on homogeneous fuel learns
    to exploit the symmetry rather than to read the fuel layer.
    """
    state = uniform_world(batch=batch, grid=grid, preset=presets[0], **kwargs)
    h, w = grid
    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)

    # Smooth noise by upsampling a coarse random field.
    coarse = torch.rand(batch, 1, max(2, int(h / scale)), max(2, int(w / scale)), generator=gen)
    field = torch.nn.functional.interpolate(coarse, size=(h, w), mode="bicubic", align_corners=False)
    field = field.squeeze(1).clamp(0, 1).to(state.device)

    ros0, mx, br, moist, model = (torch.zeros_like(field) for _ in range(5))
    edges = torch.linspace(0, 1, len(presets) + 1)
    for i, name in enumerate(presets):
        fp = PRESETS[name]
        lo, hi = float(edges[i]), float(edges[i + 1])
        mask = ((field >= lo) & (field < hi)).float()
        ros0 += mask * fp.ros0
        mx += mask * fp.moisture_ext
        br += mask * fp.burn_rate
        moist += mask * fp.moisture
        model += mask * FUEL_MODEL_FOR_PRESET[name]

    return state.replace(
        ros0=ros0, moisture_ext=mx, burn_rate=br, moisture=moist, fuel_model=model
    )


def rothermel_world(
    batch: int = 1,
    grid: tuple[int, int] = (128, 128),
    fuel_model: int | Tensor = 102,
    moisture: float = 0.06,
    cell_size: float = 30.0,
    wind: tuple[float, float] = (0.0, 0.0),
    slope: tuple[float, float] = (0.0, 0.0),
    device: str | torch.device = "cpu",
    ros_model=None,
) -> FireState:
    """A world described by a standard fuel model rather than by a preset.

    This is the builder to use with `RothermelROS`, and the one a LANDFIRE
    raster feeds: pass `fuel_model` as an `[H, W]` or `[B, H, W]` tensor of
    model numbers and the whole landscape comes from the table.

    It also fills `ros0`, `moisture_ext` and `burn_rate` *from Rothermel*, so
    the summary layers the propagator and the simpler ROS models read agree with
    the spread model rather than contradicting it. In particular `burn_rate`
    becomes Anderson's flame residence time for the fuel bed, which for fine
    fuels is a good deal shorter than the hand-picked preset values - see
    VALIDATION.md on why that no longer changes the front speed.

    `moisture` is dead 1-hour fuel moisture, the layer suppression raises. The
    coarser and live classes live on the `RothermelROS` instance; pass your own
    `ros_model` to set them.
    """
    from .ros.rothermel import RothermelROS

    state = uniform_world(
        batch=batch, grid=grid, preset="grass", cell_size=cell_size,
        wind=wind, slope=slope, device=device,
    )
    models = torch.as_tensor(fuel_model, dtype=torch.float32, device=state.device)
    state = state.replace(
        fuel_model=models.expand_as(state.ros0).contiguous(),
        moisture=torch.full_like(state.moisture, float(moisture)),
    )

    model = ros_model or RothermelROS()
    base = model.no_wind_no_slope(state)
    residence = base["residence_s"].clamp(min=1e-3)
    return state.replace(
        ros0=base["base_ros"],
        moisture_ext=base["moisture_ext"],
        burn_rate=torch.where(
            base["burnable"], 1.0 / residence, torch.zeros_like(residence)
        ),
    )
