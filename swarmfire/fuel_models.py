"""The standard fire behaviour fuel models, as a table and as batched tensors.

Anderson (1982) "Aids to Determining Fuel Models for Estimating Fire Behavior",
thirteen models numbered 1-13, and Scott & Burgan (2005) "Standard Fire Behavior
Fuel Models", forty models numbered 101-204. Both are USDA Forest Service
publications and the numbers below are theirs; 91-99 are the non-burnable
placeholders LANDFIRE uses for water, rock, urban and agriculture.

This is the table a LANDFIRE fuel-model raster indexes into, which is why
`RothermelROS` takes a per-cell model number rather than per-cell fuel-bed
parameters: the raster is the input real landscapes come in.

Units are Rothermel's, which are imperial, and are converted at the end of the
spread calculation rather than here - keeping them means every intermediate can
be checked against the published equations and against `pyretechnics` without a
conversion in the way.

`tests/test_fuel_models.py` checks every number in this table against
`pyretechnics`' copy when the `real` extra is installed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

__all__ = [
    "FUEL_MODELS",
    "FUEL_MODEL_NAMES",
    "N_SIZE_CLASSES",
    "DEAD_1H",
    "DEAD_10H",
    "DEAD_100H",
    "DEAD_HERB",
    "LIVE_HERB",
    "LIVE_WOODY",
    "FuelBedTable",
    "build_table",
    "is_burnable",
]

# The six fuel size classes every quantity below is indexed by. The first four
# are the dead category and the last two the live category; Rothermel weights
# within a category and then between categories, never across all six at once.
N_SIZE_CLASSES = 6
DEAD_1H, DEAD_10H, DEAD_100H, DEAD_HERB, LIVE_HERB, LIVE_WOODY = range(6)

# delta (ft), M_x_dead (%), h (Btu/lb / 1000),
# w_o (lb/ft^2) for dead 1h / 10h / 100h, live herbaceous, live woody,
# sigma (ft^2/ft^3) for dead 1h / 10h / 100h, live herbaceous, live woody.
#
# Dead-herbaceous is absent here: it has no load of its own and is filled at
# runtime from the live herbaceous load as that cures. See `RothermelROS`.
FUEL_MODELS: dict[int, tuple[float, ...]] = {
    # --- Anderson 13 ----------------------------------------------------------
    1:   (1.0, 12.0, 8.0, 0.0340, 0.0000, 0.0000, 0.0000, 0.0000, 3500.0,   0.0,  0.0,    0.0,    0.0),
    2:   (1.0, 15.0, 8.0, 0.0920, 0.0460, 0.0230, 0.0230, 0.0000, 3000.0, 109.0, 30.0, 1500.0,    0.0),
    3:   (2.5, 25.0, 8.0, 0.1380, 0.0000, 0.0000, 0.0000, 0.0000, 1500.0,   0.0,  0.0,    0.0,    0.0),
    4:   (6.0, 20.0, 8.0, 0.2300, 0.1840, 0.0920, 0.2300, 0.0000, 2000.0, 109.0, 30.0, 1500.0,    0.0),
    5:   (2.0, 20.0, 8.0, 0.0460, 0.0230, 0.0000, 0.0920, 0.0000, 2000.0, 109.0,  0.0, 1500.0,    0.0),
    6:   (2.5, 25.0, 8.0, 0.0690, 0.1150, 0.0920, 0.0000, 0.0000, 1750.0, 109.0, 30.0,    0.0,    0.0),
    7:   (2.5, 40.0, 8.0, 0.0520, 0.0860, 0.0690, 0.0170, 0.0000, 1750.0, 109.0, 30.0, 1550.0,    0.0),
    8:   (0.2, 30.0, 8.0, 0.0690, 0.0460, 0.1150, 0.0000, 0.0000, 2000.0, 109.0, 30.0,    0.0,    0.0),
    9:   (0.2, 25.0, 8.0, 0.1340, 0.0190, 0.0070, 0.0000, 0.0000, 2500.0, 109.0, 30.0,    0.0,    0.0),
    10:  (1.0, 25.0, 8.0, 0.1380, 0.0920, 0.2300, 0.0920, 0.0000, 2000.0, 109.0, 30.0, 1500.0,    0.0),
    11:  (1.0, 15.0, 8.0, 0.0690, 0.2070, 0.2530, 0.0000, 0.0000, 1500.0, 109.0, 30.0,    0.0,    0.0),
    12:  (2.3, 20.0, 8.0, 0.1840, 0.6440, 0.7590, 0.0000, 0.0000, 1500.0, 109.0, 30.0,    0.0,    0.0),
    13:  (3.0, 25.0, 8.0, 0.3220, 1.0580, 1.2880, 0.0000, 0.0000, 1500.0, 109.0, 30.0,    0.0,    0.0),
    # --- Non-burnable ---------------------------------------------------------
    91:  (0.0,  0.0, 0.0, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,    0.0,   0.0,  0.0,    0.0,    0.0),
    92:  (0.0,  0.0, 0.0, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,    0.0,   0.0,  0.0,    0.0,    0.0),
    93:  (0.0,  0.0, 0.0, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,    0.0,   0.0,  0.0,    0.0,    0.0),
    98:  (0.0,  0.0, 0.0, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,    0.0,   0.0,  0.0,    0.0,    0.0),
    99:  (0.0,  0.0, 0.0, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,    0.0,   0.0,  0.0,    0.0,    0.0),
    # --- Scott & Burgan 40: grass (GR) ----------------------------------------
    101: (0.4, 15.0, 8.0, 0.0046, 0.0000, 0.0000, 0.0138, 0.0000, 2200.0, 109.0, 30.0, 2000.0,    0.0),
    102: (1.0, 15.0, 8.0, 0.0046, 0.0000, 0.0000, 0.0459, 0.0000, 2000.0, 109.0, 30.0, 1800.0,    0.0),
    103: (2.0, 30.0, 8.0, 0.0046, 0.0184, 0.0000, 0.0689, 0.0000, 1500.0, 109.0, 30.0, 1300.0,    0.0),
    104: (2.0, 15.0, 8.0, 0.0115, 0.0000, 0.0000, 0.0872, 0.0000, 2000.0, 109.0, 30.0, 1800.0,    0.0),
    105: (1.5, 40.0, 8.0, 0.0184, 0.0000, 0.0000, 0.1148, 0.0000, 1800.0, 109.0, 30.0, 1600.0,    0.0),
    106: (1.5, 40.0, 9.0, 0.0046, 0.0000, 0.0000, 0.1561, 0.0000, 2200.0, 109.0, 30.0, 2000.0,    0.0),
    107: (3.0, 15.0, 8.0, 0.0459, 0.0000, 0.0000, 0.2479, 0.0000, 2000.0, 109.0, 30.0, 1800.0,    0.0),
    108: (4.0, 30.0, 8.0, 0.0230, 0.0459, 0.0000, 0.3352, 0.0000, 1500.0, 109.0, 30.0, 1300.0,    0.0),
    109: (5.0, 40.0, 8.0, 0.0459, 0.0459, 0.0000, 0.4132, 0.0000, 1800.0, 109.0, 30.0, 1600.0,    0.0),
    # --- Grass-shrub (GS) -----------------------------------------------------
    121: (0.9, 15.0, 8.0, 0.0092, 0.0000, 0.0000, 0.0230, 0.0298, 2000.0, 109.0, 30.0, 1800.0, 1800.0),
    122: (1.5, 15.0, 8.0, 0.0230, 0.0230, 0.0000, 0.0275, 0.0459, 2000.0, 109.0, 30.0, 1800.0, 1800.0),
    123: (1.8, 40.0, 8.0, 0.0138, 0.0115, 0.0000, 0.0666, 0.0574, 1800.0, 109.0, 30.0, 1600.0, 1600.0),
    124: (2.1, 40.0, 8.0, 0.0872, 0.0138, 0.0046, 0.1561, 0.3260, 1800.0, 109.0, 30.0, 1600.0, 1600.0),
    # --- Shrub (SH) -----------------------------------------------------------
    141: (1.0, 15.0, 8.0, 0.0115, 0.0115, 0.0000, 0.0069, 0.0597, 2000.0, 109.0, 30.0, 1800.0, 1600.0),
    142: (1.0, 15.0, 8.0, 0.0620, 0.1102, 0.0344, 0.0000, 0.1768, 2000.0, 109.0, 30.0,    0.0, 1600.0),
    143: (2.4, 40.0, 8.0, 0.0207, 0.1377, 0.0000, 0.0000, 0.2847, 1600.0, 109.0, 30.0,    0.0, 1400.0),
    144: (3.0, 30.0, 8.0, 0.0390, 0.0528, 0.0092, 0.0000, 0.1171, 2000.0, 109.0, 30.0, 1800.0, 1600.0),
    145: (6.0, 15.0, 8.0, 0.1653, 0.0964, 0.0000, 0.0000, 0.1331,  750.0, 109.0, 30.0,    0.0, 1600.0),
    146: (2.0, 30.0, 8.0, 0.1331, 0.0666, 0.0000, 0.0000, 0.0643,  750.0, 109.0, 30.0,    0.0, 1600.0),
    147: (6.0, 15.0, 8.0, 0.1607, 0.2433, 0.1010, 0.0000, 0.1561,  750.0, 109.0, 30.0,    0.0, 1600.0),
    148: (3.0, 40.0, 8.0, 0.0941, 0.1561, 0.0390, 0.0000, 0.1997,  750.0, 109.0, 30.0,    0.0, 1600.0),
    149: (4.4, 40.0, 8.0, 0.2066, 0.1125, 0.0000, 0.0712, 0.3214,  750.0, 109.0, 30.0, 1800.0, 1500.0),
    # --- Timber-understory (TU) -----------------------------------------------
    161: (0.6, 20.0, 8.0, 0.0092, 0.0413, 0.0689, 0.0092, 0.0413, 2000.0, 109.0, 30.0, 1800.0, 1600.0),
    162: (1.0, 30.0, 8.0, 0.0436, 0.0826, 0.0574, 0.0000, 0.0092, 2000.0, 109.0, 30.0,    0.0, 1600.0),
    163: (1.3, 30.0, 8.0, 0.0505, 0.0069, 0.0115, 0.0298, 0.0505, 1800.0, 109.0, 30.0, 1600.0, 1400.0),
    164: (0.5, 12.0, 8.0, 0.2066, 0.0000, 0.0000, 0.0000, 0.0918, 2300.0, 109.0, 30.0,    0.0, 2000.0),
    165: (1.0, 25.0, 8.0, 0.1837, 0.1837, 0.1377, 0.0000, 0.1377, 1500.0, 109.0, 30.0,    0.0,  750.0),
    # --- Timber litter (TL) ---------------------------------------------------
    181: (0.2, 30.0, 8.0, 0.0459, 0.1010, 0.1653, 0.0000, 0.0000, 2000.0, 109.0, 30.0,    0.0,    0.0),
    182: (0.2, 25.0, 8.0, 0.0643, 0.1056, 0.1010, 0.0000, 0.0000, 2000.0, 109.0, 30.0,    0.0,    0.0),
    183: (0.3, 20.0, 8.0, 0.0230, 0.1010, 0.1286, 0.0000, 0.0000, 2000.0, 109.0, 30.0,    0.0,    0.0),
    184: (0.4, 25.0, 8.0, 0.0230, 0.0689, 0.1928, 0.0000, 0.0000, 2000.0, 109.0, 30.0,    0.0,    0.0),
    185: (0.6, 25.0, 8.0, 0.0528, 0.1148, 0.2020, 0.0000, 0.0000, 2000.0, 109.0, 30.0,    0.0, 1600.0),
    186: (0.3, 25.0, 8.0, 0.1102, 0.0551, 0.0551, 0.0000, 0.0000, 2000.0, 109.0, 30.0,    0.0,    0.0),
    187: (0.4, 25.0, 8.0, 0.0138, 0.0643, 0.3719, 0.0000, 0.0000, 2000.0, 109.0, 30.0,    0.0,    0.0),
    188: (0.3, 35.0, 8.0, 0.2663, 0.0643, 0.0505, 0.0000, 0.0000, 1800.0, 109.0, 30.0,    0.0,    0.0),
    189: (0.6, 35.0, 8.0, 0.3053, 0.1515, 0.1905, 0.0000, 0.0000, 1800.0, 109.0, 30.0,    0.0, 1600.0),
    # --- Slash-blowdown (SB) --------------------------------------------------
    201: (1.0, 25.0, 8.0, 0.0689, 0.1377, 0.5051, 0.0000, 0.0000, 2000.0, 109.0, 30.0,    0.0,    0.0),
    202: (1.0, 25.0, 8.0, 0.2066, 0.1951, 0.1837, 0.0000, 0.0000, 2000.0, 109.0, 30.0,    0.0,    0.0),
    203: (1.2, 25.0, 8.0, 0.2525, 0.1263, 0.1377, 0.0000, 0.0000, 2000.0, 109.0, 30.0,    0.0,    0.0),
    204: (2.7, 25.0, 8.0, 0.2410, 0.1607, 0.2410, 0.0000, 0.0000, 2000.0, 109.0, 30.0,    0.0,    0.0),
}

FUEL_MODEL_NAMES: dict[int, str] = {
    **{n: f"R{n:02d}" for n in range(1, 14)},
    91: "NB1", 92: "NB2", 93: "NB3", 98: "NB4", 99: "NB5",
    **{100 + i: f"GR{i}" for i in range(1, 10)},
    **{120 + i: f"GS{i}" for i in range(1, 5)},
    **{140 + i: f"SH{i}" for i in range(1, 10)},
    **{160 + i: f"TU{i}" for i in range(1, 6)},
    **{180 + i: f"TL{i}" for i in range(1, 10)},
    **{200 + i: f"SB{i}" for i in range(1, 5)},
}

# Constants Rothermel treats as the same for every particle in every model.
PARTICLE_DENSITY = 32.0  # rho_p, lb/ft^3
TOTAL_MINERAL = 0.0555  # S_T, lb minerals / lb ovendry
EFFECTIVE_MINERAL = 0.01  # S_e, lb silica-free minerals / lb ovendry

# Below this a load or a surface-area-to-volume ratio counts as absent. Matches
# the tolerance `pyretechnics` uses, so the two agree on empty size classes.
ALMOST_ZERO = 1e-6


def is_burnable(number: int) -> bool:
    """LANDFIRE reserves 91-99 for water, rock, urban, agriculture and snow."""
    return not (91 <= number <= 99)


def _firemod_size_class(sigma: float) -> float:
    """Albini (1976) FIREMOD size class, page 20. Used to group `g_ij`."""
    for threshold, klass in ((1200.0, 1.0), (192.0, 2.0), (96.0, 3.0), (48.0, 4.0), (16.0, 5.0)):
        if sigma >= threshold:
            return klass
    return 6.0


@dataclass(frozen=True)
class FuelBedTable:
    """Every fuel model as dense tensors, indexed by a contiguous slot.

    A `[B, H, W]` raster of model numbers becomes a raster of slots through
    `slots_for`, and every per-cell fuel-bed quantity is then one `index_select`
    away. That is the shape LANDFIRE data arrives in and the shape a GPU wants.

    Per-slot, per-size-class (`[M, 6]`): `w_o`, `sigma`, `h`, `m_x`,
    `exp_a_sigma`. Per-slot (`[M]`): `delta`, `dynamic`, `burnable`.
    `size_class_match` is `[M, 6, 6]`, true where two classes in the same
    category share a FIREMOD size class - the grouping `g_ij` sums over.
    """

    numbers: Tensor
    delta: Tensor
    w_o: Tensor
    sigma: Tensor
    h: Tensor
    m_x: Tensor
    exp_a_sigma: Tensor
    size_class_match: Tensor
    dynamic: Tensor
    burnable: Tensor
    _slot_of: Tensor  # [max_number + 1] lookup, -1 where no model exists

    @property
    def device(self) -> torch.device:
        return self.delta.device

    def slots_for(self, numbers: Tensor) -> Tensor:
        """Map a raster of fuel model numbers to table slots.

        Unknown numbers map to slot 0, which this table always makes a
        non-burnable model, so an unrecognised code behaves as bare ground
        rather than raising in the middle of a rollout.
        """
        index = numbers.round().long()
        # Two ways to miss: a number outside the table's range, and a gap
        # inside it (there is no fuel model 50). Both go to slot 0.
        in_range = (index >= 0) & (index < self._slot_of.numel())
        slot = self._slot_of[index.clamp(0, self._slot_of.numel() - 1)]
        return torch.where(in_range, slot, torch.zeros_like(slot)).clamp(min=0)

    def to(self, device) -> "FuelBedTable":
        return FuelBedTable(
            **{k: v.to(device) for k, v in self.__dict__.items() if isinstance(v, Tensor)}
        )


def build_table(device: str | torch.device = "cpu", dtype=torch.float32) -> FuelBedTable:
    """Expand `FUEL_MODELS` into the tensors `RothermelROS` gathers from.

    Slot 0 is always a non-burnable model, so that an unknown fuel-model number
    lands somewhere harmless. Everything here is a function of the fuel model
    alone - nothing moisture-dependent is precomputed, because the dynamic load
    transfer moves fuel between size classes as the herbaceous layer cures.
    """
    numbers = [91] + sorted(FUEL_MODELS)  # slot 0 is non-burnable by construction
    n = len(numbers)

    delta = torch.zeros(n, dtype=dtype)
    w_o = torch.zeros(n, N_SIZE_CLASSES, dtype=dtype)
    sigma = torch.zeros(n, N_SIZE_CLASSES, dtype=dtype)
    h = torch.zeros(n, N_SIZE_CLASSES, dtype=dtype)
    m_x = torch.zeros(n, N_SIZE_CLASSES, dtype=dtype)
    exp_a_sigma = torch.zeros(n, N_SIZE_CLASSES, dtype=dtype)
    size_class_match = torch.zeros(n, N_SIZE_CLASSES, N_SIZE_CLASSES, dtype=torch.bool)
    dynamic = torch.zeros(n, dtype=torch.bool)
    burnable = torch.zeros(n, dtype=torch.bool)

    for slot, number in enumerate(numbers):
        row = FUEL_MODELS[number]
        d, m_x_pct, h_k = row[0], row[1], row[2]
        loads = row[3:8]  # dead 1h, 10h, 100h, live herb, live woody
        sigmas = row[8:13]

        m_x_dead = m_x_pct * 0.01
        heat = h_k * 1000.0

        # A model is dynamic when it carries a herbaceous load that can cure
        # into the dead-herbaceous class. Only the Scott & Burgan set does.
        is_dynamic = number > 100 and loads[3] > ALMOST_ZERO
        sigma_dead_herb = sigmas[3] if is_dynamic else 0.0

        delta[slot] = d
        burnable[slot] = is_burnable(number)
        dynamic[slot] = is_dynamic
        w_o[slot] = torch.tensor(
            [loads[0], loads[1], loads[2], 0.0, loads[3], loads[4]], dtype=dtype
        )
        per_class_sigma = [sigmas[0], sigmas[1], sigmas[2], sigma_dead_herb, sigmas[3], sigmas[4]]
        sigma[slot] = torch.tensor(per_class_sigma, dtype=dtype)
        h[slot] = heat
        # Live classes get their moisture of extinction at runtime; it depends
        # on how wet the dead fuel is, which is weather, not fuel.
        m_x[slot] = torch.tensor(
            [m_x_dead, m_x_dead, m_x_dead, m_x_dead if is_dynamic else 0.0, 0.0, 0.0], dtype=dtype
        )
        # Rothermel's A is -138 for dead classes and -500 for live ones.
        for j, s in enumerate(per_class_sigma):
            a = -138.0 if j < LIVE_HERB else -500.0
            exp_a_sigma[slot, j] = math.exp(a / s) if s > ALMOST_ZERO else 0.0

        classes = [_firemod_size_class(s) for s in per_class_sigma]
        for j in range(N_SIZE_CLASSES):
            for k in range(N_SIZE_CLASSES):
                same_category = (j < LIVE_HERB) == (k < LIVE_HERB)
                size_class_match[slot, j, k] = same_category and classes[j] == classes[k]

    slot_of = torch.full((max(numbers) + 1,), -1, dtype=torch.long)
    for slot, number in enumerate(numbers):
        # `numbers` starts with a duplicate 91; the real entry wins.
        if slot_of[number] < 0 or number != 91:
            slot_of[number] = slot

    table = FuelBedTable(
        numbers=torch.tensor(numbers, dtype=dtype),
        delta=delta,
        w_o=w_o,
        sigma=sigma,
        h=h,
        m_x=m_x,
        exp_a_sigma=exp_a_sigma,
        size_class_match=size_class_match,
        dynamic=dynamic,
        burnable=burnable,
        _slot_of=slot_of,
    )
    return table.to(device)
