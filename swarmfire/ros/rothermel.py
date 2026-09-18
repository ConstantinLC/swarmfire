"""Rothermel (1972) surface fire spread, batched and differentiable.

The realistic-physics slot, filled. This is the same model BehavePlus, FARSITE
and `pyretechnics` implement, and `tests/test_rothermel.py` checks it against
`pyretechnics` cell by cell across every fuel model, a range of moistures, winds
and slopes.

It is closed-form algebra - no solver, no iteration - so the whole thing is
elementwise tensor math over `[B, H, W]` and differentiates as written:

    R = I_R * xi * (1 + phi_w + phi_s) / (rho_b * epsilon * Q_ig)

reaction intensity times propagating flux ratio, scaled by the wind and slope
factors, over the heat it takes to bring the fuel ahead of the front to ignition.

Units
-----
Rothermel's equations are imperial and every published check of them is in
imperial, so the intermediates here are too: loads in lb/ft^2, surface-area-to-
volume in ft^2/ft^3, depth in ft, heat content in Btu/lb, spread in ft/min.
The conversion to m/s happens once, at the end. Moisture is a dry-weight
fraction either way.

What it needs from the state
----------------------------
`FireState.fuel_model`, a per-cell fuel model number indexing
`fuel_models.FUEL_MODELS` - which is exactly what a LANDFIRE fuel-model raster
contains - and `FireState.moisture` as dead 1-hour fuel moisture. That last is
the layer `SuppressionModel` raises, so a water drop reaches spread rate through
Rothermel's own moisture damping term rather than through a bolted-on factor.

The remaining four moisture classes are weather rather than fuel, near-uniform
over a fire-sized domain, and are constructor arguments here. They become
rasters when there is a weather model to vary them.

What it deliberately leaves out
-------------------------------
Crown fire, spotting and fire-atmosphere feedback, exactly as before. This is a
surface spread model and nothing about filling in this slot changes that.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from ..fuel_models import (
    DEAD_1H,
    DEAD_HERB,
    EFFECTIVE_MINERAL,
    LIVE_HERB,
    LIVE_WOODY,
    N_SIZE_CLASSES,
    PARTICLE_DENSITY,
    TOTAL_MINERAL,
    FuelBedTable,
    build_table,
)
from ..grid import NEIGHBOR_OFFSETS, slope_vector
from ..state import FireState
from .base import ROSModel
from .simple import length_to_breadth

# Rothermel works in feet and minutes; the package works in metres and seconds.
FT_PER_M = 3.2808398950131235
EPS = 1e-12


def _safe_div(numerator: Tensor, denominator: Tensor) -> Tensor:
    """`numerator / denominator`, zero where the denominator is.

    Rothermel is full of ratios that are genuinely zero over non-burnable
    ground - no fuel load, no bulk density, no surface area - and a plain
    division there produces NaN in the forward pass and, worse, NaN gradients
    that survive `torch.where`. Masking the denominator before dividing keeps
    both finite.
    """
    safe = torch.where(denominator.abs() > EPS, denominator, torch.ones_like(denominator))
    return torch.where(denominator.abs() > EPS, numerator / safe, torch.zeros_like(numerator))


def _safe_pow(base: Tensor, exponent: Tensor | float) -> Tensor:
    """`base ** exponent` for a non-negative base, with a finite gradient at zero."""
    return torch.where(base > EPS, base.clamp(min=EPS) ** exponent, torch.zeros_like(base))


class RothermelROS(ROSModel):
    """Rothermel (1972) surface spread with the Albini/Anderson ellipse.

    Parameters
    ----------
    moisture_10h, moisture_100h : dead fuel moisture for the coarser size
        classes, dry-weight fraction. Dead 1-hour comes from
        `FireState.moisture` per cell.
    moisture_live_herb, moisture_live_woody : live fuel moisture. The
        herbaceous value also drives the dynamic load transfer - below 30% the
        herbaceous layer is fully cured and its load moves into the dead
        category, which is what makes a grass model dangerous in late summer.
    use_wind_limit : cap the effective wind at Rothermel's `0.9 * I_R`, above
        which the model stops believing its own wind factor. On by default,
        matching BehavePlus and `pyretechnics`. `SimpleROS` has no equivalent
        and accelerates through it.
    max_length_to_breadth : cap on the fire ellipse's length/breadth ratio.
        Anderson (1983) is not supported past 8 and BehavePlus holds it there,
        so this defaults to 8 rather than to a number chosen for numerics. It
        matters more than it looks: the ratio enters as an eccentricity, and
        between 7.1 and 8 the backing rate - which goes as `1 - e` - changes by
        a quarter.
    spread_rate_adjustment : the per-fuel tuning factor operational setups apply
        to match observed spread. 1.0 is unadjusted Rothermel.
    table : a prebuilt `FuelBedTable`, if you have one on the right device.
    """

    def __init__(
        self,
        moisture_10h: float = 0.07,
        moisture_100h: float = 0.08,
        moisture_live_herb: float = 0.60,
        moisture_live_woody: float = 0.90,
        use_wind_limit: bool = True,
        max_length_to_breadth: float = 8.0,
        spread_rate_adjustment: float = 1.0,
        table: FuelBedTable | None = None,
    ):
        self.moisture_10h = moisture_10h
        self.moisture_100h = moisture_100h
        self.moisture_live_herb = moisture_live_herb
        self.moisture_live_woody = moisture_live_woody
        self.use_wind_limit = use_wind_limit
        self.max_length_to_breadth = max_length_to_breadth
        self.spread_rate_adjustment = spread_rate_adjustment
        self._table = table

    def table(self, device) -> FuelBedTable:
        """The fuel model tensors, built once and cached per device."""
        if self._table is None or self._table.device != torch.device(device):
            self._table = build_table(device)
        return self._table

    # ------------------------------------------------------------ fuel bed

    def _gather(self, state: FireState) -> dict[str, Tensor]:
        """Per-cell fuel bed parameters, `[B, H, W, 6]` unless noted.

        One `index_select` per quantity off the fuel model raster. Nothing here
        depends on moisture; that arrives in `_moisturize`.
        """
        table = self.table(state.device)
        slots = table.slots_for(state.fuel_model)
        flat = slots.reshape(-1)

        def per_class(source: Tensor) -> Tensor:
            return source.index_select(0, flat).reshape(*slots.shape, N_SIZE_CLASSES)

        return {
            "w_o": per_class(table.w_o),
            "sigma": per_class(table.sigma),
            "h": per_class(table.h),
            "m_x": per_class(table.m_x),
            "exp_a_sigma": per_class(table.exp_a_sigma),
            "size_class_match": table.size_class_match.index_select(0, flat).reshape(
                *slots.shape, N_SIZE_CLASSES, N_SIZE_CLASSES
            ),
            "delta": table.delta.index_select(0, flat).reshape(slots.shape),
            "dynamic": table.dynamic.index_select(0, flat).reshape(slots.shape),
            "burnable": table.burnable.index_select(0, flat).reshape(slots.shape),
        }

    def _moisture(self, state: FireState) -> Tensor:
        """`[B, H, W, 6]` fuel moisture by size class, dry-weight fraction.

        Dead 1-hour is the per-cell layer suppression acts on. Dead herbaceous
        takes the dead 1-hour value, since cured herbaceous fuel is fine dead
        fuel and dries at the same rate.
        """
        dead_1h = state.moisture
        like = torch.ones_like(dead_1h)
        return torch.stack(
            [
                dead_1h,
                like * self.moisture_10h,
                like * self.moisture_100h,
                dead_1h,
                like * self.moisture_live_herb,
                like * self.moisture_live_woody,
            ],
            dim=-1,
        )

    def _cure(self, bed: dict[str, Tensor]) -> Tensor:
        """Move cured herbaceous load from the live category to the dead one.

        Scott & Burgan's dynamic models carry their herbaceous fuel as live, and
        transfer it to dead as it cures. Fully green above 90% moisture, fully
        cured at or below 30%, linear between. This is the single largest
        seasonal effect in a grass model and it is why `moisture_live_herb`
        matters as much as the dead moisture does.
        """
        w_o = bed["w_o"]
        herb_load = w_o[..., LIVE_HERB]
        green = (self.moisture_live_herb / 0.9 - 1.0 / 3.0)
        green = min(max(green, 0.0), 1.0)

        cured = w_o.clone()
        is_dynamic = bed["dynamic"].unsqueeze(-1)
        transferred = torch.stack(
            [
                w_o[..., 0],
                w_o[..., 1],
                w_o[..., 2],
                herb_load * (1.0 - green),
                herb_load * green,
                w_o[..., 5],
            ],
            dim=-1,
        )
        return torch.where(is_dynamic, transferred, cured)

    def _weights(self, w_o: Tensor, sigma: Tensor, match: Tensor):
        """Rothermel's surface-area weighting factors `f_ij`, `f_i` and `g_ij`.

        Every average in the model is area-weighted, not load-weighted: a
        kilogram of fine grass presents far more burning surface than a kilogram
        of branchwood and dominates the fuel bed accordingly. `f_ij` weights
        within a category, `f_i` between the dead and live categories, and
        `g_ij` regroups by FIREMOD size class for the net-load term.
        """
        area = _safe_div(sigma * w_o, torch.full_like(w_o, PARTICLE_DENSITY))
        dead_area = area[..., :LIVE_HERB].sum(-1, keepdim=True)
        live_area = area[..., LIVE_HERB:].sum(-1, keepdim=True)

        f_ij = torch.cat(
            [
                _safe_div(area[..., :LIVE_HERB], dead_area.expand(-1, -1, -1, LIVE_HERB)),
                _safe_div(area[..., LIVE_HERB:], live_area.expand(-1, -1, -1, 2)),
            ],
            dim=-1,
        )
        total = dead_area + live_area
        f_i = torch.cat([_safe_div(dead_area, total), _safe_div(live_area, total)], dim=-1)
        # g_ij sums the f_ij of every class sharing this one's size class.
        g_ij = torch.einsum("...jk,...k->...j", match.to(f_ij.dtype), f_ij)
        return f_ij, f_i, g_ij

    @staticmethod
    def _by_category(f_ij: Tensor, values: Tensor) -> Tensor:
        """`[..., 2]` area-weighted mean of `values` within each category."""
        weighted = f_ij * values
        return torch.stack(
            [weighted[..., :LIVE_HERB].sum(-1), weighted[..., LIVE_HERB:].sum(-1)], dim=-1
        )

    def _live_moisture_of_extinction(self, bed, w_o, m_f, m_x) -> Tensor:
        """Rothermel eq. 88 with Albini (1976) App. III: live `M_x` from dead dryness.

        Live fuel has no fixed moisture of extinction. What it has is a
        threshold that falls as the dead fuel around it dries out - green fuel
        burns when the dead fuel beneath it is dry enough to carry the fire into
        it. This is the term that makes a fire in shrub suddenly viable.
        """
        loading = w_o * bed["exp_a_sigma"]
        dead_loading = loading[..., :LIVE_HERB].sum(-1)
        live_loading = loading[..., LIVE_HERB:].sum(-1)
        dead_moisture = (m_f[..., :LIVE_HERB] * loading[..., :LIVE_HERB]).sum(-1)

        m_x_dead = m_x[..., DEAD_1H]
        dead_fuel_moisture = _safe_div(dead_moisture, dead_loading)
        dead_to_live = _safe_div(dead_loading, live_loading)
        candidate = 2.9 * dead_to_live * (1.0 - _safe_div(dead_fuel_moisture, m_x_dead)) - 0.226
        m_x_live = torch.maximum(m_x_dead, candidate)

        has_both = (dead_loading > EPS) & (live_loading > EPS)
        m_x_live = torch.where(has_both, m_x_live, m_x_dead)
        return torch.cat(
            [m_x[..., :LIVE_HERB], m_x_live.unsqueeze(-1).expand(-1, -1, -1, 2)], dim=-1
        )

    # --------------------------------------------------------- the spread rate

    def no_wind_no_slope(self, state: FireState) -> dict[str, Tensor]:
        """Everything that does not depend on wind or slope.

        Returns the base spread rate in m/s along with the wind and slope factor
        coefficients, the effective-wind cap, reaction intensity and residence
        time - the quantities `fuels.rothermel_world` needs to fill in `ros0`,
        `burn_rate` and `moisture_ext` consistently with this model.
        """
        bed = self._gather(state)
        m_f = self._moisture(state)
        w_o = self._cure(bed)
        sigma, h = bed["sigma"], bed["h"]

        f_ij, f_i, g_ij = self._weights(w_o, sigma, bed["size_class_match"])
        m_x = self._live_moisture_of_extinction(bed, w_o, m_f, bed["m_x"])

        # Fuel bed characteristics.
        sigma_prime = (f_i * self._by_category(f_ij, sigma)).sum(-1)
        packing = _safe_div(
            (w_o / PARTICLE_DENSITY)[..., :LIVE_HERB].sum(-1)
            + (w_o / PARTICLE_DENSITY)[..., LIVE_HERB:].sum(-1),
            bed["delta"],
        )
        optimum_packing = torch.where(
            sigma_prime > EPS, 3.348 * _safe_pow(sigma_prime, -0.8189), torch.ones_like(sigma_prime)
        )

        # Reaction intensity: how much heat the fuel bed releases per unit area
        # per unit time, damped for mineral content and for how wet it is.
        s_e = self._by_category(f_ij, torch.full_like(w_o, EFFECTIVE_MINERAL))
        eta_s = torch.where(s_e > EPS, 0.174 * _safe_pow(s_e, -0.19), torch.ones_like(s_e))

        m_f_i = self._by_category(f_ij, m_f)
        m_x_i = self._by_category(f_ij, m_x)
        ratio = _safe_div(m_f_i, m_x_i).clamp(max=1.0)
        eta_m = 1.0 - 2.59 * ratio + 5.11 * ratio**2 - 3.52 * ratio**3
        eta_m = torch.where(m_x_i > EPS, eta_m, torch.zeros_like(eta_m))

        h_i = self._by_category(f_ij, h)
        net_load = torch.stack(
            [
                (g_ij[..., :LIVE_HERB] * w_o[..., :LIVE_HERB] * (1.0 - TOTAL_MINERAL)).sum(-1),
                (g_ij[..., LIVE_HERB:] * w_o[..., LIVE_HERB:] * (1.0 - TOTAL_MINERAL)).sum(-1),
            ],
            dim=-1,
        )
        heat_per_area = (net_load * h_i * eta_m * eta_s).sum(-1)

        a_exp = torch.where(
            sigma_prime > EPS, 133.0 * _safe_pow(sigma_prime, -0.7913), torch.zeros_like(sigma_prime)
        )
        sigma_15 = _safe_pow(sigma_prime, 1.5)
        gamma_max = _safe_div(sigma_15, 495.0 + 0.0594 * sigma_15)
        packing_ratio = _safe_div(packing, optimum_packing)
        gamma = gamma_max * _safe_pow(packing_ratio, a_exp) * torch.exp(a_exp * (1.0 - packing_ratio))
        reaction_intensity = heat_per_area * gamma  # Btu/ft^2/min

        # Propagating flux ratio: the share of that heat that actually reaches
        # the fuel ahead of the front.
        xi = _safe_div(
            torch.exp((0.792 + 0.681 * _safe_pow(sigma_prime, 0.5)) * (packing + 0.1)),
            192.0 + 0.2595 * sigma_prime,
        )

        # Heat sink: what it costs to bring that fuel to ignition.
        bulk_density = _safe_div(w_o.sum(-1), bed["delta"])
        epsilon = torch.where(
            sigma > EPS, torch.exp(-138.0 / sigma.clamp(min=EPS)), torch.zeros_like(sigma)
        )
        q_ig = torch.where(m_f > 0.0, 250.0 + 1116.0 * m_f, torch.zeros_like(m_f))
        heat_sink = bulk_density * (f_i * self._by_category(f_ij, epsilon * q_ig)).sum(-1)

        base_ft_min = _safe_div(reaction_intensity * xi, heat_sink) * self.spread_rate_adjustment
        burnable = bed["burnable"]
        base_ft_min = torch.where(burnable, base_ft_min, torch.zeros_like(base_ft_min))

        # Wind and slope factor coefficients. phi_w = C (beta/beta_op)^-E U^B
        # with U in ft/min, and phi_s = 5.275 beta^-0.3 tan^2. Both are the
        # algebraic forms `SimpleROS` approximates with fixed constants.
        b = 0.02526 * _safe_pow(sigma_prime, 0.54)
        c = 7.47 * torch.exp(-0.133 * _safe_pow(sigma_prime, 0.55))
        e = 0.715 * torch.exp(-3.59 * (sigma_prime * 1e-4))
        f = _safe_pow(packing_ratio, e)

        wind_scalar = _safe_div(c, f) * _safe_pow(torch.full_like(b, FT_PER_M), b)
        slope_factor = torch.where(
            packing > EPS, 5.275 * _safe_pow(packing, -0.3), torch.zeros_like(packing)
        )

        return {
            "base_ros": base_ft_min * 0.3048 / 60.0,  # m/s
            "wind_scalar": wind_scalar,  # phi_w = wind_scalar * U**wind_exponent, U in m/min
            "wind_exponent": b,
            "slope_factor": slope_factor,
            "max_effective_wind": 0.9 * reaction_intensity * 0.3048 / 60.0,  # m/s
            "reaction_intensity": reaction_intensity,
            "residence_s": _safe_div(torch.full_like(sigma_prime, 384.0), sigma_prime) * 60.0,
            "moisture_ext": bed["m_x"][..., DEAD_1H],
            "burnable": burnable,
        }

    def _drive(self, state: FireState, base: dict[str, Tensor]):
        """Combine wind and slope into one forcing, on the slope-tangential plane.

        Wind and slope are added as vectors, not as scalars, so a fire pushed
        north by wind and east by terrain runs northeast. Both are lifted onto
        the plane the terrain actually presents before they are added - a wind
        blowing up a hillside pushes along the hillside, not through it - which
        is what `pyretechnics` does and what makes the two agree on sloped
        ground rather than to within a cosine.

        Returns the forcing magnitude `phi_E`, the effective wind that alone
        would produce it, the unit heading as a 3D vector on the slope plane,
        and the terrain gradient.
        """
        b, h, w = state.batch, *state.grid
        grad = slope_vector(state.elevation, state.cell_size)  # [B,2,H,W], upslope
        gx, gy = grad[:, 0], grad[:, 1]

        wind = state.wind.view(b, 2, 1, 1)
        wx = wind[:, 0].expand(b, h, w)
        wy = wind[:, 1].expand(b, h, w)

        # Lift onto the slope plane: z is the rise the in-plane step makes.
        wind_3d = torch.stack([wx, wy, wx * gx + wy * gy], dim=1)
        slope_3d = torch.stack([gx, gy, gx * gx + gy * gy], dim=1)

        # vector_norm rather than sqrt-of-sum-of-squares: at zero wind on flat
        # ground every component is exactly zero, where sqrt has an infinite
        # derivative and an epsilon under the root leaks a spurious forcing into
        # the ellipse (a tiny phi_E becomes a not-tiny eccentricity, because the
        # square root amplifies it near 1).
        wind_speed = torch.linalg.vector_norm(wind_3d, dim=1)
        slope_len = torch.linalg.vector_norm(slope_3d, dim=1)

        phi_w = base["wind_scalar"] * _safe_pow(wind_speed * 60.0, base["wind_exponent"])
        phi_s = base["slope_factor"] * (gx * gx + gy * gy)

        wind_unit = _safe_div(wind_3d, wind_speed.unsqueeze(1).expand_as(wind_3d))
        slope_unit = _safe_div(slope_3d, slope_len.unsqueeze(1).expand_as(slope_3d))
        phi_e_3d = phi_w.unsqueeze(1) * wind_unit + phi_s.unsqueeze(1) * slope_unit
        phi_e = torch.linalg.vector_norm(phi_e_3d, dim=1)

        # Heading first, from the *uncapped* magnitude. The wind limit below
        # shrinks phi_E without rotating it, so normalising by the capped value
        # would leave a heading longer than one - and a unit vector that is not
        # one puts cos(offset) above 1, which sends the ellipse denominator
        # through zero and the spread rate to six figures.
        heading = _safe_div(phi_e_3d, phi_e.unsqueeze(1).expand_as(phi_e_3d))

        # Rothermel stops believing the wind factor past 0.9 * I_R. Invert phi_E
        # to the wind that alone would produce it, cap that, and convert back.
        inv_b = _safe_div(torch.ones_like(base["wind_exponent"]), base["wind_exponent"])
        effective_wind = _safe_pow(_safe_div(phi_e, base["wind_scalar"]), inv_b) / 60.0  # m/s
        if self.use_wind_limit:
            cap = base["max_effective_wind"]
            over = effective_wind > cap
            capped = base["wind_scalar"] * _safe_pow(cap * 60.0, base["wind_exponent"])
            phi_e = torch.where(over, capped, phi_e)
            effective_wind = torch.where(over, cap, effective_wind)
        north = torch.zeros_like(heading)
        north[:, 1] = 1.0
        heading = torch.where((phi_e > EPS).unsqueeze(1), heading, slope_unit)
        heading = torch.where((slope_len > EPS).unsqueeze(1) | (phi_e > EPS).unsqueeze(1), heading, north)
        return phi_e, effective_wind, heading, grad

    def __call__(self, state: FireState) -> Tensor:
        base = self.no_wind_no_slope(state)
        phi_e, effective_wind, heading, grad = self._drive(state, base)
        gx, gy = grad[:, 0], grad[:, 1]

        head_ros = base["base_ros"] * (1.0 + phi_e)

        lb = length_to_breadth(effective_wind).clamp(max=self.max_length_to_breadth)
        ecc = torch.sqrt((1.0 - 1.0 / (lb * lb)).clamp(min=0))

        # The offset angle is measured on the slope-tangential plane, not in map
        # projection: each neighbour direction is lifted onto the terrain before
        # being compared with the heading. On flat ground the two are the same;
        # on a 50% grade they differ by more than 10%.
        #
        # Note what this does *not* do: the rates come back as travel rates over
        # the slope plane, while `grid.neighbor_distances` measures map
        # distance. On steep ground the front therefore crosses a cell slightly
        # sooner than the ground distance warrants, by one over the cosine of
        # the slope. That is a property of running a cellular automaton on a map
        # projection, and it is the same approximation `SimpleROS` makes.
        cos_off = []
        for dy, dx in NEIGHBOR_OFFSETS:
            norm = math.hypot(dy, dx)
            ux, uy = dx / norm, dy / norm
            uz = ux * gx + uy * gy
            direction = torch.stack([torch.full_like(uz, ux), torch.full_like(uz, uy), uz], dim=1)
            direction = _safe_div(
                direction, torch.linalg.vector_norm(direction, dim=1, keepdim=True).expand_as(direction)
            )
            # Clamped because both vectors are unit only to float precision,
            # and the ellipse denominator is unforgiving just past 1.
            cos_off.append((direction * heading).sum(dim=1).clamp(-1.0, 1.0))
        cos_off = torch.stack(cos_off, dim=1)

        ecc = ecc.unsqueeze(1)
        return head_ros.unsqueeze(1) * (1.0 - ecc) / (1.0 - ecc * cos_off).clamp(min=1e-6)
