"""How far is this simulator from the reference implementation, and why.

Measures `swarmfire` against `pyretechnics`, which implements Rothermel (1972)
surface spread with an Eulerian level-set front tracker. Four modes:

    coefficients  Rothermel's own wind/slope/fuel parameters next to the ones
                  `SimpleROS` and `fuels.PRESETS` hard-code. Costs nothing and
                  explains most of what the other modes find.
    ros           head rate of spread against midflame wind and against slope,
                  plus the ellipse shape. Point physics only - no propagation.
    front         the propagator on trial. Both codes are given the *same*
                  directional rate of spread (see `validation.matched_world`),
                  so any disagreement is front tracking rather than physics.
    residence     front speed against flame residence time and against grid
                  resolution - two things it must not depend on, and used to.
    area          the bottom line: burned area after two hours, package as
                  shipped against the reference, with nothing matched.

    python scripts/validate_pyretechnics.py all --device cuda
    python scripts/validate_pyretechnics.py front --wind 3 --plot

Requires the reference: pip install "swarmfire[real]".
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from swarmfire import PRESETS, CAPropagator, SimpleROS, uniform_world, validation as V
from swarmfire.ros.simple import length_to_breadth

OUT = Path(__file__).resolve().parent.parent / "out"

WINDS = (0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0)
SLOPES = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7)


def _rate(value: float) -> str:
    """Render a measured spread rate, naming a stalled front as such.

    `axis_spread_rates` returns NaN when it cannot fit a front position against
    time, which in this harness means only one thing: the front never crossed
    the arrival threshold and stopped where it was.
    """
    return "  stalled" if not np.isfinite(value) else f"{value:8.4f}"


def _rule(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


# ---------------------------------------------------------------- coefficients


def coefficients(args) -> None:
    """Term-by-term: what Rothermel says these numbers should be."""
    _rule("Rothermel coefficients vs the hard-coded ones")
    print(
        "SimpleROS has the same algebraic form as Rothermel - phi_w = a*U^b and\n"
        "phi_s = c*tan^2 - so the two are comparable term by term, not only\n"
        "end to end. `a`, `b` and `c` are fuel-dependent in Rothermel and\n"
        "constant in SimpleROS, which is the root of everything below.\n"
    )
    header = f"{'preset':8} {'fuel':6} {'ros0 m/s':>18} {'wind a':>14} {'wind b':>13} {'slope c':>14} {'M_x':>12} {'residence s':>16}"
    print(header)
    for preset, fuel_model in V.FUEL_EQUIVALENTS.items():
        p = PRESETS[preset]
        c = V.rothermel_coefficients(fuel_model, args.moisture)
        fm_name = f"FM{fuel_model}"
        print(
            f"{preset:8} {fm_name:6}"
            f" {p.ros0:8.4f} /{c.ros0:8.4f}"
            f" {2.5:6.2f} /{c.wind_a:6.2f}"
            f" {1.5:5.2f} /{c.wind_b:5.2f}"
            f" {5.5:6.1f} /{c.slope_c:6.1f}"
            f" {p.moisture_ext:5.2f} /{c.moisture_ext:5.2f}"
            f" {p.residence:7.1f} /{c.residence_s:7.1f}"
        )
    print("\n(left: swarmfire, right: Rothermel for the equivalent Scott & Burgan model)")
    for preset, fuel_model in V.FUEL_EQUIVALENTS.items():
        c = V.rothermel_coefficients(fuel_model, args.moisture)
        print(
            f"  {preset:8} effective-wind cap {c.max_effective_wind:5.2f} m/s"
            "  - SimpleROS has no equivalent and accelerates past it"
        )


# ------------------------------------------------------------------ point physics


def ros(args) -> None:
    """Head rate of spread and fire shape, with no propagation involved."""
    rows: dict[str, np.ndarray] = {}

    _rule("Head rate of spread vs midflame wind, flat ground (m/s)")
    print(f"{'preset':8} {'U m/s':>7} {'swarmfire':>11} {'rothermel':>11} {'ratio':>7}")
    for preset, fuel_model in V.FUEL_EQUIVALENTS.items():
        sw, rf = [], []
        for u in WINDS:
            state = uniform_world(batch=1, grid=(8, 8), preset=preset, wind=(u, 0.0))
            sw.append(V.swarmfire_head_ros(SimpleROS(), state))
            rf.append(V.rothermel_point(fuel_model, u, 0.0, args.moisture).head_ros)
        for u, a, b in zip(WINDS, sw, rf):
            print(f"{preset:8} {u:7.1f} {a:11.4f} {b:11.4f} {a / b:7.2f}")
        rows[preset] = np.array([sw, rf])
        print()

    _rule("Head rate of spread vs slope, no wind (m/s), grass / GR2")
    print(f"{'tan phi':>8} {'swarmfire':>11} {'rothermel':>11} {'ratio':>7}")
    slope_sw, slope_rf = [], []
    for s in SLOPES:
        state = uniform_world(batch=1, grid=(8, 8), preset="grass", wind=(0.0, 0.0), slope=(s, 0.0))
        a = V.swarmfire_head_ros(SimpleROS(), state)
        b = V.rothermel_point(V.FUEL_EQUIVALENTS["grass"], 0.0, s, args.moisture).head_ros
        slope_sw.append(a)
        slope_rf.append(b)
        print(f"{s:8.2f} {a:11.4f} {b:11.4f} {a / b:7.2f}")

    _rule("Ellipse length/breadth vs midflame wind")
    print(
        "Both use Anderson (1983); swarmfire's coefficients are the same numbers\n"
        "in m/s that pyretechnics carries in mph. They agree exactly until\n"
        "Rothermel's effective-wind cap bites, which swarmfire does not have.\n"
    )
    print(f"{'U m/s':>7} {'swarmfire':>11} {'rothermel':>11}")
    for u in (0.0, 1.0, 2.0, 3.0, 5.0, 6.0, 8.0, 12.0):
        a = float(length_to_breadth(torch.tensor(u)))
        b = V.rothermel_point(V.FUEL_EQUIVALENTS["grass"], u, 0.0, args.moisture).length_to_breadth
        print(f"{u:7.1f} {a:11.3f} {b:11.3f}")

    if args.plot:
        _plot_ros(rows, np.array([slope_sw, slope_rf]))


def _plot_ros(wind_rows, slope_rows) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(17, 3.8))
    for ax, (preset, arr) in zip(axes, wind_rows.items()):
        ax.plot(WINDS, arr[0], "o-", label="swarmfire SimpleROS")
        ax.plot(WINDS, arr[1], "s-", label="pyretechnics Rothermel")
        ax.set_title(f"{preset} / FM{V.FUEL_EQUIVALENTS[preset]}")
        ax.set_xlabel("midflame wind (m/s)")
        ax.set_ylabel("head ROS (m/s)")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[3].plot(SLOPES, slope_rows[0], "o-", label="swarmfire")
    axes[3].plot(SLOPES, slope_rows[1], "s-", label="pyretechnics")
    axes[3].set_title("slope, no wind (grass)")
    axes[3].set_xlabel("tan(slope)")
    axes[3].set_ylabel("head ROS (m/s)")
    axes[3].grid(alpha=0.3)
    axes[3].legend(fontsize=8)
    fig.tight_layout()
    path = OUT / "validation_ros.png"
    fig.savefig(path, dpi=130)
    print(f"\nwrote {path}")


# -------------------------------------------------------------- front geometry


def front(args) -> None:
    """The propagator alone, with the ROS difference removed.

    Both codes get Rothermel's directional rates (`validation.matched_world`
    makes `SimpleROS` reproduce them to float precision), so what is left is
    the difference between a smooth cellular automaton and a level set.
    """
    fuel_model = V.FUEL_EQUIVALENTS[args.preset]
    grid = (args.grid, args.grid)
    duration = args.minutes * 60.0

    reference_point = V.rothermel_point(fuel_model, args.wind, 0.0, args.moisture)
    _rule(f"Front geometry, {args.preset} / FM{fuel_model}, {args.wind} m/s midflame, flat")
    print(
        f"grid {grid[0]}x{grid[1]} at {args.cell:g} m, {args.minutes:g} min, dt {args.dt:g} s\n"
        f"identical directional ROS in both codes: head {reference_point.head_ros:.4f} m/s, "
        f"L/B {reference_point.length_to_breadth:.2f}\n"
    )
    if reference_point.wind_limited:
        print("warning: above Rothermel's effective-wind cap - the two ROS fields differ here\n")

    pyr = V.pyretechnics_arrival_map(
        fuel_model, grid, args.cell, args.wind, duration, moisture=args.moisture
    )
    pyr_axes = V.axis_spread_rates(pyr, cell_size=args.cell)

    print(f"{'front tracker':34} {'head':>8} {'flank':>8} {'back':>8} {'cells':>7} {'IoU':>6}")
    print(
        f"{'pyretechnics level set':34} {_rate(pyr_axes['head'])} {_rate(pyr_axes['flank'])}"
        f" {_rate(pyr_axes['back'])} {int(np.isfinite(pyr).sum()):7d} {1.0:6.3f}"
    )

    maps = {"pyretechnics": pyr}
    variants = [
        ("swarmfire CA, Rothermel residence", None),
        (f"swarmfire CA, {args.preset} preset residence", PRESETS[args.preset].residence),
        ("swarmfire CA, no fuel burnout", 1e9),
    ]
    for label, residence in variants:
        state, ros_model, _ = V.matched_world(
            fuel_model,
            grid,
            args.cell,
            (args.wind, 0.0),
            moisture=args.moisture,
            residence_s=residence,
            device=args.device,
        )
        arrival = V.swarmfire_arrival_map(state, CAPropagator(ros_model), duration, args.dt)
        axes = V.axis_spread_rates(arrival, cell_size=args.cell)
        print(
            f"{label:34} {_rate(axes['head'])} {_rate(axes['flank'])} {_rate(axes['back'])}"
            f" {int(np.isfinite(arrival).sum()):7d} {V.iou(arrival, pyr, duration):6.3f}"
        )
        maps[label] = arrival

    print(
        "\nThe three swarmfire rows should be identical: front speed is sourced\n"
        "from whether a cell has ignited, not from how fast it is consuming fuel,\n"
        "so burnout time cannot reach it. Head and back land within 6 and 8\n"
        "percent; the flank runs about a third fast because eight-neighbour\n"
        "contributions combine with a p-norm rather than along the fastest path,\n"
        "which is the remaining known defect in this class."
    )
    if args.plot:
        _plot_front(maps, duration, args.cell)


def _plot_front(maps, duration, cell_size) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Crop every panel to the same window - the union of the burn scars, with a
    # small margin - so the panels are comparable at a glance.
    burnt = np.zeros_like(next(iter(maps.values())), dtype=bool)
    for arrival in maps.values():
        burnt |= np.isfinite(arrival)
    ys, xs = np.nonzero(burnt)
    pad = 6
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, burnt.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, burnt.shape[1])

    fig, axes = plt.subplots(1, len(maps) + 1, figsize=(3.4 * (len(maps) + 1), 3.8))
    levels = np.linspace(0, duration / 60.0, 13)
    for ax, (label, arrival) in zip(axes, maps.items()):
        ax.contourf(arrival[y0:y1, x0:x1] / 60.0, levels=levels, cmap="inferno")
        ax.contour(arrival[y0:y1, x0:x1] / 60.0, levels=levels, colors="k", linewidths=0.3)
        ax.set_title(label.replace("swarmfire CA, ", "swarmfire: "), fontsize=9)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

    # Perimeters at four times, reference against the best-case CA.
    overlay = axes[-1]
    best = max(maps.items(), key=lambda kv: np.isfinite(kv[1]).sum() if "swarmfire" in kv[0] else -1)
    for minutes in (duration / 60.0 * f for f in (0.25, 0.5, 0.75, 1.0)):
        overlay.contour(
            maps["pyretechnics"][y0:y1, x0:x1] / 60.0, levels=[minutes], colors="k", linewidths=1.1
        )
        overlay.contour(
            best[1][y0:y1, x0:x1] / 60.0, levels=[minutes], colors="C3", linewidths=1.1,
            linestyles="--",
        )
    overlay.set_title("perimeters: reference (black)\nvs " + best[0].replace("swarmfire CA, ", "swarmfire "), fontsize=9)
    overlay.set_aspect("equal")
    overlay.set_xticks([])
    overlay.set_yticks([])

    fig.suptitle(
        "time of arrival (min) - both codes given identical directional rate of spread", fontsize=11
    )
    fig.tight_layout()
    path = OUT / "validation_front.png"
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")


# ------------------------------------------------------------- residence sweep


def residence(args) -> None:
    """Front speed against burnout time - a dependence that must not exist.

    It did. A cell ignites when `ignition` reaches 1, and `ignition` used to
    accumulate at `I * R / L` from each lit neighbour, with `I` the neighbour's
    *combustion* rate. Travel time over one cell is `L / R` - about 100 s for
    grass at 30 m - while flame residence for fine fuels is around ten seconds,
    so the source went dark long before the front had arrived. `front_source`
    now supplies a has-ignited gate instead, and these columns are flat.

    Every residence time runs as one batch element, which is the whole point of
    the `[B, H, W]` layout.
    """
    fuel_model = V.FUEL_EQUIVALENTS[args.preset]
    grid = (args.grid, args.grid)
    duration = args.minutes * 60.0
    times = (10.0, 20.0, 40.0, 60.0, 100.0, 150.0, 300.0, 600.0, 1e9)

    reference = V.rothermel_point(fuel_model, args.wind, 0.0, args.moisture)
    coeffs = V.rothermel_coefficients(fuel_model, args.moisture)
    travel = args.cell / reference.head_ros

    _rule(f"Front speed vs flame residence time, {args.preset}, {args.wind} m/s midflame")
    print(
        f"prescribed head ROS {reference.head_ros:.4f} m/s\n"
        f"time for the front to cross one {args.cell:g} m cell: {travel:.0f} s\n"
        f"Rothermel flame residence for this fuel: {coeffs.residence_s:.0f} s\n"
        f"{args.preset} preset residence: {PRESETS[args.preset].residence:.0f} s\n"
    )

    state, ros_model, _ = V.matched_world(
        fuel_model,
        grid,
        args.cell,
        (args.wind, 0.0),
        moisture=args.moisture,
        batch=len(times),
        device=args.device,
    )
    burn_rate = torch.tensor([1.0 / t for t in times], device=state.device).view(-1, 1, 1)
    state = state.replace(burn_rate=torch.ones_like(state.burn_rate) * burn_rate)

    arrival = V.swarmfire_arrival_map(state, CAPropagator(ros_model), duration, args.dt)
    print(f"{'residence s':>12} {'residence/travel':>17} {'head m/s':>10} {'fraction of ROS':>17}")
    fractions = []
    for t, layer in zip(times, arrival):
        head = V.axis_spread_rates(layer, cell_size=args.cell)["head"]
        fractions.append(head / reference.head_ros)
        label = "inf" if t > 1e8 else f"{t:.0f}"
        ratio = "inf" if t > 1e8 else f"{t / travel:.2f}"
        fraction = "  stalled" if not np.isfinite(head) else f"{head / reference.head_ros:17.2f}"
        print(f"{label:>12} {ratio:>17} {_rate(head):>10} {fraction:>17}")

    print(
        "\nFlat, and it has to be: residence time is a property of the fuel and\n"
        "the front is not entitled to see it. This column used to run from 0.29\n"
        "to 0.94 - and, once the smouldering floors came out of `combustion`, to\n"
        "stall outright below a residence of one cell crossing."
    )

    # The controlling ratio was residence over travel time, and travel time is
    # cell size over rate of spread - so the same defect used to read as a
    # resolution dependence, which is the form in which it actually bit.
    _rule(f"The same thing as a resolution dependence, {args.preset} preset residence")
    print(f"{'cell m':>7} {'travel s':>9} {'residence/travel':>17} {'head m/s':>10} {'fraction':>9}")
    for cell, side, step in ((10.0, 512, 2.0), (15.0, 384, 3.0), (20.0, 256, 5.0), (30.0, 192, 5.0), (45.0, 128, 8.0)):
        state, ros_model, _ = V.matched_world(
            fuel_model,
            (side, side),
            cell,
            (args.wind, 0.0),
            moisture=args.moisture,
            residence_s=PRESETS[args.preset].residence,
            device=args.device,
        )
        arrival = V.swarmfire_arrival_map(state, CAPropagator(ros_model), duration, step)
        head = V.axis_spread_rates(arrival, cell_size=cell)["head"]
        crossing = cell / reference.head_ros
        fraction = "  stalled" if not np.isfinite(head) else f"{head / reference.head_ros:9.2f}"
        print(
            f"{cell:7.0f} {crossing:9.0f} {PRESETS[args.preset].residence / crossing:17.2f}"
            f" {_rate(head):>10} {fraction:>9}"
        )
    print(
        "\nAlso flat, across a four-fold range of cell size. `EnvConfig.cell_size`\n"
        "defaults to 30 m to match LANDFIRE rasters; that row used to read 0.45\n"
        "against 0.95 at 10 m, which meant the simulator was reporting its grid\n"
        "rather than its fuel. The residual 5 percent is the arrival-threshold\n"
        "ramp - see the `dt` and softness notes in VALIDATION.md - not resolution."
    )

    if args.plot:
        _plot_residence(times, fractions, travel, coeffs.residence_s, PRESETS[args.preset].residence)


def _plot_residence(times, fractions, travel_s, rothermel_s, preset_s) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    finite = [(t, f if np.isfinite(f) else 0.0) for t, f in zip(times, fractions) if t < 1e8]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.semilogx([t for t, _ in finite], [f for _, f in finite], "o-", zorder=3)
    ax.axhline(1.0, color="k", ls="--", lw=0.8, label="rate of spread the front was given")
    ax.axhline(fractions[-1], color="C1", ls=":", lw=0.9, label="ceiling, fuel never runs out")
    ax.set_ylim(-0.05, 1.1)
    ax.axvline(travel_s, color="C7", lw=0.8)
    ax.annotate(
        f"front crosses one cell\nin {travel_s:.0f} s",
        (travel_s, 0.42), xytext=(4, 0), textcoords="offset points", fontsize=8, color="C7",
    )
    for seconds, label, colour in (
        (rothermel_s, "Rothermel", "C3"),
        (preset_s, "preset", "C2"),
    ):
        ax.axvline(seconds, color=colour, lw=1.0, ls="-.")
        ax.annotate(
            label, (seconds, 0.95), xytext=(3, 0), textcoords="offset points",
            fontsize=8, color=colour,
        )
    ax.set_xlabel("flame residence time (s)")
    ax.set_ylabel("measured head speed / prescribed")
    ax.set_title("the front only runs at its own rate once cells stay lit\nlonger than the front takes to cross them", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    path = OUT / "validation_residence.png"
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")


# ----------------------------------------------------------------- bottom line


def area(args) -> None:
    """Burned area after `--minutes`, package as shipped against the reference.

    Nothing is matched here: default `SimpleROS`, default presets, default
    propagator. This is the number the caveats in the README are about, and it
    is not the product of the two error sources but rather their interaction -
    grass agrees on rate of spread to within a few percent and still burns a
    fifth of the area, because the front never reaches that rate.
    """
    grid = (args.grid, args.grid)
    duration = args.minutes * 60.0
    cell_area_ha = args.cell**2 / 1e4

    _rule(f"Burned area after {args.minutes:g} min, as shipped (hectares)")
    print(f"{'preset':8} {'U m/s':>7} {'reference':>10} {'swarmfire':>10} {'ratio':>7} {'IoU':>7}")
    for preset, fuel_model in V.FUEL_EQUIVALENTS.items():
        for wind in (2.0, 3.0, 5.0):
            reference = V.pyretechnics_arrival_map(
                fuel_model, grid, args.cell, wind, duration, moisture=args.moisture
            )
            state = uniform_world(
                batch=1,
                grid=grid,
                preset=preset,
                cell_size=args.cell,
                wind=(wind, 0.0),
                device=args.device,
            )
            ours = V.swarmfire_arrival_map(
                state, CAPropagator(SimpleROS()), duration, args.dt
            )
            a_ref = np.isfinite(reference).sum() * cell_area_ha
            a_ours = np.isfinite(ours).sum() * cell_area_ha
            print(
                f"{preset:8} {wind:7.1f} {a_ref:10.1f} {a_ours:10.1f}"
                f" {a_ours / a_ref:7.2f} {V.iou(ours, reference, duration):7.3f}"
            )


# --------------------------------------------------------------------- driver


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "mode", choices=["all", "coefficients", "ros", "front", "residence", "area"]
    )
    parser.add_argument("--preset", default="grass", choices=list(V.FUEL_EQUIVALENTS))
    parser.add_argument("--wind", type=float, default=3.0, help="midflame wind, m/s")
    parser.add_argument("--grid", type=int, default=256)
    parser.add_argument("--cell", type=float, default=20.0, help="metres per cell")
    parser.add_argument("--minutes", type=float, default=120.0)
    parser.add_argument("--dt", type=float, default=5.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--plot", action="store_true", help="write figures to out/")
    args = parser.parse_args()
    args.moisture = V.STANDARD_MOISTURE

    if not V.PYRETECHNICS_AVAILABLE:
        raise SystemExit('pyretechnics is not installed. Run: pip install "swarmfire[real]"')
    OUT.mkdir(exist_ok=True)

    modes = {
        "coefficients": coefficients,
        "ros": ros,
        "front": front,
        "residence": residence,
        "area": area,
    }
    for name in modes if args.mode == "all" else [args.mode]:
        modes[name](args)


if __name__ == "__main__":
    main()
