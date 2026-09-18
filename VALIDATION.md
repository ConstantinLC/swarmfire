# Validation against pyretechnics

How far is this simulator from a reference implementation, and why.

[`pyretechnics`](https://pyregence.github.io/pyretechnics/) implements Rothermel
(1972) surface spread with the Albini/Anderson elliptical template and an
Eulerian level-set front tracker. It is actively maintained by Spatial
Informatics Group as part of the Pyregence effort, it is the same model
BehavePlus and FARSITE are built on, and — unlike them — it is a library that
can be called cell by cell. That makes it the right yardstick, and the README
has listed measuring against it as a step in *Going realistic* from the start.

Reproduce everything below with:

```bash
pip install -e ".[real,viz,dev]"
python scripts/validate_pyretechnics.py all --plot
pytest tests/test_validation.py tests/test_rothermel.py
```

## The bottom line first

Uniform fuel, flat ground, point ignition, two hours, 20 m cells, nothing tuned.
`SimpleROS` runs on `fuels.PRESETS`; `RothermelROS` runs on the equivalent
Scott & Burgan model:

| preset | wind | reference (ha) | **Rothermel** | ratio | IoU | *SimpleROS* | *ratio* | *IoU* |
|---|---|---|---|---|---|---|---|---|
| grass | 2 m/s | 102.7 | 123.6 | 1.20 | 0.83 | *101.7* | *0.99* | *0.87* |
| grass | 3 m/s | 222.1 | 239.2 | 1.08 | 0.89 | *189.8* | *0.85* | *0.79* |
| grass | 5 m/s | 306.2 | 271.2 | 0.89 | 0.80 | *240.6* | *0.79* | *0.74* |
| shrub | 2 m/s | 1.0 | 2.3 | 2.23 | 0.45 | *46.9* | *45.1* | *0.02* |
| shrub | 3 m/s | 2.1 | 3.5 | 1.67 | 0.60 | *85.8* | *41.2* | *0.02* |
| shrub | 5 m/s | 4.4 | 5.3 | 1.20 | 0.81 | *147.4* | *33.5* | *0.03* |
| timber | 2 m/s | 61.6 | 79.1 | 1.28 | 0.78 | *7.4* | *0.12* | *0.12* |
| timber | 3 m/s | 122.3 | 140.4 | 1.15 | 0.87 | *12.5* | *0.10* | *0.10* |
| timber | 5 m/s | 224.1 | 194.9 | 0.87 | 0.79 | *20.9* | *0.09* | *0.09* |

With `RothermelROS` every fuel lands within about 30 % of the reference, except
shrub at low wind where the whole fire is one or two hectares and two cells of
discretisation is the entire difference. With the presets, shrub is out by a
factor of 45 and timber by a factor of 10.

Grass is the exception that proves the point: `SimpleROS` matches it almost
exactly, by accident, because two large errors cancel across that wind band
(§1). Nothing about that transfers to another fuel.

What is left is the front tracker rather than the physics — §2.

## Why the comparison is split in two

An under-propagating front and an over-fast rate of spread produce the same
burned-area curve as a correct simulator. So the two are measured separately:

1. **Point physics (§1).** Does the ROS model return the rate of spread
   Rothermel returns, for the same fuel, moisture, midflame wind and slope? A
   question about `ros/` only.
2. **Front geometry (§2).** Given rates that already match, does `CAPropagator`
   move the front at them, and into the right shape? A question about
   `propagate.py` only.

The second needs both codes driven by identical rates, and there are now two
independent ways to arrange that. `RothermelROS` reproduces the reference's
rates directly. `validation.matched_world` gets there differently: Rothermel's
wind factor is `φ_w = C(β/β_op)^−E · U^B` and its slope factor `φ_s = G·tan²φ` —
exactly the algebraic form `SimpleROS` already uses — so handing `SimpleROS`
Rothermel's own coefficients for one fuel and one moisture state makes the two
agree to float precision (asserted in `tests/test_validation.py`).

§2 runs both. They should give the same answer, and if they ever stop, one of
them has a bug.

## 1. Point physics: both rate-of-spread models, one metric

```bash
python scripts/validate_pyretechnics.py ros
```

Both models are measured the same way: **relative error against `pyretechnics`
in all eight directions**, over one grid — the three fuels the package ships,
four dead 1-hour moistures (4, 6, 9, 18 %), and twelve wind and slope
combinations from dead calm to 20 m/s and a 50 % grade. Each model is driven
through the world the package actually builds for it: `SimpleROS` on a
`fuels.PRESETS` world, `RothermelROS` on the equivalent Scott & Burgan model
through `fuels.rothermel_world`.

![head rate of spread against wind and against moisture](out/validation_ros.png)

| fuel | model | cases | median | 90th pct | max |
|---|---|---|---|---|---|
| grass / FM102 | `SimpleROS` | 288 | 2.1e-1 | 1.3e+0 | 7.7e+0 |
| | **`RothermelROS`** | 288 | **8.4e-5** | **2.2e-4** | **2.4e-4** |
| shrub / FM142 | `SimpleROS` | 288 | 6.0e+0 | 1.5e+1 | 4.8e+1 |
| | **`RothermelROS`** | 288 | **5.1e-5** | **2.0e-4** | **3.1e-4** |
| timber / FM163 | `SimpleROS` | 384 | 6.6e-1 | 7.7e-1 | 8.6e-1 |
| | **`RothermelROS`** | 384 | **4.9e-5** | **1.6e-4** | **1.9e-4** |

Four orders of magnitude between them at the median, and `RothermelROS` is at
float32 — which is what closed-form algebra should manage, and means the number
there measures accumulation rather than modelling.

### 1.1 Is that a fair grid?

The sweep varies fuel moisture, and **`SimpleROS` has no moisture term at all** —
`ros/simple.py` never reads `state.moisture`. Pinning moisture at 6 %, which is
what the presets were implicitly written for, removes that axis:

| fuel | model | median | 90th pct | max |
|---|---|---|---|---|
| grass / FM102 | `SimpleROS` | 2.1e-1 | 1.6e+0 | 5.9e+0 |
| | `RothermelROS` | 8.4e-5 | 2.2e-4 | 2.2e-4 |
| shrub / FM142 | `SimpleROS` | 6.1e+0 | 1.5e+1 | 2.4e+1 |
| | `RothermelROS` | 5.1e-5 | 1.6e-4 | 2.0e-4 |
| timber / FM163 | `SimpleROS` | 7.2e-1 | 7.7e-1 | 8.5e-1 |
| | `RothermelROS` | 4.9e-5 | 1.6e-4 | 1.9e-4 |

It barely moves. `SimpleROS` is not being punished for an axis it does not
model; it is wrong on the axes it does.

That missing moisture term is worth dwelling on, because it is load-bearing for
this project. Grass, 3 m/s, flat:

| dead 1-h moisture | `SimpleROS` | `RothermelROS` | reference |
|---|---|---|---|
| 3 % | 0.27981 | 0.35228 | 0.35229 |
| 6 % | 0.27981 | 0.29399 | 0.29400 |
| 12 % | 0.27981 | 0.19422 | 0.19422 |
| 20 % | 0.27981 | **0.00000** | **0.00000** |

GR2's dead moisture of extinction is 15 %. Past it the fuel cannot carry fire
and the reference goes to zero; `SimpleROS` returns the same number it returns
for bone-dry grass. The README says suppression "acts through fuel moisture, not
as a special case — water raises a cell's moisture; past the fuel's moisture of
extinction, the fuel stops carrying fire. That mechanism is already in
Rothermel." Under `SimpleROS` it is not: a drop can only act through the
separate multiplicative gate in `SuppressionModel`. `RothermelROS` is the first
model in this package that actually provides the advertised pathway.

### 1.2 Rothermel across every fuel model

```bash
python scripts/validate_pyretechnics.py rothermel
```

The table above covers the three fuels `SimpleROS` can express. `RothermelROS`
is not limited to those, so it is also checked across **13,920 directional
comparisons** — all 53 burnable fuel models (the Anderson 13 and the Scott &
Burgan 40), three moistures, twelve wind and slope combinations, all eight
directions:

| | relative error |
|---|---|
| maximum | **3.7e-4** |
| median | **4.9e-5** |
| 99.9th percentile | 3.7e-4 |

By family, so the agreement is visibly not carried by one convenient group:

| family | cases | max rel | median |
|---|---|---|---|
| Anderson 13 | 432 | 3.6e-4 | 5.7e-5 |
| GR — grass | 276 | 3.0e-4 | 5.9e-5 |
| GS — grass-shrub | 120 | 3.7e-4 | 5.7e-5 |
| SH — shrub | 276 | 3.2e-4 | 5.2e-5 |
| TU — timber-understory | 168 | 3.4e-4 | 6.7e-5 |
| TL — timber litter | 324 | 3.4e-4 | 8.0e-5 |
| SB — slash-blowdown | 144 | 2.0e-4 | 5.0e-5 |

The error grows with wind — 5.6e-5 at dead calm, 1.4e-4 at 3 m/s, 3.7e-4 above
12 m/s — which is float32 running through the wind factor's exponentials, at a
scale five orders of magnitude below anything physical.

In absolute terms, at 6 % moisture on flat ground, swarmfire / reference. Four
decades of spread rate, agreeing to five figures:

| model | U = 0 | U = 3 m/s | U = 8 m/s |
|---|---|---|---|
| GR2 | 0.00782 / 0.00782 | 0.29399 / 0.29400 | 0.65434 / 0.65431 |
| GR4 | 0.01578 / 0.01578 | 0.59787 / 0.59788 | 2.44691 / 2.44696 |
| SH2 | 0.00140 / 0.00140 | 0.02500 / 0.02501 | 0.09365 / 0.09365 |
| SH5 | 0.01213 / 0.01213 | 0.44911 / 0.44911 | 1.41430 / 1.41431 |
| TU3 | 0.00773 / 0.00773 | 0.21701 / 0.21702 | 0.80403 / 0.80405 |
| TL3 | 0.00107 / 0.00107 | 0.01386 / 0.01386 | 0.01729 / 0.01728 |
| SB4 | 0.02073 / 0.02073 | 0.45071 / 0.45072 | 1.87848 / 1.87853 |
| R01 | 0.02339 / 0.02339 | 0.94551 / 0.94556 | 1.50926 / 1.50919 |
| R10 | 0.00601 / 0.00601 | 0.09536 / 0.09536 | 0.36955 / 0.36956 |

GR2 at 8 m/s is 0.654 against a no-wind 0.0078 — an 84-fold wind factor that has
stopped growing, because Rothermel's effective-wind limit has taken hold. TL3
barely moves from 3 to 8 m/s for the same reason at a much lower cap. The first
panel of the figure shows both: `RothermelROS` flattens where the reference
does and `SimpleROS` climbs straight through.

**The fuel model table.** `swarmfire/fuel_models.py` is transcribed published
data — Anderson (1982) and Scott & Burgan (2005), both USDA — and transcription
is exactly the sort of thing that is silently wrong for months.
`test_fuel_model_table_matches_the_reference` checks every field of every model
against `pyretechnics`' copy: depth, load and surface-area-to-volume per size
class, heat content, moisture of extinction, and the dynamic flag.

### 1.3 Where SimpleROS's error comes from

`SimpleROS` has the right algebraic *shape*. `phi_w = a·U^b` and
`phi_s = c·tan²φ` are Rothermel's own forms, its wind exponent `b = 1.5` is
close to the fuel-dependent 1.36–1.46 Rothermel produces, and its ellipse is
Anderson (1983) with the same coefficients expressed in m/s rather than mph
(`0.1147 × 2.23694 = 0.25657`) — agreeing to four decimals at every wind below
the effective-wind cap. The directional template `(1−e)/(1−e·cos θ)`, the polar
ellipse with the ignition at a focus, is character-for-character the reference's.

What it lacks is any dependence on the fuel. Every coefficient is a constant
where Rothermel's is a function of the fuel bed:

| | `SimpleROS` | Rothermel GR2 | SH2 | TL3 |
|---|---|---|---|---|
| `ros0` (m/s) | 0.020 / 0.012 / 0.004 | 0.0078 | 0.0014 | 0.0077 |
| wind `a` | 2.5 — all fuels | 7.40 | 3.66 | 6.06 |
| wind `b` | 1.5 — all fuels | 1.46 | 1.39 | 1.36 |
| slope `c` | 5.5 — all fuels | 36.5 | 19.9 | 28.6 |
| moisture of extinction | 0.12 / 0.20 / 0.25 | 0.15 | 0.15 | 0.30 |
| flame residence (s) | 60 / 150 / 400 | 12.7 | 13.8 | 14.3 |
| effective-wind cap | *none* | 5.25 m/s | 10.6 m/s | 16.3 m/s |

Which gives four distinct failures, on top of the missing moisture term in §1.1:

* **`ros0` is off by a different factor per fuel, in different directions** —
  grass 2.6× too fast, shrub 8.6× too fast, timber 0.52× too slow. The presets
  were not picked from a fuel model, and it shows. There is no single correction.
* **One wind and slope constant cannot serve every fuel.** GR2 and SH2's wind
  scalars differ by 2×. The slope constant is 3.6–6.6× too weak everywhere, so
  steep ground under-spreads badly: at a 50 % grade in zero wind grass spreads
  at 60 % of the reference rate, at 70 % half.
* **There is no effective-wind limit.** Rothermel stops believing its own wind
  factor above `0.9 I_R`; `SimpleROS` accelerates through it — 2.45× the
  reference at 10 m/s on grass, and visibly diverging in the first panel of the
  figure. Because the ellipse is driven off the same uncapped effective wind,
  the *shape* runs away too: at 12 m/s `length_to_breadth` returns 20 where the
  reference holds at 3.41, and `SimpleROS.max_eccentricity` clamps what is
  realised to L/B ≈ 7.1, still twice the reference.
* **Residence times are 5–28× Rothermel's.** This one used to matter a great
  deal more than it looks; see §2.1.

#### The accident worth knowing about

The one case where `SimpleROS` looks fine — and the reason the summary table in
§1 reports a median rather than a single headline number. Head rate of spread,
grass, flat ground:

| midflame wind | `SimpleROS` | reference | ratio |
|---|---|---|---|
| 0 m/s | 0.0200 | 0.0078 | 2.56 |
| 1 m/s | 0.0700 | 0.0657 | 1.07 |
| 3 m/s | 0.2798 | 0.2940 | 0.95 |
| 5 m/s | 0.5790 | 0.6096 | 0.95 |
| 8 m/s | 1.1514 | 0.6543 | 1.76 |
| 10 m/s | 1.6011 | 0.6543 | 2.45 |

Between 1 and 5 m/s, `SimpleROS` on grass is within 10 % of Rothermel. That is
not calibration: `ros0` is 2.6× too high and the wind factor 3× too weak, and
across that one band the two errors cancel. Outside it they stop. The `grass`
demos in this repository sit inside the band, which is why the fires look about
right — and why a harness was worth building rather than eyeballing GIFs.

Shrub and timber have no such band: shrub runs 6.4–7.6× fast at every wind
speed, timber 0.25–0.29× slow at every wind speed.

## 2. Front geometry: the propagator

```bash
python scripts/validate_pyretechnics.py front --plot
```

Same question as §1, one layer up: given directional rates that already match
the reference, does `CAPropagator` move the front at them? Two independent ways
to arrange that, and they should agree:

* **`RothermelROS`**, which reproduces the reference's rates directly (§1.2).
  This is the configuration anyone running the package for real numbers uses.
* **`validation.matched_world`**, which hands `SimpleROS` Rothermel's own
  coefficients for one fuel and one moisture state. It predates `RothermelROS`
  and is kept precisely because it reaches the same place by a different route.

Grass / FM102, 3 m/s midflame, flat, 20 m cells, 2 hours:

![arrival time maps](out/validation_front.png)

| front tracker | head (m/s) | flank (m/s) | back (m/s) | IoU |
|---|---|---|---|---|
| pyretechnics level set | 0.2949 | 0.0429 | 0.0236 | 1.000 |
| *prescribed by the ROS model* | *0.2940* | *0.0433* | *0.0234* | — |
| swarmfire CA, **`RothermelROS`** | 0.2777 | 0.0577 | 0.0253 | **0.887** |
| swarmfire CA, matched `SimpleROS` | 0.2777 | 0.0577 | 0.0253 | **0.887** |

The two swarmfire rows are identical to four figures, which is the point of
running both: what is left is the front tracker and not the rate-of-spread
model. Head 6 % slow, back 8 % fast, flank 33 % fast.

Read the first two rows together as well — the reference level set reproduces
the rates it was handed to within 1 % on all three axes. That is the yardstick
working.

### 2.1 Three bugs, and what they were hiding

**The p-norm differentiated to NaN.** `ignition_drive` combined the eight
neighbour contributions as `x.pow(6).sum().pow(1/6)`. Forward that is fine;
backward, `d/ds [s^(1/6)]` is infinite at `s = 0`, and `s` is exactly zero at
every cell with no lit neighbour — most of the grid. So
`burned_area().backward()` returned NaN for every rollout, and the
differentiable-simulation path the package is built around did not work at all.
`torch.linalg.vector_norm` is forward-identical and defines the subgradient at
the origin as zero.

**Two gates never reached zero.** A sigmoid is never zero, and both gates in
`combustion` were plain sigmoids:

| gate | at the physical zero | meaning |
|---|---|---|
| `sigmoid((ignition − 1) / 0.08)` | `3.7e-6` at zero progress | every cell in the domain burns a little, forever |
| `sigmoid((fuel − 0.03) / 0.03)` | `0.269` at zero fuel | a cell with **no fuel left** reports 27 % of full combustion |

The first is a bias: `burned_area()` accumulated over the whole grid whether or
not anything was alight, and `active()` on an unlit 192×192 grid read 0.13
against a `done` threshold of `1e-3` — so `info["extinguished"]` could never
become true and no rollout ever stopped early. The second was load-bearing in a
way nobody intended, and getting to it is the interesting part.

**The front was sourced from combustion.** `ignition_drive` took its source term
from a neighbour's `intensity` — the rate that neighbour is consuming fuel.
That rate collapses on the flame residence timescale, which is a property of the
fuel and says nothing about the grid. The front needs `cell_size / R` seconds to
cross a cell: 100 s for grass on a 30 m cell, against a flame residence of 13 s.
So the source went dark long before the neighbour it was pushing had arrived.

Those two interacted. The `0.269` floor meant a burnt-out cell never actually
stopped driving, which supplied exactly the missing drive — badly, at a strength
set by the floor rather than by the rate of spread. **The fire was spreading
because burnt fuel never stopped burning.** Removing the floor alone, before
touching the source, took a grass fire at the default 30 m resolution from
spreading at 45 % of its stated rate to not spreading at all: the nearest unlit
cell climbed to `ignition = 0.937` and asymptoted there, never crossing the
arrival threshold.

```
t=  60s  cells ignited:   9   best unlit progress: 0.911   total intensity: 0.4459
t= 300s  cells ignited:   9   best unlit progress: 0.936   total intensity: 0.1125
t=1200s  cells ignited:   9   best unlit progress: 0.937   total intensity: 0.0233
```

The fix is `CAPropagator.front_source`: drive the front from whether a cell has
*ignited* — which `ignition` already records and which does not un-happen —
gated by flammability so that a doused cell stops pushing its neighbours.
Combustion intensity keeps its own job, consuming fuel. Once the two are
separated, front speed stops depending on burnout time:

| residence | residence / cell crossing | head (m/s) | fraction of prescribed |
|---|---|---|---|
| 10 s | 0.15 | 0.2777 | 0.94 |
| 20 s | 0.29 | 0.2777 | 0.94 |
| 60 s | 0.88 | 0.2777 | 0.94 |
| 300 s | 4.41 | 0.2777 | 0.94 |
| ∞ | ∞ | 0.2777 | 0.94 |

and stops depending on the grid, which is the form the same defect took in
practice:

| cell size | crossing time | head (m/s) | fraction | *before the fix* |
|---|---|---|---|---|
| 10 m | 34 s | 0.2796 | 0.95 | 0.95 |
| 15 m | 51 s | 0.2796 | 0.95 | 0.92 |
| 20 m | 68 s | 0.2777 | 0.94 | 0.69 |
| **30 m** | 102 s | 0.2810 | 0.96 | **0.45** |
| 45 m | 153 s | 0.2806 | 0.95 | 0.37 |

`EnvConfig.cell_size` defaults to 30 m, to match LANDFIRE rasters.

**What a latch does *not* cost.** A burnt-out cell keeps radiating, so a fire can
eventually cross a barrier whose suppressant has evaporated with nothing left
burning against it. That is the wrong reason for a barrier to fail and a
minimum-arrival-time propagator would remove it — but it is not introduced here.
The previous formulation had the same artefact through the `0.269` fuel-gate
floor, and the two are within 8 % of each other: on a 2 mm water line across a
5 m/s grass fire the barrier failed at **12720 s before and 11696 s after**.

There is no measure isolating the propagator on which the new version is worse.
Head and back speed, residence independence, resolution independence, perimeter
IoU against the reference, and whether gradients survive at all: every one moved
toward `pyretechnics`. The only composite number that got *worse* is shrub
burned area (13–19× the reference, now 33–45×), and that is `PRESETS["shrub"]`
carrying a spread rate 8.6× SH2's with the compensating brake removed — a
`fuels.py` error the old propagator was hiding, not a new one.

### 2.2 What is still wrong: flanks run about a third too fast

The flank runs at 0.0577 m/s against a prescribed 0.0433 — 33 % fast — while
the head is 6 % slow. The rightmost panel of the arrival-map figure shows the
result: perimeters too fat and too blunt where the reference is a teardrop.

The cause is in `ignition_drive`. Neighbour contributions combine with a p-norm,
which the docstring describes as approximating "the front arrives along the
fastest path". At `p = 6` it does not: a flank cell with two or three lit
neighbours at comparable rates accumulates faster than the single fastest path
would give. Sharpening the norm barely helps, and pushing it far enough to
matter destroys the head instead:

| `pnorm` | head / prescribed | flank / prescribed |
|---|---|---|
| 1 | 1.41 | 2.43 |
| 2 | 1.00 | 1.57 |
| 4 | 0.95 | 1.36 |
| **6** (default) | 0.94 | 1.33 |
| 12 | 0.94 | 1.31 |
| 24 | 0.80 | — (front breaks up) |

There is no setting that gives both. The p-norm is the wrong instrument here,
not a badly-tuned one — a genuine minimum-arrival-time update, or a
differentiable soft-*min* over per-direction arrival times rather than a soft-max
over rates, is what the docstring is actually describing.

### 2.3 And a smaller one: the head loses a few percent per timestep

Well inside the stability limit `check_stability()` reports (27 s for this
scenario), the head rate still drifts with `dt` — 0.2837 m/s at `dt = 2 s`,
0.2778 at 5 s, 0.2508 at 20 s, a 12 % spread. The flanks and back barely move.
`EnvConfig.dt` defaults to 5 s, so this is a percent or two on top of everything
above rather than a headline, but it is the same family of problem: the
propagator's answer depends on how it was discretised. It is also most of the
residual 6 % gap in the head.

## What to do about it, in order

1. ~~Fix the propagator before the physics.~~ **Done.** `front_source` separates
   arrival from combustion, `soft_gate` closes both smouldering floors, and
   `torch.linalg.vector_norm` keeps the backward pass finite. Front speed is now
   independent of burnout time and of resolution, and gradients survive the
   simulator.
2. ~~Implement `ros/rothermel.py`.~~ **Done.** Rothermel (1972) over the
   Anderson 13 and Scott & Burgan 40 fuel models, agreeing with `pyretechnics`
   to float32 across 13,920 directional comparisons — §1.1. `fuels.rothermel_world` builds a world from a
   fuel-model raster and fills `ros0`, `moisture_ext` and `burn_rate` from the
   same model, so the propagator's fuel accounting and the spread model describe
   one fire rather than two.
3. **Load real landscapes.** `RothermelROS` already takes a per-cell fuel model
   number, which is exactly what a LANDFIRE fuel-model raster contains, so this
   is now a data-loading job rather than a modelling one: elevation into
   `FireState.elevation`, the fuel-model raster into `FireState.fuel_model`, and
   `landfire` to fetch both. Canopy cover and height are not read yet; they
   belong with the wind-adjustment factor, which this model does not apply
   because `FireState.wind` is already a midflame wind.
4. **Replace the scalar weather moistures.** `RothermelROS` takes the 10-hour,
   100-hour and live moistures as constructor arguments because they are
   weather, near-uniform over a fire-sized domain, and there is no weather model
   to vary them. Dead 1-hour is already a per-cell layer, which is the one that
   matters: it is what suppression raises.
5. **Finish the front tracker.** §2.2 and §2.3 are what remain — the p-norm's
   33 % flank overshoot and a few percent of timestep drift. Both want a
   minimum-arrival-time scheme rather than more tuning, and both now dominate
   the error budget.
6. **Calibrate the suppression constants** in `SuppressionModel` against
   drop-test coverage-level data. With Rothermel in place, a drop reaches spread
   rate through the published moisture-damping term, so the thing left to
   calibrate is how much moisture a given coverage level actually adds.
7. **Re-run the suppression study.** Every result in the README predates all of
   this.

## What this harness is

| file | what it holds |
|---|---|
| `swarmfire/validation.py` | the `pyretechnics` adapter — coefficient extraction, `matched_world`, arrival maps from both codes, axis spread rates, IoU |
| `scripts/validate_pyretechnics.py` | six modes: `coefficients`, `ros` (both models, one metric), `rothermel` (all 53 fuel models), `front`, `residence`, `area` |
| `tests/test_validation.py` | the exact identities as assertions, the current disagreement as regression guards |
| `tests/test_rothermel.py` | `RothermelROS` and the fuel model table against the reference, cell by cell |

The tolerances in `test_validation.py` are measurements, not targets. They are
there so that a change in the physics shows up as a failing test and gets looked
at, not so that they pass. `test_rothermel.py` is different: those are float32
tolerances on closed-form algebra, and they should never need loosening.

### Comparison conditions

Held fixed everywhere so that differences are attributable to the models:
dead 1/10/100-hour moisture 6/7/8 %, live herbaceous 60 %, live woody 90 %; no
canopy; `FireState.wind` read as midflame wind and inverted to the 10 m wind the
reference's gridded entry points expect via the Andrews (2012) adjustment;
ignition at the grid centre. Spotting, crown fire and fire–atmosphere feedback
are off in the reference, since this package has none.

Three differences are not controlled and are small here: the reference ignites a
single cell where `state.ignite` seeds a soft blob of radius 1.5 cells; the two
codes take different timesteps; and `pyretechnics` works on the slope-tangential
plane where this package works in map projection, so on sloped ground the
matched-ROS bridge is exact only to the projection factor — about 2 % at a 20 %
grade. Every propagation comparison above is on flat ground, where it is exact.
Axis spread rates are read as the slope of a distance-against-arrival-time fit
rather than final extent over elapsed time, which removes the ignition offset
entirely.
