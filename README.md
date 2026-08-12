# swarmfire

Differentiable, GPU-batched wildfire propagation with suppressant dynamics, and
a platform abstraction for delivering that suppressant under realistic
operational constraints.

The current dynamics are a simple Rothermel Rate-Of-Spead fire propagation model, with simple moisture accumulation mechanisms for fire suppression. The goal is to progressively increment to more and more realistic simulations, while allowing for learnt aircraft strategies.

This is a baseground for the development of an optimal automatic drone-based wildfire response, with the target of being operational on real events.

This work is a proposition to bridge different (currently isolated) aspects of wildfire mitigation models, in order to discover optimal wildfire response strategies. In the next table, we make an inventory of existing works :

| work | fire spread model | suppressant → fire coupling | delivery / platform realism | learned or optimised strategy | differentiable | batched · GPU | code available |
|---|---|---|---|---|---|---|---|
| Dada & Barakos 2025 — real-time CFD of aerial firefighting [1] | ✗ | ✗ | ✓ resolved drop breakup and fall | ✗ (pilot training) | ✗ | ✗ | ✗ |
| Plucinski & Sullivan 2024 — combustion wind tunnel [2] | ◐ bench-scale flame spread | ✓ measured, direct *and* indirect attack | ✗ (applied by hand) | ✗ | ✗ | ✗ | n/a (experimental) |
| Lovellette — AT-802 ground pattern guide [3] | ✗ | ✗ | ✓ measured coverage-level footprints | ✗ | ✗ | ✗ | n/a (field data) |
| Martins et al. 2026 — water drops on wildfire spread [4] | ✓ | ✓ | ◐ drop as a prescribed footprint | ✗ | ✗ | ✗ | ? |
| **swarmfire (this work)** | ✓ smooth CA, Rothermel-shaped ROS | ✓ through fuel moisture, water *and* retardant | ✓ capacity, footprint, reload, transit | ◐ RL + gradient interfaces ready, no trained policy yet | ✓ end-to-end through hundreds of steps | ✓ `[B,H,W]` tensor ops | ✓ |

✓ does it · ◐ partially · ✗ does not · ? not established from the source

Each existing work is excellent at one column and silent on the others: the
drop is modelled without the fire, the fire without the aircraft, and the
suppressant chemistry without either. Nothing in that list produces a gradient
or a strategy. This repository is the attempt to close that loop — an
environment where a controller can be *trained*, with each column swappable for
the higher-fidelity version the corresponding paper already provides. The
calibration targets are [2] and [3]; [4] is the closest prior art on the
coupling itself.

[1] `dada2025real` · [2] `plucinski2024methodologies` · [3] Lovellette, USDA
Forest Service · [4] `martins2026beyond` — full entries in
[`litterature.md`](litterature.md).

Built to the plan in [`goals.md`](goals.md): get the pipeline working on simple
fire and water dynamics first, but with every seam already in place for the
realistic version.

```bash
pip install -e ".[viz,dev]"
python scripts/demo.py --device cuda
pytest
```

```python
from swarmfire import FireEnv, EnvConfig

env = FireEnv("drone_swarm", config=EnvConfig(batch=256, grid=(192, 192), device="cuda"))
env.reset()
print(env.rollout()["burned_ha"].mean())   # unsuppressed baseline
```

## The three design decisions

**Fire state is continuous, not a burning/not-burning flag.** Every gate in the
physics is smooth, so gradients survive hundreds of fire steps. You can
backpropagate `burned_area()` straight into the network that chose where to
drop water — a far stronger signal than a scalar reward, and the direct route
to the convolution-based controller in `goals.md`. It also stops fires from
coming out octagonal. This is the one choice that is genuinely expensive to
retrofit.

**Suppression acts through fuel moisture, not as a special case.** Water raises
a cell's moisture; past the fuel's moisture of extinction, the fuel stops
carrying fire. That mechanism is already in Rothermel, so the toy model and the
eventual real model respond to a drop through the same pathway. Water works
hard but boils off in minutes; retardant is weaker but persists for hours and
is not consumed by the fire — which is exactly why one is dropped *on* the
front and the other *ahead* of it, and what makes the control problem
interesting.

**Everything is one batched tensor op.** No Python loop over environments. The
whole propagation step is a 3×3 stencil plus elementwise algebra.

## Layout

| file | what it owns |
|---|---|
| `state.py` | the layer stack — terrain, fuel, weather, fire, suppressant, all as `[B,H,W]` rasters |
| `grid.py` | neighbour geometry, the `shift` stencil, terrain gradients |
| `ros/simple.py` | wind + slope → elliptical directional spread rates |
| `ros/rothermel.py` | **the realistic-physics slot** — empty, documented, one-line swap |
| `propagate.py` | the smooth CA step |
| `suppression.py` | water/retardant dynamics and the flammability coupling |
| `platforms.py` | tanker / helicopter / drone swarm → deposition footprints |
| `env.py` | batched `reset`/`step`/`rollout`, RL and differentiable modes |
| `fuels.py` | fuel presets, homogeneous and patchy worlds |

## The seams

Each is an interface with a working implementation behind it, swappable
without touching anything else:

| seam | now | next |
|---|---|---|
| `ROSModel` | `SimpleROS` | `RothermelROS` + LANDFIRE fuel models |
| `Propagator` | `CAPropagator` | level set / minimum arrival time |
| `Platform` | tanker, helicopter, 32-drone swarm | ground crews, dozer lines, refill logistics |
| `SuppressionModel` | water, retardant | foam, gel, dose-response calibration |

## Platforms

All three reduce to the same primitive — add a mass field to the water or
retardant layer. They differ only in footprint shape, capacity, reload time,
and how fast the delivery point moves.

| | capacity | footprint | reload | agent |
|---|---|---|---|---|
| tanker | 12,000 L | 300 × 30 m line | 30 min | retardant |
| helicopter | 3,000 L | 25 m disc | 5 min | water |
| drone swarm | 32 × 20 L | 6 m discs | 2 min | water |

Footprints are built from continuous coordinates with soft edges, so they are
differentiable with respect to drop position and heading — a policy gets a
gradient telling it to move a drop fifteen metres north.

## Going realistic

1. Implement `ros/rothermel.py` (closed-form algebra — the module docstring
   lists every term and every new state layer it needs).
2. Validate it cell-by-cell against [`pyretechnics`](https://pyregence.github.io/pyretechnics/),
   which implements the same model and is actively maintained. Agreeing with it
   to a few percent beats any hand-rolled unit test.
3. Load real fuel/canopy/elevation rasters with `landfire`.
4. Calibrate the suppression constants in `SuppressionModel` against drop-test
   coverage-level data.

Steps 1–3 change one constructor argument and the contents of `fuels.py`.
Nothing else in the package moves.

## Caveats

- The `SimpleROS` coefficients are plausible orders of magnitude, not
  calibrated values. Numbers out of this model are not predictions.
- Explicit propagation is conditionally stable. `env.check_stability()` returns
  the largest safe `dt`; call it after changing wind, fuel, or resolution.
- Reload and empty-tank gates are hard (non-differentiable) by nature. Drop
  position, heading, volume, and all of the physics are smooth.
- No crown fire, no spotting, no fire-atmosphere feedback. Those belong with
  the Rothermel work.
