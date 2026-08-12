# swarmfire

Differentiable, GPU-batched wildfire propagation with suppressant dynamics, and
a platform abstraction for delivering that suppressant under realistic
operational constraints.

The current dynamics are a simple Rothermel Rate-Of-Spead fire propagation model, with simple moisture accumulation mechanisms for fire suppression. The goal is to progressively increment to more and more realistic simulations, while allowing for learnt aircraft strategies.

This is a baseground for the development of an optimal automatic drone-based wildfire response, with the target of being operational on real events.

## Visualization 

![Free burn, helicopter and drone swarm on the same ignition](out/comparison.gif)

The same fire under three regimes, sharing one ignition and one clock: grass,
wind (5, 2) m/s, 192 × 192 cells at 30 m, 2.8 hours. The fleet is grounded for
the first ten minutes (`dispatch_s = 600`), so all three panels are identical
until it launches. The white cross is the vehicle, dimming as its tank empties
and brightening on reload. Regenerate with:

```bash
python scripts/suppression_study.py compare --dispatch 600 --upscale 2 --record-every 20
```

## How to get the code running ?

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

## Related work

This work is a proposition to bridge different (currently isolated) aspects of wildfire mitigation models, in order to discover optimal wildfire response strategies. In the next table, we make an inventory of existing works :

| work | fire spread model | suppressant → fire coupling | delivery / platform realism | learned or optimised strategy | differentiable | batched · GPU | code available |
|---|---|---|---|---|---|---|---|
| Dada & Barakos 2025 — real-time CFD of aerial firefighting [1] | ✗ | ✗ | ✓ resolved drop breakup and fall | ✗ (pilot training) | ✗ | ✗ | ✗ |
| Calbrix et al. 2023 — CFD drops, CL-415 and Dash-8 [2] | ✗ | ✗ | ✓ per-airframe drop simulation | ✗ | ✗ | ✗ | ✗ |
| Wu et al. 2024 — review of airtanker drop characteristics [3] | ✗ | ◐ surveys effectiveness studies | ✓ survey of drop patterns | ✗ | ✗ | ✗ | n/a (review) |
| Lovellette — AT-802 ground pattern guide [4] | ✗ | ✗ | ✓ measured coverage-level footprints | ✗ | ✗ | ✗ | n/a (field data) |
| Plucinski & Sullivan 2024 — combustion wind tunnel [5] | ◐ bench-scale flame spread | ✓ measured, direct *and* indirect attack | ✗ (applied by hand) | ✗ | ✗ | ✗ | n/a (experimental) |
| Martins et al. 2026 — water drops on wildfire spread [6] | ✓ | ✓ | ◐ drop as a prescribed footprint | ✗ | ✗ | ✗ | ? |
| Murray et al. 2024 — deep RL for firebreak placement [7] | ✓ | ✗ (firebreaks, not suppressant) | ✗ (no platform, no logistics) | ✓ trained DRL policy | ✗ (RL through a black-box sim) | ◐ | ? |
| Matei et al. 2026 — aerial suppression planning, hybrid CNN–CA [8] | ◐ frozen CNN emits the CA's parameters | ◐ water *and* retardant, but multiplied onto a surrogate never trained on drops | ✓ 18-airframe fleet, real capacities and turnaround | ◐ 3,000 Adam steps per scenario, no policy | ✓ | ◐ vectorised over 300 MC samples | ✗ |
| **swarmfire (this work)** | ✓ explicit smooth CA, Rothermel-shaped ROS | ✓ through fuel moisture against moisture of extinction | ◐ three generic classes: capacity, footprint, reload, transit | ◐ RL + gradient interfaces ready, no trained policy yet | ✓ end-to-end through hundreds of steps | ✓ `[B,H,W]` tensor ops | ✓ |

✓ does it · ◐ partially · ✗ does not · ? not established from the source

Most of these works are excellent in one column and silent on the rest: [1–4]
model the drop without the fire, [5] the suppressant chemistry without either,
[6] the coupling without an aircraft or a controller. [7] trains a policy, but
over firebreaks rather than suppressant and through a non-differentiable
simulator.

[8] is the nearest neighbour of this work and the honest benchmark for it. It is
*ahead* of this repository on operational realism — an eighteen-airframe fleet
with real capacities and turnaround times, a real fire (the 2020 Bear Fire) on
real terrain, and explicit aleatoric and epistemic uncertainty — and it already
separates the two agents, water scaling the current burning probability and
retardant decaying a persistent fuel field.

Three things remain different here. Its fire model is a frozen CNN that emits the
parameters of a cellular automaton, trained without suppression in the data, so
the drop is multiplied onto dynamics that never saw one; in this repository
suppressant raises fuel moisture and the fire stops where moisture passes the
fuel's moisture of extinction — a mechanism already inside the spread model, and
therefore calibratable against [4] and [5] rather than assumed, and inspectable
cell by cell against a reference implementation. Its optimiser runs three
thousand Adam steps per scenario over drop poses; this environment is built to
train one amortised policy that generalises across fires. And its code is not
released. Fidelity upgrades come from [1], [2] and [6]; the fleet in [8] is the
model for extending `platforms.py` beyond three generic classes.

[1] `dada2025real` · [2] `calbrix2023numerical` · [3] `wu2024review` ·
[4] Lovellette, USDA Forest Service · [5] `plucinski2024methodologies` ·
[6] `martins2026beyond` · [7] `murray2024advancing` · [8] `mateidifferentiable`
([arXiv:2606.13633](https://arxiv.org/abs/2606.13633)) — full entries in
[`litterature.md`](litterature.md).

## Realistic Design Decisions

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
| helicopter | 3,000 L | 25 m disc | 10 min | water |
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
- Reload, empty-tank and dispatch gates are hard (non-differentiable) by
  nature. Drop position, heading, volume, and all of the physics are smooth.
- `EnvConfig.dispatch_s` holds the fleet on the ground for the first
  `dispatch_s` seconds - detection, reporting and the flight out - so the fire
  burns unopposed however good the policy is. It defaults to 0; a real initial
  attack is tens of minutes.
- No crown fire, no spotting, no fire-atmosphere feedback. Those belong with
  the Rothermel work.
