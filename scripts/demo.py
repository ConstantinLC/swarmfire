"""End-to-end pipeline check.

Runs three scenarios on the same ignition and compares them:

  1. free burn                - the baseline every controller must beat
  2. helicopter, scripted     - drops water on the head of the fire
  3. tanker, scripted         - lays a retardant line ahead of the head

Then benchmarks throughput and demonstrates that a gradient flows from burned
area back to a drop coordinate.

    python scripts/demo.py --device cuda --out out/
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from swarmfire import EnvConfig, FireEnv, render


def head_seeking_policy(env: FireEnv, lead_cells: float = 0.0):
    """Aim at the most intense cell, offset downwind by `lead_cells`.

    Crude on purpose - it exists to prove the action plumbing works and to give
    a learned policy something to beat, not to be good.
    """
    h, w = env.cfg.grid
    wind = torch.tensor(env.cfg.wind, dtype=torch.float32)
    lead = wind / wind.norm().clamp(min=1e-6) * lead_cells

    def policy(obs):
        grid = obs["grid"]
        intensity = grid[:, 2]  # OBS_LAYERS index of "intensity"
        b = intensity.shape[0]
        flat = intensity.flatten(1).argmax(dim=1)
        y = (flat // w).float() + lead[1]
        x = (flat % w).float() + lead[0]

        action = torch.zeros(b, env.n_agents, env.action_dim, device=grid.device)
        action[..., 0] = (y / (h - 1) * 2 - 1).unsqueeze(-1)
        action[..., 1] = (x / (w - 1) * 2 - 1).unsqueeze(-1)
        action[..., 2] = 1.0  # always release when able
        if env.action_dim > 3:  # tanker: lay the line across the wind
            action[..., 3] = 0.5
        return action.clamp(-1, 1)

    return policy


def run(name: str, env: FireEnv, policy=None, record=False):
    env.reset()
    limit, ok = env.check_stability()
    t0 = time.perf_counter()
    out = env.rollout(policy=policy, record=record)
    dt = time.perf_counter() - t0

    cells = env.cfg.batch * env.cfg.grid[0] * env.cfg.grid[1] * out["steps"]
    print(
        f"  {name:<22} burned {out['burned_ha'].mean():7.1f} ha   "
        f"return {out['return'].mean():8.1f}   "
        f"{out['steps']:3d} steps in {dt:5.2f}s  "
        f"({cells / dt / 1e6:6.1f} M cell-steps/s)"
        + ("" if ok else f"   [UNSTABLE: dt must be <= {limit:.2f}s]")
    )
    return env.state, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--grid", type=int, default=192)
    ap.add_argument("--steps", type=int, default=360)
    ap.add_argument("--out", default="out")
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    base = dict(
        batch=args.batch,
        grid=(args.grid, args.grid),
        horizon=args.steps,
        wind=(5.0, 2.0),
        preset="grass",
        device=args.device,
        dt=4.0,
    )

    print(f"\nswarmfire demo  |  device={args.device}  batch={args.batch}  grid={args.grid}^2\n")

    states = {}
    free_env = FireEnv("helicopter", config=EnvConfig(**base))
    states["free burn"], free_out = run("free burn", free_env, None, record=True)

    heli_env = FireEnv("helicopter", config=EnvConfig(**base))
    states["helicopter (water)"], _ = run("helicopter (water)", heli_env, head_seeking_policy(heli_env))

    tanker_env = FireEnv("tanker", config=EnvConfig(**base))
    states["tanker (retardant)"], _ = run(
        "tanker (retardant)", tanker_env, head_seeking_policy(tanker_env, lead_cells=12)
    )

    swarm_env = FireEnv("drone_swarm", config=EnvConfig(**base))
    states["drone swarm (water)"], _ = run("drone swarm (water)", swarm_env, head_seeking_policy(swarm_env))

    render.panel(states, outdir / "scenarios.png")
    render.save_gif(free_out["frames"], outdir / "free_burn.gif")
    print(f"\n  wrote {outdir/'scenarios.png'} and {outdir/'free_burn.gif'}")

    # ---- gradient through the simulator, the whole point of the soft state
    print("\n  differentiable-sim check:")
    env = FireEnv("helicopter", config=EnvConfig(**{**base, "batch": 2, "horizon": 60}))
    env.reset()
    drop = torch.zeros(2, 1, 3, device=args.device, requires_grad=True)
    obs = env.observe()
    for _ in range(60):
        obs, _, _, _ = env.step(drop.expand(2, env.n_agents, env.action_dim), detach=False)
    loss = env.state.burned_area().sum()
    loss.backward()
    print(f"    d(burned ha)/d(drop y, x, release) = {drop.grad[0, 0].tolist()}")
    print("    non-zero gradient => a CNN can be trained by backprop through the fire\n")


if __name__ == "__main__":
    main()
