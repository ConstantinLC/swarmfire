"""The batched environment.

There is no per-environment Python loop anywhere. `batch` independent fires
advance in lockstep as one set of tensor ops, which is what turns eight A6000s
into thousands of simultaneous episodes.

Two ways to use it:

* **RL** - `reset` / `step`, with `detach=True` (the default) so no autograd
  graph accumulates.
* **Differentiable simulation** - `detach=False`, then backpropagate a loss
  such as `state.burned_area().sum()` straight through the fire physics into
  whatever produced the actions. This is the path the convolution-based
  controller in `goals.md` wants; it is strictly more informative than a scalar
  reward, and it is available because every gate in the model is smooth.

A Gymnasium adapter lives in `gym_wrapper.py` for off-the-shelf algorithms, but
train against this class directly when throughput matters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor

from . import fuels
from .platforms import Platform, make_platform
from .propagate import CAPropagator, Propagator
from .ros import ROSModel, SimpleROS
from .state import FireState, ignite
from .suppression import SuppressionModel


@dataclass
class EnvConfig:
    batch: int = 8
    grid: tuple[int, int] = (128, 128)
    cell_size: float = 30.0
    dt: float = 5.0
    horizon: int = 400
    preset: str = "grass"
    wind: tuple[float, float] = (4.0, 0.0)
    slope: tuple[float, float] = (0.0, 0.0)
    heterogeneous: bool = False
    ignition: tuple[int, int] | None = None  # defaults to grid centre
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # Reward weights, in hectares-equivalent per unit.
    water_cost_per_kl: float = 0.02
    seed: int | None = None


class FireEnv:
    """Fire propagation plus one suppression platform, batched."""

    def __init__(
        self,
        platform: Platform | str = "helicopter",
        ros_model: ROSModel | None = None,
        suppression: SuppressionModel | None = None,
        propagator: Propagator | None = None,
        config: EnvConfig | None = None,
    ):
        self.cfg = config or EnvConfig()
        self.platform = make_platform(platform) if isinstance(platform, str) else platform
        self.suppression = suppression or SuppressionModel()
        self.ros_model = ros_model or SimpleROS()
        self.propagator = propagator or CAPropagator(self.ros_model, self.suppression)
        self.state: FireState
        self._step = 0
        self._prev_burned: Tensor

    # ------------------------------------------------------------ lifecycle

    def reset(self) -> dict[str, Tensor]:
        cfg = self.cfg
        builder = fuels.patchy_world if cfg.heterogeneous else fuels.uniform_world
        kwargs = dict(
            batch=cfg.batch,
            grid=cfg.grid,
            cell_size=cfg.cell_size,
            wind=cfg.wind,
            slope=cfg.slope,
            device=cfg.device,
        )
        if cfg.heterogeneous:
            kwargs["seed"] = cfg.seed
        else:
            kwargs["preset"] = cfg.preset
        state = builder(**kwargs)

        y, x = cfg.ignition or (cfg.grid[0] // 2, cfg.grid[1] // 2)
        self.state = ignite(state, y, x)
        self.platform.reset(cfg.batch, cfg.grid, torch.device(cfg.device))
        self._step = 0
        self._prev_burned = self.state.burned_area()
        return self.observe()

    # ---------------------------------------------------------- observation

    def observe(self) -> dict[str, Tensor]:
        """Grid layers for a conv net, plus the fleet's own proprioception."""
        h, w = self.state.grid
        pos = self.platform.position.clone()
        fleet = torch.cat(
            [
                pos[..., 0:1] / h * 2 - 1,
                pos[..., 1:2] / w * 2 - 1,
                (self.platform.load / self.platform.spec.capacity_l).unsqueeze(-1),
                (self.platform.cooldown / self.platform.spec.reload_s).unsqueeze(-1),
            ],
            dim=-1,
        )
        return {"grid": self.state.stack(), "fleet": fleet}

    @property
    def action_dim(self) -> int:
        return self.platform.action_dim

    @property
    def n_agents(self) -> int:
        return self.platform.spec.n_agents

    def zero_action(self) -> Tensor:
        """A no-release action of the right shape, useful as a baseline."""
        a = torch.zeros(
            self.cfg.batch, self.n_agents, self.action_dim, device=torch.device(self.cfg.device)
        )
        a[..., 2] = -1.0
        return a

    # ----------------------------------------------------------------- step

    def step(self, action: Tensor, detach: bool = True):
        """Advance one timestep.

        Returns `(obs, reward, done, info)`. Reward is negative hectares newly
        burned, minus the cost of suppressant used, so a do-nothing policy
        scores the full loss and any useful drop improves on it.

        Set `detach=False` to keep the autograd graph for differentiable
        simulation; wrap a fixed number of steps and call `.backward()` on the
        accumulated loss (truncated BPTT), then `state.detach()`.
        """
        water, retardant = self.platform.step(action, self.state, self.cfg.dt)
        state = self.suppression.deposit(self.state, water=water, retardant=retardant)
        state = self.propagator.step(state, self.cfg.dt)
        if detach:
            state = state.detach()
            self.platform.detach()
        self.state = state

        burned = self.state.burned_area()
        newly_burned = burned - self._prev_burned
        self._prev_burned = burned

        volume_kl = (water.sum(dim=(-2, -1)) * self.state.cell_area) / 1000.0
        reward = -newly_burned - self.cfg.water_cost_per_kl * volume_kl

        self._step += 1
        out = self.state.active() < 1e-3
        done = out | (self._step >= self.cfg.horizon)

        info = {
            "burned_ha": burned,
            "active": self.state.active(),
            "extinguished": out,
            "step": self._step,
        }
        return self.observe(), reward, done, info

    # ------------------------------------------------------------- rollouts

    def rollout(
        self,
        policy: Callable[[dict[str, Tensor]], Tensor] | None = None,
        steps: int | None = None,
        detach: bool = True,
        record: bool = False,
    ) -> dict:
        """Run to the horizon. `policy` maps an observation to an action.

        With `policy=None` the fire burns unsuppressed, which is the baseline
        every controller has to beat.
        """
        obs = self.observe()
        steps = steps or self.cfg.horizon
        total = torch.zeros(self.cfg.batch, device=self.state.device)
        frames: list[Tensor] = []

        for _ in range(steps):
            action = self.zero_action() if policy is None else policy(obs)
            obs, reward, done, info = self.step(action, detach=detach)
            total = total + reward
            if record:
                frames.append(self.state.stack().detach().cpu())
            if bool(done.all()):
                break

        return {
            "return": total,
            "burned_ha": self.state.burned_area(),
            "steps": self._step,
            "frames": frames,
            "info": info,
        }

    # -------------------------------------------------------------- sanity

    def check_stability(self) -> tuple[float, bool]:
        """`(max stable dt, whether the configured dt is safe)`.

        Explicit propagation is only conditionally stable. Call this after
        changing wind, fuel, or resolution - an unstable step shows up as a
        fire that spreads faster than any wind justifies.
        """
        limit = self.ros_model.max_stable_dt(self.state)
        return limit, self.cfg.dt <= limit
