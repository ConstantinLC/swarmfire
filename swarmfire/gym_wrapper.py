"""Gymnasium adapter, for reaching for an off-the-shelf algorithm.

Use it to sanity-check against a known-good PPO implementation. Do not use it
for real training runs: it moves tensors to numpy every step and throws away
the batch dimension that makes this simulator fast. Train against `FireEnv`
directly, or vectorise it behind whatever your RL library calls a vec env.

`pip install gymnasium` - it is not a hard dependency of the package.
"""

from __future__ import annotations

import numpy as np
import torch

from .env import EnvConfig, FireEnv
from .state import OBS_LAYERS


def make_gym_env(platform: str = "helicopter", config: EnvConfig | None = None):
    import gymnasium as gym
    from gymnasium import spaces

    class SingleFireEnv(gym.Env):
        metadata = {"render_modes": ["rgb_array"]}

        def __init__(self):
            cfg = config or EnvConfig()
            cfg.batch = 1  # gym is single-env by construction
            self.env = FireEnv(platform, config=cfg)
            h, w = cfg.grid
            n, a = self.env.n_agents, self.env.action_dim

            self.observation_space = spaces.Dict(
                {
                    "grid": spaces.Box(-np.inf, np.inf, (len(OBS_LAYERS), h, w), np.float32),
                    "fleet": spaces.Box(-np.inf, np.inf, (n, 5), np.float32),
                }
            )
            self.action_space = spaces.Box(-1.0, 1.0, (n, a), np.float32)

        def _obs(self, obs):
            return {k: v[0].detach().cpu().numpy().astype(np.float32) for k, v in obs.items()}

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return self._obs(self.env.reset()), {}

        def step(self, action):
            a = torch.as_tensor(action, dtype=torch.float32, device=self.env.state.device)
            obs, reward, done, info = self.env.step(a.unsqueeze(0))
            truncated = bool(self.env._step >= self.env.cfg.horizon)
            terminated = bool(done[0]) and not truncated
            scalars = {k: (v[0].item() if torch.is_tensor(v) else v) for k, v in info.items()}
            return self._obs(obs), float(reward[0]), terminated, truncated, scalars

        def render(self):
            from .render import composite

            return (composite(self.env.state.stack()) * 255).astype(np.uint8)

    return SingleFireEnv()
