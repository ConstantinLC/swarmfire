"""swarmfire - differentiable, batched wildfire propagation and suppression.

    from swarmfire import FireEnv, EnvConfig

    env = FireEnv("helicopter", config=EnvConfig(batch=64, device="cuda"))
    env.reset()
    out = env.rollout()            # unsuppressed baseline
    print(out["burned_ha"].mean())

The pieces are meant to be replaced one at a time:

    ros_model    SimpleROS -> RothermelROS      (physics realism)
    propagator   CAPropagator -> level set      (front geometry)
    platform     helicopter -> tanker / swarm   (action constraints)
    suppression  water/retardant -> foam, gel   (agent chemistry)
"""

from .env import EnvConfig, FireEnv
from .fuels import PRESETS, patchy_world, uniform_world
from .platforms import DiscPlatform, LineDropPlatform, Platform, PlatformSpec, make_platform
from .propagate import CAPropagator, Propagator
from .ros import ConstantROS, ROSModel, RothermelROS, SimpleROS
from .state import DYNAMIC_LAYERS, OBS_LAYERS, STATIC_LAYERS, FireState, ignite
from .suppression import SuppressionModel

__version__ = "0.1.0"

__all__ = [
    "FireEnv", "EnvConfig",
    "FireState", "ignite", "OBS_LAYERS", "DYNAMIC_LAYERS", "STATIC_LAYERS",
    "uniform_world", "patchy_world", "PRESETS",
    "ROSModel", "SimpleROS", "ConstantROS", "RothermelROS",
    "Propagator", "CAPropagator",
    "SuppressionModel",
    "Platform", "PlatformSpec", "DiscPlatform", "LineDropPlatform", "make_platform",
]
