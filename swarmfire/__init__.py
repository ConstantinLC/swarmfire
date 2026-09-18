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
from .fuel_models import FUEL_MODEL_NAMES, FUEL_MODELS
from .fuels import PRESETS, patchy_world, rothermel_world, uniform_world
from .platforms import DiscPlatform, LineDropPlatform, Platform, PlatformSpec, make_platform
from .propagate import CAPropagator, Propagator
from .ros import ConstantROS, ROSModel, RothermelROS, SimpleROS
from .state import DYNAMIC_LAYERS, OBS_LAYERS, STATIC_LAYERS, FireState, ignite
from .suppression import SuppressionModel

__version__ = "0.1.0"

__all__ = [
    "FireEnv", "EnvConfig",
    "FireState", "ignite", "OBS_LAYERS", "DYNAMIC_LAYERS", "STATIC_LAYERS",
    "uniform_world", "patchy_world", "rothermel_world", "PRESETS",
    "FUEL_MODELS", "FUEL_MODEL_NAMES",
    "ROSModel", "SimpleROS", "ConstantROS", "RothermelROS",
    "Propagator", "CAPropagator",
    "SuppressionModel",
    "Platform", "PlatformSpec", "DiscPlatform", "LineDropPlatform", "make_platform",
]
