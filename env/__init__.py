"""SwarmRL environment package.

Exposes the PettingZoo Parallel environment and its configuration:

    from env import SwarmEnv, EnvConfig

    env = SwarmEnv(EnvConfig())
    observations, infos = env.reset()
"""

from env.config import EnvConfig
from env.swarm_env import SwarmEnv

__all__ = ["EnvConfig", "SwarmEnv"]
