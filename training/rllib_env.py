"""RLlib adapter around the PettingZoo swarm environment.

RLlib's current API stack drives environments through
:class:`ray.rllib.env.multi_agent_env.MultiAgentEnv`. The PettingZoo
environment in :mod:`env` already follows the multi-agent ``reset`` /
``step`` contract, so this module only adds what RLlib requires on top:

* the ``MultiAgentEnv`` base class (space accessors, ``_agent_ids``)
  and the ``__all__`` episode flag RLlib reads from every step, and
* optional collection of :class:`training.metrics.EpisodeMetrics`
  when an episode finishes.

The adapter is registered with RLlib by
:func:`training.register_swarm_env`, which is what makes
``config.environment(env="swarm")`` work.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from env import EnvConfig, SwarmEnv
from training.metrics import EpisodeMetrics, MetricsAggregator

ALL_AGENTS_KEY = "__all__"


class SwarmMultiAgentEnv(MultiAgentEnv):
    """``MultiAgentEnv`` view of :class:`env.SwarmEnv`.

    Args:
        env_config: Environment settings forwarded to
            :meth:`env.config.EnvConfig.from_dict`. Unknown keys are
            ignored; RLlib adds worker metadata to the mapping it
            passes to the environment creator.
        collector: Optional collector that receives one
            :class:`~training.metrics.EpisodeMetrics` per finished
            episode. Collection only works for environments created in
            the collector's own process (training runs configured with
            ``num_env_runners: 0``).
    """

    metadata = {"name": "swarm_v1", "render_modes": []}

    def __init__(
        self,
        env_config: Mapping[str, Any] | None = None,
        *,
        collector: MetricsAggregator | None = None,
    ) -> None:
        super().__init__()
        self._swarm = SwarmEnv(EnvConfig.from_dict(env_config))
        self.agents: list[str] = []
        self.possible_agents = list(self._swarm.possible_agents)
        self._agent_ids = set(self.possible_agents)
        self.observation_spaces = {
            agent: self._swarm.observation_space(agent) for agent in self.possible_agents
        }
        self.action_spaces = {
            agent: self._swarm.action_space(agent) for agent in self.possible_agents
        }
        self.observation_space = self._swarm.observation_space(self.possible_agents[0])
        self.action_space = self._swarm.action_space(self.possible_agents[0])
        self.render_mode = None
        self._collector = collector

    @property
    def swarm(self) -> SwarmEnv:
        """The wrapped PettingZoo environment."""
        return self._swarm

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
        """Start a new episode and mirror the live agent list."""
        observations, infos = self._swarm.reset(seed=seed, options=options)
        self.agents = list(self._swarm.agents)
        return observations, infos

    def step(
        self, action_dict: Mapping[str, np.ndarray]
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        """Advance the swarm one step and add RLlib's ``__all__`` flags.

        On the final step of an episode the wrapped environment drops its
        agents and returns no observations; every agent that just acted is
        given its last observation back, because RLlib needs it to bootstrap
        the value function of a truncated episode.
        """
        observations, rewards, terminations, truncations, infos = self._swarm.step(
            dict(action_dict)
        )
        self.agents = list(self._swarm.agents)
        missing = [agent for agent in rewards if agent not in observations]
        if missing:
            observations = {
                **observations,
                **{agent: self._swarm.observe(agent) for agent in missing},
            }
        terminateds = dict(terminations)
        truncateds = dict(truncations)
        terminateds[ALL_AGENTS_KEY] = bool(terminateds) and all(terminateds.values())
        truncateds[ALL_AGENTS_KEY] = bool(truncateds) and all(truncateds.values())
        if terminateds[ALL_AGENTS_KEY] or truncateds[ALL_AGENTS_KEY]:
            self._record_episode(infos)
        return observations, rewards, terminateds, truncateds, infos

    def close(self) -> None:
        """Release the wrapped environment."""
        self._swarm.close()

    def _record_episode(self, infos: Mapping[str, Mapping[str, Any]]) -> None:
        """Store one :class:`EpisodeMetrics` record for a finished episode."""
        if self._collector is None or not infos:
            return
        samples = list(infos.values())
        self._collector.add(
            EpisodeMetrics(
                episode=len(self._collector),
                steps=int(max(sample["step"] for sample in samples)),
                mean_reward=float(
                    np.mean([sample["episode_reward"] for sample in samples])
                ),
                explored_pct=float(
                    np.mean([sample["explored_pct"] for sample in samples])
                ),
                collisions=int(max(sample["collisions"] for sample in samples)),
            )
        )


__all__ = ["ALL_AGENTS_KEY", "SwarmMultiAgentEnv"]
