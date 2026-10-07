"""Shared RLlib configuration builder used by the training entry points.

Two layers are provided:

* :func:`build_algorithm_config` returns a plain, JSON-friendly
  dictionary. It documents the full configuration contract and powers
  ``--print-config`` without importing Ray RLlib.
* :func:`build_ppo_config` translates the same
  :class:`~training.config.TrainingConfig` into a live RLlib
  ``PPOConfig`` that can be handed to ``AlgorithmConfig.build()``.

Ray objects are only created by :func:`build_ppo_config` (which imports
RLlib lazily) and by the algorithm built from it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

from env import EnvConfig, SwarmEnv
from training.config import TrainingConfig

if TYPE_CHECKING:
    from ray.rllib.algorithms.ppo import PPOConfig

POLICY_ID = "swarm_policy"

NUM_ITERATIONS_KEY = "num_iterations"
"""Entry-point setting: how many times ``Algorithm.train()`` is called."""

TRAINING_KEYS = frozenset(
    {
        "clip_param",
        "entropy_coeff",
        "entropy_coeff_schedule",
        "gamma",
        "grad_clip",
        "grad_clip_by",
        "kl_coeff",
        "kl_target",
        "lambda_",
        "lr",
        "minibatch_size",
        "model",
        "num_epochs",
        "optimizer",
        "shuffle_batch_per_epoch",
        "train_batch_size",
        "train_batch_size_per_learner",
        "use_critic",
        "use_gae",
        "use_kl_loss",
        "vf_clip_param",
        "vf_loss_coeff",
    }
)
"""Settings forwarded to ``PPOConfig.training()``."""

ENV_RUNNER_KEYS = frozenset(
    {
        "batch_mode",
        "num_env_runners",
        "num_envs_per_env_runner",
        "rollout_fragment_length",
    }
)
"""Settings forwarded to ``PPOConfig.env_runners()``."""

LEGACY_TRAINING_KEYS = {
    "gae_lambda": "lambda_",
    "num_workers": "num_env_runners",
    "sgd_minibatch_size": "minibatch_size",
}
"""Pre-Week-2 configuration names and their RLlib equivalents."""

SUPPORTED_TRAIN_KEYS = (
    TRAINING_KEYS | ENV_RUNNER_KEYS | {NUM_ITERATIONS_KEY} | set(LEGACY_TRAINING_KEYS)
)


def map_agent_to_policy(agent_id: str, episode: Any = None, **kwargs: Any) -> str:
    """Map every drone to the shared swarm policy (module-level so the
    mapping stays picklable)."""
    return POLICY_ID


def build_algorithm_config(config: TrainingConfig) -> dict[str, Any]:
    """Assemble an RLlib-style algorithm configuration dictionary.

    The returned mapping contains everything a training run
    needs: registry name, environment kwargs, spaces for the policy
    spec, model settings, multi-agent layout, trainer settings, and the
    global seed. No Ray objects are created here; use
    :func:`build_ppo_config` for the RLlib ``AlgorithmConfig``.
    """
    env = SwarmEnv(EnvConfig.from_dict(config.env_config))
    seed = config.seed
    return {
        "algo_class_name": config.algorithm,
        "env": config.env_name,
        "env_config": dict(config.env_config),
        "framework": config.framework,
        "seed": seed,
        "torch_seed": seed,
        "observation_space": env.observation_space(env.possible_agents[0]),
        "action_space": env.action_space(env.possible_agents[0]),
        "model": dict(config.model),
        "multi_agent": {
            "policies": {POLICY_ID: None},
            "policy_mapping_fn": map_agent_to_policy,
            "policy_ids": [POLICY_ID],
        },
        "train": dict(config.train),
        "checkpoint_dir": config.checkpoint_dir,
    }


def translate_train_settings(train: Mapping[str, Any]) -> dict[str, Any]:
    """Translate a ``train`` mapping into RLlib ``AlgorithmConfig`` calls.

    Legacy names (:data:`LEGACY_TRAINING_KEYS`) are renamed, entry-point
    settings (``num_iterations``) are dropped, and unknown names raise a
    ``ValueError`` that lists every supported setting.

    Args:
        train: Raw ``train`` block of a training configuration.

    Returns:
        Settings split by target: ``ENV_RUNNER_KEYS`` land in
        :meth:`PPOConfig.env_runners`, everything else in
        :meth:`PPOConfig.training`.

    Raises:
        ValueError: If a setting is not recognised.
    """
    settings: dict[str, Any] = {}
    for key, value in train.items():
        target = LEGACY_TRAINING_KEYS.get(key, key)
        if target == NUM_ITERATIONS_KEY:
            continue
        if target not in TRAINING_KEYS | ENV_RUNNER_KEYS:
            raise ValueError(
                f"unknown training setting {key!r}; supported settings are "
                f"{sorted(SUPPORTED_TRAIN_KEYS)}"
            )
        settings[target] = value
    return settings


def build_ppo_config(config: TrainingConfig) -> "PPOConfig":
    """Build the RLlib ``PPOConfig`` for an IPPO run.

    The environment spaces are taken from a freshly constructed
    :class:`env.SwarmEnv`, and the shared policy is registered with
    explicit spaces so RLlib does not have to infer them from the
    vectorized environment.

    Args:
        config: Resolved training configuration.

    Returns:
        A configured, not yet built, RLlib ``PPOConfig``.
    """
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.policy.policy import PolicySpec

    settings = translate_train_settings(config.train)
    env_runner_settings = {
        key: settings.pop(key) for key in list(settings) if key in ENV_RUNNER_KEYS
    }
    # "auto" derives the fragment length from the train batch size, which keeps
    # RLlib's batch-size validation happy for any batch size we are given.
    env_runner_settings.setdefault("rollout_fragment_length", "auto")

    env = SwarmEnv(EnvConfig.from_dict(config.env_config))
    observation_space = env.observation_space(env.possible_agents[0])
    action_space = env.action_space(env.possible_agents[0])

    return (
        PPOConfig()
        .environment(env=config.env_name, env_config=dict(config.env_config))
        .framework(framework=config.framework)
        .debugging(seed=config.seed)
        .rl_module(model_config=dict(config.model))
        .multi_agent(
            policies={
                POLICY_ID: PolicySpec(
                    observation_space=observation_space,
                    action_space=action_space,
                )
            },
            policy_mapping_fn=map_agent_to_policy,
        )
        .env_runners(**env_runner_settings)
        .training(**settings)
    )


def num_iterations(config: TrainingConfig) -> int:
    """Number of ``Algorithm.train()`` calls requested by the config.

    Raises:
        ValueError: If the configured value is not a positive integer.
    """
    value = int(config.train.get(NUM_ITERATIONS_KEY, 1))
    if value < 1:
        raise ValueError(f"{NUM_ITERATIONS_KEY} must be >= 1, got {value}")
    return value


def summarize(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return a JSON-friendly summary of an algorithm configuration."""
    summary: dict[str, Any] = {}
    for key, value in config.items():
        if key in {"observation_space", "action_space", "policy_mapping_fn"}:
            summary[key] = str(value)
        else:
            summary[key] = value
    return summary


__all__ = [
    "ENV_RUNNER_KEYS",
    "NUM_ITERATIONS_KEY",
    "POLICY_ID",
    "SUPPORTED_TRAIN_KEYS",
    "TRAINING_KEYS",
    "build_algorithm_config",
    "build_ppo_config",
    "map_agent_to_policy",
    "num_iterations",
    "summarize",
    "translate_train_settings",
]
