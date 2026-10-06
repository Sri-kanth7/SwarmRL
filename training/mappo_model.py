"""Network architecture specifications for the training layer.

Week 1 describes the actor/critic network shapes that MAPPO and
IPPO will use. No weights are created here; building the actual
PyTorch modules and the centralized critic is Week 2 work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class NetworkSpec:
    """Feed-forward network description.

    Attributes:
        input_dim: Dimension of the network input.
        output_dim: Dimension of the network output.
        hidden_sizes: Sizes of the hidden layers.
        activation: Activation name (``relu``, ``tanh``, ``elu``).
    """

    input_dim: int
    output_dim: int
    hidden_sizes: tuple[int, ...] = (256, 256)
    activation: str = "relu"

    def to_dict(self) -> dict[str, Any]:
        """Serializable representation of the specification."""
        return {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "hidden_sizes": list(self.hidden_sizes),
            "activation": self.activation,
        }


@dataclass(frozen=True)
class ActorCriticSpec:
    """Actor/critic pair used by the training layer.

    Week 1: both networks use per-drone inputs. The centralized
    critic (global/central input) is introduced in Week 2 by
    replacing ``critic`` with a central-input specification.
    """

    actor: NetworkSpec
    critic: NetworkSpec
    notes: str = field(default="week1: decentralized actor + per-agent critic")

    def to_dict(self) -> dict[str, Any]:
        """Serializable representation of the specification."""
        return {
            "actor": self.actor.to_dict(),
            "critic": self.critic.to_dict(),
            "notes": self.notes,
        }


def default_spec(
    observation_dim: int,
    action_dim: int,
    model_config: Mapping[str, Any] | None = None,
) -> ActorCriticSpec:
    """Build the default actor/critic specification.

    Args:
        observation_dim: Environment observation dimension.
        action_dim: Environment action dimension.
        model_config: Optional ``fcnet_hiddens``/``activation`` overrides.
    """
    model_config = model_config or {}
    hidden = tuple(int(v) for v in model_config.get("fcnet_hiddens", (256, 256)))
    activation = str(model_config.get("activation", "relu"))
    actor = NetworkSpec(observation_dim, action_dim, hidden, activation)
    critic = NetworkSpec(observation_dim, 1, hidden, activation)
    return ActorCriticSpec(actor=actor, critic=critic)
