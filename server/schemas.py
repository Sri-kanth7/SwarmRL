"""Pydantic message schemas for the backend.

These models are the single source of truth for the WebSocket and
HTTP payloads documented in ``docs/contracts/websocket_schema.md``
and ``docs/contracts/metrics_schema.md``. The TypeScript types in
``frontend/src/types/`` mirror them field for field.

Week 1 covers the state/metrics messages. Control messages (pause,
speed, camera focus) are added in later weeks without changing the
fields defined here.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class BaseMessage(BaseModel):
    """Base class adding version-agnostic serialization."""

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-dict form of the message (pydantic v1/v2)."""
        dump = getattr(self, "model_dump", None)
        if callable(dump):
            return dump()
        return self.dict()


class AgentState(BaseMessage):
    """State of a single drone for one streamed step.

    Field semantics match the environment's y-up coordinate system:
    ``x``/``z`` span the ground plane, ``y`` is height, angles are
    in radians.
    """

    id: int = Field(ge=0, description="Drone index (0..agent_count-1)")
    x: float = Field(description="World x position")
    y: float = Field(description="World y position (height)")
    z: float = Field(description="World z position")
    pitch: float = Field(default=0.0, description="Nose pitch in radians")
    yaw: float = Field(default=0.0, description="Heading in radians")


class StepMessage(BaseMessage):
    """Per-step swarm state streamed to the frontend.

    Field contract:

    - ``step``: monotonically increasing step counter (int >= 0)
    - ``agents``: one :class:`AgentState` per drone
    - ``newly_visited_cells``: flat coverage-cell indices explored
      during this step (``row * cols + col``)
    - ``collisions``: collision events during this step (int >= 0)
    - ``explored_pct``: explored coverage in ``[0, 100]`` (float)
    """

    step: int = Field(default=0, ge=0)
    agents: list[AgentState] = Field(default_factory=list)
    newly_visited_cells: list[int] = Field(default_factory=list)
    collisions: int = Field(default=0, ge=0)
    explored_pct: float = Field(default=0.0, ge=0.0, le=100.0)


class MetricsSnapshot(BaseMessage):
    """Latest metrics snapshot (see ``docs/contracts/metrics_schema.md``).

    - ``explored_pct``: explored coverage in ``[0, 100]``
    - ``collisions``: cumulative collision events of the episode
    - ``mean_reward``: mean per-drone reward of the latest step
    """

    step: int = Field(default=0, ge=0)
    explored_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    collisions: int = Field(default=0, ge=0)
    mean_reward: float = Field(default=0.0)
