"""Backend metrics collection.

Keeps a bounded history of :class:`server.schemas.MetricsSnapshot`
records and computes the mean per-drone reward of the latest step.
Full time-series streaming and dashboards are later work; Week 1
only fixes the shape of the data.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Sequence

from server.schemas import MetricsSnapshot


class MetricsCollector:
    """Collects metrics snapshots produced by the state generator."""

    def __init__(self, max_history: int = 1000) -> None:
        if max_history < 1:
            raise ValueError("max_history must be >= 1")
        self._history: deque[MetricsSnapshot] = deque(maxlen=max_history)
        self._latest: MetricsSnapshot | None = None

    @property
    def latest(self) -> MetricsSnapshot | None:
        """Most recent snapshot, or ``None`` before the first record."""
        return self._latest

    def __len__(self) -> int:
        return len(self._history)

    def record(
        self,
        *,
        step: int,
        explored_pct: float,
        collisions: int,
        rewards: Sequence[float] = (),
    ) -> MetricsSnapshot:
        """Record one snapshot and return it.

        Args:
            step: Current step counter.
            explored_pct: Explored coverage percentage in ``[0, 100]``.
            collisions: Collision count to store.
            rewards: Per-drone rewards of the latest step; their mean
                becomes ``mean_reward`` (0.0 when empty).
        """
        mean_reward = float(sum(rewards) / len(rewards)) if rewards else 0.0
        snapshot = MetricsSnapshot(
            step=int(step),
            explored_pct=float(explored_pct),
            collisions=int(collisions),
            mean_reward=mean_reward,
        )
        self._latest = snapshot
        self._history.append(snapshot)
        return snapshot

    def history(self) -> list[MetricsSnapshot]:
        """Recorded snapshots, oldest first."""
        return list(self._history)

    def summary(self) -> dict[str, Any]:
        """Aggregate view of the recorded history."""
        if not self._history:
            return {
                "samples": 0,
                "step": 0,
                "explored_pct": 0.0,
                "collisions": 0,
                "mean_reward": 0.0,
                "best_explored_pct": 0.0,
            }
        latest = self._history[-1]
        return {
            "samples": len(self._history),
            "step": latest.step,
            "explored_pct": latest.explored_pct,
            "collisions": latest.collisions,
            "mean_reward": latest.mean_reward,
            "best_explored_pct": max(s.explored_pct for s in self._history),
        }

    def reset(self) -> None:
        """Clear collected history."""
        self._history.clear()
        self._latest = None
