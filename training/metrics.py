"""Training metrics aggregation.

Week 1 defines the episode-metric record and a small aggregator that
will be fed by training runs and evaluations. The metric names match
``docs/contracts/metrics_schema.md``:

- ``mean_reward``: mean per-drone total reward of an episode,
- ``explored_pct``: explored coverage percentage in ``[0, 100]``,
- ``collisions``: collision events recorded during the episode.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class EpisodeMetrics:
    """Metrics of a single finished episode.

    Attributes:
        episode: Episode counter (0-based).
        steps: Number of steps in the episode.
        mean_reward: Mean per-drone total reward over the episode.
        explored_pct: Explored coverage percentage in ``[0, 100]``.
        collisions: Collision events during the episode.
    """

    episode: int
    steps: int
    mean_reward: float
    explored_pct: float
    collisions: int

    def to_dict(self) -> dict[str, Any]:
        """Serializable representation."""
        return asdict(self)


class MetricsAggregator:
    """Collects episode metrics and produces run-level summaries."""

    def __init__(self) -> None:
        self._episodes: list[EpisodeMetrics] = []

    def __len__(self) -> int:
        return len(self._episodes)

    @property
    def episodes(self) -> list[EpisodeMetrics]:
        """All recorded episodes, in insertion order."""
        return list(self._episodes)

    def add(self, metrics: EpisodeMetrics) -> None:
        """Record one finished episode."""
        self._episodes.append(metrics)

    def extend(self, metrics: Iterable[EpisodeMetrics]) -> None:
        """Record several finished episodes."""
        for item in metrics:
            self.add(item)

    def reset(self) -> None:
        """Discard all recorded episodes."""
        self._episodes.clear()

    def summary(self) -> dict[str, float]:
        """Aggregate statistics over the recorded episodes."""
        if not self._episodes:
            return {
                "episodes": 0.0,
                "mean_reward": 0.0,
                "mean_explored_pct": 0.0,
                "mean_collisions": 0.0,
                "mean_steps": 0.0,
                "best_explored_pct": 0.0,
            }
        count = len(self._episodes)
        return {
            "episodes": float(count),
            "mean_reward": sum(e.mean_reward for e in self._episodes) / count,
            "mean_explored_pct": sum(e.explored_pct for e in self._episodes) / count,
            "mean_collisions": sum(e.collisions for e in self._episodes) / count,
            "mean_steps": sum(e.steps for e in self._episodes) / count,
            "best_explored_pct": max(e.explored_pct for e in self._episodes),
        }
