"""Unit tests for metrics aggregation (training) and metrics
collection (backend)."""

import pytest

from server.metrics import MetricsCollector
from training.metrics import EpisodeMetrics, MetricsAggregator


def test_aggregator_empty_summary():
    aggregator = MetricsAggregator()
    assert len(aggregator) == 0
    summary = aggregator.summary()
    assert summary["episodes"] == 0.0
    assert summary["mean_reward"] == 0.0


def test_aggregator_summary_aggregates_episodes():
    aggregator = MetricsAggregator()
    aggregator.add(
        EpisodeMetrics(
            episode=0, steps=100, mean_reward=1.5, explored_pct=40.0, collisions=3
        )
    )
    aggregator.add(
        EpisodeMetrics(
            episode=1, steps=120, mean_reward=2.5, explored_pct=60.0, collisions=1
        )
    )
    summary = aggregator.summary()
    assert summary["episodes"] == 2.0
    assert summary["mean_reward"] == pytest.approx(2.0)
    assert summary["mean_explored_pct"] == pytest.approx(50.0)
    assert summary["mean_collisions"] == pytest.approx(2.0)
    assert summary["mean_steps"] == pytest.approx(110.0)
    assert summary["best_explored_pct"] == pytest.approx(60.0)
    assert len(aggregator.episodes) == 2


def test_aggregator_reset_clears_history():
    aggregator = MetricsAggregator()
    aggregator.add(
        EpisodeMetrics(
            episode=0, steps=10, mean_reward=0.0, explored_pct=1.0, collisions=0
        )
    )
    aggregator.reset()
    assert len(aggregator) == 0


def test_collector_records_mean_reward_and_bounded_history():
    collector = MetricsCollector(max_history=3)
    first = collector.record(
        step=1, explored_pct=10.0, collisions=0, rewards=[1.0, 2.0]
    )
    assert first.mean_reward == pytest.approx(1.5)
    assert collector.latest is first
    for step in range(2, 6):
        collector.record(step=step, explored_pct=step * 10.0, collisions=0)
    assert len(collector) == 3
    summary = collector.summary()
    assert summary["samples"] == 3
    assert summary["step"] == 5
    assert summary["best_explored_pct"] == 50.0


def test_collector_empty_summary_and_reset():
    collector = MetricsCollector()
    assert collector.latest is None
    assert collector.summary()["samples"] == 0
    collector.record(step=1, explored_pct=0.0, collisions=0, rewards=[])
    collector.reset()
    assert collector.latest is None
    assert len(collector) == 0
