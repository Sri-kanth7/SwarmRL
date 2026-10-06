"""Unit tests for the WebSocket/metrics message schemas."""

import pytest
from pydantic import ValidationError

from server.schemas import AgentState, MetricsSnapshot, StepMessage


def test_agent_state_requires_id():
    with pytest.raises(ValidationError):
        AgentState(x=1.0, y=2.0, z=3.0)


def test_agent_state_accepts_contract_fields():
    agent = AgentState(id=0, x=1.5, y=2.0, z=3.0, pitch=0.1, yaw=-0.2)
    assert agent.id == 0
    assert agent.pitch == 0.1


def test_step_message_defaults():
    message = StepMessage()
    assert message.step == 0
    assert message.agents == []
    assert message.newly_visited_cells == []
    assert message.collisions == 0
    assert message.explored_pct == 0.0


def test_step_message_round_trip_matches_contract_fields():
    message = StepMessage(
        step=3,
        agents=[AgentState(id=0, x=1.5, y=2.0, z=3.0, pitch=0.1, yaw=-0.2)],
        newly_visited_cells=[4, 5],
        collisions=1,
        explored_pct=12.5,
    )
    data = message.to_dict()
    assert set(data) == {
        "step",
        "agents",
        "newly_visited_cells",
        "collisions",
        "explored_pct",
    }
    assert data["agents"][0]["id"] == 0
    restored = StepMessage(**data)
    assert restored.explored_pct == pytest.approx(12.5)
    assert restored.newly_visited_cells == [4, 5]


def test_step_message_rejects_out_of_contract_values():
    with pytest.raises(ValidationError):
        StepMessage(explored_pct=101.0)
    with pytest.raises(ValidationError):
        StepMessage(explored_pct=-0.5)
    with pytest.raises(ValidationError):
        StepMessage(collisions=-1)
    with pytest.raises(ValidationError):
        StepMessage(step=-1)


def test_metrics_snapshot_contract():
    snapshot = MetricsSnapshot(
        step=10, explored_pct=50.0, collisions=2, mean_reward=-0.25
    )
    data = snapshot.to_dict()
    assert set(data) == {"step", "explored_pct", "collisions", "mean_reward"}
    assert data["mean_reward"] == pytest.approx(-0.25)
    with pytest.raises(ValidationError):
        MetricsSnapshot(explored_pct=150.0)
