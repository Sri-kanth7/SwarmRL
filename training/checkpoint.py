"""Checkpoint handoff foundation.

Week 1 defines the checkpoint directory layout and the metadata
record that travels with every future policy checkpoint. Writing
model weights (``torch`` state dicts, Ray RLlib checkpoints) and
loading them for inference are Week 2 work; the metadata helpers
below are already usable by tooling and tests.

See ``docs/contracts/checkpoint_contract.md`` for the full contract.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

CHECKPOINT_FORMAT_VERSION = 1
METADATA_FILENAME = "metadata.json"


@dataclass(frozen=True)
class CheckpointMetadata:
    """Versioned description of a checkpoint directory.

    Attributes:
        format_version: Version of the checkpoint contract.
        algorithm: Algorithm that produced the checkpoint (MAPPO/IPPO).
        step: Training iteration the checkpoint corresponds to.
        observation_dim: Observation-vector dimension of the policy.
        action_dim: Action-vector dimension of the policy.
        agent_count: Number of drones the policy was trained for.
        seed: Seed of the training run.
        created_at: UTC ISO-8601 creation timestamp.
        config_path: Training configuration file used, if any.
        weights_file: Policy-weights file name (None until Week 2).
    """

    format_version: int = CHECKPOINT_FORMAT_VERSION
    algorithm: str = ""
    step: int = 0
    observation_dim: int = 0
    action_dim: int = 0
    agent_count: int = 0
    seed: int = 0
    created_at: str = ""
    config_path: str | None = None
    weights_file: str | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CheckpointMetadata":
        """Build metadata from a mapping, ignoring unknown keys."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def to_dict(self) -> dict[str, Any]:
        """Serializable representation."""
        return asdict(self)


def checkpoint_dir(base_dir: str | Path, run_id: str = "latest") -> Path:
    """Directory that holds one checkpoint."""
    return Path(base_dir) / run_id


def metadata_path(directory: str | Path) -> Path:
    """Path of the metadata file inside a checkpoint directory."""
    return Path(directory) / METADATA_FILENAME


def save_metadata(directory: str | Path, metadata: CheckpointMetadata) -> Path:
    """Write ``metadata.json`` into ``directory`` and return its path."""
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    path = metadata_path(target)
    payload = metadata.to_dict()
    if not payload.get("created_at"):
        payload["created_at"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_metadata(directory: str | Path) -> CheckpointMetadata:
    """Read ``metadata.json`` from ``directory``."""
    path = metadata_path(directory)
    data = json.loads(path.read_text(encoding="utf-8"))
    return CheckpointMetadata.from_dict(data)
