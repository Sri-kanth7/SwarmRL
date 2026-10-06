# Checkpoint Contract

Interface between the **training** layer (producer) and the
**backend** inference layer (consumer).

Implementation: `training/checkpoint.py` (metadata helpers, Week 1).
Actual weight files are produced in later weeks; this document
fixes the handoff format so both sides can be built against it.

## Directory Layout

```text
training/checkpoints/           # SWARMRL_CHECKPOINT_DIR / project.yaml
└── <run_id>/                   # e.g. "mappo_2026w2" or "latest"
    ├── metadata.json           # required, this contract
    ├── policy.pt               # policy weights (later week)
    └── config_snapshot.yaml    # copy of the training config (later week)
```

`training/checkpoints/` is git-ignored (only `.gitkeep` is tracked).

## Metadata File (`metadata.json`)

Produced by `training.checkpoint.save_metadata`, read back with
`training.checkpoint.load_metadata`.

| Field | Type | Required | Meaning |
|---|---|---|---|
| `format_version` | `integer` | yes | Checkpoint-contract version; `1` for this document |
| `algorithm` | `string` | yes | `"MAPPO"` or `"IPPO"` |
| `step` | `integer >= 0` | yes | Training iteration the checkpoint reflects |
| `observation_dim` | `integer >= 0` | yes | Observation-vector dimension of the policy |
| `action_dim` | `integer >= 0` | yes | Action-vector dimension (`3` for this project) |
| `agent_count` | `integer >= 0` | yes | Swarm size the policy was trained for |
| `seed` | `integer` | yes | Seed of the training run |
| `created_at` | `string` (ISO-8601 UTC) | yes | Creation timestamp (written on save) |
| `config_path` | `string \| null` | no | Training YAML used |
| `weights_file` | `string \| null` | no | File name of the policy weights; `null` until produced |

Example:

```json
{
  "format_version": 1,
  "algorithm": "MAPPO",
  "step": 0,
  "observation_dim": 21,
  "action_dim": 3,
  "agent_count": 50,
  "seed": 42,
  "created_at": "2026-01-01T00:00:00+00:00",
  "config_path": "training/configs/mappo.yaml",
  "weights_file": null
}
```

## Handoff Rules

1. A checkpoint directory is **complete** only when `metadata.json`
   exists and `weights_file` names an existing file inside it.
2. Consumers must check `format_version` and refuse versions they
   do not understand.
3. Consumers must check `observation_dim`, `action_dim`, and
   `agent_count` against the runtime environment before loading.
4. Metadata is JSON only — no pickled objects in `metadata.json`.
5. Directories may be archived or deleted freely; the directory
   layout is recreated by the producer.

## Consumer Flow (later weeks)

```text
resolve checkpoint dir (SWARMRL_CHECKPOINT_DIR)
  -> load metadata.json, validate format_version
  -> verify observation/action dims against EnvConfig
  -> load policy weights (weights_file)
  -> backend inference engine serves StepMessage as before
```

## Out of Scope (later weeks)

Weight serialization format, checkpoint rotation/cleanup, remote
artifact storage, resume-from-checkpoint training.
