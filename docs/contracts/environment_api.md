# Environment API Contract

Stable interface between the **environment** and the **training**
layers (and any other consumer of simulation state).

Implementation: `env/swarm_env.py` (`SwarmEnv`), configuration in
`env/config.py` (`EnvConfig`).

## Type: PettingZoo ParallelEnv

`SwarmEnv` subclasses `pettingzoo.utils.env.ParallelEnv`:

- all agents act simultaneously,
- one `step(actions)` advances the whole swarm,
- spaces are per-agent and identical across agents.

```python
from env import SwarmEnv, EnvConfig

env = SwarmEnv(EnvConfig(agent_count=50))
observations, infos = env.reset(seed=42)
observations, rewards, terminations, truncations, infos = env.step(actions)
```

## Agent Identifiers

- Type: `str`
- Pattern: `"drone_0" ... "drone_{n-1}"` where
  `n = EnvConfig.agent_count` (default **50**).
- `env.possible_agents` is the full list; `env.agents` holds the
  currently live agents (empty after an episode ends).
- The integer index inside the identifier (`env.agent_index(agent)`)
  is the same `id` used in WebSocket messages.

## reset()

```python
observations, infos = env.reset(seed=None, options=None)
```

| Argument | Type | Meaning |
|---|---|---|
| `seed` | `int \| None` | Reseeds the spawn generator; omitted uses the configured seed |
| `options` | `dict \| None` | Reserved, ignored in Week 1 |

Returns `({agent: obs}, {agent: info})` for **every** agent.
Reset restores: fresh spawn states (uniform inside the world minus
`spawn_margin`, zero velocity, zero pitch, uniform yaw), an empty
coverage grid, and zeroed counters.

## step(actions)

```python
observations, rewards, terminations, truncations, infos = env.step(actions)
```

| Argument | Type | Meaning |
|---|---|---|
| `actions` | `dict[str, ndarray]` | Exactly one action vector per live agent; missing/extra keys raise `ValueError` |

Return value:

| Element | Type | Semantics |
|---|---|---|
| `observations` | `dict[str, ndarray]` | Only agents still alive (empty at episode end) |
| `rewards` | `dict[str, float]` | Step reward for every agent that acted |
| `terminations` | `dict[str, bool]` | Always `False` in Week 1 (no terminal state yet) |
| `truncations` | `dict[str, bool]` | `True` when `step_count >= max_steps` |
| `infos` | `dict[str, dict]` | Per-agent info (see below) for every agent that acted |

Episode lifecycle: `reset()` → `step()` × `max_steps` → all agents
truncated, `env.agents == []`, next `reset()` starts a new episode.

### Info keys

| Key | Type | Meaning |
|---|---|---|
| `step` | `int` | Steps elapsed so far in the episode |
| `explored_pct` | `float` | Explored coverage `[0, 100]` |
| `collisions` | `int` | Cumulative collision events of the episode |
| `step_collisions` | `int` | Collision events of this step |
| `collision` | `bool` | This agent was involved in an agent-agent collision |
| `boundary_contact` | `bool` | This agent contacted the world boundary |
| `newly_visited_cells` | `list[int]` | Flat coverage indices this agent explored |
| `episode_reward` | `float` | Cumulative reward of this agent |
| `agent_index` | `int` | Integer index matching the wire-format `id` |

## observation_space(agent)

`gymnasium.spaces.Box`, `shape = (13 + lidar_rays,)` (default
**21**), `dtype = float32`. Layout (all values clipped to bounds):

| Block | Content | Dim | Range |
|---|---|---|---|
| 1 | Own position `(x, y, z) / world_size` | 3 | `[0, 1]` |
| 2 | Own velocity `(vx, vy, vz) / max_speed` | 3 | `[-1, 1]` |
| 3 | Own attitude `pitch / max_pitch`, `yaw / pi` | 2 | `[-1, 1]` |
| 4 | Nearest neighbour: relative offset `/ world_size` (3), distance / world diagonal (1) | 4 | `[-1, 1]`, `[0, 1]` |
| 5 | LiDAR-like ranges: `lidar_rays` horizontal rays to the world boundary / diagonal | `K` | `[0, 1]` |
| 6 | Local coverage: visited fraction of cells within `local_coverage_radius` | 1 | `[0, 1]` |

Block 4 uses `[0, 0, 0, 1]` when the agent is alone (no neighbour).
Layout source of truth: `env/observations.py`.

## action_space(agent)

`gymnasium.spaces.Box`, `shape = (3,)`, `dtype = float32`:

| Index | Name | Unit | Range | Semantics |
|---|---|---|---|---|
| 0 | `velocity` | m/s | `[-max_speed, max_speed]` | Forward speed along the drone's heading |
| 1 | `pitch` | rad | `[-max_pitch, max_pitch]` | **Absolute** commanded pitch (clipped) |
| 2 | `yaw` | rad | `[-pi, pi]` | **Absolute** commanded heading (wrapped) |

Commands are absolute, not incremental. Movement per step:
`position += heading(pitch, yaw) * velocity * dt`, where
`heading = (cos p cos y, sin p, cos p sin y)`. If the move leaves
the world box the position is clipped to the boundary and the step
reports a boundary contact.

## Configuration Parameters (`EnvConfig`)

| Field | Default | Meaning |
|---|---|---|
| `agent_count` | `50` | Swarm size |
| `world_size` | `(100, 50, 100)` | `(x, y, z)` extents, y-up |
| `seed` | `42` | Default spawn seed |
| `dt` | `0.1` | Seconds per step |
| `max_steps` | `500` | Truncation horizon |
| `max_speed` | `5.0` | Velocity action bound |
| `max_pitch` | `0.6` | Pitch action bound |
| `collision_radius` | `1.5` | Pairwise collision distance |
| `lidar_rays` | `8` | Horizontal range rays |
| `coverage_cell_size` | `5.0` | Coverage cell edge length |
| `local_coverage_radius` | `10.0` | Local coverage query radius |
| `spawn_margin` | `5.0` | Spawn distance from walls |
| `reward` | `RewardConfig()` | Reward weights (see below) |

`EnvConfig.from_dict(...)` builds a config from YAML/RLlib
`env_config` mappings (unknown keys ignored); `to_dict()` is the
inverse.

## Reward Interface (`env/rewards.py`)

| Term | Default weight | Applied when |
|---|---|---|
| `exploration` | `+1.0` per newly explored cell | Agent discovers new coverage |
| `collision` | `-10.0` | Agent involved in an agent-agent collision |
| `boundary` | `-0.1` | Agent contacts the world boundary |
| `time` | `-0.01` | Every step |
| `energy` | `0.0` (reserved) | Speed-dependent term (later) |
| `spread` | `0.0` (reserved) | Swarm-spreading shaping (later) |

Collision events for metrics: one event per colliding pair plus one
per boundary contact.

## Determinism

Same seed ⇒ same spawn states and same observation sequence.
`reset(seed=k)` reseeds the generator; otherwise the configured
seed applies on first use.

## Out of Scope (later weeks)

Dynamic obstacles, wind, curriculum, terminal states for full
coverage, reward tuning, performance optimization.
