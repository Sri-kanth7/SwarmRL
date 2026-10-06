# SwarmRL

## Multi-Agent Deep Reinforcement Learning Simulator

SwarmRL is a multi-agent deep reinforcement learning simulator
designed to coordinate a swarm of 50 autonomous drones in a
simulated disaster zone.

The objective is to train the drones to search and cover the
environment cooperatively while minimizing collisions and
producing an emergent, decentralized search pattern.

---

## Project Overview

Training a single autonomous agent in an environment is a
well-established problem. Coordinating a large swarm of
autonomous agents introduces additional challenges such as:

- Multi-agent coordination
- Collision avoidance
- Collective area coverage
- Decentralized decision making
- Continuous 3D navigation
- Scalable reinforcement learning

SwarmRL addresses these challenges using Multi-Agent
Proximal Policy Optimization (MAPPO) with a centralized
critic during training and decentralized actors during
inference.

---

## System Architecture

```text
                   SwarmRL
                      |
        +-------------+-------------+
        |             |             |
        v             v             v
   Environment     RL Engine     Backend
        |             |             |
        |             |             |
   PettingZoo     Ray RLlib      Python
   NumPy          PyTorch        WebSocket
        |             |             |
        +-------------+-------------+
                      |
                      v
                3D Frontend
                      |
                React + Three.js