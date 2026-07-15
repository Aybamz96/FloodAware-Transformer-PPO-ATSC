# FloodAware-Transformer-PPO-ATSC

## Flood-Aware Adaptive Traffic Signal Control Using a Lightweight Transformer, Proximal Policy Optimization, and an Antifragility Index

This repository contains the official implementation of the research project:

> **Antifragile Traffic Signal Control Under Progressive Flood Disruption: A Transformer-Enhanced Proximal Policy Optimization Framework**

The project develops a flood-aware adaptive traffic signal controller capable of maintaining efficient traffic operation under progressively increasing roadway flooding using:

- Lightweight Transformer encoder
- Proximal Policy Optimization (PPO)
- SUMO microscopic traffic simulation
- TraCI interface
- Antifragility Index (AI)

---

# Overview

Urban flooding significantly degrades traffic operations by reducing roadway capacity, increasing congestion, and disrupting traffic signal coordination.

This project introduces an adaptive traffic signal control framework that combines temporal traffic representation with reinforcement learning to improve traffic performance under flood conditions.

Unlike conventional adaptive controllers, the proposed framework continuously learns traffic behavior while quantifying system adaptation using an Antifragility Index.

---

# Repository Structure

```text
FloodAware-Transformer-PPO-ATSC
│
├── src/
├── simulation/
├── experiments/
├── figures/
├── results/
│   ├── csv/
│   ├── plots/
│   └── logs/
├── docs/
├── README.md
├── LICENSE
└── requirements.txt
```

---

# Methodology

The framework consists of five major components:

1. Progressive flood scenario generation
2. SUMO microscopic traffic simulation
3. Lightweight Transformer temporal encoder
4. PPO adaptive signal controller
5. Antifragility evaluation

The controller receives a 16-dimensional traffic state vector every second through TraCI, encodes the recent traffic history using an 8-step Transformer encoder, and selects adaptive signal timings using PPO.

---

# Flood Scenarios

| Scenario | Flood Depth |
|-----------|------------:|
| Baseline | 0.00 m |
| Mild | 0.12 m |
| Moderate | 0.28 m |
| Severe | 0.55 m |

---

# Performance Metrics

The framework evaluates:

- Antifragility Index (AI)
- Service Rate
- Throughput
- Waiting Time
- Queue Length
- Reward

---

# Requirements

- Python 3.11+
- SUMO
- TraCI
- PyTorch
- Stable-Baselines3
- NumPy
- Pandas
- Matplotlib

---

# Running the Simulation

```bash
python experiments/flood_sumo_integration_v65.py
```

---

# Results

The paper evaluates the controller using:

- Progressive flood scenarios
- Five independent random seeds
- Statistical performance analysis
- Comparative evaluation

---

# Citation

If you use this repository, please cite:

```bibtex
Citation will be added after publication.
```

---

# License

MIT License.
