# OSAC: Optical Integrated Sensing and Communication

This repository contains the implementation for the paper: **"Optical Integrated Sensing and Communication: Deep Reinforcement Learning-based Beam Tracking for Dynamic Vehicle-to-Everything Networks"**.

## Abstract

This project investigates **Deep Reinforcement Learning (DRL)** for continuous optical beam tracking in **Vehicle-to-Everything (V2X)** scenarios under realistic channel impairments, including atmospheric turbulence, pointing jitter, and line-of-sight blockages. The beam tracking problem is formulated as a Markov Decision Process (MDP) with an 18-dimensional continuous state space and continuous action space. A shaped reward function is utilized to jointly optimize Signal-to-Noise Ratio (SNR) for Vehicle-to-Vehicle (V2V) and Vehicle-to-Infrastructure (V2I) links while ensuring robust connectivity in dynamic environments.

## Features

- **Custom Gymnasium Environment**: `OSAC_V2X_Env` simulates vehicle physics, optical channel characteristics (path loss, pointing error, turbulence), and traffic dynamics.
- **Single-Agent RL Support**: Compatible with Stable Baselines3 algorithms (PPO, SAC, DDPG, TRPO, A2C).
- **Realistic Channel Model**: Includes geometric loss, atmospheric turbulence (log-normal fading), and blockage penalties.
- **Visualization**: Real-time rendering using Pygame to visualize beam alignment and vehicle interactions.
- **Benchmarking**: Scripts to compare different RL algorithms and continuous control baselines.

## Setup

This project is managed with `uv` for dependency resolution and virtual environment management.

### Prerequisites

- Python >= 3.12
- [uv](https://github.com/astral-sh/uv)

### Installation

1. **Clone the repository:**

   ```bash
   git clone <repository_url>
   cd osac
   ```

2. **Install dependencies:**

   ```bash
   uv sync
   ```

   This will create a virtual environment and install all required packages specified in `pyproject.toml`.

## Usage

### Training

To train an RL agent, run one of the training scripts located in `osac/training_scripts/`. For example, to train a PPO agent:

```bash
uv run python osac/training_scripts/training_PPO.py
```

Other available algorithms:

- `training_SAC.py`
- `training_DDPG.py`
- `training_TRPO.py`
- `training_a2c.py`

Logs and models will be saved in the `osac/` directory.

### Visualization & Evaluation

To visualize a trained model or run a baseline evaluation:

```bash
uv run python osac/demo.py
```

To generate performance plots (results will be saved to `osac/results/`):

```bash
uv run python osac/algo_comparison.py
uv run python osac/power_vs_time.py
uv run python osac/policy_comparision.py
```

## Project Structure

```text
osac/
├── osac_env.py             # Main Gymnasium Environment
├── training_scripts/       # Training scripts for different RL algorithms
│   ├── training_PPO.py
│   ├── training_SAC.py
│   └── ...
├── results/                # Generated plots and visualizations
├── models/                 # Saved model checkpoints
├── algo_comparison.py      # Plot learning curves
├── power_vs_time.py        # Plot simplified power analysis
└── policy_comparision.py   # Compare final policy performance
```
