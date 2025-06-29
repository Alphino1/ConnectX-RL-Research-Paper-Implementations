# AlphaZero for ConnectX and Other Research Implementations

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)

## Table of Contents

1. [Overview](#overview)
2. [Objectives](#objectives)
3. [Methodology](#methodology)
4. [System Design](#system-design)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Repository Structure](#repository-structure)
8. [Multi-Paper Research Hub](#multi-paper-research-hub)
9. [License](#license)
10. [References](#references)

---

## Overview

This repository provides an implementation of the **AlphaZero** algorithm adapted to the **ConnectX** environment on Kaggle. The implementation is based on self-play reinforcement learning with Monte Carlo Tree Search (MCTS) guided by a residual neural network.

> **Note:** This repository is structured to serve as a multi-paper research hub. Additional reimplementations of state-of-the-art papers will be added as separate subdirectories.

---

## Objectives

- **ConnectX Adaptation**: Implement the AlphaZero paradigm on the 6×7 ConnectX grid.
- **Baseline Foundation**: Provide a compute-efficient, reproducible implementation.
- **Multi-Paper Repository**: Expand the repo with further deep learning and RL research paper reimplementations.
- **Extensibility**: Ensure modular and documented design for easy integration of new ideas.

---

## Methodology (AlphaZero)

### 1. Self-Play Data Generation
Agents generate training data by playing against themselves using MCTS guided by neural priors.

### 2. Neural Network Architecture
- **Input**: Two-channel tensor for current player and opponent.
- **Backbone**: 5 residual blocks with 128 filters and batch normalization.
- **Heads**:
  - Policy head: outputs action probabilities.
  - Value head: evaluates the current board state.

### 3. MCTS Enhancements
- **PUCT**: Balances exploration/exploitation.
- **Dirichlet Noise**: Injected at the root to encourage exploration.
- **Value Propagation**: Uses alternating signs for perspective switching.

### 4. Training Loop
- Iterative: Self-play → data aggregation → training.
- **Loss Function**: Combined policy (cross-entropy), value (MSE), and L2 regularization.

---

## System Design


| Module                  | Description                                      |
|-------------------------|--------------------------------------------------|
| `game/ConnectXState.py` | Game logic and fast win detection                |
| `mcts.py`               | MCTS algorithm with exploration enhancements     |
| `network.py`            | Residual CNN with dual heads                     |
| `self_play.py`          | Orchestrates self-play data generation           |
| `train.py`              | Handles batching, loss computation, training     |
| `evaluate.py`           | Elo-style evaluation against baselines           |

---

## Installation

```bash
## Installation

```bash
git clone https://github.com/Alphino1/ConnectX-RL-Research-Paper-Implementations.git
cd ConnectX-RL-Research-Paper-Implementations
pip install -r requirements.txt

> Requires: Python ≥ 3.8, PyTorch ≥ 1.9, NumPy, tqdm




```
## Usage

1. Run Training (AlphaZero)

python train.py --iterations 5 --self_play_games 50 --mcts_simulations 200

2. Run Evaluation

python evaluate.py --checkpoint checkpoints/iter_5.pth --episodes 100

3. Explore Notebook

Open the following notebook in Jupyter:

notebook/alphazero_connectx.ipynb

for a step-by-step walkthrough and visualizations.





---

## Multi-Paper Research Hub

This repository will evolve into a consolidated library of multiple research paper reimplementations. Each paper will be added under its own directory, maintaining:

1. Interactive Jupyter notebooks


2. Modular scripts and training code


3.  Well-documented README files


4. (Optional) Unit tests



Example Future Additions:

paper_muzero/

paper_alphago/



This design enables structured, scalable growth of the repository for both learning and contribution.


---

## License

This project is licensed under the MIT License.


---

## References

1. Silver, D., Hubert, T., Schrittwieser, J., et al. Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm, Science, 2018.


2. Kaggle ConnectX Competition – https://www.kaggle.com/competitions/connectx



