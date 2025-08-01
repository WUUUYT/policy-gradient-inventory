# Reinforcement Learning for Two-Echelon Inventory Control

A policy-gradient (REINFORCE) implementation that learns optimal ordering policies for a two-stage inventory system, minimizing holding, shortage and transportation costs over a finite horizon.

---

## Features

- **Custom Gym-style Environment**  
  - `TwoEchelonEnv` implements a two-stage inventory model with stochastic demand, fixed/setup and per-unit purchase costs, holding/shortage costs, and inter-installation transfer costs.  
- **REINFORCE Agent**  
  - `ReinforceAgent` samples continuous order quantities via Normal distributions, computes discounted returns, and updates policy parameters (`θ₁`, `θ₂`) with vanilla policy-gradient.  
- **Configurable Hyperparameters**  
  - All model costs, demand rate, horizon (T), discount (γ), learning rate, etc. are read from a single `config/default.yaml`.  
- **Modular Code Structure**  
  - Clear separation of environment, agent, training script, and utilities for logging & plotting.  

---

## Project Structure

    two-echelon-inv-rl/
    ├── config/
    │   └── default.yaml        # hyperparameters & env settings
    ├── envs/
    │   └── two_echelon_env.py  # TwoEchelonEnv: reset(), step(a1,a2) → (state, reward, done)
    ├── agents/
    │   └── reinforce_agent.py  # ReinforceAgent: select_action(), update()
    ├── scripts/
    │   └── train.py            # training loop: batch episodes → agent.update()
    ├── utils/
    │   ├── logger.py           # CSV/TensorBoard logging
    │   └── plot.py             # plot loss & policy-parameter curves
    ├── outputs/                # checkpoints/ & logs/
    ├── requirements.txt        # Python dependencies
    ├── .gitignore
    └── README.md

---

## Installation

    # 1. Clone repo
    git clone https://github.com/YOUR_USERNAME/two-echelon-inv-rl.git
    cd two-echelon-inv-rl

    # 2. Create & activate virtualenv
    python3 -m venv .venv
    source .venv/bin/activate     # macOS/Linux

    # 3. Install dependencies
    pip install -r requirements.txt

---

## Configuration

Edit `config/default.yaml` to tweak:

    env:
      h1:             2.0        # holding cost @ installation 1
      p1:             5.0        # shortage cost @ installation 1
      h2:             1.0        # holding cost @ installation 2
      demand_lambda:  5          # demand mean
      init_x1:        5.0
      init_w1:        5.0
      init_x2:        10.0
      init_w2:        0.0
      K:              0.0
      c:              1.0
      c1:             1.0
      T:              100        # episode length
    agent:
      theta1:         20.0
      theta2:         25.0
      sigma1:         0.6
      sigma2:         0.6
      lr:             0.02
      gamma:          0.9
    train:
      batch_size:     10
      num_batches:    2400

---

## Training

    # Run the training loop
    python scripts/train.py --config config/default.yaml

---

## Visualization

    from utils.plot import plot_training_curve
    plot_training_curve("outputs/logs/training.csv")

---

## License

This project is licensed under the [MIT License](LICENSE).

---

> _Feel free to open issues or pull requests!_  
> _Author: Yitong Wu_  
> _Last updated: 2025-08-01_