"""Single source of truth: paths, seeds, training budgets, hyperparameters.

Everything that varies between smoke / demo / full modes lives here so
the rest of the codebase reads constants without branching on mode itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# --- Paths --------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
RESULTS_DIR = ARTIFACTS / "results"
CHECKPOINTS_DIR = ARTIFACTS / "checkpoints"
TB_LOGS_DIR = ARTIFACTS / "tb_logs"
FIGURES_DIR = ARTIFACTS / "figures"

for _d in (RESULTS_DIR, CHECKPOINTS_DIR, TB_LOGS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Seeds --------------------------------------------------------------------

SEEDS_FULL = (0, 1, 2)         # 3 seeds for the cached full run
SEEDS_DEMO = (0,)              # 1 seed for demo / live retrain
SEEDS_SMOKE = (0,)             # 1 seed for the crash test

# --- Training budgets ---------------------------------------------------------

Mode = Literal["smoke", "cache", "demo", "full"]


@dataclass(frozen=True)
class Budget:
    """Training budget per (algo family, mode) — used by the trainers."""

    tabular_episodes: int          # number of episodes for Q-learning / SARSA
    deep_timesteps: int            # number of env steps for DQN / PPO / SAC
    eval_episodes: int             # episodes used at the end for evaluation


# `cache` shares the demo budget — used when the notebook is in cache mode
# but a viz cell needs to train a fresh-and-fast single agent on the side.
BUDGETS: dict[Mode, Budget] = {
    "smoke": Budget(tabular_episodes=200, deep_timesteps=1_000, eval_episodes=5),
    "cache": Budget(tabular_episodes=500, deep_timesteps=5_000, eval_episodes=10),
    "demo":  Budget(tabular_episodes=500, deep_timesteps=5_000, eval_episodes=10),
    "full":  Budget(tabular_episodes=1500, deep_timesteps=30_000, eval_episodes=20),
}


def seeds_for(mode: Mode) -> tuple[int, ...]:
    return {"smoke": SEEDS_SMOKE, "cache": SEEDS_DEMO,
            "demo": SEEDS_DEMO, "full": SEEDS_FULL}[mode]


# --- Algorithm hyperparameters ------------------------------------------------
# Kept here rather than scattered in each agent file: one place to tune.

TABULAR_HP = {
    "alpha": 0.1,                  # learning rate
    "gamma": 0.99,                 # discount
    "epsilon_start": 1.0,
    "epsilon_end": 0.05,
    "epsilon_decay_episodes": 800, # linear decay over this many episodes
    "discretizer_bins": (40, 40),  # (position_bins, velocity_bins)
}

DQN_HP = {
    "learning_rate": 4e-4,
    "buffer_size": 50_000,
    "learning_starts": 1_000,
    "batch_size": 128,
    "gamma": 0.99,
    "target_update_interval": 600,
    "train_freq": 16,
    "gradient_steps": 8,
    "exploration_fraction": 0.2,
    "exploration_final_eps": 0.07,
    "policy_kwargs": {"net_arch": [256, 256]},
}

PPO_HP = {
    "learning_rate": 3e-4,
    "n_steps": 1024,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "policy_kwargs": {"net_arch": [64, 64]},
}

SAC_HP = {
    "learning_rate": 3e-4,
    "buffer_size": 50_000,
    "learning_starts": 1_000,
    "batch_size": 256,
    "gamma": 0.99,
    "tau": 0.005,
    "train_freq": 1,
    "gradient_steps": 1,
    "policy_kwargs": {"net_arch": [256, 256]},
}

# --- Environment constants (reproduced from gymnasium for analysis use) -------

POS_MIN, POS_MAX = -1.2, 0.6
VEL_MIN, VEL_MAX = -0.07, 0.07
GOAL_POSITION = 0.5
GRAVITY_DISCRETE = 0.0025          # cos(3x) * G  in the discrete dynamics
FORCE_DISCRETE = 0.001
POWER_CONTINUOUS = 0.0015
GRAVITY_CONTINUOUS = 0.0025
