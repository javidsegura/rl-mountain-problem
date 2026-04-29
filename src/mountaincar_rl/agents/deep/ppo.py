"""PPO (Schulman et al., 2017) — works with both discrete and continuous actions.

The single algorithm that bridges all 4 scenarios in our matrix.
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO

from mountaincar_rl.config import PPO_HP


def make(env: gym.Env, seed: int, tb_log_dir: Path | None = None) -> PPO:
    """Build a fresh PPO model targeting `env` (discrete or continuous)."""
    return PPO(
        "MlpPolicy",
        env,
        seed=seed,
        verbose=0,
        tensorboard_log=str(tb_log_dir) if tb_log_dir else None,
        **PPO_HP,
    )
