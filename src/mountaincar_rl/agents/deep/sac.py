"""SAC (Haarnoja et al., 2018) — continuous action spaces only.

Off-policy maximum-entropy actor-critic. Strong sample efficiency on
continuous control; the standard pick for MountainCarContinuous.
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
from stable_baselines3 import SAC

from mountaincar_rl.config import SAC_HP


def make(env: gym.Env, seed: int, tb_log_dir: Path | None = None) -> SAC:
    """Build a fresh SAC model targeting `env`."""
    return SAC(
        "MlpPolicy",
        env,
        seed=seed,
        verbose=0,
        tensorboard_log=str(tb_log_dir) if tb_log_dir else None,
        **SAC_HP,
    )
