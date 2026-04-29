"""DQN (Mnih et al., 2015) — discrete action spaces only.

We use SB3's MlpPolicy on the raw 2-d (pos, vel) observation. Hyperparameters
come from `mountaincar_rl.config.DQN_HP` so the rest of the codebase can stay
agnostic to algo internals.
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
from stable_baselines3 import DQN

from mountaincar_rl.config import DQN_HP


def make(env: gym.Env, seed: int, tb_log_dir: Path | None = None) -> DQN:
    """Build a fresh DQN model targeting `env`."""
    return DQN(
        "MlpPolicy",
        env,
        seed=seed,
        verbose=0,
        tensorboard_log=str(tb_log_dir) if tb_log_dir else None,
        **DQN_HP,
    )
