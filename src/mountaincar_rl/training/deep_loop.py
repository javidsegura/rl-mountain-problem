"""Deep training loop — wraps SB3's `.learn()` and snapshots episode metrics.

We attach a custom callback that records episode reward + length at every
episode end (drained from `info["episode"]` which the Monitor wrapper writes).
This gives us learning curves on the same shape as the tabular trainer's output,
so plotting code stays unified.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordEpisodeStatistics
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback

from mountaincar_rl.config import GOAL_POSITION


@dataclass
class TrainResult:
    """Mirrors tabular_loop.TrainResult so viz code is shared."""

    rewards: np.ndarray
    lengths: np.ndarray
    successes: np.ndarray
    final_epsilon: float = 0.0  # not meaningful for SB3 algos


class _EpisodeRecorder(BaseCallback):
    """Drains episode stats logged by `RecordEpisodeStatistics`."""

    def __init__(self):
        super().__init__()
        self.rewards: list[float] = []
        self.lengths: list[int] = []
        self.successes: list[bool] = []

    def _on_step(self) -> bool:
        # SB3 forwards the underlying gym `info` dicts in `self.locals["infos"]`
        infos = self.locals.get("infos", [])
        for info in infos:
            ep = info.get("episode")
            if ep is not None:
                self.rewards.append(float(ep["r"]))
                self.lengths.append(int(ep["l"]))
                # Final obs lives at info["terminal_observation"] when the
                # episode ended naturally; otherwise we infer from current obs.
                term_obs = info.get("terminal_observation")
                if term_obs is not None:
                    self.successes.append(bool(term_obs[0] >= GOAL_POSITION))
                else:
                    self.successes.append(False)
        return True


def train_deep(
    model: BaseAlgorithm,
    env: gym.Env,
    n_timesteps: int,
    *,
    progress: bool = True,
    desc: str | None = None,
) -> TrainResult:
    """Train SB3 `model` on `env` for `n_timesteps` env-steps."""
    # `RecordEpisodeStatistics` populates info["episode"] = {"r": .., "l": ..}
    # which our callback drains. Monkey-patch the live env in the model.
    if not isinstance(env, RecordEpisodeStatistics):
        env = RecordEpisodeStatistics(env)
        model.set_env(env)

    recorder = _EpisodeRecorder()
    model.learn(
        total_timesteps=n_timesteps,
        callback=recorder,
        progress_bar=progress,
        log_interval=10,
        reset_num_timesteps=True,
        tb_log_name=desc or "run",
    )

    return TrainResult(
        rewards=np.asarray(recorder.rewards, dtype=np.float64),
        lengths=np.asarray(recorder.lengths, dtype=np.int64),
        successes=np.asarray(recorder.successes, dtype=bool),
    )
