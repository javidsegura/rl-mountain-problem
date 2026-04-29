"""Per-agent evaluation: roll out N greedy episodes and report metrics.

Also: a `load_results()` helper that loads cached training-run JSONs
matching a glob.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np

from mountaincar_rl.agents.factory import is_tabular
from mountaincar_rl.config import GOAL_POSITION, RESULTS_DIR


@dataclass
class EvalMetrics:
    """Greedy-policy evaluation summary."""

    mean_reward: float
    std_reward: float
    success_rate: float
    mean_steps: float
    std_steps: float
    n_episodes: int


def evaluate_agent(agent_or_model, env: gym.Env, n_episodes: int = 20,
                   seed: int = 1234) -> EvalMetrics:
    """Run `n_episodes` greedy rollouts and return summary statistics."""
    rewards: list[float] = []
    lengths: list[int] = []
    successes: list[bool] = []

    is_tab = hasattr(agent_or_model, "name") and is_tabular(getattr(agent_or_model, "name", ""))

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        ep_reward = 0.0
        ep_len = 0
        terminated = truncated = False
        while not (terminated or truncated):
            if is_tab:
                action = agent_or_model.act(obs, greedy=True)
            else:
                action, _ = agent_or_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += float(reward)
            ep_len += 1
        rewards.append(ep_reward)
        lengths.append(ep_len)
        successes.append(bool(terminated and obs[0] >= GOAL_POSITION))

    return EvalMetrics(
        mean_reward=float(np.mean(rewards)),
        std_reward=float(np.std(rewards)),
        success_rate=float(np.mean(successes)),
        mean_steps=float(np.mean(lengths)),
        std_steps=float(np.std(lengths)),
        n_episodes=n_episodes,
    )


def load_results(pattern: str = "*.json", root: Path | None = None) -> list[dict]:
    """Load all result JSONs matching `pattern` from `RESULTS_DIR`."""
    root = root or RESULTS_DIR
    return [json.loads(p.read_text()) for p in sorted(root.glob(pattern))]
