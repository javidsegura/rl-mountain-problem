"""Explicit ε-greedy training loop for the from-scratch tabular agents.

Returns a TrainResult with per-episode rewards (for learning curves) and the
final trained agent. tqdm provides a per-episode progress bar.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
from tqdm.auto import tqdm

from mountaincar_rl.agents.tabular import QLearningAgent, SARSAAgent
from mountaincar_rl.config import GOAL_POSITION


@dataclass
class TrainResult:
    """Per-episode metrics; consumed by viz + evaluation."""

    rewards: np.ndarray         # length = n_episodes
    lengths: np.ndarray         # length = n_episodes
    successes: np.ndarray       # bool, length = n_episodes
    final_epsilon: float


def _is_success(obs, terminated: bool) -> bool:
    """An episode is a success when it terminates with the car at/past the flag."""
    return bool(terminated and obs[0] >= GOAL_POSITION)


def train_tabular(
    agent: QLearningAgent | SARSAAgent,
    env: gym.Env,
    n_episodes: int,
    *,
    seed: int = 0,
    desc: str | None = None,
) -> TrainResult:
    """Train `agent` on `env` for `n_episodes` episodes."""
    rewards = np.zeros(n_episodes, dtype=np.float64)
    lengths = np.zeros(n_episodes, dtype=np.int64)
    successes = np.zeros(n_episodes, dtype=bool)

    pbar = tqdm(range(n_episodes), desc=desc or agent.name, leave=False)
    for ep in pbar:
        obs, _ = env.reset(seed=seed + ep)
        agent.decay_epsilon(ep)

        # SARSA needs the next action chosen up front; Q-learning doesn't.
        is_sarsa = isinstance(agent, SARSAAgent)
        action = agent.act(obs) if is_sarsa else None

        ep_reward = 0.0
        ep_len = 0
        terminated = truncated = False
        while not (terminated or truncated):
            if not is_sarsa:
                action = agent.act(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += float(reward)
            ep_len += 1

            if is_sarsa:
                next_action = agent.act(next_obs) if not (terminated or truncated) else 0
                agent.update(obs, action, reward, next_obs, next_action,
                             done=terminated or truncated)
                action = next_action
            else:
                agent.update(obs, action, reward, next_obs,
                             done=terminated or truncated)
            obs = next_obs

        rewards[ep] = ep_reward
        lengths[ep] = ep_len
        successes[ep] = _is_success(obs, terminated)

        if (ep + 1) % 50 == 0:
            mean_recent = float(rewards[max(0, ep - 49):ep + 1].mean())
            sr = float(successes[max(0, ep - 49):ep + 1].mean())
            pbar.set_postfix(rew=f"{mean_recent:.1f}", succ=f"{sr:.0%}",
                             eps=f"{agent.epsilon:.2f}")

    return TrainResult(
        rewards=rewards, lengths=lengths,
        successes=successes, final_epsilon=agent.epsilon,
    )
