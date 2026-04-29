"""Tabular Q-learning (Watkins, 1989).

    Q(s, a) ← Q(s, a) + α [ r + γ·max_a' Q(s', a') − Q(s, a) ]

Off-policy TD control: the bootstrap target uses max over next actions,
regardless of the action that the behavior policy (ε-greedy) actually picks.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mountaincar_rl.agents.base import Agent
from mountaincar_rl.config import TABULAR_HP
from mountaincar_rl.representations.discretizer import Discretizer


class QLearningAgent(Agent):
    name = "q_learning"

    def __init__(
        self,
        n_actions: int = 3,
        discretizer: Discretizer | None = None,
        hp: dict | None = None,
        seed: int = 0,
    ):
        self.n_actions = n_actions
        self.discretizer = discretizer or Discretizer(*TABULAR_HP["discretizer_bins"])
        self.hp = {**TABULAR_HP, **(hp or {})}
        self.rng = np.random.default_rng(seed)

        n_pos, n_vel = self.discretizer.shape
        self.Q = np.zeros((n_pos, n_vel, n_actions), dtype=np.float64)
        self.epsilon = self.hp["epsilon_start"]

    # --- ε-greedy ----------------------------------------------------------------

    def act(self, obs: np.ndarray, *, greedy: bool = False) -> int:
        i, j = self.discretizer.encode(obs)
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return int(np.argmax(self.Q[i, j]))

    # --- single-transition update -----------------------------------------------

    def update(self, obs, action, reward, next_obs, done):
        i, j = self.discretizer.encode(obs)
        ni, nj = self.discretizer.encode(next_obs)
        target = reward + (0.0 if done else self.hp["gamma"] * np.max(self.Q[ni, nj]))
        td_error = target - self.Q[i, j, action]
        self.Q[i, j, action] += self.hp["alpha"] * td_error
        return td_error

    # --- exploration schedule ----------------------------------------------------

    def decay_epsilon(self, episode: int) -> None:
        eps_s, eps_e = self.hp["epsilon_start"], self.hp["epsilon_end"]
        frac = min(1.0, episode / self.hp["epsilon_decay_episodes"])
        self.epsilon = eps_s + frac * (eps_e - eps_s)

    # --- (de)serialization -------------------------------------------------------

    def save(self, path: Path) -> None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, Q=self.Q, epsilon=self.epsilon)

    def load(self, path: Path) -> None:
        data = np.load(path)
        self.Q = data["Q"]
        self.epsilon = float(data["epsilon"])
