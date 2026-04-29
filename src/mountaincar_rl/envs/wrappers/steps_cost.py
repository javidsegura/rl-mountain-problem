"""Step cost — for scenario (4) continuous + min-steps.

PDF p17 specifies: 'costs linearly proportional to number of non-null actions taken'.
We add a small −cost on every step (regardless of action magnitude) so the agent
is rewarded for finishing fast, on top of the env's native fuel-cost reward.
"""

from __future__ import annotations

import gymnasium as gym


class StepsCostWrapper(gym.Wrapper):
    """Adds `−cost` to the reward at every step (acts like a min-steps objective)."""

    def __init__(self, env: gym.Env, cost: float = 0.1):
        super().__init__(env)
        self.cost = cost

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, float(reward) - self.cost, terminated, truncated, info
