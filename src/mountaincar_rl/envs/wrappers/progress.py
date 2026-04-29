"""Progress-based shaping: bonus on Δposition toward the goal.

Naive but informative — encourages monotone rightward motion. Note this
shaping is *not* potential-based and can distort the optimal policy
(may push the car directly right against gravity instead of oscillating).
We include it precisely to *show* that effect in the comparative analysis.
"""

from __future__ import annotations

import gymnasium as gym

from mountaincar_rl.config import GOAL_POSITION


class ProgressShapingWrapper(gym.Wrapper):
    """Adds `scale * (x' - x)` to the reward."""

    def __init__(self, env: gym.Env, scale: float = 10.0):
        super().__init__(env)
        self.scale = scale
        self._prev_x: float | None = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_x = float(obs[0])
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        x = float(obs[0])
        bonus = self.scale * (x - (self._prev_x or x))
        self._prev_x = x
        # Optional small one-shot bonus on reaching the goal (helps DQN bootstrap)
        if terminated and x >= GOAL_POSITION:
            bonus += 10.0
        info["shaping_bonus"] = bonus
        return obs, float(reward) + bonus, terminated, truncated, info
