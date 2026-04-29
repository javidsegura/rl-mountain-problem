"""Velocity-based shaping: bonus on rightward velocity (positive v).

Encourages the car to *move right* with intent — mild compared to energy/progress
shaping. Useful as an intermediate baseline in the comparative analysis.
"""

from __future__ import annotations

import gymnasium as gym


class VelocityShapingWrapper(gym.Wrapper):
    """Adds `scale * max(0, v)` to the reward."""

    def __init__(self, env: gym.Env, scale: float = 10.0):
        super().__init__(env)
        self.scale = scale

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        v = float(obs[1])
        bonus = self.scale * max(0.0, v)
        info["shaping_bonus"] = bonus
        return obs, float(reward) + bonus, terminated, truncated, info
