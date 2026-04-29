"""Fuel cost — for scenario (3) discrete + min-fuel.

PDF p17 specifies: 'costs proportional to number of [right, left] actions taken'
(i.e. the no-op action is free; throttling either way costs).

For continuous action spaces we use cost ∝ |a|, but the canonical
MountainCarContinuous-v0 already does this (−0.1·a²) so we leave it alone.
"""

from __future__ import annotations

import gymnasium as gym


class FuelCostWrapper(gym.Wrapper):
    """Adds an extra `−cost` whenever a non-zero throttle action is taken (discrete envs)."""

    NO_OP_DISCRETE = 1   # action 1 in MountainCar-v0 is "don't accelerate"

    def __init__(self, env: gym.Env, cost: float = 1.0):
        super().__init__(env)
        self.cost = cost

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if int(action) != self.NO_OP_DISCRETE:
            penalty = -self.cost
            reward = float(reward) + penalty
            info["fuel_penalty"] = penalty
        return obs, float(reward), terminated, truncated, info
