"""Energy-based reward shaping: bonus proportional to Δ(PE + KE).

Encourages the agent to gain mechanical energy — the physical insight that
solves MountainCar. PE = sin(3x), KE = 0.5*v² (in the env's natural units).
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np


def total_energy(state: np.ndarray) -> float:
    """Mechanical energy (potential + kinetic) for a (pos, vel) state."""
    x, v = float(state[0]), float(state[1])
    return float(np.sin(3.0 * x) + 0.5 * v * v * 100.0)
    # The factor 100 brings KE onto the same scale as PE (v_max²·0.5 ≈ 2.5e-3,
    # much smaller than PE ≈ 1). Empirically helps shaping have an effect.


class EnergyShapingWrapper(gym.Wrapper):
    """Adds `scale * (E(s') - E(s))` to the reward at every step."""

    def __init__(self, env: gym.Env, scale: float = 1.0):
        super().__init__(env)
        self.scale = scale
        self._prev_energy: float | None = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_energy = total_energy(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        e_new = total_energy(obs)
        bonus = self.scale * (e_new - (self._prev_energy or e_new))
        self._prev_energy = e_new
        info["shaping_bonus"] = bonus
        return obs, float(reward) + bonus, terminated, truncated, info
