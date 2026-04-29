"""Potential-based reward shaping (Ng, Harada & Russell, 1999).

    F(s, a, s') = γ·Φ(s') - Φ(s)

This is the *theory-safe* shaping: it preserves the optimal policy of the
underlying MDP. We use total mechanical energy as the potential function,
so the agent is rewarded for *being* in higher-energy states (consistent
with the physical insight) without any policy distortion.
"""

from __future__ import annotations

import gymnasium as gym

from mountaincar_rl.envs.wrappers.energy import total_energy


class PotentialShapingWrapper(gym.Wrapper):
    """Reward + γ·Φ(s') − Φ(s), with Φ = total mechanical energy."""

    def __init__(self, env: gym.Env, gamma: float = 0.99, scale: float = 1.0):
        super().__init__(env)
        self.gamma = gamma
        self.scale = scale
        self._prev_phi: float | None = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_phi = total_energy(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        phi_new = total_energy(obs)
        bonus = self.scale * (self.gamma * phi_new - (self._prev_phi or phi_new))
        self._prev_phi = phi_new
        info["shaping_bonus"] = bonus
        return obs, float(reward) + bonus, terminated, truncated, info
