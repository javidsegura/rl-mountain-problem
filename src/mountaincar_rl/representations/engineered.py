"""Engineered features — physics-informed augmentations of the raw state.

We add: total mechanical energy (PE + KE) and the local slope angle.
This is the "engineered" representation called out in the rubric (PDF p7).
"""

from __future__ import annotations

import numpy as np

from mountaincar_rl.representations.base import Representation


class EngineeredFeatures(Representation):
    """Returns [pos, vel, energy, slope_angle] as a 4-d feature vector."""

    @staticmethod
    def potential_energy(x: float) -> float:
        return float(np.sin(3.0 * x))

    @staticmethod
    def kinetic_energy(v: float) -> float:
        return float(0.5 * v * v * 100.0)  # scale to PE range; see wrappers/energy.py

    @staticmethod
    def slope_angle(x: float) -> float:
        # slope of the hill y = sin(3x) is 3·cos(3x); arctan gives the angle
        return float(np.arctan(3.0 * np.cos(3.0 * x)))

    def encode(self, obs: np.ndarray) -> np.ndarray:
        x, v = float(obs[0]), float(obs[1])
        pe = self.potential_energy(x)
        ke = self.kinetic_energy(v)
        slope = self.slope_angle(x)
        return np.array([x, v, pe + ke, slope], dtype=np.float32)
