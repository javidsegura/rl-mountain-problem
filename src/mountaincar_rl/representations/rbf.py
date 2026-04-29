"""Radial-basis-function features over the (pos, vel) state space.

Each feature is exp(-‖s − c_k‖² / (2σ²)) for a grid of centers c_k. Smooth
generalization, often better than tile coding for small problems.
"""

from __future__ import annotations

import numpy as np

from mountaincar_rl.config import POS_MAX, POS_MIN, VEL_MAX, VEL_MIN
from mountaincar_rl.representations.base import Representation


class RBFFeatures(Representation):
    """Gaussian RBF features on a regular grid of centers."""

    def __init__(self, n_pos: int = 8, n_vel: int = 8, sigma: float = 0.1):
        pos = np.linspace(POS_MIN, POS_MAX, n_pos)
        vel = np.linspace(VEL_MIN, VEL_MAX, n_vel)
        pp, vv = np.meshgrid(pos, vel, indexing="ij")
        self._centers = np.stack([pp.ravel(), vv.ravel()], axis=1)  # (K, 2)
        # Different scales per dim (pos range much larger than vel range)
        self._scales = np.array([POS_MAX - POS_MIN, VEL_MAX - VEL_MIN]) * sigma

    @property
    def n_features(self) -> int:
        return self._centers.shape[0]

    def encode(self, obs: np.ndarray) -> np.ndarray:
        diff = (np.asarray(obs, dtype=np.float32) - self._centers) / self._scales
        return np.exp(-0.5 * np.sum(diff * diff, axis=1)).astype(np.float32)
