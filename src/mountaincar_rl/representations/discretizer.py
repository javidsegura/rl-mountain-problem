"""Discretizer — uniform grid over (position, velocity).

Used by tabular agents (Q-learning, SARSA) to bucket the continuous
state space into a finite number of (i, j) cells.
"""

from __future__ import annotations

import numpy as np

from mountaincar_rl.config import POS_MAX, POS_MIN, VEL_MAX, VEL_MIN
from mountaincar_rl.representations.base import Representation


class Discretizer(Representation):
    """Uniform grid binning. Encodes (x, v) to (i, j) integer index."""

    def __init__(self, n_pos: int = 40, n_vel: int = 40):
        self.n_pos = n_pos
        self.n_vel = n_vel
        self._pos_edges = np.linspace(POS_MIN, POS_MAX, n_pos + 1)
        self._vel_edges = np.linspace(VEL_MIN, VEL_MAX, n_vel + 1)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.n_pos, self.n_vel)

    def encode(self, obs: np.ndarray) -> tuple[int, int]:
        x, v = float(obs[0]), float(obs[1])
        i = int(np.clip(np.digitize(x, self._pos_edges) - 1, 0, self.n_pos - 1))
        j = int(np.clip(np.digitize(v, self._vel_edges) - 1, 0, self.n_vel - 1))
        return i, j

    def decode_centers(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the (pos, vel) coordinate of each bin's center — for plotting."""
        pos_centers = 0.5 * (self._pos_edges[:-1] + self._pos_edges[1:])
        vel_centers = 0.5 * (self._vel_edges[:-1] + self._vel_edges[1:])
        return pos_centers, vel_centers
