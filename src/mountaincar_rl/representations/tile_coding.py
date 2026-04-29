"""Tile coding — overlapping uniform grids with offsets.

Classic Sutton & Barto (Ch 9) feature representation. Each tile gives a
binary feature; multiple tilings (offset by a fraction of a tile width)
provide overlapping coverage and good generalization.
"""

from __future__ import annotations

import numpy as np

from mountaincar_rl.config import POS_MAX, POS_MIN, VEL_MAX, VEL_MIN
from mountaincar_rl.representations.base import Representation


class TileCoder(Representation):
    """`n_tilings` overlapping grids, each `(tiles_per_dim, tiles_per_dim)`."""

    def __init__(self, n_tilings: int = 8, tiles_per_dim: int = 8):
        self.n_tilings = n_tilings
        self.tiles_per_dim = tiles_per_dim
        self._n_features = n_tilings * tiles_per_dim * tiles_per_dim
        self._pos_range = POS_MAX - POS_MIN
        self._vel_range = VEL_MAX - VEL_MIN

    @property
    def n_features(self) -> int:
        return self._n_features

    def encode(self, obs: np.ndarray) -> np.ndarray:
        x, v = float(obs[0]), float(obs[1])
        out = np.zeros(self._n_features, dtype=np.float32)
        for t in range(self.n_tilings):
            offset_pos = (t / self.n_tilings) * (self._pos_range / self.tiles_per_dim)
            offset_vel = (t / self.n_tilings) * (self._vel_range / self.tiles_per_dim)
            i = int(((x - POS_MIN + offset_pos) / self._pos_range) * self.tiles_per_dim)
            j = int(((v - VEL_MIN + offset_vel) / self._vel_range) * self.tiles_per_dim)
            i = max(0, min(i, self.tiles_per_dim - 1))
            j = max(0, min(j, self.tiles_per_dim - 1))
            idx = t * self.tiles_per_dim * self.tiles_per_dim + i * self.tiles_per_dim + j
            out[idx] = 1.0
        return out
