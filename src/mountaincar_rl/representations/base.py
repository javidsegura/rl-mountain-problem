"""Abstract Representation — Strategy pattern interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Representation(ABC):
    """Maps a 2-d MountainCar observation to features (continuous or discrete)."""

    @abstractmethod
    def encode(self, obs: np.ndarray) -> Any:
        """Return the encoded form. Type depends on subclass:
        - Discretizer → tuple[int, int] (a state index)
        - TileCoder / RBFFeatures / EngineeredFeatures → np.ndarray
        """
