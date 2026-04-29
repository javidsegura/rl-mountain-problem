"""State representation strategies (Strategy pattern).

Each Representation maps a raw (pos, vel) observation to a feature vector
or a discrete index. Tabular agents use Discretizer; deep agents use Raw
(default) or Engineered (adds energy + slope).
"""

from mountaincar_rl.representations.base import Representation
from mountaincar_rl.representations.discretizer import Discretizer
from mountaincar_rl.representations.engineered import EngineeredFeatures
from mountaincar_rl.representations.rbf import RBFFeatures
from mountaincar_rl.representations.tile_coding import TileCoder

__all__ = [
    "Representation",
    "Discretizer",
    "EngineeredFeatures",
    "RBFFeatures",
    "TileCoder",
]
