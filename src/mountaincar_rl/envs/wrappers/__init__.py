"""Reward-shaping wrappers (gym.Wrapper subclasses).

Each wrapper modifies the reward signal but never the observation/action spaces.
This keeps agents agnostic to which shaping is active.
"""

from mountaincar_rl.envs.wrappers.energy import EnergyShapingWrapper
from mountaincar_rl.envs.wrappers.fuel import FuelCostWrapper
from mountaincar_rl.envs.wrappers.potential import PotentialShapingWrapper
from mountaincar_rl.envs.wrappers.progress import ProgressShapingWrapper
from mountaincar_rl.envs.wrappers.steps_cost import StepsCostWrapper
from mountaincar_rl.envs.wrappers.velocity import VelocityShapingWrapper

__all__ = [
    "EnergyShapingWrapper",
    "FuelCostWrapper",
    "PotentialShapingWrapper",
    "ProgressShapingWrapper",
    "StepsCostWrapper",
    "VelocityShapingWrapper",
]
