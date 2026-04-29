"""Env factory — single entry point that maps (scenario, shaping) to a ready env.

This is the only place the rest of the codebase calls gym.make() with a wrapped
env. Everything else just receives the resulting env.
"""

from __future__ import annotations

from typing import Literal

import gymnasium as gym

from mountaincar_rl.envs.scenarios import SPECS, Scenario
from mountaincar_rl.envs.wrappers import (
    EnergyShapingWrapper,
    FuelCostWrapper,
    PotentialShapingWrapper,
    ProgressShapingWrapper,
    StepsCostWrapper,
    VelocityShapingWrapper,
)

Shaping = Literal["none", "energy", "progress", "velocity", "potential"]


def make_env(
    scenario: Scenario | str,
    shaping: Shaping = "none",
    seed: int | None = None,
    render_mode: str | None = None,
) -> gym.Env:
    """Build a MountainCar env for `scenario` with optional reward shaping.

    Order of wrapping (outer to inner): [shaping] → [scenario adaptation] → base env.
    Seeded via env.reset(seed=...) by the trainer, not here.
    """
    if isinstance(scenario, str):
        scenario = Scenario(scenario)
    spec = SPECS[scenario]

    env = gym.make(spec.gym_id, render_mode=render_mode)

    # 1) Scenario adaptation wrappers (3 = discrete+fuel, 4 = continuous+steps)
    if spec.needs_fuel_wrapper:
        env = FuelCostWrapper(env, cost=1.0)
    if spec.needs_steps_wrapper:
        env = StepsCostWrapper(env, cost=0.1)

    # 2) Reward shaping wrapper (analyst's choice; "none" = unchanged)
    if shaping == "energy":
        env = EnergyShapingWrapper(env, scale=10.0)
    elif shaping == "progress":
        env = ProgressShapingWrapper(env, scale=10.0)
    elif shaping == "velocity":
        env = VelocityShapingWrapper(env, scale=10.0)
    elif shaping == "potential":
        env = PotentialShapingWrapper(env, gamma=0.99, scale=10.0)
    elif shaping != "none":
        raise ValueError(f"Unknown shaping: {shaping}")

    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)

    return env
