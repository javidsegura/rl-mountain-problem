"""The 4 scenarios from the assignment, as a single enum.

Scenario matrix (PDF p17):

                 Minimum steps        Minimum fuel
    Discrete     (1) MountainCar-v0   (3) MountainCar-v0 + fuel cost
    Continuous   (4) MCC-v0 + steps   (2) MountainCarContinuous-v0
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Scenario(str, Enum):
    """Identifier for one of the 4 problem variants."""

    DISCRETE_STEPS = "discrete_steps"      # (1) canonical MountainCar-v0
    CONTINUOUS_FUEL = "continuous_fuel"    # (2) canonical MountainCarContinuous-v0
    DISCRETE_FUEL = "discrete_fuel"        # (3) discrete + adapted fuel cost
    CONTINUOUS_STEPS = "continuous_steps"  # (4) continuous + adapted step cost


@dataclass(frozen=True)
class ScenarioSpec:
    """Static metadata for a scenario — used by factory + viz/labels."""

    scenario: Scenario
    gym_id: str
    is_discrete: bool
    needs_fuel_wrapper: bool
    needs_steps_wrapper: bool
    label: str


SPECS: dict[Scenario, ScenarioSpec] = {
    Scenario.DISCRETE_STEPS: ScenarioSpec(
        scenario=Scenario.DISCRETE_STEPS,
        gym_id="MountainCar-v0",
        is_discrete=True,
        needs_fuel_wrapper=False,
        needs_steps_wrapper=False,
        label="Discrete · min steps",
    ),
    Scenario.CONTINUOUS_FUEL: ScenarioSpec(
        scenario=Scenario.CONTINUOUS_FUEL,
        gym_id="MountainCarContinuous-v0",
        is_discrete=False,
        needs_fuel_wrapper=False,
        needs_steps_wrapper=False,
        label="Continuous · min fuel",
    ),
    Scenario.DISCRETE_FUEL: ScenarioSpec(
        scenario=Scenario.DISCRETE_FUEL,
        gym_id="MountainCar-v0",
        is_discrete=True,
        needs_fuel_wrapper=True,
        needs_steps_wrapper=False,
        label="Discrete · min fuel (adapted)",
    ),
    Scenario.CONTINUOUS_STEPS: ScenarioSpec(
        scenario=Scenario.CONTINUOUS_STEPS,
        gym_id="MountainCarContinuous-v0",
        is_discrete=False,
        needs_fuel_wrapper=False,
        needs_steps_wrapper=True,
        label="Continuous · min steps (adapted)",
    ),
}


def discrete_scenarios() -> tuple[Scenario, ...]:
    return tuple(s for s, spec in SPECS.items() if spec.is_discrete)


def continuous_scenarios() -> tuple[Scenario, ...]:
    return tuple(s for s, spec in SPECS.items() if not spec.is_discrete)
