"""Analysis: extract policy on grid, surrogate interpretability, physics."""

from mountaincar_rl.analysis.interpretability import (
    feature_importance,
    fit_surrogate_tree,
)
from mountaincar_rl.analysis.physics import (
    energy_grid,
    kinetic_energy,
    potential_energy,
    total_energy,
)
from mountaincar_rl.analysis.policy_grid import (
    extract_action_grid,
    extract_value_grid,
    rollout_trajectory,
    state_grid,
)

__all__ = [
    "extract_action_grid",
    "extract_value_grid",
    "rollout_trajectory",
    "state_grid",
    "fit_surrogate_tree",
    "feature_importance",
    "potential_energy",
    "kinetic_energy",
    "total_energy",
    "energy_grid",
]
