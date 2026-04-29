"""Surrogate interpretability — fit a simple model to a learned policy.

Per the rubric (PDF p7): "feature importance, regression on policy".
We fit a shallow Decision Tree to (state → action) pairs sampled from the
learned policy on a grid, then compute permutation importance to attribute
each feature's role.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from mountaincar_rl.analysis.policy_grid import state_grid
from mountaincar_rl.representations.engineered import EngineeredFeatures


@dataclass
class SurrogateResult:
    """Output of a surrogate fit + interpretability pass."""

    feature_names: list[str]
    importances: np.ndarray             # mean over permutations
    importances_std: np.ndarray
    surrogate_score: float              # accuracy (discrete) or R² (continuous)
    surrogate: Any                      # the fitted sklearn model (Tree)


def _build_features(states: np.ndarray, use_engineered: bool):
    if not use_engineered:
        return states.astype(np.float32), ["position", "velocity"]
    feat = EngineeredFeatures()
    rows = np.stack([feat.encode(s) for s in states], axis=0)
    return rows.astype(np.float32), ["position", "velocity", "energy", "slope_angle"]


def fit_surrogate_tree(agent_or_model, *, is_continuous: bool = False,
                       use_engineered: bool = True, max_depth: int = 5,
                       n_pos: int = 60, n_vel: int = 60) -> SurrogateResult:
    """Sample policy on a grid, fit a Decision Tree, compute permutation importance."""
    _, _, states = state_grid(n_pos, n_vel)

    is_tab = hasattr(agent_or_model, "name")
    if is_tab:
        actions = np.array([agent_or_model.act(s, greedy=True) for s in states])
    else:
        preds, _ = agent_or_model.predict(states, deterministic=True)
        if preds.ndim == 2:
            preds = preds[:, 0]
        actions = preds

    X, names = _build_features(states, use_engineered)

    if is_continuous:
        model = DecisionTreeRegressor(max_depth=max_depth, random_state=0)
        model.fit(X, actions)
        score = float(model.score(X, actions))
    else:
        model = DecisionTreeClassifier(max_depth=max_depth, random_state=0)
        model.fit(X, actions.astype(int))
        score = float(model.score(X, actions.astype(int)))

    pi = permutation_importance(model, X, actions if is_continuous else actions.astype(int),
                                n_repeats=10, random_state=0)

    return SurrogateResult(
        feature_names=names,
        importances=pi.importances_mean,
        importances_std=pi.importances_std,
        surrogate_score=score,
        surrogate=model,
    )


def feature_importance(agent_or_model, *, is_continuous: bool = False) -> dict:
    """Convenience wrapper returning a JSON-serializable summary."""
    res = fit_surrogate_tree(agent_or_model, is_continuous=is_continuous)
    return {
        "feature_names": res.feature_names,
        "importances": res.importances.tolist(),
        "importances_std": res.importances_std.tolist(),
        "surrogate_score": res.surrogate_score,
    }
