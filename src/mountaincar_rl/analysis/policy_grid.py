"""Extract a learned policy / value function over a regular (pos, vel) grid.

Used by the visualization module to render heatmaps, surfaces, and the
side-by-side comparison plots.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from mountaincar_rl.config import GOAL_POSITION, POS_MAX, POS_MIN, VEL_MAX, VEL_MIN


def state_grid(n_pos: int = 80, n_vel: int = 80) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (pos_axis, vel_axis, states) where states.shape == (n_pos*n_vel, 2)."""
    pos = np.linspace(POS_MIN, POS_MAX, n_pos)
    vel = np.linspace(VEL_MIN, VEL_MAX, n_vel)
    pp, vv = np.meshgrid(pos, vel, indexing="ij")
    states = np.stack([pp.ravel(), vv.ravel()], axis=1).astype(np.float32)
    return pos, vel, states


def extract_action_grid(agent_or_model, n_pos: int = 80, n_vel: int = 80,
                        is_continuous: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (pos, vel, action_grid) where action_grid[i,j] = chosen action at (pos[i], vel[j])."""
    pos, vel, states = state_grid(n_pos, n_vel)
    is_tab = hasattr(agent_or_model, "name")

    actions = np.zeros((n_pos, n_vel), dtype=np.float32 if is_continuous else np.int32)
    if is_tab:
        for k, s in enumerate(states):
            i, j = divmod(k, n_vel)
            actions[i, j] = agent_or_model.act(s, greedy=True)
    else:
        # SB3 batch predict
        preds, _ = agent_or_model.predict(states, deterministic=True)
        if preds.ndim == 2:                # continuous: shape (N, 1)
            preds = preds[:, 0]
        actions = preds.reshape(n_pos, n_vel)

    return pos, vel, actions


def extract_value_grid(agent, n_pos: int = 80, n_vel: int = 80) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For tabular agents only — V(s) = max_a Q(s, a) on the (coarser) discretizer grid.

    Returns the agent's own (n_pos, n_vel) — i.e. the full Q-table value layer,
    not interpolated to `n_pos`/`n_vel` arguments.
    """
    if not hasattr(agent, "Q"):
        raise TypeError("extract_value_grid requires a tabular agent with a .Q table")
    pos_centers, vel_centers = agent.discretizer.decode_centers()
    V = agent.Q.max(axis=2)  # (n_pos_bins, n_vel_bins)
    return pos_centers, vel_centers, V


def rollout_trajectory(agent_or_model, env: gym.Env, *, seed: int = 0,
                       max_steps: int = 1000) -> tuple[np.ndarray, float, bool]:
    """Run one greedy episode and return (states[T,2], total_reward, success)."""
    obs, _ = env.reset(seed=seed)
    states = [obs.copy()]
    is_tab = hasattr(agent_or_model, "name")
    total = 0.0
    terminated = truncated = False
    while not (terminated or truncated) and len(states) < max_steps:
        if is_tab:
            action = agent_or_model.act(obs, greedy=True)
        else:
            action, _ = agent_or_model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        states.append(obs.copy())
        total += float(reward)
    return np.asarray(states), total, bool(terminated and obs[0] >= GOAL_POSITION)
