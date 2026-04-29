"""Phase portrait of episodes in (position, velocity) space.

Optionally overlays the energy contours from `analysis.physics.energy_grid`
so the viewer can see the agent climbing the energy "hills" of the state space.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

from mountaincar_rl.analysis.physics import energy_grid


def phase_portrait(
    trajectories: list[np.ndarray],
    *,
    rewards: list[float] | None = None,
    title: str = "Phase portrait",
    show_energy_contours: bool = True,
    ax: plt.Axes | None = None,
    save_path: Path | None = None,
) -> plt.Figure:
    """Plot N trajectories of shape (T, 2) in (pos, vel) space."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    if show_energy_contours:
        pos, vel, E = energy_grid(80, 80)
        cs = ax.contour(pos, vel, E.T, levels=10, alpha=0.3, cmap="Greys")
        ax.clabel(cs, inline=True, fontsize=7, fmt="%.1f")

    cmap = plt.get_cmap("plasma")
    n = len(trajectories)
    for k, traj in enumerate(trajectories):
        color = cmap(k / max(1, n - 1))
        label = f"R={rewards[k]:.0f}" if rewards is not None else None
        ax.plot(traj[:, 0], traj[:, 1], "-", color=color, alpha=0.8,
                lw=1.2, label=label)
        ax.plot(traj[0, 0], traj[0, 1], "o", color=color, ms=4)
        ax.plot(traj[-1, 0], traj[-1, 1], "s", color=color, ms=4)

    ax.axhline(0, color="k", lw=0.4, alpha=0.3)
    ax.axvline(0, color="k", lw=0.4, alpha=0.3)
    ax.axvline(0.5, color="g", lw=1, ls="--", alpha=0.5, label="goal x=0.5")
    ax.set_xlabel("position")
    ax.set_ylabel("velocity")
    ax.set_title(title)
    if rewards is not None or show_energy_contours:
        ax.legend(loc="best", fontsize=7)

    if save_path is not None:
        fig.tight_layout()
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig
