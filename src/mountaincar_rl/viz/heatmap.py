"""Action / visitation heatmaps over the (pos, vel) state space."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


def action_heatmap(pos: np.ndarray, vel: np.ndarray, action_grid: np.ndarray,
                   *, title: str = "", is_continuous: bool = False,
                   ax: plt.Axes | None = None, save_path: Path | None = None) -> plt.Figure:
    """Heatmap of the greedy action at each (pos, vel) state.

    Discrete: 3 colors (left / no-op / right). Continuous: diverging RdBu colormap.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.figure

    if is_continuous:
        im = ax.imshow(action_grid.T, origin="lower",
                       extent=[pos[0], pos[-1], vel[0], vel[-1]],
                       aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("action (force)")
    else:
        # Map 0=left (red), 1=no-op (cyan), 2=right (green)
        cmap = ListedColormap(["#d62728", "#1f77b4", "#2ca02c"])
        im = ax.imshow(action_grid.T, origin="lower",
                       extent=[pos[0], pos[-1], vel[0], vel[-1]],
                       aspect="auto", cmap=cmap, vmin=-0.5, vmax=2.5)
        cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2])
        cbar.set_ticklabels(["left", "no-op", "right"])

    ax.set_xlabel("position")
    ax.set_ylabel("velocity")
    ax.set_title(title)
    ax.axhline(0, color="k", lw=0.5, alpha=0.3)
    ax.axvline(0, color="k", lw=0.5, alpha=0.3)

    if save_path is not None:
        fig.tight_layout()
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


def visitation_heatmap(pos_centers: np.ndarray, vel_centers: np.ndarray,
                       counts: np.ndarray, *, title: str = "Visitation",
                       ax: plt.Axes | None = None,
                       save_path: Path | None = None) -> plt.Figure:
    """Log-scale heatmap of how often each (pos, vel) cell was visited.

    `counts` is the absolute count from the Q-table-derived visitation
    histogram (or any equivalent). We use log10(1 + counts) for contrast.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.figure

    im = ax.imshow(np.log10(1.0 + counts.T), origin="lower",
                   extent=[pos_centers[0], pos_centers[-1],
                           vel_centers[0], vel_centers[-1]],
                   aspect="auto", cmap="Blues")
    fig.colorbar(im, ax=ax, label="log10(1 + visits)")
    ax.set_xlabel("position")
    ax.set_ylabel("velocity")
    ax.set_title(title)

    if save_path is not None:
        fig.tight_layout()
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig
