"""Side-by-side policy comparison across algorithms.

Renders a grid of action-heatmaps, one per algorithm, on the same axes
so the rubric's "Comparative Policy Analysis" item is immediately visible.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mountaincar_rl.viz.heatmap import action_heatmap


def compare_policies(
    policies: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    is_continuous: bool = False,
    suptitle: str = "Side-by-side learned policies",
    save_path: Path | None = None,
) -> plt.Figure:
    """`policies` maps algo_label -> (pos_axis, vel_axis, action_grid)."""
    n = len(policies)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 4.5 * rows),
                             squeeze=False)

    for k, (label, (pos, vel, actions)) in enumerate(policies.items()):
        r, c = divmod(k, cols)
        action_heatmap(pos, vel, actions, title=label,
                       is_continuous=is_continuous, ax=axes[r, c])

    # Hide unused axes
    for k in range(len(policies), rows * cols):
        r, c = divmod(k, cols)
        axes[r, c].set_visible(False)

    fig.suptitle(suptitle, fontsize=13, y=1.02)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig
