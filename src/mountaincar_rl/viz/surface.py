"""3-D value surface V(x, v) — required by the rubric."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def value_surface(pos: np.ndarray, vel: np.ndarray, V: np.ndarray,
                  *, title: str = "Value surface V(x, v)",
                  cmap: str = "viridis",
                  save_path: Path | None = None) -> plt.Figure:
    """Wireframe / surface plot of V over the (pos, vel) grid."""
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    pp, vv = np.meshgrid(pos, vel, indexing="ij")
    surf = ax.plot_surface(pp, vv, V, cmap=cmap, edgecolor="none", alpha=0.9)
    ax.set_xlabel("position")
    ax.set_ylabel("velocity")
    ax.set_zlabel("V(x, v)")
    ax.set_title(title)
    fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1)
    if save_path is not None:
        fig.tight_layout()
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig
