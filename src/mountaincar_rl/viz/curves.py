"""Learning curves with mean ± std shaded bands."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mountaincar_rl.evaluation.statistics import smoothed_mean_std


def learning_curves(
    grouped: dict,
    *,
    metric_label: str = "Episode reward",
    title: str = "Learning curves (mean ± std over seeds)",
    smooth_window: int = 25,
    ax: plt.Axes | None = None,
    save_path: Path | None = None,
) -> plt.Figure:
    """Plot one mean±std band per (algo, scenario) group.

    `grouped` comes from `evaluation.statistics.aggregate_seeds(results, "rewards")`.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))
    else:
        fig = ax.figure

    cmap = plt.get_cmap("tab10")
    for k, ((algo, scen_label), payload) in enumerate(grouped.items()):
        x, mean, std = smoothed_mean_std(payload["series"], window=smooth_window)
        color = cmap(k % 10)
        label = f"{algo} · {scen_label}"
        ax.plot(x, mean, color=color, label=label, lw=1.5)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)

    ax.set_xlabel("episode")
    ax.set_ylabel(metric_label)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    if save_path is not None:
        fig.tight_layout()
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig
