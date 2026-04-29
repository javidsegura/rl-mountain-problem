"""Multi-seed aggregation: mean ± std curves with optional smoothing.

`smoothed_mean_std()` produces the (mean, std) arrays that learning-curve
plots consume to draw central line + shaded band.
"""

from __future__ import annotations

import numpy as np


def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(x) < window:
        return x.astype(np.float64)
    kernel = np.ones(window) / window
    # Convolve in 'valid' mode then pad start with the head's mean to keep length
    smoothed = np.convolve(x, kernel, mode="valid")
    pad = np.full(len(x) - len(smoothed), smoothed[0])
    return np.concatenate([pad, smoothed])


def smoothed_mean_std(
    series_per_seed: list[np.ndarray],
    window: int = 25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack `series_per_seed`, smooth each, return (x, mean, std)."""
    # Truncate to the shortest series so we can stack
    min_len = min(len(s) for s in series_per_seed)
    stacked = np.stack([_moving_average(np.asarray(s[:min_len]), window)
                        for s in series_per_seed], axis=0)
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)
    return np.arange(min_len), mean, std


def aggregate_seeds(results: list[dict], key: str = "rewards") -> dict:
    """Group `results` by (algo, scenario) and stack the named series across seeds.

    Returns a dict keyed by (algo, scenario_label) with values:
        {seeds: [..], series: [np.ndarray, ...], shaping: ...}
    """
    grouped: dict[tuple[str, str], dict] = {}
    for r in results:
        k = (r["algo"], r.get("scenario_label", r["scenario"]))
        grouped.setdefault(k, {"seeds": [], "series": [], "shaping": r.get("shaping")})
        grouped[k]["seeds"].append(r["seed"])
        grouped[k]["series"].append(np.asarray(r[key], dtype=np.float64))
    return grouped
