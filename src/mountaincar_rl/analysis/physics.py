"""Physics — energy / oscillation interpretation of MountainCar.

The PDF appendix frames MountainCar via the Forced Harmonic Oscillator:

    x'' + 2β x' + ω₀² x = F₀ cos(ω t)

In MountainCar the dynamics are:
    v_{t+1} = v_t + (a − 1) F − cos(3 x_t) G        (discrete)
    x_{t+1} = x_t + v_{t+1}

Linearizing cos(3x) ≈ 1 − (3x)²/2 around x ≈ 0 gives a near-harmonic potential
with characteristic angular frequency ω = √(3·G) ≈ 0.0866 (in env units),
period T = 2π/ω ≈ 72 steps. So an episode of <72 steps means the agent finished
faster than one natural oscillation — essentially impossible from a cold start;
typical optimal solutions take ≈110-130 steps.
"""

from __future__ import annotations

import numpy as np

from mountaincar_rl.config import (
    GRAVITY_DISCRETE,
    POS_MAX,
    POS_MIN,
    VEL_MAX,
    VEL_MIN,
)


def potential_energy(x: float | np.ndarray) -> np.ndarray:
    """PE(x) = sin(3x) — height of the hill at position x."""
    return np.sin(3.0 * np.asarray(x, dtype=np.float64))


def kinetic_energy(v: float | np.ndarray) -> np.ndarray:
    """KE(v) = ½ v² (scaled to PE range — see envs/wrappers/energy.py)."""
    return 0.5 * np.asarray(v, dtype=np.float64) ** 2 * 100.0


def total_energy(x, v) -> np.ndarray:
    return potential_energy(x) + kinetic_energy(v)


def natural_frequency() -> float:
    """ω = √(3·G) — small-angle oscillation frequency (env units)."""
    return float(np.sqrt(3.0 * GRAVITY_DISCRETE))


def natural_period() -> float:
    return 2.0 * np.pi / natural_frequency()


def energy_grid(n_pos: int = 80, n_vel: int = 80) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (pos, vel, E) for plotting an energy-surface contour overlay."""
    pos = np.linspace(POS_MIN, POS_MAX, n_pos)
    vel = np.linspace(VEL_MIN, VEL_MAX, n_vel)
    pp, vv = np.meshgrid(pos, vel, indexing="ij")
    return pos, vel, total_energy(pp, vv)
