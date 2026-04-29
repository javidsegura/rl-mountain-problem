"""Smoke test — make sure every algorithm's training loop runs end-to-end.

Trains 1 algo × 1 scenario × 1 seed × tiny budget, then exercises the viz
primitives on the resulting policy. Fails loudly if anything is broken.
Total budget: ≤2 minutes.
"""

from __future__ import annotations

import time
import warnings

import numpy as np
from rich.console import Console
from rich.table import Table

warnings.filterwarnings("ignore")

from mountaincar_rl.agents import make_agent  # noqa: E402
from mountaincar_rl.analysis import (  # noqa: E402
    extract_action_grid,
    fit_surrogate_tree,
    rollout_trajectory,
)
from mountaincar_rl.envs import Scenario, make_env  # noqa: E402
from mountaincar_rl.training.deep_loop import train_deep  # noqa: E402
from mountaincar_rl.training.tabular_loop import train_tabular  # noqa: E402
from mountaincar_rl.viz import (  # noqa: E402
    action_heatmap,
    compare_policies,
    learning_curves,
    phase_portrait,
    value_surface,
)

CONSOLE = Console()

CASES = [
    ("q_learning",  Scenario.DISCRETE_STEPS,    "tabular", 200),
    ("sarsa",       Scenario.DISCRETE_STEPS,    "tabular", 200),
    ("dqn",         Scenario.DISCRETE_STEPS,    "deep",    1_000),
    ("ppo",         Scenario.DISCRETE_STEPS,    "deep",    1_000),
    ("ppo",         Scenario.CONTINUOUS_FUEL,   "deep",    1_000),
    ("sac",         Scenario.CONTINUOUS_FUEL,   "deep",    1_000),
]


def _train_one(algo: str, scenario: Scenario, family: str, budget: int):
    env = make_env(scenario, shaping="potential" if algo in {"dqn", "sac"} else "none",
                   seed=0)
    agent = make_agent(algo, env, seed=0)
    if family == "tabular":
        result = train_tabular(agent, env, n_episodes=budget, seed=0,
                               desc=f"{algo}/{scenario.value}")
    else:
        result = train_deep(agent, env, n_timesteps=budget,
                            progress=False, desc=f"{algo}/{scenario.value}")
    return env, agent, result


def main() -> int:
    table = Table(title="Smoke test", show_lines=False)
    table.add_column("algo", style="cyan")
    table.add_column("scenario", style="magenta")
    table.add_column("episodes")
    table.add_column("mean R", justify="right")
    table.add_column("wall (s)", justify="right")
    table.add_column("status", justify="center")

    overall_ok = True
    trained = []

    for algo, scenario, family, budget in CASES:
        t0 = time.time()
        try:
            env, agent, result = _train_one(algo, scenario, family, budget)
            trained.append((algo, scenario, env, agent))
            mean_r = float(np.mean(result.rewards)) if len(result.rewards) else float("nan")
            n_ep = len(result.rewards)
            table.add_row(algo, scenario.value, str(n_ep),
                          f"{mean_r:.2f}", f"{time.time() - t0:.1f}",
                          "[green]OK[/green]")
        except Exception as e:  # noqa: BLE001
            overall_ok = False
            table.add_row(algo, scenario.value, "-", "-", f"{time.time() - t0:.1f}",
                          f"[red]FAIL: {e}[/red]")

    CONSOLE.print(table)

    # Exercise viz primitives on a single trained policy
    CONSOLE.print("\n[bold]Exercising viz primitives...[/bold]")
    try:
        algo, scenario, env, agent = trained[0]
        is_cont = not env.unwrapped.action_space.__class__.__name__.startswith("Discrete")
        pos, vel, actions = extract_action_grid(agent, n_pos=40, n_vel=40,
                                                is_continuous=is_cont)
        action_heatmap(pos, vel, actions, title=f"smoke · {algo}",
                       is_continuous=is_cont)

        # Surface from tabular Q
        if hasattr(agent, "Q"):
            from mountaincar_rl.analysis import extract_value_grid
            pc, vc, V = extract_value_grid(agent)
            value_surface(pc, vc, V, title="smoke · V(x,v)")

        # Phase portrait from rollouts
        rollouts = []; rewards = []
        for s in range(3):
            traj, r, _ = rollout_trajectory(agent, env, seed=100 + s, max_steps=200)
            rollouts.append(traj); rewards.append(r)
        phase_portrait(rollouts, rewards=rewards, title="smoke · phase")

        # Comparison from a couple of trained policies (re-extract grids)
        grids = {}
        for algo_, sc_, env_, ag_ in trained[:3]:
            is_c = not env_.unwrapped.action_space.__class__.__name__.startswith("Discrete")
            if is_c:
                continue
            pos_, vel_, act_ = extract_action_grid(ag_, n_pos=40, n_vel=40,
                                                   is_continuous=False)
            grids[f"{algo_}/{sc_.value}"] = (pos_, vel_, act_)
        if len(grids) >= 2:
            compare_policies(grids, is_continuous=False, suptitle="smoke · compare")

        # Learning curve smoke
        from mountaincar_rl.evaluation.statistics import aggregate_seeds
        fake = [{"algo": "q_learning", "scenario": "x", "scenario_label": "x",
                 "seed": s, "rewards": np.random.randn(100).tolist()} for s in range(3)]
        learning_curves(aggregate_seeds(fake, "rewards"), title="smoke · curves")

        # Surrogate / interpretability
        s_res = fit_surrogate_tree(trained[0][3], is_continuous=False, n_pos=20, n_vel=20)
        CONSOLE.print(f"   surrogate score = {s_res.surrogate_score:.3f}; "
                      f"importances = {dict(zip(s_res.feature_names, s_res.importances))}")

        import matplotlib.pyplot as plt
        plt.close("all")
        CONSOLE.print("[green]   viz primitives OK[/green]")
    except Exception as e:  # noqa: BLE001
        overall_ok = False
        CONSOLE.print(f"[red]   viz primitives FAILED: {e}[/red]")

    if overall_ok:
        CONSOLE.print("\n[bold green]✓ Smoke test passed[/bold green]")
        return 0
    CONSOLE.print("\n[bold red]✗ Smoke test FAILED[/bold red]")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
