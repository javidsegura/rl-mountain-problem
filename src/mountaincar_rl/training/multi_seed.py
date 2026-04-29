"""Multi-seed runner — `make train` invokes this as a script.

For each (algo, scenario, seed) tuple:
  1. Build env with shaping (we default to "potential" for deep agents on the
     hardest scenarios; tabular gets the raw env to keep the demo honest).
  2. Build the agent / model.
  3. Train.
  4. Save the model + a JSON of per-episode rewards/lengths/successes.

A single tqdm bar tracks completed runs across the whole matrix. Each per-run
inner progress bar comes from the trainer (SB3's own tqdm or our tabular tqdm).
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Literal

from tqdm.auto import tqdm

from mountaincar_rl.agents import make_agent
from mountaincar_rl.agents.factory import is_tabular
from mountaincar_rl.config import (
    BUDGETS,
    CHECKPOINTS_DIR,
    RESULTS_DIR,
    TB_LOGS_DIR,
    Mode,
    seeds_for,
)
from mountaincar_rl.envs import Scenario, make_env
from mountaincar_rl.envs.scenarios import SPECS
from mountaincar_rl.training.deep_loop import TrainResult as DeepResult
from mountaincar_rl.training.deep_loop import train_deep
from mountaincar_rl.training.tabular_loop import TrainResult as TabResult
from mountaincar_rl.training.tabular_loop import train_tabular

# Algo → (set of compatible scenarios)
ALGO_SCENARIOS: dict[str, tuple[Scenario, ...]] = {
    "q_learning": (Scenario.DISCRETE_STEPS, Scenario.DISCRETE_FUEL),
    "sarsa":      (Scenario.DISCRETE_STEPS, Scenario.DISCRETE_FUEL),
    "dqn":        (Scenario.DISCRETE_STEPS, Scenario.DISCRETE_FUEL),
    "ppo":        (Scenario.DISCRETE_STEPS, Scenario.DISCRETE_FUEL,
                   Scenario.CONTINUOUS_FUEL, Scenario.CONTINUOUS_STEPS),
    "sac":        (Scenario.CONTINUOUS_FUEL, Scenario.CONTINUOUS_STEPS),
}

GROUPS: dict[str, tuple[str, ...]] = {
    "tabular": ("q_learning", "sarsa"),
    "deep":    ("dqn", "ppo", "sac"),
    "all":     ("q_learning", "sarsa", "dqn", "ppo", "sac"),
}

# Default shaping per (algo, scenario) — energy shaping helps DQN/SAC find the
# goal during the smoke / demo budgets; PPO works fine on raw rewards.
DEFAULT_SHAPING = {
    "q_learning": "none",
    "sarsa":      "none",
    "dqn":        "potential",
    "ppo":        "none",
    "sac":        "potential",
}


def _result_path(algo: str, scenario: Scenario, seed: int, mode: Mode) -> Path:
    return RESULTS_DIR / f"{mode}__{algo}__{scenario.value}__seed{seed}.json"


def _checkpoint_path(algo: str, scenario: Scenario, seed: int, mode: Mode) -> Path:
    return CHECKPOINTS_DIR / f"{mode}__{algo}__{scenario.value}__seed{seed}"


def _save_result(result: TabResult | DeepResult, path: Path,
                 *, algo: str, scenario: Scenario, seed: int, mode: Mode,
                 wall_clock_s: float, shaping: str) -> None:
    payload = {
        "algo": algo,
        "scenario": scenario.value,
        "scenario_label": SPECS[scenario].label,
        "seed": seed,
        "mode": mode,
        "shaping": shaping,
        "wall_clock_s": wall_clock_s,
        "n_episodes": int(len(result.rewards)),
        "rewards": result.rewards.tolist(),
        "lengths": result.lengths.tolist(),
        "successes": [bool(s) for s in result.successes.tolist()],
        "final_epsilon": float(getattr(result, "final_epsilon", 0.0)),
        "mean_reward_last10pct": float(
            result.rewards[-max(1, len(result.rewards) // 10):].mean()
        ),
        "success_rate_last10pct": float(
            sum(result.successes[-max(1, len(result.successes) // 10):])
            / max(1, len(result.successes) // 10)
        ),
    }
    path.write_text(json.dumps(payload))


def run_one(algo: str, scenario: Scenario, seed: int, mode: Mode,
            *, shaping: str | None = None, save: bool = True) -> dict:
    """Train one (algo, scenario, seed). Returns metadata dict."""
    shaping = shaping or DEFAULT_SHAPING.get(algo, "none")
    env = make_env(scenario, shaping=shaping, seed=seed)
    tb_dir = TB_LOGS_DIR / f"{mode}__{algo}__{scenario.value}"
    agent = make_agent(algo, env, seed=seed, tb_log_dir=tb_dir)
    budget = BUDGETS[mode]

    t0 = time.time()
    desc = f"{algo}/{scenario.value}/s{seed}"
    if is_tabular(algo):
        result = train_tabular(agent, env, n_episodes=budget.tabular_episodes,
                               seed=seed, desc=desc)
    else:
        result = train_deep(agent, env, n_timesteps=budget.deep_timesteps,
                            progress=True, desc=desc)
    wall = time.time() - t0

    if save:
        ckpt = _checkpoint_path(algo, scenario, seed, mode)
        if is_tabular(algo):
            agent.save(ckpt.with_suffix(".npz"))
        else:
            agent.save(str(ckpt))  # SB3 appends .zip

        _save_result(
            result, _result_path(algo, scenario, seed, mode),
            algo=algo, scenario=scenario, seed=seed, mode=mode,
            wall_clock_s=wall, shaping=shaping,
        )
    env.close()
    return {"algo": algo, "scenario": scenario.value, "seed": seed,
            "wall_clock_s": wall,
            "mean_reward_last10pct":
                float(result.rewards[-max(1, len(result.rewards) // 10):].mean()),
            "success_rate_last10pct":
                float(sum(result.successes[-max(1, len(result.successes) // 10):])
                      / max(1, len(result.successes) // 10))}


def expand_matrix(algos: Iterable[str], mode: Mode) -> list[tuple[str, Scenario, int]]:
    """Expand (algos × compatible-scenarios × seeds) into a flat list."""
    out: list[tuple[str, Scenario, int]] = []
    seeds = seeds_for(mode)
    for algo in algos:
        for scenario in ALGO_SCENARIOS[algo]:
            for seed in seeds:
                out.append((algo, scenario, seed))
    return out


def run_matrix(algos: Iterable[str], mode: Mode = "full") -> list[dict]:
    """Run all (algo, scenario, seed) combinations and return summary list."""
    matrix = expand_matrix(algos, mode)
    summaries: list[dict] = []

    for algo, scenario, seed in tqdm(matrix, desc=f"matrix({mode})", unit="run"):
        summary = run_one(algo, scenario, seed, mode)
        summaries.append(summary)

    return summaries


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--all", action="store_true", help="Run the full matrix")
    p.add_argument("--group", choices=list(GROUPS), help="Restrict to a group")
    p.add_argument("--mode", choices=("smoke", "demo", "full"), default="full")
    args = p.parse_args()

    group_key = "all" if args.all or args.group is None else args.group
    algos = GROUPS[group_key]
    mode: Mode = args.mode

    print(f"\n=> Running matrix: algos={algos} mode={mode} "
          f"seeds={seeds_for(mode)}\n")
    summaries = run_matrix(algos, mode=mode)

    # Summary table
    print("\n=== Run summary ===")
    print(f"{'algo':<12} {'scenario':<22} {'seed':<5} {'wall':<7} "
          f"{'mean_R':<10} {'succ%':<6}")
    for s in summaries:
        print(f"{s['algo']:<12} {s['scenario']:<22} {s['seed']:<5} "
              f"{s['wall_clock_s']:<7.1f} {s['mean_reward_last10pct']:<10.2f} "
              f"{100 * s['success_rate_last10pct']:<6.1f}")
    total = sum(s["wall_clock_s"] for s in summaries)
    print(f"\nTotal wall-clock: {total:.1f}s ({total / 60:.1f} min) "
          f"across {len(summaries)} runs.")


if __name__ == "__main__":
    main()
