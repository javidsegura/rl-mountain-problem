"""Factory: name → agent instance (tabular) or SB3 model (deep).

Tabular agents own their training loop in `mountaincar_rl.training.tabular_loop`.
Deep agents are SB3 models trained via `mountaincar_rl.training.deep_loop`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym

from mountaincar_rl.agents.deep import DEEP_AGENT_FACTORIES
from mountaincar_rl.agents.tabular import QLearningAgent, SARSAAgent

TABULAR_NAMES = {"q_learning", "sarsa"}
DEEP_NAMES = set(DEEP_AGENT_FACTORIES.keys())
ALL_NAMES = TABULAR_NAMES | DEEP_NAMES


def is_tabular(name: str) -> bool:
    return name in TABULAR_NAMES


def is_deep(name: str) -> bool:
    return name in DEEP_NAMES


def make_agent(
    name: str,
    env: gym.Env,
    seed: int = 0,
    tb_log_dir: Path | None = None,
    **kwargs: Any,
):
    """Build the agent / model identified by `name`.

    Tabular agents return a custom Agent. Deep agents return an SB3 model.
    The training loops know which type they're dealing with.
    """
    if name == "q_learning":
        return QLearningAgent(n_actions=env.action_space.n, seed=seed, **kwargs)
    if name == "sarsa":
        return SARSAAgent(n_actions=env.action_space.n, seed=seed, **kwargs)
    if name in DEEP_AGENT_FACTORIES:
        return DEEP_AGENT_FACTORIES[name](env, seed=seed, tb_log_dir=tb_log_dir)
    raise ValueError(f"Unknown agent name: {name}. Known: {sorted(ALL_NAMES)}")
