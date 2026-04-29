"""Agents package.

Strategy + Factory pattern: every agent implements `Agent` (act/learn/save/load),
constructed via `make_agent(name, env, ...)`.
"""

from mountaincar_rl.agents.base import Agent
from mountaincar_rl.agents.factory import make_agent

__all__ = ["Agent", "make_agent"]
