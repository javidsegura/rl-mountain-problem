"""Abstract Agent — the Strategy interface every algo (tabular + deep) implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np


class Agent(ABC):
    """Common interface for tabular and deep agents.

    Tabular agents implement `learn` themselves (their training loop lives in
    `mountaincar_rl.training.tabular_loop`). Deep agents (SB3 wrappers) delegate
    `learn` to SB3's `.learn()` via `mountaincar_rl.training.deep_loop`.
    """

    name: str

    @abstractmethod
    def act(self, obs: np.ndarray, *, greedy: bool = False) -> Any:
        """Select an action. `greedy=True` disables exploration (used at eval)."""

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist learned parameters to disk."""

    @abstractmethod
    def load(self, path: Path) -> None:
        """Load previously-saved parameters."""
