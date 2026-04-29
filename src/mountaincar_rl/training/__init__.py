"""Training package: tabular loop, deep loop, multi-seed runner."""

from mountaincar_rl.training.deep_loop import train_deep
from mountaincar_rl.training.tabular_loop import train_tabular

__all__ = ["train_tabular", "train_deep"]
