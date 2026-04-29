"""Deep agents (thin wrappers around Stable-Baselines3 algorithms).

Each module exposes a `make(env, seed, tb_log_dir)` factory that returns an
SB3 model. The training loop in `mountaincar_rl.training.deep_loop` calls
`.learn(total_timesteps, progress_bar=True)` on it.
"""

from mountaincar_rl.agents.deep.dqn import make as make_dqn
from mountaincar_rl.agents.deep.ppo import make as make_ppo
from mountaincar_rl.agents.deep.sac import make as make_sac

DEEP_AGENT_FACTORIES = {
    "dqn": make_dqn,
    "ppo": make_ppo,
    "sac": make_sac,
}

__all__ = ["make_dqn", "make_ppo", "make_sac", "DEEP_AGENT_FACTORIES"]
