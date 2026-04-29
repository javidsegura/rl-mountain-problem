# MountainCar RL — Tinder for RL (RLI 22.00)

> A modular RL testbed comparing 5 algorithms (Q-learning, SARSA, DQN, PPO, SAC) across 4 MountainCar scenarios, with reward shaping, multi-seed statistical evaluation, policy interpretability, and physical (Forced Harmonic Oscillator) analysis.

---

## What it is

The MountainCar problem (Moore, 1990) cast as a testbed: an under-powered car must learn to leverage potential energy by oscillating to escape a valley. We treat it as a Forced Harmonic Oscillator and analyze how 5 RL algorithms — 2 tabular + 3 deep — discover qualitatively different policies under 4 reward / action-space variations.

The deliverable is a single Jupyter notebook ([docs/deliverables/task1/mountaincar_analysis.ipynb](docs/deliverables/task1/mountaincar_analysis.ipynb)) that imports all logic from a `src/mountaincar_rl/` package, narrates the analysis paper-style, and renders cached results in ~30 seconds.

---

## Tech stack

| Layer | Technology |
|---|---|
| Language | Python 3.13 |
| Env | Gymnasium (`MountainCar-v0`, `MountainCarContinuous-v0`) |
| Tabular agents | NumPy (from scratch) |
| Deep agents | Stable-Baselines3 |
| NN backend | PyTorch (SB3 dependency) |
| Monitoring | TensorBoard |
| Interpretability | scikit-learn (DecisionTree, permutation importance) |
| Plots | Matplotlib + Seaborn |
| Progress bars | tqdm |
| Notebook | JupyterLab |
| Pkg manager | uv |

---

## Repository structure

```
.
├── Makefile                            # Dev commands (make help)
├── pyproject.toml                      # uv-managed deps
├── requirements.txt                    # Pinned deps for the notebook's !pip install path
├── src/
│   └── mountaincar_rl/
│       ├── config.py                   # Single source of truth (seeds, paths, hyperparams)
│       ├── envs/                       # Scenario factory + reward wrappers
│       │   ├── scenarios.py
│       │   ├── factory.py
│       │   └── wrappers/               # energy / progress / velocity / potential / fuel
│       ├── representations/            # State representations (Strategy pattern)
│       │   ├── base.py
│       │   ├── discretizer.py
│       │   ├── tile_coding.py
│       │   ├── rbf.py
│       │   └── engineered.py
│       ├── agents/                     # Strategy + Factory patterns
│       │   ├── base.py
│       │   ├── factory.py
│       │   ├── tabular/                # From-scratch (Q-learning, SARSA)
│       │   └── deep/                   # SB3 wrappers (DQN, PPO, SAC)
│       ├── training/                   # Tabular loop, deep loop, multi-seed runner
│       ├── evaluation/                 # Metrics + statistics (mean ± std)
│       ├── analysis/                   # Policy grids, interpretability, physics
│       └── viz/                        # Heatmap, surface, phase, curves, compare
├── docs/
│   ├── deliverables/
│   │   ├── task1/                      # The notebook (Part 01)
│   │   ├── task2/                      # Paper analysis (Part 02)
│   │   └── task3/                      # Presentation deck
│   └── instructions/                   # Source material
├── scripts/
│   ├── smoke.py                        # Pre-train smoke test
│   └── build_submission.sh             # Builds RLI_22_00 - Group XX.zip
└── artifacts/                          # Gitignored except results/
    ├── checkpoints/                    # SB3 .zip model saves
    ├── tb_logs/                        # TensorBoard event files
    ├── figures/                        # Cached PNGs for the notebook
    └── results/                        # JSON metrics per (algo, scenario, seed)
```

---

## RL pipeline

### Algorithms × scenarios

| Group | Algo | Action space | Scenarios covered |
|---|---|---|---|
| Tabular | Q-learning | discrete | 1, 3 |
| Tabular | SARSA | discrete | 1, 3 |
| Deep | DQN | discrete | 1, 3 |
| Deep | PPO | discrete + continuous | 1, 2, 3, 4 |
| Deep | SAC | continuous | 2, 4 |

Scenarios:
1. `MountainCar-v0` — discrete actions, minimum steps (default −1/step)
2. `MountainCarContinuous-v0` — continuous actions, minimum fuel (default −0.1·a²)
3. Discrete + minimum fuel (adapted via wrapper)
4. Continuous + minimum steps (adapted via wrapper)

### Reward wrappers
- **Default** — −1/step (discrete) or −0.1·a² (continuous)
- **Energy** — bonus on Δ(PE+KE)
- **Progress** — bonus on Δposition toward goal
- **Velocity** — bonus on rightward velocity
- **Potential** — potential-based shaping (theory-safe; preserves optimal policy)
- **Fuel** — penalize action magnitude (used for scenarios 3 & 4)

### State representations
Raw, discretized (N=20/40/100), tile coding, RBF features, engineered (energy + slope angle).

### Statistical evaluation
3 seeds per (algo, scenario), mean ± std reported with error bands on every learning curve.

---

## Setup

### Prerequisites
- Python 3.13 (managed by `uv`, see `.python-version`)
- ~500 MB disk for deps + artifacts

### Quickstart for graders (3 commands)

```bash
make install          # ~30 sec: uv sync
make notebook         # opens JupyterLab on the deliverable
# In the notebook: Run All. Default MODE="cache" renders cached results in ~30 sec.
```

Alternative (no `uv` installed): the first cell of the notebook does `!pip install -r requirements.txt` and works in any Python 3.13 venv.

### Verifying the training loop (optional, for graders)

Open the notebook, change the first cell's `MODE = "cache"` to `MODE = "demo"`, Run All. Re-trains every algorithm with reduced budget (1 seed, 5k timesteps). Total ≤5 min.

---

## Dev commands

```bash
make help            # List all targets
make install         # uv sync
make smoke           # ~2 min: 1k-step training per algo to verify no crashes
make train-tabular   # ~2 min: tabular agents, full config
make train           # ~15-18 min: FULL matrix, regenerates cached artifacts/
make tensorboard     # opens TensorBoard on artifacts/tb_logs
make notebook        # launches JupyterLab on docs/deliverables/task1/
make clean           # removes tb_logs + checkpoints + figures (keeps results JSON)
make clean-all       # removes ALL artifacts including the cached results
GROUP=XX make zip    # builds RLI_22_00 - Group XX.zip
```

---

## Three notebook execution modes

| Mode | Action | Time |
|---|---|---|
| `cache` (default) | Loads pre-computed JSON + PNG, renders only | ~30 sec |
| `demo` | Trains all algos at reduced budget (1 seed, 5k steps) | ≤5 min |
| `full` | Trains 3 seeds × 30k steps; this is what we use to regenerate the cache | ~15-18 min |

Set the mode via `MODE = "..."` in the notebook's first cell.

---

## Submission

Final submission file: `RLI_22_00 - Group XX.zip`. Built via `GROUP=XX make zip`. Contains: `src/`, the notebook, `requirements.txt`, the cached `artifacts/results/` + `artifacts/figures/`, `README.md`, `Makefile`.

---

## Team

| Name | Role |
|---|---|
| _TBD_ | _TBD_ |

---

## References

- Moore, A. W. (1990). *Efficient Memory-based Learning for Robot Control* — original MountainCar.
- Sutton & Barto (2018). *Reinforcement Learning: An Introduction* — Q-learning, SARSA.
- Mnih et al. (2015). *Human-level control through deep reinforcement learning* — DQN.
- Schulman et al. (2017). *Proximal Policy Optimization Algorithms* — PPO.
- Haarnoja et al. (2018). *Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL with a Stochastic Actor* — SAC.
- Stable-Baselines3 docs: https://stable-baselines3.readthedocs.io
- Gymnasium MountainCar: https://gymnasium.farama.org/environments/classic_control/mountain_car/
