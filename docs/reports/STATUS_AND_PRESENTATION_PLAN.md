# Project status + presentation plan — Group 6

> Read this end-to-end before opening anything else. It explains what's already done, what you need to do, and exactly which slide each of you owns.

---

## 1 · What's done (don't redo this)

- **Part 01 — MountainCar** is finished and wrapped in a single notebook: [`docs/deliverables/task1/mountaincar_analysis.ipynb`](../deliverables/task1/mountaincar_analysis.ipynb). 50 cells, 14 sections, all figures embedded. Runs end-to-end in a fresh Python 3.13 venv.
- **Part 02 — paper analysis** of *Mnih et al., "Playing Atari with Deep Reinforcement Learning"* is written in [`docs/deliverables/task2/text.md`](../deliverables/task2/text.md). 14 sections, slide-ready.
- **Submission zip** already built: `RLI_22_00 - Group 6.zip` (5.3 MB) at the project root.

The only thing left is **the presentation deck** (Part 03), and that's what this doc is for.

---

## 2 · How to deliver the deck

- **Length:** 21 slides total, ≈ 18 min of speaking time. Prof allows 15-20 min.
- **Format:** PPTX (or Google Slides → export as PDF for submission). Single deck covering both Part 01 and Part 02.
- **Visual style:** every slide = 1 idea + 1 image (or 1 table). No walls of text.
- **Where to find the source content** for each slide: see the table in §4. The notebook (Part 01) and `task2/text.md` (Part 02) are the canonical references — copy headlines and lift figures from there.
- **Where the 12 cached figures live:** [`artifacts/figures/`](../../artifacts/figures/). All filenames listed in §4.
- **Two figures need to be grabbed from the DQN paper itself** (Mnih et al., 2013): the CNN architecture diagram (paper Figure 1) and the Seaquest value-function plot (paper Figure 3). Paper PDF is freely available via DeepMind / arXiv.

---

## 3 · Speaker assignments

Distribute roughly evenly so every member presents at least 2 slides. The prof can question any of you about any section, so even slides you don't *speak* you should *understand*.

The following slides are the technically denser ones and should go to the people most comfortable with the material:

- **Luis** — slides 8, 9 (deep RL learning curves: DQN/PPO/SAC on discrete + continuous)
- **Javi** — slides 3, 10, 11 (conceptual overview, comparative policy analysis, reward shaping)
- **Nikoloz** — slides 12, 13 (interpretability + physical / FHO interpretation)
- **Alex** — slides 17, 18 (DQN paper: RL formulation + CNN architecture / experience replay)
- **Diego** — slides 19, 20 (DQN paper: experimental setup + results / Seaquest interpretation)

Remaining slides (1, 2, 4, 5, 6, 7, 14, 15, 16, 21) split between Juan, Alp, Jad — pick what you like, just make sure each person presents at least 2.

---

## 4 · Slide-by-slide blueprint

For each slide: *what it says* + *what image to use* + *speaker notes (≈30 sec each unless noted)*.

### Part 01 — MountainCar (slides 1–15)

| # | Title / one-liner | Image | Speaker notes (≈ what to say out loud) |
|---|---|---|---|
| 1 | **Title slide** — RLI 22.00, Group 6, "MountainCar RL — algorithm comparison" + 8 names | none | "Hi, we're group 6. We compared 5 RL algorithms on the MountainCar problem and analyzed the DQN-Atari paper. Both parts will tie together at the end." (15 sec) |
| 2 | **Abstract** — 4 bullets: problem, approach, key Part-01 finding, Part-02 paper picked | none | Read the executive summary from notebook §1. "Headline: DQN solves discrete MountainCar with 83% success, SAC solves continuous robustly, PPO works on continuous but struggles on discrete sparse rewards." (45 sec) |
| 3 | **Why MountainCar is a hard RL problem** — 5 bullets (sparse reward / adversarial physics / continuous state / discrete+continuous action variants / ~3 swings within horizon) | `artifacts/figures/03_engineered_features.png` (energy + slope landscape) | Don't explain "what is RL". Explain why this *specific* problem requires the algorithms we use. Mention: "gravity > engine, so naive 'push right' fails — agent must oscillate to gain energy". (60 sec — give this one extra time, it sets up everything) |
| 4 | **Problem framing — MDP + 4 scenarios** — the MDP table from notebook §2.1 + the 2×2 scenario table | none (just two tables) | "State is (position, velocity). Discrete action: left / no-op / right. Continuous: real-valued force in [-1,1]. The prof asks for 4 scenarios — 2 from gym, 2 we adapted with wrappers." (45 sec) |
| 5 | **Algorithm × scenario matrix** — table: Q-learning, SARSA, DQN, PPO, SAC × which scenarios each handles | none | "5 algorithms chosen to cover tabular vs deep, on-policy vs off-policy, discrete vs continuous. PPO is the only one spanning all 4 scenarios." (30 sec) |
| 6 | **Reward shaping** — 5 wrapper variants (none / energy / progress / velocity / potential) | none (table from notebook §4) | "Sparse reward is hard, so we test 5 shaping schemes. Only 'potential' is mathematically guaranteed to preserve the optimal policy. The others can distort it — and we measure how much later." (30 sec) |
| 7 | **Tabular: Q-learning vs SARSA** — learning curves + 1 phase portrait panel | `05_tabular_learning_curves.png` + Q-learning panel of `09_phase_portraits.png` | "Tabular at 1500 episodes is under-budgeted — both algos plateau without solving. Q-learning slightly beats SARSA on the fuel variant. The takeaway: tabular RL needs a lot of episodes for sparse reward, which motivates moving to deep methods." (45 sec) |
| 8 | **Deep discrete: DQN vs PPO** — learning curves | `06_deep_discrete_learning_curves.png` | "DQN climbs to 83% success — the off-policy replay buffer reuses lucky goal-reaches. PPO plateaus at -180 — its on-policy updates can't bootstrap from rare successes. Same shaping, same scenario, very different outcomes." (45 sec) |
| 9 | **Deep continuous: PPO vs SAC** — learning curves | `07_deep_continuous_learning_curves.png` | "SAC dominates — positive returns in ~10 episodes with near-zero variance. SAC's entropy bonus auto-tunes exploration, perfect for sparse continuous control. PPO catches up eventually but slower and noisier." (45 sec) |
| 10 | **Comparative policy analysis (discrete)** — side-by-side heatmaps | `10_compare_discrete.png` | "Same MDP, four structurally different policies. DQN learns the textbook X-shape: push with the velocity to amplify each swing. PPO converges to a naive 'push right' policy. Tabular agents are noisy versions of the right pattern." (60 sec — this is the rubric headline, dwell) |
| 11 | **Reward shaping: shaped return vs objective** — bar chart | `11_shaping_vs_objective.png` | "Critical methodological lesson. We trained PPO under each shaping scheme and measured both the shaped return AND the *objective* steps-to-goal. Shaping inflates the shaped return without improving the objective. You only see this if you measure both." (60 sec) |
| 12 | **Interpretability — what features matter** | `12_feature_importance.png` | "We fit a Decision Tree to each agent's policy and ranked feature importance. DQN uses position+velocity (consistent with X-shape). SAC is mostly position-driven. The engineered 'energy' feature isn't dominant — deep agents learn the relevant physics from raw (x,v)." (45 sec) |
| 13 | **Physics — energy contours + SAC trajectories** | `13_energy_overlay.png` | "Grey curves are constant-energy contours. SAC's trajectories visibly cross multiple contours — this is the energy-pumping behavior the FHO model predicts. Linear FHO predicts ~72-step natural period; observed periods are in the same ballpark when the agent oscillates." (45 sec) |
| 14 | **Continuous policy comparison + cross-action-space convergence** | `10_compare_continuous.png` | "PPO-cont outputs near-zero everywhere — drift policy. SAC's diagonal force field is the smooth-action equivalent of DQN's X-shape. Two independent algorithms found the same control idea — push with the velocity — in different action spaces." (45 sec) |
| 15 | **Part-01 conclusions** — 3 bullets: best per scenario / sample efficiency / shaping verdict | none | Read directly from notebook §14. "DQN wins discrete, SAC wins continuous, shaping helps but you must measure objective." (30 sec) |

### Part 02 — DQN paper (slides 16–20)

| # | Title / one-liner | Image | Speaker notes |
|---|---|---|---|
| 16 | **Why this paper** — DQN/Atari is landmark; 1-line pitch + 7-game collage | screenshots from any Atari game (free online: Pong / Breakout / Space Invaders) | "We picked Mnih 2013 — DQN. It's the paper that started modern deep RL. They showed an agent could learn to play 7 Atari games from raw pixels alone, no hand-crafted features, beating prior methods on 6 of 7 games." (45 sec) |
| 17 | **RL formulation in the paper** — agent/env/state/action/reward/policy table | none (table from `task2/text.md` §3) | "Standard RL framework. Notable: state is a *stack of 4 frames* because one frame doesn't tell you which way the ball is moving. Reward is just the change in game score." (45 sec) |
| 18 | **Method — CNN + experience replay** | DQN paper Figure 1 (CNN architecture) | "Q-learning where Q is a CNN. The key innovation is *experience replay*: store every transition in a buffer, sample mini-batches randomly. Three benefits: data efficiency, decorrelated samples, stable training distribution. This is the trick that made deep RL work." (60 sec) |
| 19 | **Experimental setup + results** — 7 games + comparison-vs-baselines table | table from `task2/text.md` §8 + §9 | "Same architecture and hyperparameters across all 7 games — DQN is general, not per-game tuned. Results: beats prior RL on 6/7 games, beats human on 3 (Breakout, Pong, Enduro)." (45 sec) |
| 20 | **Interpretability — Seaquest value function** | DQN paper Figure 3 (Seaquest value plot) | "Powerful interpretability example. The predicted value rises when an enemy appears, peaks just before a torpedo hits, drops after. The network has learned meaningful event→reward associations from pixels alone." (45 sec) |

### Conclusion + Q&A (slide 21)

| # | Title / one-liner | Image | Speaker notes |
|---|---|---|---|
| 21 | **Strengths · Limitations · Personal take · Q&A** — 2-column strengths/limitations + 2-line opinion: "in 2026 we'd use Rainbow DQN; DQN can't do continuous control which is why our SAC results matter" | none | "Strengths and limitations from `task2/text.md` §11+§12. Then our take: DQN is the foundation of modern off-policy deep RL, but it's superseded by Rainbow today, and it can't do continuous action — for that you need policy-gradient methods like the SAC we used in Part 01. That's why both parts of this assignment matter." (60 sec, then open Q&A) |

---

## 5 · Where to copy content from (one place per source)

- **Part 01 facts, numbers, plots:** the notebook (`docs/deliverables/task1/mountaincar_analysis.ipynb`). Each markdown cell has the analysis prose ready to paraphrase.
- **Part 02 facts, structure:** [`docs/deliverables/task2/text.md`](../deliverables/task2/text.md). 14 numbered sections, slide-ready.
- **All Part 01 figures:** [`artifacts/figures/`](../../artifacts/figures/) — 12 PNGs, named `01_…` through `13_…`.
- **DQN paper figures (CNN diagram + Seaquest):** download from arXiv `1312.5602`.

---

## 6 · Final checklist before presenting

- [ ] Run the notebook once locally (Run All) — make sure it renders end-to-end.
- [ ] Open the deck in presenter mode and time it. Target 18 min.
- [ ] Each member rehearses their slides at least once — being able to answer questions on adjacent slides matters too.
- [ ] During Q&A: have the notebook open in a tab. If the prof asks about a number, you can show the cell.
- [ ] On submission day: only **one** group member uploads. The submission file is `RLI_22_00 - Group 6.zip` (already built, in project root). The deck (PDF or PPTX) goes alongside it per prof instructions.
