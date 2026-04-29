
# Part 2 — Analysis of a Reinforcement Learning Paper

## "Playing Atari with Deep Reinforcement Learning" — Mnih et al., DeepMind, 2013

---

## 1. Introduction and motivation

For the second part of the assignment, we analyse the paper *"Playing Atari with Deep Reinforcement Learning"* by Mnih et al. (DeepMind, 2013). The paper is a landmark contribution because it introduced the **Deep Q-Network (DQN)** and showed, for the first time, that an RL agent can learn successful control policies directly from high-dimensional visual input — the raw Atari game pixels — without any hand-crafted features or game-specific information.

This paper is a strong choice for the assignment because it cleanly connects the theoretical RL framework (states, actions, rewards, policies) to a real implementation. The authors formulate Atari gameplay as an RL problem, apply a Q-learning-based method, and evaluate the agent across multiple games.

The deeper importance is that DQN demonstrated RL could move beyond low-dimensional toy environments (like our MountainCar) and operate directly on complex sensory data — by letting a CNN learn the visual representation jointly with the policy.

---

## 2. Problem description

The problem is to train an agent to play Atari 2600 games using only:
- the raw game screen,
- the reward signal (game score change),
- the set of legal actions.

The agent does not access the emulator's internal state. It sees what a human player sees.

This is hard for several reasons:
1. **High-dimensional input.** Each observation is a full image.
2. **Sparse, noisy, delayed rewards.** Many actions don't immediately produce score changes.
3. **Correlated observations.** Adjacent video frames are nearly identical, which makes naive online learning unstable.
4. **Non-stationary data distribution.** As the agent's policy improves, the kinds of states it visits change.

The goal is to *learn a policy that maximises long-term reward*, not to classify images or predict labels.

---

## 3. Reinforcement learning formulation

The Atari setting fits the standard RL framework:

| RL component | Definition in the paper |
|---|---|
| **Agent** | The Deep Q-Network player |
| **Environment** | The Atari 2600 emulator (Arcade Learning Environment) |
| **Observation / state** | Stack of the last 4 preprocessed game frames |
| **Actions** | Legal joystick / button actions per game |
| **Reward** | Change in game score |
| **Policy** | $\arg\max_a Q(s, a)$, with ε-greedy exploration during training |
| **Objective** | Maximise expected discounted future reward |
| **Learning method** | Q-learning with neural function approximator + experience replay |

A single frame doesn't fully describe game state — for example, in Pong or Breakout you can't tell from one frame which way the ball is moving. The paper addresses this by stacking the **last 4 preprocessed frames** as input to the network. This gives the agent temporal information to infer motion.

---

## 4. Methodology — the Deep Q-Network

The method is built on **Q-learning**, where the agent learns an action-value function:

$$Q(s, a) = \mathbb{E}\left[\sum_{t} \gamma^t r_t \,\Big|\, s_0 = s, a_0 = a\right]$$

For environments with small, discrete state spaces, $Q$ can be stored in a table. For Atari, the visual state space is effectively infinite, so the paper replaces the table with a neural network:

$$Q(s, a; \theta)$$

where $\theta$ are the network weights. The network takes the state and outputs one Q-value per action.

The method is:
- **Model-free** — no model of environment dynamics is learned.
- **Off-policy** — the target uses $\max_{a'} Q(s', a')$, so the agent learns the greedy policy while collecting data with an ε-greedy behaviour policy.

---

## 5. Neural network architecture

DQN uses a **convolutional neural network** because the input is visual. CNNs naturally learn spatial patterns (object positions, edges, motion-relevant structures).

Frames are preprocessed:
- converted to grayscale,
- downsampled and cropped to **84 × 84**,
- stacked in groups of 4 most recent frames → input shape **84 × 84 × 4**.

Architecture overview:

| Layer | Role |
|---|---|
| Input | Receives stacked preprocessed frames |
| Convolutional layers | Extract visual features |
| Fully connected layer | Combines features into a decision representation |
| Output | One Q-value per legal action |

A clean design choice: the network outputs **all action Q-values in one forward pass**, so the agent can compare actions cheaply at every step.

---

## 6. Experience replay

One of the paper's most important methodological contributions.

In standard online RL, the agent learns from experiences in the order they happen. This is unstable for two reasons:
- consecutive samples are highly correlated (adjacent frames look almost identical),
- the data distribution shifts as the policy changes.

DQN fixes this with a **replay memory**: every transition $(s_t, a_t, r_t, s_{t+1})$ is stored, and training mini-batches are randomly sampled from this memory.

Three benefits:
1. **Data efficiency** — every transition can be re-used many times.
2. **Decorrelated samples** — random sampling breaks the temporal correlation.
3. **Stable training distribution** — the buffer averages over many past behaviours.

Experience replay is the central reason DQN was able to train a deep network with RL signals.

---

## 7. Exploration strategy

DQN uses **ε-greedy exploration**:
- with probability $1 - \varepsilon$, take the greedy action,
- with probability $\varepsilon$, take a random action.

ε is annealed over training (high at the start, low at the end) — the agent explores broadly early and exploits its learned policy later.

---

## 8. Experimental setup

DQN is evaluated on **7 Atari games**: Beam Rider, Breakout, Enduro, Pong, Q*bert, Seaquest, Space Invaders.

A key strength of the design: **the same architecture, learning algorithm, and hyperparameters are used across all games** (with one minor frame-skipping adjustment for Space Invaders). This supports the claim that DQN is a *general* approach rather than a per-game tuning exercise.

DQN is compared against several baselines:

| Baseline | Description |
|---|---|
| Random | Uniform random actions |
| SARSA | On-policy TD with hand-engineered features |
| Contingency | Feature-based with extra learned representation |
| Human expert | Human gameplay benchmark |
| HNeat | Evolutionary policy search |
| **DQN** | The proposed method |

The main evaluation metric is **average total reward** (the game score).

---

## 9. Results

DQN strongly outperforms previous methods. Key headline numbers from the paper:
- DQN beats all previous RL approaches on **6 of 7 games**.
- DQN surpasses a human expert on **3 games**.
- Standout examples: Breakout (DQN 168 vs Human 31, SARSA 5.2), Pong (DQN +20 vs Human −3).

These results matter because DQN achieves them **without any hand-crafted visual features** — it learns directly from pixels. Previous methods relied on engineered representations of the Atari screen.

DQN is **far below human performance** on Q*bert, Seaquest, and Space Invaders. The paper attributes this to those games requiring strategies over longer time scales.

---

## 10. Interpreting the learned value function

The paper provides a powerful interpretability example using **Seaquest**. Visualising the predicted value across one episode shows that:

- the value rises when an enemy appears,
- rises further as the agent fires a torpedo,
- peaks just before the torpedo hits the enemy,
- falls after the enemy disappears.

This shows the network has learned **meaningful relationships between visual events and expected future reward** — it is not reacting randomly to pixels. The Seaquest visualisation is direct evidence that the learned value function captures useful structure in the environment.

---

## 11. Strengths

1. **Learns directly from raw pixels.** No engineered features, no object detectors, no privileged emulator access.
2. **Combines deep learning + RL.** The CNN solves the visual representation problem; Q-learning estimates long-term value.
3. **Experience replay.** Fundamentally addresses the stability issues of online RL with neural networks.
4. **Generality.** Same architecture and hyperparameters across 7 distinct games.
5. **Strong empirical evaluation.** Compared against random, classical RL, evolutionary methods, and humans.

---

## 12. Limitations

1. **Sample inefficiency.** Trained on 10 million frames per game — orders of magnitude more than humans need.
2. **Discrete actions only.** DQN doesn't directly extend to continuous-control problems (robotics, etc.).
3. **Reward clipping.** All rewards are clipped to {−1, 0, +1}, removing magnitude information. Stabilises training but loses signal.
4. **Weak long-horizon planning.** Worst on games (Q*bert, Seaquest) that require multi-step strategies.
5. **No theoretical convergence guarantees.** The paper itself notes that combining nonlinear function approximation with Q-learning can be unstable in general — DQN works empirically, not by proof.

---

## 13. Personal evaluation — what we'd change today

Looking at DQN through a 2026 lens, three things stand out.

**What still holds up.** The combination of (i) CNN representation learning, (ii) Q-learning, and (iii) experience replay is the foundation of essentially all modern off-policy deep RL. Every algorithm we used in Part 01 (DQN, SAC) descends from this paper. Reading it carefully gave us the conceptual scaffolding for our own DQN implementation on MountainCar — just with a 2-D state instead of 84×84×4.

**What we'd improve.** Most of the original DQN's known weaknesses have published fixes:
- **Maximisation bias** in the target $\max_{a'} Q(s', a'; \theta)$ — solved by **Double DQN** (Van Hasselt et al., 2016) using a separate target network for the argmax.
- **Naive replay sampling** — solved by **Prioritised Experience Replay** (Schaul et al., 2016) sampling transitions with high TD error more often.
- **Single value head conflating "where I am" and "what I do"** — solved by **Dueling DQN** (Wang et al., 2016) with separate streams for state-value and advantage.
- All of these are combined in **Rainbow DQN** (Hessel et al., 2018), which is what we'd actually use today on Atari.

**What it can't do.** DQN fundamentally cannot handle continuous action spaces — for that you need policy-gradient methods (PPO, SAC, TD3). Our Part 01 SAC results on MountainCarContinuous show why this matters: continuous-control problems are everywhere in robotics, autonomous vehicles, and process control.

**Overall.** The paper's importance is not just that DQN learned to play Atari, but that it showed a *scalable direction* — combining deep representation learning with classical RL — that the entire field now follows. It's the right paper to read first when getting into deep RL.

---

## 14. Conclusion

DQN is a strong case study because it shows how RL can be applied to complex visual control tasks. By combining Q-learning, convolutional neural networks, and experience replay, the paper learns from pixel input and achieves strong Atari performance without hand-crafted features. Its lasting importance is not the Atari results themselves, but the demonstration of a scalable recipe for deep RL in high-dimensional environments — a recipe that every modern deep RL algorithm builds on.

---

## Notes for slides (team reference)

- Section count: **14**, designed to map ~1 section per slide for the Part-02 portion of the deck (with §11 + §12 likely combined into a single "strengths vs limitations" slide).
- §13 has been rewritten to give a clear *personal* perspective (modern critique + bridge to Part 01) rather than repeat §11. The prof asks for personal evaluation explicitly — keep this section.
- External figures to grab from the paper itself: **Figure 1** (CNN architecture) for §5, **Figure 3** (Seaquest value function) for §10.
