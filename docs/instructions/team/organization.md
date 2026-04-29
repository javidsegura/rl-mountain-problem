RLI 22.00 — Tinder for RL
Due Apr 29
6 days
MountainCar Group Assignment — 8 members — max 140 pts

Overview
Technical plan
Roles
Timeline
Grading
70%
Code + report
30%
Presentation
140
Max points
Part 01 — MountainCar 70% weight
Multiple RL algorithms (tabular + deep)
Discrete + continuous action versions
Custom reward shaping + gym wrappers
Multiple state representations
Policy analysis + visualization
Tensorboard monitoring
Statistical evaluation (seeds, variance)
Interpretability (feature importance, regression)
Part 02 — RL Paper analysis 30% weight
Pick a landmark RL paper or project
Define states, actions, rewards
Explain the algorithm used
Analyze results + methodology
Personal evaluation of why it's exemplary
Only presentation doc needed (no code)
Suggested picks: DQN (Atari), AlphaGo, SAC, PPO, AlphaStar, DeepMind data center cooling

Deliverables checklist
1. Jupyter Notebook (.ipynb) — Part 01
Standalone, reproducible, all models + results. Submit as part of the ZIP.

2. Presentation deck (PDF or PPTX) — Part 01 + 02
Paper-style: abstract, design choices, visualizations, comparisons, conclusions. 15-20 min.

Submission file
RLI_22_00 – Group {XY}.zip — ONE file, coordinator sends it. Identical across all members.


RL algorithms to implement
Algorithm	Environment	State repr.	Priority
Q-learning (tabular)	MountainCar-v0	Discretized grid	Must have
SARSA (tabular)	MountainCar-v0	Discretized grid	Strong add
DQN	MountainCar-v0	Raw (pos, vel)	Must have
A2C / PPO	MountainCar-v0	Raw + engineered	Must have
SAC or TD3	MountainCarContinuous-v0	Raw (pos, vel)	Must have
DDPG	MountainCarContinuous-v0	Raw	Nice to have
State representations
Raw: (position, velocity) as-is
Discretized: grid N×N (try N=20, 40, 100)
Tile coding: overlapping tilings (good baseline for tabular)
Engineered: add energy = PE + KE, slope angle
RBF features: radial basis functions over state space
Reward shaping variants
Default: -1 per step
Energy-based: reward for gaining kinetic or potential energy
Progress-based: reward proportional to Δposition toward goal
Velocity-based: reward for rightward velocity
Sparse + potential: default + potential-based shaping
Analysis + visualization requirements (key for full marks)
Policy visualization
Q-table heatmap (position vs velocity)
Phase portrait (trajectories in state space)
Value function surface (3D plot)
Action map overlay
Performance metrics
Mean episode reward over training
Success rate (% episodes solved)
Steps to goal distribution
Run with 5+ seeds, report std dev
Interpretability
Feature importance (regression on policy)
Decision boundary visualization
Physical interpretation of learned policy
Compare policies across algorithms
Tooling stack
Gymnasium, PyTorch or Stable-Baselines3, TensorBoard, Matplotlib/Seaborn, Scikit-learn (for interpretability), NumPy. Wrap environments using gym.Wrapper for reward shaping.

What gets you from 90 to 100
Comparative policy analysis across algorithms
Don't just show each algorithm works. Plot their policies side-by-side in the same phase space. Explain WHY they differ physically (e.g. Q-learning tends to use more aggressive left-right oscillation vs SAC's smoother continuous control).

Physical interpretation
The professor cares about connecting RL to the physics of the problem. Interpret the learned policy in terms of energy, oscillation, and resonance. The assignment even shows the Forced Harmonic Oscillator as the relevant physics model.

Statistical rigor
Report mean ± std over 5 seeds for all key metrics. Use error bars on your learning curves. This shows the variance of the method, not just the best run.

Reward shaping analysis — objective vs engineered reward
The rubric explicitly calls out "analysis of objective performance vs engineered reward." Show both. Plot actual steps-to-goal alongside the shaped reward. Be honest about what shaping does and doesn't help.

Presentation quality
All members must visibly participate. Every person will be asked questions on any section. Rehearse once with time constraint. Keep it instructional — treat the class as the audience.

Common ways to lose points
Code doesn't run standalone in a clean venv
Only showing 1-2 algorithms with no comparison
No statistical analysis (single seed runs)
Slides are just bullet points with no visualizations
Part 02 is superficial (just describes the paper, no real evaluation)
One person dominates the presentation and others can't answer questions