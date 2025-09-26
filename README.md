# 🎰 Monte Carlo Blackjack with Pygame Simulation

An interactive **Reinforcement Learning** project where an agent learns Blackjack via **Monte Carlo (MC) control** in OpenAI Gym’s `Blackjack-v1`, with a real-time **Pygame** visualizer that exposes the agent’s reasoning: states, ε-greedy choices, Q-values, and strategy hints.

---

## What This Project Does

- **Learns purely from experience** (model-free). The agent observes full episodes of Blackjack and updates its beliefs about action quality from realized returns—no dynamics model or value bootstrapping.
- **Tracks and improves a policy** over time using ε-greedy exploration:
  - Early training favors **exploration** (high ε), gradually shifting to **exploitation** as knowledge accumulates.
- **Surfaces internals live**:
  - Per-state **Q-values** for `STICK` and `HIT`
  - **Preferred action** and **confidence** (|ΔQ|)
  - **Exploration rate (ε)**, **win rate**, and **state coverage**
  - Human-readable **strategy insights** (e.g., “Dealer strong—consider HIT”)

---

## Environment & State/Action Space

- **State**: `(player_sum ∈ {0…31}, dealer_upcard ∈ {1…10}, usable_ace ∈ {0,1})`
- **Actions**: `STICK = 0`, `HIT = 1`
- **Rewards**: terminal +1 / 0 / −1

---

## Learning Algorithms

### Monte Carlo Prediction (Action Values)
Estimates \( Q(s,a) \) by averaging the **empirical returns** from full episodes generated under a behavior policy.

### Constant-α Monte Carlo Control
Incremental policy improvement using ε-greedy exploration and constant-step updates:

$$
Q(s,a) \leftarrow Q(s,a) + \alpha \big(G - Q(s,a)\big)
$$

- \( G \): **return** from the first occurrence of \((s,a)\) to episode end (undiscounted or discounted by \( \gamma \))
- \( \alpha \): constant step size for stability and continual adaptation
- **Policy improvement**: after each update, choose \( \arg\max_a Q(s,a) \) with probability \(1-\varepsilon\); otherwise explore.

> Notes:
> - In Blackjack, **first-visit** and **every-visit** MC are equivalent in expectation.
> - Decaying \( \varepsilon \) (down to a floor) balances coverage and convergence.

---

## Pygame Visualization

- **Fixed, readable layout**: table view, dealer/agent panels, and right-side dashboards.
- **Live episode playback**: “thinking” phase (shows Q-table lookup & confidence) → action → terminal outcome with a short **game breakdown**.
- **Learning dashboard**: episodes completed, win/loss/draw counts, ε, states learned, α, and current algorithm (“MC Control”).
- **Q-values panel**: highlights best action for the current state, confidence, and whether the agent is **exploring** or **exploiting**.

---

## What We Learned / Insights

- **Monte Carlo can learn solid Blackjack behavior without a model** by leveraging complete-episode returns and ε-greedy control.
- **Transparency matters**: the visualizer turns Q-tables into intuition—why the agent prefers `HIT` vs `STICK` in a given state and how confident it is.
- **Exploration scheduling** is crucial: high ε early prevents brittle policies; a gentle decay sustains discovery while letting the policy stabilize.
- **State coverage** drives learning speed: the UI’s “States Learned” metric correlates with policy quality; sparse regions lag and benefit from additional exploration.
- **Practical heuristics emerge** naturally:
  - High hard totals (e.g., 17+) trend toward **STICK**,
  - Very low totals (≤11) trend toward **HIT**,
  - Mid-range depends on the **dealer upcard**—mirroring textbook strategy.
- With modest training budgets (e.g., ~thousands of episodes), the policy becomes **qualitatively reasonable**; longer runs tighten Q-estimates and reduce oscillations.

---

## Code Highlights

- **Safety shims** for Gym API differences (`reset_env_safe`, `step_env_safe`).
- Clear separation of concerns:
  - Episode generation via current ε-greedy policy
  - MC return computation and **constant-α** updates
  - Real-time rendering & lightweight UI controls (step/auto/reset/speed)
- Minimal dependencies: `gym`, `numpy`, `pygame`.

---

## Use Cases

- **Teaching RL**: a concrete, end-to-end example of MC control with transparent internals.
- **Debugging policies**: visualize when/why exploration overrides exploitation.
- **Ablations**: vary \( \alpha \), \( \varepsilon \) decay, and \( \gamma \); observe convergence, stability, and sample efficiency.

---

## License

MIT
