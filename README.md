# 🎰 Monte Carlo Blackjack with Pygame Simulation

An interactive **Reinforcement Learning** project where an agent learns Blackjack with **Monte Carlo (MC) control** in OpenAI Gym’s `Blackjack-v1`, visualized live in **Pygame**. The UI exposes Q-values, ε-greedy decisions, confidence (|ΔQ|), state coverage, and evolving win rate—turning a Q-table into something you can see and reason about.

---

## What It Shows

- **Model-free learning from episodes.** The agent improves using only sampled returns—no transition model, no TD bootstrapping.
- **Policy improvement via ε-greedy control.** Early exploration (high ε) gradually gives way to exploitation as estimates stabilize.
- **Transparent internals.** For each state, you can see `STICK`/`HIT` Q-values, the preferred action, confidence, exploration vs exploitation, and concise strategy hints.

---

## Environment & Spaces

- **State**: `(player_sum ∈ {0…31}, dealer_upcard ∈ {1…10}, usable_ace ∈ {0,1})`  
- **Actions**: `STICK = 0`, `HIT = 1`  
- **Rewards**: terminal `+1 / 0 / −1`

---

## Learning Mechanics (MC)

**Episode return** from the first visit to a state–action pair:

$$
G_t \;=\; \sum_{k=0}^{T-t-1} \gamma^{\,k}\, R_{t+1+k}
$$

**Constant-α Monte Carlo control (incremental update):**

$$
Q(s,a) \;\leftarrow\; Q(s,a) \;+\; \alpha \,\big[\,G \;-\; Q(s,a)\,\big]
$$

**Greedy policy wrt Q (used for improvement/exploitation):**

$$
\pi(s) \;=\; \arg\max_{a} \, Q(s,a)
$$

- `alpha (α)`: constant step size for stability and continual adaptation  
- `gamma (γ)`: discount factor (often 1.0 in episodic Blackjack)  
- `epsilon (ε)`: exploration rate with gentle decay to a minimum floor

> In Blackjack, first-visit and every-visit MC yield equivalent estimates in expectation; constant-α updates keep learning responsive as more episodes arrive.

---

## Visualization Highlights

- **Thinking → Action** phases each step, with the current state, the Q-lookup, confidence, and the chosen move.
- **Dashboards** for episodes, W/L/D, ε, states learned, and α.
- **Q-values panel** shows `Q(s, STICK)` vs `Q(s, HIT)`, preferred action, and whether the agent is exploring or exploiting.
- **Strategy cues** that connect learned behavior to human intuition  
  (e.g., hard 17+ → usually STICK; ≤11 → usually HIT; mid-range depends on dealer strength).

---

## Key Takeaways

- **Monte Carlo can learn strong play without a model** by averaging full-episode returns under a well-scheduled ε-greedy policy.
- **Exploration scheduling matters.** A slow ε decay improves state coverage and reduces premature convergence to brittle policies.
- **Visibility accelerates understanding.** Watching Q-values and confidence evolve makes it clear *why* the agent prefers `HIT` vs `STICK` in specific contexts.
- **Emergent heuristics mirror textbook strategy.** With sufficient episodes, the policy approximates standard Blackjack guidance (high hard totals stick; low totals hit; mid-range is dealer-dependent).
- **Ablations are simple.** Vary α, ε-decay, and γ to study convergence speed, variance, and stability.

---

## Code Structure (Essentials)

- **Episode generation:** ε-greedy rollouts from the current Q.  
- **Returns & updates:** compute `G` per first visit and apply the constant-α update to `Q(s,a)`.  
- **Safety wrappers:** handle Gym API differences (`reset_env_safe`, `step_env_safe`).  
- **Renderer:** fixed, legible Pygame layout for table, dashboards, Q-panel, and hints.

---

## License

MIT
