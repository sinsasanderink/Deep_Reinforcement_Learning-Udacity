# 🎰 Monte Carlo Blackjack with Pygame Simulation

An interactive **Reinforcement Learning** project where an agent learns to play Blackjack via **Monte Carlo (MC) control** in OpenAI Gym’s `Blackjack-v1`, with a real-time **Pygame** visualizer that reveals the agent’s internal state: Q-values, ε-greedy choices, confidence, exploration rate, and strategy hints.

---

## What This Project Does

- **Model-free learning from full episodes.** The agent improves purely from sampled returns—no transition model, no bootstrapping.
- **Policy improvement via ε-greedy control.** Early training explores widely; ε decays to favor exploitation as knowledge accumulates.
- **Transparent learning.** The UI surfaces per-state `STICK`/`HIT` Q-values, preferred action and confidence (‖ΔQ‖), running win rate, ε, and state coverage—turning tables into intuition.

---

## Environment & Spaces

- **State**: `(player_sum ∈ {0…31}, dealer_upcard ∈ {1…10}, usable_ace ∈ {0,1})`  
- **Actions**: `STICK = 0`, `HIT = 1`  
- **Terminal rewards**: `+1 / 0 / −1`

---

## Learning Overview

### Monte Carlo Returns
For a first occurrence of state–action pair \((s_t, a_t)\) in an episode of horizon \(T\), the (discounted) return is
\[
G_t \;=\; \sum_{k=0}^{T-t-1} \gamma^{k}\, R_{t+1+k}.
\]

### Constant-α MC Control (Incremental)
The action-value estimate is updated as
$$
Q(s,a) \;\leftarrow\; Q(s,a) + \alpha \big( G - Q(s,a) \big),
$$
where \(G\) is the Monte Carlo return, \(\alpha\) is a constant step size, and \(\gamma\) is the discount factor.

### Behavior & Improvement
- **Behavior policy**: ε-greedy over current \(Q\) to ensure coverage.  
- **Improved target policy**: greedy w.r.t. \(Q\).  
- **Action selection**: with probability \(1-\varepsilon\) choose \(\arg\max_a Q(s,a)\); otherwise explore.

> In this Blackjack setting, first-visit and every-visit MC produce equivalent estimates in expectation; constant-\(\alpha\) updates enable continual adaptation.

---

## Visualization (Pygame)

- **Episode playback** with “thinking” and “action” phases.
- **Dashboards**: episodes, win/loss/draw tallies, \(\varepsilon\), states learned, \(\alpha\), algorithm label.
- **Q-values panel**: highlights best action, shows confidence \( |Q(s,\text{STICK})-Q(s,\text{HIT})| \), and whether the agent is exploring or exploiting.
- **Strategy hints** to connect learned behavior to human blackjack intuition (e.g., “Hard 17+ → STICK”, “≤11 → HIT”, mid-totals depend on dealer upcard).

---

## Key Takeaways

- **Monte Carlo can learn strong Blackjack behavior** without a model by averaging episode returns under ε-greedy exploration.
- **Exploration scheduling matters**: slower ε decay yields broader state coverage and more stable policies.
- **Readable internals accelerate understanding**: visualizing \(Q\), confidence, and ε makes it clear *why* the agent prefers `HIT` vs `STICK` in a given state.
- **Heuristics emerge end-to-end**: with enough episodes, the learned policy mirrors textbook strategy (high hard totals stick; low totals hit; mid-range depends on dealer strength).
- **Ablations are easy**: vary \(\alpha\), \(\varepsilon\) decay, and \(\gamma\) to study convergence speed, variance, and stability.

---

## Code Highlights

- Safe wrappers for Gym API differences (`reset_env_safe`, `step_env_safe`).
- Clean separation of concerns:
  - ε-greedy episode generation from \(Q\)
  - MC return computation & constant-\(\alpha\) updates
  - Real-time rendering with fixed, legible layout

---

## License

MIT
