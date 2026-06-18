

============= CELL 45 =============

## 8. Discussion

### 8.1 MDP Characterisation

Both environments are **episodic**, **fully observable**, **discrete** MDPs.

| Property | EmptyEnv | KeyDoorLavaEnv |
|---|---|---|
| Termination | Goal reached or max-steps truncated | Goal reached, lava contact, or max-steps |
| Horizon (`max_steps`) | 256 | 512 |
| Reward | Sparse (+1 at goal only) | Sparse (+1 at goal) + shaped sub-task bonuses |
| Observability | Full — position + direction | Full — position, direction, key/door/goal flags |
| Core difficulty | Medium — random start, unimodal reward | High — four sequential sub-tasks; sparse without shaping |

The decisive structural difference is the **sequential sub-task dependency** in KeyDoorLavaEnv (key → door → lava gap → goal). Random exploration that completes all four sub-tasks in order is exponentially unlikely, which is why event-based reward shaping is essential.

### 8.2 Algorithm Comparison

**Monte Carlo** updates only at episode end. In EmptyEnv this is fine — episodes are short and a single successful trajectory pushes useful Q-values back along the whole path. In KeyDoorLavaEnv MC is noisier: returns include the full variance of the rollout, and most early episodes return only the small shaping bonuses, so per-update signal-to-noise is low.

**SARSA** updates online using the *behaviour policy's* next action. Because behaviour is ε-greedy, SARSA's value estimates near lava reflect the cost of ε-greedy missteps and the learned policy stays one cell away from the lava column. This is a feature near the lava and a mild cost elsewhere (slightly longer paths than Q-Learning).

**Q-Learning** decouples behaviour (ε-greedy) from evaluation (greedy max), propagating the optimal-future value immediately. It reaches a near-optimal greedy policy in fewer episodes than SARSA. Training Q-values for cells next to lava can look optimistic, but at inference (ε = 0) it produces the shortest paths in both environments.

### 8.3 Hyperparameter Choices

Hyperparameters were chosen deliberately for each algorithm based on their update mechanics. The key insight from extensive experimentation is that **no single γ works well for all three algorithms** on KeyDoorLavaEnv.

#### EmptyEnv (shared settings)

| Parameter | Value | Reasoning |
|---|---|---|
| α | 0.1 | Small table (768 entries) converges quickly; modest α avoids oscillation |
| γ | 0.99 | High discount keeps the goal reward visible from the random start position |
| ε start | 0.5 | Modest exploration — random walks find the fixed goal quickly |
| ε decay | 0.995 | Reaches ε_min after ~600 episodes, leaving ~400 for exploitation |
| ε min | 0.05 | Small residual exploration prevents premature convergence |
| q_init | 0.5 | Optimistic init drives systematic state-space coverage |
| Episodes | 1 000 | Sufficient: state space is only 256 states × 3 actions |

#### KeyDoorLavaEnv — Per-Algorithm Design

Each algorithm received different hyperparameters because their update mechanics impose different requirements:

| Parameter | Monte Carlo | SARSA | Q-Learning | Why different |
|---|---|---|---|---|
| α | **0.3** | 0.1 | 0.1 | MC's high-variance episode-end returns need a faster learning rate to make progress; TD methods update per-step and need a smaller α to avoid oscillation |
| γ | **0.99** | **0.96** | **0.96** | MC computes full returns: with γ=0.96, the goal reward at step 50 is discounted to 0.96⁵⁰ ≈ 0.13 — nearly invisible. With γ=0.99 it is 0.99⁵⁰ ≈ 0.61. TD methods with large shaping bonuses (3.0, 2.4) risk over-optimistic Q-values at γ=0.99; γ=0.96 dampens propagation |
| q_init | **0.5** | **0.05** | **0.05** | MC selects actions using Q-values throughout the episode (no per-step updates), so optimistic init drives exploration. SARSA/Q-Learning update per-step and discover good actions quickly even from near-zero init |
| ε start | 1.0 | 1.0 | 1.0 | Full exploration at the start is needed to discover all four sub-tasks |
| ε decay | 0.995 | 0.995 | 0.995 | ε reaches 0.05 by episode ~600, giving 49 400 exploitation episodes — enough to refine Q-values across all randomised layouts |
| Episodes | 50 000 | 50 000 | 50 000 | Experimentally determined sweet spot: 10k is too few (under-exploited), while 100k offers no improvement and unnecessarily increases training time |

**Why ε_decay=0.995 outperforms 0.9999:** With v2 state, the problem is a proper MDP — no aliased states. Once the agent discovers the optimal path it can exploit it reliably. Fast decay (ε at 0.05 by episode 600) maximises the exploitation window, whereas slow decay (ε still at 0.37 at episode 10 000) keeps injecting noise that corrupts already-learned Q-values.

**Why 50k episodes instead of 100k:** Empirical testing showed that agents converge fully by 50k episodes (with SARSA and Q-Learning reliably hitting ~100% success). Running for 100k episodes yielded no additional improvement and, in fact, showed slight performance regressions (e.g., MC dropping from 59% to 55%) likely due to accumulated noise from the occasional random exploratory step. 50k represents a sufficient budget for convergence without wasting compute.

### 8.4 State Representation

The state function must encode enough information to make the MDP Markovian (no hidden variables) while keeping the table size tractable.

#### EmptyEnv

```
state = (agent_x, agent_y, agent_dir)
```

The agent position and orientation fully determine what action to take next. No other information is needed since the goal is fixed at the bottom-right corner.

- **State space:** 8 × 8 grid × 4 directions = **256 states**
- **Q-table size:** 256 × 3 actions = **768 entries**
- **Actions used:** `left (0)`, `right (1)`, `forward (2)` — `pickup`, `toggle`, and `done` are no-ops in an empty room.

#### KeyDoorLavaEnv

The task has four sequential sub-tasks: pick up the key → open the door → cross the lava gap → reach the goal. A single state tuple cannot efficiently represent all phases, because the relevant target changes across phases. We use a **phase-conditioned state** that includes the agent's current position, the current phase index, the position of the active sub-goal, and the direction:

| Phase | Trigger | State tuple | Active target |
|---|---|---|---|
| 0 — Find key | Default (key not carried) | `(ax, ay, 0, kx, ky, d)` | Key position |
| 1 — Open door | Key carried, door closed | `(ax, ay, 1, dx, dy, d)` | Door position |
| 2 — Reach goal | Door open | `(ax, ay, 2, gy, d)` | Goal row |

All positions come directly from environment attributes (`env.agent_pos`, `env.current_key_pos()`, `env.door_pos`, `env.goal_pos`) — no distance computations, no geometric signals.

**Why this representation makes the MDP Markovian:** Without the key/door position, two states that look identical to the agent (same cell, same direction) but differ in key/door location would map to the same Q-table entry. The optimal action differs (e.g., move left vs. right toward the key), causing Q-value conflicts that prevent convergence. Including the sub-goal position removes all such ambiguity.

**Why phase-conditioned rather than concatenating all positions:** Including all entity positions in every state would create an unnecessarily large table and would require the agent to learn irrelevant correlations (e.g., door position during phase 2 doesn't matter). By conditioning on the current phase, each phase has its own compact state space.

**Actions used:** `left (0)`, `right (1)`, `forward (2)`, `pickup (3)`, `toggle (5)` — `drop (4)` and `done (6)` serve no purpose in this task.

**State space size (actual reachable bounds):** Because the agent cannot pass the partition wall until the door is open, the effective state space is much smaller than a naive 10×10 grid calculation:
- **Phase 0:** Agent confined to left room (3×8 = 24 cells). Key is in left room (24 cells). 4 directions. (24 × 24 × 4 = 2,304 states)
- **Phase 1:** Agent in left room (24 cells). Door is on partition wall (fixed x, 8 possible y positions). 4 directions. (24 × 8 × 4 = 768 states)
- **Phase 2:** Both rooms reachable (~56 empty cells). Goal is in one of two corners (2 positions). 4 directions. (56 × 2 × 4 = 448 states)
- **Total reachable states:** 2,304 + 768 + 448 = **3,520 states**

- **Q-table size:** 3,520 states × 5 actions = **17,600 entries**

### 8.5 Reward Shaping Analysis

Without shaping, KeyDoorLavaEnv has a purely sparse reward: +1 only at the goal. The probability of a random agent completing all four sub-tasks in one episode is negligibly small, so most early episodes return only the step penalties — the agent receives no useful gradient signal and cannot learn.

We apply **event-based shaping** that triggers a one-time bonus at each sub-task milestone:

| Event | Bonus | Implementation |
|---|---|---|
| Pick up the key | **+3.0** | `is_carrying_key()` rising-edge, guarded by `_got_key_bonus` flag |
| Cross the lava gap | **+2.4** | `has_crossed_lava()` rising-edge, guarded by `_got_cross_bonus` flag |
| Step penalty | **−0.0005** | Applied every step |

#### Why large bonuses (3.0, 2.4)?

Discovered via Optuna hyperparameter search (25 trials, 20 000 episodes each). Runs with bonuses capped at 0.3–0.5 produced noticeably lower success rates. The key-pickup bonus must be large enough to dominate the noise in early training episodes. With the step penalty at 0.0005, the worst-case total penalty over 512 steps is only 0.256 — well below the 3.0 key bonus, so the signal is never overwhelmed.

**Ratio check:** `key_bonus (3.0) ≫ max_step_penalty (0.0005 × 512 = 0.256)` ✅

#### What the shaping encourages

The agent treats key pickup and lava crossing as highly valuable sub-goals, even in early episodes where it never reaches the final goal. This creates a **curriculum effect**: the agent first learns to find and pick up the key, then learns to open the door, then learns to cross the lava — each stage building on the previous.

#### Potential unintended effects and how they are prevented

| Risk | Prevention |
|---|---|
| Agent repeatedly drops and picks up the key to farm the bonus | One-shot flag (`_got_key_bonus`) resets only at episode start; the bonus fires at most once per episode |
| Agent learns to cross lava repeatedly | Same rising-edge guard (`_got_cross_bonus`) |
| Step penalty discourages necessary exploration | Penalty magnitude (0.0005) is small enough that any sub-task bonus (≥2.4) easily outweighs many penalty steps |

#### Shaping magnitude and algorithm sensitivity

An interesting finding from experimentation: γ and the bonus magnitude interact differently per algorithm. With large bonuses and γ=0.99, Q-Learning's greedy-max bootstrap propagates the 3.0 key bonus aggressively forward, creating over-optimistic Q-values in states near the lava. SARSA's on-policy updates include the cost of ε-greedy missteps, naturally suppressing over-optimism. MC averages over whole trajectories, making it robust to local Q-value inflation but sensitive to the overall scale of returns.


============= CELL 46 =============

## 9. Best Settings

The tables below summarise the optimal configurations discovered and used for the final training runs.

### EmptyEnv (All Algorithms)

| Hyperparameter | Value |
|---|---|
| α (learning rate) | 0.1 |
| γ (discount) | 0.99 |
| ε start | 0.5 |
| ε decay (per episode) | 0.995 |
| ε minimum | 0.05 |
| Q-table initialisation | 0.5 (optimistic) |
| Training episodes | 1 000 |
| Actions used | left (0), right (1), forward (2) |
| `max_steps` | 256 |

### KeyDoorLavaEnv

As discussed in Section 8.3, no single configuration works optimally for all algorithms due to their different update mechanics.

#### Environment Settings (Shared)
| Environment Property | Value |
|---|---|
| Actions used | left (0), right (1), forward (2), pickup (3), toggle (5) |
| `max_steps` | 512 |
| Shaping: key pickup bonus | **+3.0** |
| Shaping: lava cross bonus | **+2.4** |
| Step penalty | **−0.0005** |

#### Algorithm Hyperparameters
| Hyperparameter | Monte Carlo | SARSA | Q-Learning |
|---|---|---|---|
| α (learning rate) | 0.3 | 0.1 | 0.1 |
| γ (discount) | 0.99 | 0.96 | 0.96 |
| ε start | 1.0 | 1.0 | 1.0 |
| ε decay (per episode) | 0.995 | 0.995 | 0.995 |
| ε minimum | 0.05 | 0.05 | 0.05 |
| Q-table initialisation | 0.5 | 0.05 | 0.05 |
| Training episodes | 50 000 | 50 000 | 50 000 |
