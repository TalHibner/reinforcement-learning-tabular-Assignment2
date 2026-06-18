import json
import base64

with open('HW2_2026B_KeyDoorLava_final.ipynb', 'r') as f:
    nb = json.load(f)

md_lines = [
    "# Reinforcement Learning: Mid Semester Project - 2026 B",
    "**Gilad Ticher 318770039 & Tal Hibner 026548446**",
    "\n---",
    "\n# Environments",
    "\nTwo custom MiniGrid environments are defined below: **`EmptyEnv`** and **`KeyDoorLavaEnv`**. Both",
    "expose all 7 MiniGrid actions and use a sparse reward (`+1` on goal, `0` otherwise). Each env's",
    "section below explains the layout and the rules on what you may edit.",
    "\n**Action IDs.** Both envs use the standard MiniGrid action space (`Discrete(7)`):",
    "\n| ID | Name | Effect |",
    "|---|---|---|",
    "| 0 | `left` | turn 90° to the left (no move) |",
    "| 1 | `right` | turn 90° to the right (no move) |",
    "| 2 | `forward` | move one cell in the facing direction |",
    "| 3 | `pickup` | pick up an object in the cell directly in front |",
    "| 4 | `drop` | drop the carried object into the cell directly in front |",
    "| 5 | `toggle` | activate the object in front (e.g. open a door with the matching key) |",
    "| 6 | `done` | no-op in these envs |",
    "\n**Coordinates.** Cells are `(x, y)` with `x` growing **rightward** and `y` growing **downward**",
    "(so `y=1` is the top interior row and `y=H-2` is the bottom). The agent's facing direction",
    "`env.agent_dir` is `0` = right, `1` = down, `2` = left, `3` = up.",
    "\n# Environment 1: EmptyEnv",
    "\n**EmptyEnv Training Progress**",
    "![EmptyEnv Plot](final_report_graph_0.png)",
    "\n# Environment 2: KeyDoorLavaEnv",
    "\n**KeyDoorLavaEnv Training Progress**",
    "![KeyDoorLavaEnv Plot](final_report_graph_1.png)",
    "\n```text",
    "============================================================",
    "  EmptyEnv — Greedy Evaluation (100 episodes each)",
    "============================================================",
    "  Algorithm         Avg Reward    Avg Steps   Success Rate",
    "  -------------------------------------------------------",
    "  Monte Carlo            0.990         18.4         99.0%",
    "  SARSA                  0.990         11.8         99.0%",
    "  Q-Learning             0.970         16.6         97.0%",
    "============================================================",
    "\n============================================================",
    "  KeyDoorLavaEnv — Greedy Evaluation (100 episodes each)",
    "============================================================",
    "  Algorithm         Avg Reward    Avg Steps   Success Rate",
    "  -------------------------------------------------------",
    "  Monte Carlo            4.960        227.4         61.0%",
    "  SARSA                  6.388         24.7        100.0%",
    "  Q-Learning             6.387         25.2        100.0%",
    "============================================================",
    "```\n"
]

# Append Discussion and Settings
for cell in nb['cells']:
    if cell['cell_type'] == 'markdown':
        src = "".join(cell.get('source', []))
        if "8. Discussion" in src or "8.1" in src or "8.2" in src or "8.3" in src or "8.4" in src or "8.5" in src or "9. Best Settings" in src:
            # We must convert state space bullets to a table as requested by the user
            old_bounds = "- **Phase 0:** Agent confined to left room (3×8 = 24 cells). Key is in left room (24 cells). 4 directions. (24 × 24 × 4 = 2,304 states)\n- **Phase 1:** Agent in left room (24 cells). Door is on partition wall (fixed x, 8 possible y positions). 4 directions. (24 × 8 × 4 = 768 states)\n- **Phase 2:** Both rooms reachable (~56 empty cells). Goal is in one of two corners (2 positions). 4 directions. (56 × 2 × 4 = 448 states)\n- **Total reachable states:** 2,304 + 768 + 448 = **3,520 states**"
            new_bounds_table = "| Phase | Agent Domain | Target Domain | Directions | States |\n|---|---|---|---|---|\n| **0 — Key Search** | Left room (24 cells) | Key in left room (24 cells) | 4 | 2,304 |\n| **1 — Door Open** | Left room (24 cells) | Door on wall (8 positions) | 4 | 768 |\n| **2 — Cross Lava** | Both rooms (~56 cells) | Goal in corners (2 positions) | 4 | 448 |\n| **Total** | | | | **3,520** |"
            src = src.replace(old_bounds, new_bounds_table)
            md_lines.append(src + "\n")

with open('report_026548446_318770039.md', 'w') as f:
    f.write("\n".join(md_lines))

print("Markdown rebuilt.")
