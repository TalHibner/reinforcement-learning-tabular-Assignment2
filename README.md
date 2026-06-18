# Tabular Reinforcement Learning in MiniGrid

**Authors:** Gilad Ticher (318770039) & Tal Hibner (026548446)  
**Course:** Reinforcement Learning: Mid Semester Project - 2026 B

## Overview
This repository contains the implementation and comparative analysis of three tabular reinforcement learning algorithms—**Monte Carlo**, **SARSA**, and **Q-Learning**—applied to custom MiniGrid environments. 

The agents were evaluated on two environments:
1. **`EmptyEnv`**: A simple navigation task in an empty room.
2. **`KeyDoorLavaEnv`**: A complex, sequential puzzle requiring the agent to locate a key, unlock a door, and cross a lava gap.

While all three algorithms easily solved the simple environment, the complex environment presented significant exploration challenges due to its sparse rewards and high-penalty termination states. We demonstrate that through targeted event-based reward shaping and algorithm-specific hyperparameter tuning (via **Optuna**), temporal-difference methods (SARSA and Q-Learning) successfully achieve optimal policies with a 100% success rate. In contrast, Monte Carlo methods struggled with high-variance truncation penalties, highlighting the structural advantages of bootstrapping in environments with long horizons.

## Repository Structure

- `HW2_2026B_KeyDoorLava_finalfinal.ipynb`: The primary Jupyter Notebook containing the full implementation of the environments, the RL agents, reward shaping logic, and Optuna hyperparameter optimization.
- `report_026548446_318770039.pdf`: The final compiled scientific report detailing the Markov Decision Process (MDP) characteristics, algorithm comparisons, and reward shaping analysis.
- `report_026548446_318770039.md`: The raw Markdown source for the final report.
- `videos/`: Rendered MP4 recordings demonstrating the trained agents' policies in both environments.
- `img_*.png` / `final_report_img_*.png`: Generated matplotlib graphs used in the final report.

## Setup & Execution

### Prerequisites
To run the notebook locally, ensure you have the required dependencies installed:
```bash
pip install minigrid gymnasium numpy matplotlib tqdm optuna imageio imageio-ffmpeg
```

### Running the Notebook
Open `HW2_2026B_KeyDoorLava_finalfinal.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab. The notebook is fully self-contained and structured to execute sequentially from environment setup through training, evaluation, and video rendering.
