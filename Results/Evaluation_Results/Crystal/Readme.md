Each subfolder is named after the NTS approaches used during the evaluation.

## NTS Approaches

- **EI (Expected Improvement)**: Maximizes the expected improvement, focusing on areas of the parameter space that could yield significant improvements in the objective.
- **GP-LCB (Gaussian Process Lower Confidence Bound)**: Balances exploration and exploitation by utilizing Gaussian Process models to consider both predicted performance and uncertainty.
- **GEL (Greedy for Exploration)**: Focuses on exploring the parameter space extensively, aiming for broader coverage.
- **GER (Greedy for Exploitation)**: Prioritizes testing parameters that are expected to perform well based on current knowledge.
- **GUC (Greedy for Uncertainty)**: Concentrates on areas of high uncertainty in the parameter space to refine predictions.
- **RL-Any**: Utilizes reinforcement learning to explore the parameter space, can move from one state to any other state.
- **RL-Step**: A reinforcement learning approach focusing on stepwise exploration of the parameter space.

---

Each NTS approach subfolder contains evaluation results specific to that method for Crytsal protocol.
