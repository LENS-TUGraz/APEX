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

# Application Requirements (ARs)

The subfolders inside each NTS approach represent different **Application Requirements (ARs)**. Each AR defines a specific optimization goal with a constraint. These ARs are designed to evaluate the performance of the protocol under various scenarios.

## AR Table

The following table lists the considered ARs along with their respective goals and constraints:

| **Ref** | **Goal**           | **Constraint**    |
|---------|--------------------|-------------------|
| AR_1     | Minimize \( E_c \) | PRR ≥ 65%         |
| AR_2     | Minimize \( E_c \) | PRR ≥ 92%         |
| AR_3     | Minimize \( E_c \) | PRR ≥ 96%         |
| AR_4     | Maximize PRR       | \( E_c \) ≤ 210 J |
| AR_5     | Maximize PRR       | \( E_c \) ≤ 190 J |
| AR_6     | Maximize PRR       | \( E_c \) ≤ 170 J |
| AR_13    | Minimize \( E_c \) | PRR ≥ 0.975       |
| AR_14    | Maximize PRR       | \( E_c \) ≤ 168 J |

### Notes:
- **Goal**: Indicates whether the focus is on minimizing energy consumption (\( E_c \)) or maximizing the Packet Reception Rate (PRR).
- **Constraint**: Specifies the conditions under which the goal must be achieved.
- \( E_c \): Represents energy consumption in joules (J).
- PRR: Stands for Packet Reception Rate and is expressed as a percentage.

Each AR folder contains results specific to that application requirement for the given NTS approach.

Inside each AR folder, the following files can be found:

- **`AR_X_goal_value.json`**: Contains the goal value returned as a nested dictionary:
  - The outer keys represent the iteration numbers.
  - The inner keys represent the number of testbed trials in the respective iteration.

- **`AR_X_parameter_set.json`**: Contains the returned parameter set as a nested dictionary:
  - The outer keys represent the iteration numbers.
  - The inner keys represent the number of testbed trials in the respective iteration.
  - The parameter set is given as a list, where each element corresponds to parameter values in the order defined in the respective configuration file.

