# Synthetic_Data

This folder contains the synthetic dataset, scripts, and evaluation results used to assess APEX in high-dimensional optimization settings.

## Files

- **Synthetic_data_creation_script.py**  
  Script used to generate the synthetic dataset. It extends the RPL protocol configuration space by adding two additional parameters, creating a five-parameter setup. The script uses rule-based models and real testbed observations, combined with controlled noise, to mimic realistic wireless network behavior.

- **real_synthetic_combined_972_5.json**  
  Combined dataset that merges real testbed data and generated synthetic data, used for evaluating the framework’s optimization performance in a high-dimensional setting.

- **filtered_real_testbed_data.json**  
  Filtered subset of real testbed data extracted from the main dataset, used as the basis to train and generate the synthetic dataset.

## Subfolder

- **Evaluation_Results**  
  Contains the evaluation results obtained by running APEX and baseline optimization methods on the synthetic dataset.  
  Inside, you will find:
  - One folder per NTS approach (e.g., EI, GP-LCB, GEL, GER, GUC, RL-Any, RL-Step, RL-GP, SVM).
  - Inside each NTS approach folder, the results are reported for **Application Requirement AR_9**.
    - **AR_9_goal_value.json**: The goal values returned during the experiment, organized as a nested dictionary (outer key: iteration number, inner key: trial number).
    - **AR_9_parameter_set.json**: The parameter sets returned, organized similarly, where each parameter set is stored as a list following the order defined in the configuration.
