# APEX
This folder contains the core functionalities required to execute the framework. Below is an overview of the key files and their purposes:

## Files

### `Main.py`
The main entry point of the APEX framework. This script connects all components and executes the workflow based on the provided settings file. The `SETTINGS_FILE` variable specifies the input configuration and can be set to:
- `../config/RPL_config.yaml` for the RPL protocol
- `../config/crystal_config.yaml` for the Crystal protocol

### `Next_Testpoint_Selection.py`
Implements the Next Test Point Selection (NTS) algorithms, which are pivotal to the framework. Key approaches include:
- **EI (Expected Improvement)**: Focuses on maximizing the expected improvement to find the optimal configuration efficiently.
- **GP-LCB (Gaussian Process Lower Confidence Bound)**: Balances exploration and exploitation, leveraging Gaussian Process models for optimization.

Other methods included in this script serve as baseline approaches.

### `model_fitting.py`
Provides the model fitting functionality. The framework uses:
- **Linear Regression (LR)**: Utilized for simpler, greedy approaches.
- **Gaussian Process (GP)**: Employed in APEX approaches for advanced parameter optimization.

### `ResultsStorage.py` and `ResultsStorage_LR.py`
Handles the storage and management of results. These scripts provide functionalities to store experiment data, update fitted models, and retrieve relevant statistics for decision-making:
- **`ResultsStorage.py`**: Handles operations related to Gaussian Process-based approaches.
- **`ResultsStorage_LR.py`**: Handles operations related to Linear Regression-based approaches.

### `Utilities.py`
Offers additional utility functions used across various processes, such as parameter management, data transformation, and evaluation of thresholds and goals.

### `RecordedTestEnvironment.py`, `DCubeTestEnvironment.py`
Provide specific implementations for testing protocols:
- **RecordedTestEnvironment.py**: Utilizes pre-recorded data for evaluation.
- **DCubeTestEnvironment.py**: Executes real-world experimentation using the D-Cube testbed.

### `AbstractTestEnvironment.py`
Defines an abstract base class for creating custom test environments. It requires implementing the `execute_test` method to define the behavior of a specific environment.

