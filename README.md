# APEX Framework

APEX (Automated Parameter Exploration) is a framework designed for optimizing protocol parameters in low-power wireless systems. This project provides tools to optimize protocol parameters for a given application requirements (ARs).
## Project Structure

- **APEX/**: Contains the main functionalities of the framework, including core modules for test environments, result storage, utility functions, model fitting, and next test point selection algorithms. Detailed information about this folder is provided in its own README file.
- **Results**:
  - Contains recorded results used for evaluation.
  - Includes a subfolder `Evaluation_Results` with protocol evaluation results for different ARs.
- **config**:
  - Contains files for specifying user requirements, including application goals, parameter ranges, and termination criteria for optimization.
  - Example inputs include selecting the test environment, defining optimization targets, setting constraints, and configuring the next test point selection algorithm.
- **Binaries**:
  - Contains firmware and related files for scheduling experiments in the D-Cube testbed.
  - Ensure the correct API key is updated in `config/dcubeKey.yaml` if using this feature.

## Usage

### Requirements
Make sure you have the required dependencies installed. You can set up the environment by running:

```bash
pip install -r requirements.txt
```

### Configuration
Before running the framework, configure the following:

1.  **Input Parameters**: Specify your requirements and inputs in the configuration files located in the `config` folder.
2.  **API Key**: Update the `config/dcubeKey.yaml` file with your D-Cube API key if you plan to schedule experiments on the D-Cube testbed.

### Running the Framework
To run the main program, execute:

```bash
python Main.py
```

### Selecting a Test Environment
Choose one of the following test environments:

- **RecordedTestEnvironment**: Use pre-recorded data.
- **DCubeTestEnvironment**: Run tests in the D-Cube testbed.

### Next Test Point Selection Algorithms
The framework includes several algorithms for next test point selection, implemented in the `APEX` folder. Key methods include:

- **EI (Expected Improvement)**: Maximizes the expected improvement to focus on areas of high potential gain.
- **GP-LCB (Gaussian Process Lower Confidence Bound)**: Explores the parameter space while considering uncertainty.
- **Baseline Approaches**:
  - **GEL (Greedy for Exploration)**: A straightforward method for exploring parameter space.
  - **GUC (Greedy for Uncertainty)**: Focuses on uncertain regions in the parameter space.
  - **etc.**

## Results

The results of the experiments are stored in the `Results` folder, which contains recorded results for the evaluated protocols. The evaluation results related to the paper for different application requirements (ARs) can be found in the `Evaluation_Results` subfolder.

## Contribution

Feel free to contribute to the project by submitting pull requests or reporting issues.

---

### Additional Notes

- If running on the D-Cube testbed, ensure the `Binaries` folder contains the correct firmware and related files.
