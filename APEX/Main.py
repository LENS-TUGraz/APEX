"""
Main Module for the APEX Framework.

This script serves as the entry point for the APEX framework. It connects all core components
and executes the optimization workflow based on the specified settings file. The settings file
defines the protocol (e.g., RPL, Crystal), application requirements (ARs), and other configuration parameters.

Usage:
- Update the `SETTINGS_FILE` variable to specify the configuration file (e.g., RPL_config.yaml or crystal_config.yaml).
- Run this script to start the framework.

Key Features:
- Loads and connects core modules such as test environments, NTS approaches, and result storage.
- Executes the optimization process iteratively based on the selected protocol and configuration.

Note:
Ensure all dependencies are installed and configurations are correctly set before execution.
"""

import copy
import random
import statistics
import sys

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler

import Utilities
from Next_Testpoint_Selection import Next_Testpoint_Selection
from ResultsStorage import ResultStorage
from ResultsStorage_LR import ResultStorage_LR
from model_fitting import fit_model

#SETTINGS_FILE = '../config/RPL_config.yaml' # The settings file to use for RPL
SETTINGS_FILE = '../config/crystal_config.yaml' # The settings file to use for Crystal

class Main:
    def __init__(self, settings_file):
        self.settings = dict()
        with open(settings_file) as f:
            self.settings = yaml.full_load(f)
        Utilities.reformat_settings(self.settings) # Reformat the setting file to extract information
        print(f'Main: Selected settings file: {settings_file}', flush=True)
        self.result_storage = ResultStorage(self.settings) # Create an instance of the result storage
        self.result_storage_LR = ResultStorage_LR(self.settings)  # Create an instance of the result storage for Linear Regression
        self.test_environment = Utilities.create_instance(self.settings['main']['testEnvironment'], self.settings) # Create an instance of the test environment
        self.next_testpoint_selection = Next_Testpoint_Selection(self.settings) # Create an instance of the next test point selection
        self.Param_ranges = self.settings['main']['parameterRanges'] # Get the parameter ranges
        self.base_params = list(self.settings['main']['parameterNames'].keys()) # base of the parameters
        NTS = self.settings['main']['nextPointAlgo'] # Get the next point algorithm
        if hasattr(self.next_testpoint_selection, NTS): # If the next test point selection has the attribute given in the settings
            self.next_params_algorithm = getattr(self.next_testpoint_selection, NTS) # Get the attribute from the next test point selection class
        else:
            print(
                f"Error: The attribute '{NTS}' does not exist in 'Next_Testpoint_Selection'. Please check the spelling or create the respective NTS method")
            sys.exit(1)  # Exit the program with a non-zero status to indicate an error
        self.next_params_algorithm = getattr(self.next_testpoint_selection, self.settings['main']['nextPointAlgo']) # Get the next parameter algorithm from the next test point selection class
        self.model_type = self.settings['main']['model_type'] # Get the model type
        self.n_init_runs = self.settings['main']['totalInitTests'] # Get the total number of initial tests
        self.scaler = StandardScaler() # Create a common instance of the standard scaler for scaling the data
        anon_name = self.settings['main']['anonymousNames'] # Get the anonymous names to find the anonymous variable list
        self.anon_var_list = list(anon_name.values()) # Get the list of anonymous variable names
        if 'Max_number_of_testbed_trials' in self.settings['main']['termination_criteria']: # check whether the termination criteria is given in terms of max number of testbed trials
            self.total_runs = self.settings['main']['termination_criteria']['Max_number_of_testbed_trials'] # set the total number of runs to the max number of testbed trials
        else: # Otherwise
            self.total_runs = self.settings['main']['totalTests'] # set the total number of runs to the total number of tests
        self.min_test_needed = Utilities.get_min_test_needed(self.settings) # Get the minimum test needed based on the confidence level and percentile requirement
        self.cumulative_worst_regret = {} # This is the cumulative worst regret so far
        self.max_runs = self.min_test_needed # This is the maximum number of runs for each tests (exhasutive search).
                                             # Here it is set to the minimum test needed. Can be changed to any other value but has to be greater than the minimum test needed
        self.best_goal_itr = {} # Keep track of the best goal so far for each iteration
        self.best_params_itr = {} # Keep track of the best parameter set so far for each iteration

    def perform_initial_tests(self): # Perform random initial tests
        if self.settings['main']['model_type'] == 'GP':  # Gaussian Process
            init_runs = 0
            if self.settings['main']['suggested_parameter_sets'][
                'Enabled'] == True:  # If there are suggested parameter sets
                List_of_suggested_parameter_sets = [tuple(item) for item in self.settings['main']['suggested_parameter_sets'][
                    'List'] ] # Get the list of suggested parameter sets
                for suggested_param in List_of_suggested_parameter_sets:  # For each suggested parameter
                    suggested_param_dict = dict(
                        zip(self.base_params, suggested_param))  # Create a dictionary of the suggested parameter
                    print(f'Main: Initial test run {init_runs + 1} with suggested parameter set {suggested_param_dict}')
                    current_result = self.test_environment.execute_test(
                        suggested_param_dict)  # Execute the test with the suggested parameter
                    self.result_storage.add_single_test(current_result,
                                                        used=True)  # Add the test result to the result storage
                    init_runs += 1  # Increment the initial runs
                self.result_storage.update_fits()  # Update the fits
                self.settings['main']['totalInitTests'] = self.settings['main']['totalInitTests'] - init_runs  # Subtract the initial runs from the total initial tests
            for init_test_run_nr in range(self.settings['main']['totalInitTests']):  # For each initial test run
                parameters_to_draw_from = copy.deepcopy(self.settings['main'][
                                                            'listOfParamDicts'])  # Create a deep copy of the list of parameter dictionaries
                information_gain = False
                while not information_gain: # While there is no information gain
                    if len(parameters_to_draw_from) > 0: # If the length of the list of parameter dictionaries is greater than 0
                        random_index = random.randrange(len(parameters_to_draw_from)) # Randomly select an index from the list of parameter dictionaries
                        random_params = parameters_to_draw_from.pop(random_index) # Remove the randomly selected parameter dictionary from the list of parameter dictionaries
                        # added information would be redundant (linear independence):
                        if init_test_run_nr < self.settings['main']['fitFreedomDegrees']: # If the initial test run number is less than the freedom degrees
                            information_gain = self.result_storage.check_for_information_gain(random_params) # Check for information gain
                        else:
                            information_gain = True  # don't care anymore system is overdetermined anyways!  Set information gain to True
                    else:
                        print(f'Main: Warning, the parameter space is too small to provide enough information '
                              f'for the specified fit function. Your fits might make little sense this way.')
                        random_params = Utilities.pick_random_params(self.settings) # Pick random parameters
                        information_gain = True # Set information gain to True
                print(f'Main: Initial test run {init_runs +init_test_run_nr + 1} with random parameter set {random_params}')
                current_result = self.test_environment.execute_test(random_params) # Execute the test with the selected random parameter
                self.result_storage.add_single_test(current_result, used=True) # Add the test result to the result storage
            self.result_storage.update_fits() # Update the fits
            anon_name = self.settings['main']['anonymousNames'] # Get the anonymous names
            self.anon_var_list = list(anon_name.values()) # Get the list of anonymous names
            self.X_values = self.result_storage.table[self.anon_var_list].values.tolist() # Get the X values which is the features (input) to the model
            self.Y_values = self.result_storage.table['goal'].values.tolist() # Get the Y values which is the target (output) of the model

        elif self.settings['main']['model_type'] == 'LR': # Linear Regression
            init_runs = 0
            if self.settings['main']['suggested_parameter_sets']['Enabled'] == True:
                List_of_suggested_parameter_sets = [tuple(item) for item in self.settings['main']['suggested_parameter_sets']['List']]
                for suggested_param in List_of_suggested_parameter_sets:
                    suggested_param_dict = dict(zip(self.base_params, suggested_param))
                    print(f'Main: Initial test run {init_runs + 1} with suggested parameter set {suggested_param_dict}')
                    current_result = self.test_environment.execute_test(suggested_param_dict)
                    self.result_storage_LR.add_single_test(current_result, used=True)
                    init_runs += 1
                self.result_storage_LR.update_fits()
                self.settings['main']['totalInitTests'] = self.settings['main']['totalInitTests'] - init_runs
            for init_test_run_nr in range(self.settings['main']['totalInitTests']): # For each initial test run
                parameters_to_draw_from = copy.deepcopy(self.settings['main']['listOfParamDicts']) # Create a deep copy of the list of parameter dictionaries
                information_gain = False
                random_params = None
                while not information_gain: # While there is no information gain
                    if len(parameters_to_draw_from) > 0: # If the length of the list of parameter dictionaries is greater than 0
                        random_index = random.randrange(len(parameters_to_draw_from))   # Randomly select an index from the list of parameter dictionaries
                        random_params = parameters_to_draw_from.pop(random_index) # Remove the randomly selected parameter dictionary from the list of parameter dictionaries
                        # test if adding this parameter set would add information to the fit or if the
                        # added information would be redundant (linear independence):
                        if init_test_run_nr < self.settings['main']['fitFreedomDegrees']:
                            information_gain = self.result_storage_LR.check_for_information_gain(random_params)
                        else:
                            information_gain = True  # don't care anymore, system is overdetermined anyways!
                    else:
                        print(f'Main: Warning, the parameter space is too small to provide enough information '
                              f'for the specified fit function. Your fits might make little sense this way.')
                        random_params = Utilities.pick_random_params(self.settings) # Pick random parameters
                        information_gain = True # Set information gain to True
                print(f'Main: Initial test run {init_runs+init_test_run_nr + 1} with random parameter set {random_params}')
                current_result = self.test_environment.execute_test(random_params) # Execute the test with the selected random parameter
                self.result_storage_LR.add_single_test(current_result, used=True) # Add the test result to the result storage
            self.result_storage_LR.update_fits() # Update the LR fits

    def main_loop(self):
        if self.settings['main']['model_type'] == 'GP': # Gaussian Process
            n_itr = self.total_runs - self.n_init_runs # Number of iterations
            grouped_results_itr = self.result_storage.current_constained_table.groupby(self.anon_var_list) # Group the results by the anonymous variable list
            result_dict_itr = {key: group['goal'].tolist() for key, group in grouped_results_itr} # Create a dictionary of the results
            if len(result_dict_itr) == 0: # If the length of the dictionary of results is 0
                best_goal_so_far = float('inf') # Assign infinity to the best goal so far
            else:
                best_goal_so_far = min(value[0] for value in result_dict_itr.values()) # Get the minimum value of the goal from the dictionary of results
            for i in range(n_itr): # For each iteration
                terminated_with_constraints_conf = False # just to keep track of the termination criteria
                current_thresh_conf, param_with_highest_conf, highest_conf = Utilities.calculated_threshold_confidence(
                    self.settings, grouped_results_itr, self.best_params_itr)  # Calculate the threshold confidence
                if self.settings['main']['termination_criteria']['Confidence_in_satisfying_constraints'] != 'None':
                    if highest_conf >= self.settings['main']['termination_criteria'][
                        'Confidence_in_satisfying_constraints']:  # If the threshold confidence is greater than or equal to the confidence in satisfying constraints
                        current_best_parameter_set = dict(zip(self.anon_var_list, param_with_highest_conf))
                        current_thresh_conf = highest_conf
                        terminated_with_constraints_conf = True
                        break
                X_init_scaled = self.scaler.fit_transform(self.X_values) # Fit the scaler and transform the X values
                fitted_model = fit_model(X_init_scaled, self.Y_values, self.model_type) # Fit the model
                x_next = self.next_params_algorithm(fitted_model, self.scaler, self.result_storage, i, best_goal_so_far, self.cumulative_worst_regret, self.max_runs) # Get the next parameter
                print(f'Iteration: {i + 1 + self.n_init_runs}')
                y_next_results = self.test_environment.execute_test(x_next) # Execute the test with the next parameter
                if type(y_next_results) is dict and y_next_results is not None: # If the type of the results is a dictionary and the results are not None
                    y_next_results = pd.Series(y_next_results) # Convert the results to a pandas series
                x_next_list = y_next_results[self.anon_var_list].values.tolist() # Get the anonymous variable list values
                self.result_storage.add_single_test(y_next_results, used=True) # Add the test results to the result storage
                y_next = y_next_results['goal'] # Get the goal value from the test results as the target
                self.X_values.append(x_next_list) # Append the next parameter to the X values
                self.Y_values.append(y_next) # Append the goal value to the Y values
                self.result_storage.update_fits() # Update the fits
                grouped_results_itr = self.result_storage.current_constained_table.groupby(self.anon_var_list) # Group the results by the anonymous variable list
                result_dict_itr = {key: group['goal'].tolist() for key, group in grouped_results_itr} # Create a dictionary of the results
                n_itr_t = i + self.n_init_runs # Get the iteration number
                current_optimality_conf = Utilities.calculate_optimality_confidence(self.cumulative_worst_regret) # Calculate the optimality confidence
                if self.settings['main']['termination_criteria']['Confidence_in_optimality'] != 'None': # If the confidence in optimality is not None
                    if current_optimality_conf >= self.settings['main']['termination_criteria']['Confidence_in_optimality']:
                        break
                if bool(result_dict_itr): # If the dictionary of results is not empty
                    max_value_length = max(len(value) for value in result_dict_itr.values()) # Get the maximum length of the values
                    if max_value_length < self.min_test_needed: # If the maximum length of the values is less than the minimum test needed
                        max_value_length = max_value_length # Set the maximum value length to the maximum value length
                    else: # Otherwise
                        max_value_length = self.min_test_needed  # Set the maximum value length to the minimum test needed
                    # Keep only the dictionaries with values of the maximum length
                    filtered_dict_max_length = {key: value for key, value in result_dict_itr.items() if
                                                len(value) >= max_value_length}
                    median_results_itr_max_length = {key: statistics.median(val) for key, val in
                                                     filtered_dict_max_length.items()}
                    try:
                        # Check if it's a single dictionary or a list of dictionaries
                        if isinstance(median_results_itr_max_length, dict):
                            self.best_goal_itr[n_itr_t] = min(median_results_itr_max_length.values())
                        else:
                            # It's a list of dictionaries
                            values_to_check = [value for values in median_results_itr_max_length for value in values]
                            self.best_goal_itr[n_itr_t] = min(values_to_check) # Get the minimum value of the values to check
                        self.best_params_itr[n_itr_t] = min(median_results_itr_max_length, key=median_results_itr_max_length.get) # Get the parameter set resulting in the minimum value
                    except ValueError: # If there is a value error (i.e. If the threshold is not met for any of the parameters)
                        self.best_goal_itr[n_itr_t] = None # Assign None to the best goal so far
                        self.best_params_itr[n_itr_t] = None # Assign None to the best parameter set so far
                else: # If the dictionary of results is empty (i.e. If the threshold is not met for any of the parameters)
                    self.best_goal_itr[n_itr_t] = None  # Assign None to the best goal so far
                    self.best_params_itr[n_itr_t] = None # Assign None to the best parameter set so far
                if len(result_dict_itr) == 0: # If the length of the dictionary of results is 0
                    # assign infinity to the best goal so far
                    best_goal_so_far = float('inf') # Assign infinity to the best goal so far
                else:   # Otherwise
                    best_goal_so_far = min(value[0] for value in result_dict_itr.values()) # Get the minimum value of the goal from the dictionary of results
                current_best_parameter_set = self.best_params_itr[n_itr_t]
            if terminated_with_constraints_conf:
                print(f'Main: Optimal parameter set: {current_best_parameter_set}')
                print(f'Main: Confidence in satisfying constraints: {current_thresh_conf}')
            else:
                print(f'Main: Optimal parameter set: {current_best_parameter_set}')
                print(f'Main: Confidence in satisfying constraints: {current_thresh_conf}')
                print(f'Main: Confidence in optimality: {current_optimality_conf}')

        elif self.settings['main']['model_type'] == 'LR': # Linear Regression
            if not self.next_params_algorithm.__name__ == "GER": # If the next parameter algorithm is not GER
                n_itr = self.total_runs - self.n_init_runs # Number of iterations
                for i in range(n_itr):
                    actual_itr = i + self.n_init_runs
                    x_next = self.next_params_algorithm(result_storage=self.result_storage_LR) # Get the next parameter as per the selected NTS algorithm
                    print(f'Iteration: {i+ 1 + self.n_init_runs}')
                    y_next_results = self.test_environment.execute_test(x_next) # Execute the test with the next parameter
                    self.result_storage_LR.add_single_test(y_next_results, used=True) # Add the test results to the result storage
                    self.result_storage_LR.update_fits() # Update the fits
                    grouped_results_itr = self.result_storage_LR.current_constained_table.groupby(self.anon_var_list) # Group the results by the anonymous variable list
                    result_dict_itr = {key: group['goal'].tolist() for key, group in grouped_results_itr} # Create a dictionary of the results
                    if bool(result_dict_itr): # If the dictionary of results is not empty
                        max_value_length = max(len(value) for value in result_dict_itr.values()) # Get the maximum length of the values
                        if max_value_length < self.min_test_needed: # If the maximum length of the values is less than the minimum test needed
                            max_value_length = max_value_length # Set the maximum value length to the maximum value length
                        else: # Otherwise
                            max_value_length = self.min_test_needed # Set the maximum value length to the minimum test needed
                        # Keep only the dictionaries with values of the maximum length
                        filtered_dict_max_length = {key: value for key, value in result_dict_itr.items() if
                                                    len(value) >= max_value_length}
                        median_results_itr_max_length = {key: statistics.median(val) for key, val in
                                                         filtered_dict_max_length.items()}
                        try:
                            if isinstance(median_results_itr_max_length, dict): # If the median results is a dictionary
                                self.best_goal_itr[actual_itr] = min(median_results_itr_max_length.values()) # Get the minimum value of the values
                            else:
                                # It's a list of dictionaries
                                values_to_check = [value for values in median_results_itr_max_length for value in values] # Get the values to check
                                self.best_goal_itr[n_itr] = min(values_to_check) # Get the minimum value of the values to check
                            self.best_params_itr[actual_itr] = min(median_results_itr_max_length,
                                                                key=median_results_itr_max_length.get) # Get the parameter set resulting in the minimum value
                        except ValueError:  # If there is a value error (i.e. If the threshold is not met for any of the parameters)
                            self.best_goal_itr[actual_itr] = None  # Assign None to the best goal so far
                    else: # If the dictionary of results is empty (i.e. If the threshold is not met for any of the parameters)
                        self.best_goal_itr[actual_itr] = None # Assign None to the best goal so far
                if actual_itr in self.best_params_itr:
                    print(f'Main: Optimal parameter set: {self.best_params_itr[actual_itr]}')
                else:
                    print('None of the parameter sets met the threshold as per the given requirements and criteria')

            else:
                n_itr = self.total_runs - self.n_init_runs
                i = 0
                while i < n_itr:
                    list_of_param_dicts = self.settings['main']['listOfParamDicts'] # Get the list of parameter dictionaries
                    random.shuffle(list_of_param_dicts)
                    for param_dict in list_of_param_dicts:
                        # break if we have reached the maximum number of iterations
                        if i >= n_itr:
                            break
                        x_next = param_dict
                        print(f'Iteration: {i + 1 + self.n_init_runs}')
                        y_next_results = self.test_environment.execute_test(x_next)
                        if type(y_next_results) is dict:
                            y_next_results = pd.Series(y_next_results)
                        self.result_storage_LR.add_single_test(y_next_results, used=True)
                        self.result_storage_LR.update_fits()
                        n_itr_t = i + self.n_init_runs
                        grouped_results_itr = self.result_storage_LR.current_constained_table.groupby(self.anon_var_list)
                        result_dict_itr = {key: group['goal'].tolist() for key, group in grouped_results_itr}
                        median_results_itr = {key: statistics.median(val) for key, val in result_dict_itr.items()}
                        if bool(result_dict_itr):
                            max_value_length = max(len(value) for value in result_dict_itr.values())
                            if max_value_length < self.min_test_needed:
                                max_value_length = max_value_length
                            else:
                                max_value_length = self.min_test_needed
                            # Keep only the dictionaries with values of the maximum length
                            filtered_dict_max_length = {key: value for key, value in result_dict_itr.items() if len(value) >= max_value_length}
                            median_results_itr_max_length = {key: statistics.median(val) for key, val in filtered_dict_max_length.items()}
                            try:
                                # Check if it's a single dictionary or a list of dictionaries
                                if isinstance(median_results_itr_max_length, dict):
                                    self.best_goal_itr[n_itr_t] = min(median_results_itr_max_length.values())
                                else:
                                    # It's a list of dictionaries
                                    values_to_check = [value for values in median_results_itr_max_length for value in values]
                                    self.best_goal_itr[n_itr_t] = min(values_to_check)
                                self.best_params_itr[n_itr_t] = min(median_results_itr_max_length, key=median_results_itr_max_length.get)
                            except ValueError:
                                self.best_goal_itr[n_itr_t] = None
                        else:
                            self.best_goal_itr[n_itr_t] = None
                            try:
                                self.best_params_itr[n_itr_t] = min(median_results_itr, key=median_results_itr.get)
                            except ValueError:
                                self.best_params_itr[n_itr_t] = None
                        i += 1
                print(f'Main: Optimal parameter set: {self.best_params_itr[n_itr_t]}')

        elif self.settings['main']['model_type'] == 'RL-Any': # Reinforcement Learning with action include moving to any parameter set (state)
            n_itr = self.total_runs # Number of iterations
            parameter_values_list = self.settings['main']['parameter_values'] # Get the parameter values
            # Extract parameter names and their respective possible values
            param_names = list(parameter_values_list.keys())
            param_values = list(parameter_values_list.values())
            # Initialize a Q-table with dimensions based on the number of parameters and their possible values
            q_table_shape = tuple(len(values) for values in param_values)
            q_table = np.zeros(q_table_shape + q_table_shape)  # Add shape for state-action pairs
            # Learning parameters
            learning_rate = 0.1  # Alpha
            discount_factor = 0.9  # Gamma
            epsilon = 0.05  # Exploration-exploitation balance
            # Choose a random initial state (random parameter combination)
            current_state = tuple(random.randint(0, len(param_values[i]) - 1) for i in range(len(param_names))) # Random initial state
            i = 0
            while i < n_itr:
                print(f'Main: Starting iteration {i}', flush=True)
                grouped_results_itr = self.result_storage_LR.current_constained_table.groupby(
                    self.anon_var_list)  # Group the current results
                result_dict = {key: group['goal'].tolist() for key, group in grouped_results_itr}  # Create a dictionary of the results
                filtered_dict = {key: value for key, value in result_dict.items() if len(value) >= self.max_runs}  # exhaustively tested parameter sets
                exhausted_params = list(filtered_dict.keys())  # Get the exhausted parameter sets
                # Exploration-exploitation trade-off
                if random.uniform(0, 1) < epsilon:
                    # Explore: randomly choose a new action (next state)
                    action = tuple(random.randint(0, len(param_values[i]) - 1) for i in range(len(param_names)))
                else:
                    # Exploit: choose the action (next state) with the highest Q-value from the current state
                    action = np.unravel_index(np.argmax(q_table[current_state]), q_table[current_state].shape)
                # Convert the tuple of indices (action) into real parameter values
                def map_action_to_values(action, parameters):
                    # Zip the parameter names with their corresponding index in the action tuple
                    return {param: parameters[param][index] for param, index in zip(parameters.keys(), action)}
                # Convert the action tuple (2, 1) into real values
                real_values = map_action_to_values(action, parameter_values_list)
                if tuple(real_values.values()) in exhausted_params:
                    # add -inf to the Q-value to avoid choosing this action again
                    q_table[current_state][action] += -np.inf
                    continue # Skip the rest of the loop and start the next iteration
                # Get the results based on the action taken (i.e., the new parameter configuration)
                current_result = self.test_environment.execute_test(real_values)
                # update the result storage
                self.result_storage_LR.add_single_test(current_result, used=True)
                grouped_results_itr_2 = self.result_storage_LR.current_constained_table.groupby(self.anon_var_list)
                result_dict_itr_2 = {key: group['goal'].tolist() for key, group in grouped_results_itr_2}
                median_results_itr = {key: statistics.median(val) for key, val in result_dict_itr_2.items()}
                # Get the reward based on the action taken (i.e., the new parameter configuration)
                reward = Utilities.reward_calculator(self.settings, self.result_storage_LR, current_result)
                # Get the current Q-value for the state-action pair (current_state, action)
                current_q_value = q_table[current_state][action]
                # Find the best future Q-value from the next state's best possible action
                best_future_q = np.max(q_table[action])  # The best Q-value from the next state
                # Update the Q-value for the state-action pair
                q_table[current_state][action] += learning_rate * (
                        reward + discount_factor * best_future_q - current_q_value)
                # Move to the next state (the action becomes the next state)
                current_state = action
                if bool(result_dict_itr_2): # If the dictionary of results is not empty
                    max_value_length = max(len(value) for value in result_dict_itr_2.values()) # Get the maximum length of the values
                    if max_value_length < self.min_test_needed: # If the maximum length of the values is less than the minimum test needed
                        max_value_length = max_value_length # Set the maximum value length to the maximum value length
                    else:
                        max_value_length = self.min_test_needed # Set the maximum value length to the minimum test needed
                    if bool(self.best_params_itr): # If the best parameter set is not empty
                        current_best_param = self.best_params_itr[i - 1] # Get the current best parameter
                        # number of results for the current best parameter
                        if current_best_param is not None:
                            current_best_param_results = len(result_dict_itr_2.get(current_best_param, [])) # Get the number of results for the current best parameter
                            max_value_length = current_best_param_results # Set the maximum value length to the current best parameter results
                    # Keep only the dictionaries with values of the maximum length
                    filtered_dict_max_length = {key: value for key, value in result_dict_itr_2.items() if
                                                len(value) >= max_value_length}
                    median_results_itr_max_length = {key: statistics.median(val) for key, val in
                                                     filtered_dict_max_length.items()}
                    try:
                        # Check if it's a single dictionary or a list of dictionaries
                        if isinstance(median_results_itr_max_length, dict):
                            self.best_goal_itr[i] = min(median_results_itr_max_length.values())
                        else:
                            # It's a list of dictionaries
                            values_to_check = [value for values in median_results_itr_max_length for value in values]
                            self.best_goal_itr[i] = min(values_to_check)
                        self.best_params_itr[i] = min(median_results_itr_max_length,
                                                      key=median_results_itr_max_length.get)
                    except ValueError: # If there is a value error (i.e. If the threshold is not met for any of the parameters)
                        self.best_goal_itr[i] = None
                else: # If the dictionary of results is empty (i.e. If the threshold is not met for any of the parameters)
                    self.best_goal_itr[i] = None
                    try:
                        self.best_params_itr[i] = min(median_results_itr, key=median_results_itr.get)
                    except ValueError: # If there is a value error (i.e. If the threshold is not met for any of the parameters)
                        self.best_params_itr[i] = None
                i += 1
            print(f'Main: Optimal parameter set: {self.best_params_itr[i-1]}')

        elif self.settings['main']['model_type'] == 'RL-Step': # Reinforcement Learning with action include only moving one step in a given direction for only one parameter at a time
            def get_neighboring_actions(state):
                actions = []
                for i, index in enumerate(state):
                    # Increment action for parameter i
                    if index < len(param_values[i]) - 1:
                        neighbor = list(state)
                        neighbor[i] += 1
                        actions.append((tuple(neighbor), i * 2))  # i * 2 for "increment" action
                    # Decrement action for parameter i
                    if index > 0:
                        neighbor = list(state)
                        neighbor[i] -= 1
                        actions.append((tuple(neighbor), i * 2 + 1))  # i * 2 + 1 for "decrement" action
                return actions
            n_itr = self.total_runs # Number of iterations
            parameter_values_list = self.settings['main']['parameter_values']
            # Extract parameter names and their respective possible values
            param_values = list(parameter_values_list.values())
            # Initialize a Q-table with dimensions based on the number of parameters and their possible values
            q_table_shape = tuple(len(values) for values in param_values)  # Shape for all possible states
            num_actions = 2 * len(param_values)  # Each parameter has two actions (increment, decrement)
            # Initialize Q-table with the new shape (state space + action space)
            q_table = np.zeros(q_table_shape + (num_actions,))
            # Learning parameters
            learning_rate = 0.1  # Alpha
            discount_factor = 0.9  # Gamma
            epsilon = 0.05  # Exploration-exploitation balance
            # Choose a random initial state (random parameter combination)
            current_state = tuple(random.randint(0, len(param_values[i]) - 1) for i in range(len(param_values))) # Random initial state
            i = 0 # Initialize the iteration counter
            loop_counter = 0 # used to detect if the algorithm is stuck in a loop
            while i < n_itr:
                grouped_results_itr = self.result_storage_LR.current_constained_table.groupby(
                    self.anon_var_list)  # Group the current results
                result_dict = {key: group['goal'].tolist() for key, group in
                               grouped_results_itr}  # Create a dictionary of the results
                filtered_dict = {key: value for key, value in result_dict.items() if
                                 len(value) >= self.max_runs}  # exhaustively tested parameter sets
                neighbors = get_neighboring_actions(current_state)
                print(f'Main: Starting iteration {i}', flush=True)
                # Exploration vs. exploitation
                if random.uniform(0, 1) < epsilon:
                    # Explore: choose a random neighboring action
                    action, action_index = random.choice(neighbors)
                else:
                    # Exploit: choose the action with the highest Q-value
                    action, action_index = max(neighbors, key=lambda x: q_table[current_state][x[1]])
                # Convert the tuple of indices (action) into real parameter values
                def map_action_to_values(action, parameters):
                    # Zip the parameter names with their corresponding index in the action tuple
                    return {param: parameters[param][index] for param, index in zip(parameters.keys(), action)}

                # Convert the action tuple (2, 1) into real values
                real_values = map_action_to_values(action, parameter_values_list)

                if tuple(real_values.values()) in filtered_dict: # if the parameter set has been exhaustively tested
                    if loop_counter > num_actions: # if the algorithm is stuck in a loop (i.e., all actions have been tried)
                        # if the algorithm is stuck in a loop, change the state to random
                        current_state = tuple(random.randint(0, len(param_values[i]) - 1) for i in range(len(param_values))) # Random initial state
                    # add -inf to the Q-value
                    q_table[current_state][action_index] += -np.inf # Avoid choosing this action again
                    loop_counter += 1 # Increment the loop counter
                    continue
                # Get the results based on the action taken (i.e., the new parameter configuration)
                current_result = self.test_environment.execute_test(real_values)
                # update the result storage
                self.result_storage_LR.add_single_test(current_result, used=True)
                grouped_results_itr_2 = self.result_storage_LR.current_constained_table.groupby(self.anon_var_list)
                result_dict_itr_2 = {key: group['goal'].tolist() for key, group in grouped_results_itr_2}
                median_results_itr = {key: statistics.median(val) for key, val in result_dict_itr_2.items()}
                # Get the reward based on the action taken (i.e., the new parameter configuration)
                reward = Utilities.reward_calculator(self.settings, self.result_storage_LR, current_result)
                # Get the current Q-value for the state-action pair
                current_q_value = q_table[current_state][action_index]
                # Calculate the best future Q-value for the next state
                best_future_q = max(q_table[action])
                # Q-learning update
                q_table[current_state][action_index] += learning_rate * (
                        reward + discount_factor * best_future_q - current_q_value)
                # Move to the next state (the action becomes the next state)
                current_state = action
                if bool(result_dict_itr_2):
                    max_value_length = max(len(value) for value in result_dict_itr_2.values())
                    if max_value_length < self.min_test_needed:
                        max_value_length = max_value_length
                    else:
                        max_value_length = self.min_test_needed
                    # Keep only the dictionaries with values of the maximum length
                    filtered_dict_max_length = {key: value for key, value in result_dict_itr_2.items() if
                                                len(value) >= max_value_length}
                    median_results_itr_max_length = {key: statistics.median(val) for key, val in
                                                     filtered_dict_max_length.items()}
                    try:
                        # Check if it's a single dictionary or a list of dictionaries
                        if isinstance(median_results_itr_max_length, dict):
                            self.best_goal_itr[i] = min(median_results_itr_max_length.values())
                        else:
                            # It's a list of dictionaries
                            values_to_check = [value for values in median_results_itr_max_length for value in values]
                            self.best_goal_itr[i] = min(values_to_check)
                        self.best_params_itr[i] = min(median_results_itr_max_length,
                                                      key=median_results_itr_max_length.get)
                    except ValueError:
                        self.best_goal_itr[i] = None
                else:
                    self.best_goal_itr[i] = None
                    try:
                        self.best_params_itr[i] = min(median_results_itr, key=median_results_itr.get)
                    except ValueError:
                        self.best_params_itr[i] = None
                i += 1
                loop_counter = 0
            print(f'Main: Optimal parameter set: {self.best_params_itr[i-1]}')

if __name__ == '__main__':
    main = Main(SETTINGS_FILE)
    main.perform_initial_tests()
    main.main_loop()
