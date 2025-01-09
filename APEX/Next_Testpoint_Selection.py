import numpy as np
import statistics
from scipy.stats import norm
import Utilities
import warnings
from sklearn.exceptions import ConvergenceWarning
import random


class Next_Testpoint_Selection:

    def __init__(self, settings_ref):
        self._settings = settings_ref # Reference to the settings dictionary
        anon_name = self._settings['main']['anonymousNames'] # Get the anonymous names of the parameters
        self.anon_var_list = list(anon_name.values()) # List of the anonymous names
        self.sub_opptimality_vector =  []  # Initialize the suboptimality vector that keep track of the suboptimality of the best goal so far used to calculate the confidence metric
        self.max_min_value_dict = {} # Initialize the dictionary that keeps track of the maximum and minimum values of the results
        self.th_sc_param = {} # Initialize the dictionary that keeps track of the threshold score of the parameters
        self.n_init_runs = self._settings['main']['totalInitTests'] # Get the number of initial runs
        self.Param_ranges = self._settings['main']['parameterRanges'] # Get the parameter ranges
        self.mean_results_itr_2 = {} # Initialize the dictionary that keeps track of the mean results from GP LCB
        self.base_params = list(self._settings['main']['parameterNames'].keys())

    def EI(self, model, scaler, result_storage,i,best_goal_so_far,cumulative_worst_regret, max_run): # EI - Expected Improvment
        grouped_results = result_storage.current_constained_table.groupby(self.anon_var_list) # Group the results that satisfy the constraints
        result_dict_itr = {key: group['goal'].tolist() for key, group in grouped_results} # Create a dictionary with the results
        mean_results_itr = {key: statistics.mean(val) for key, val in result_dict_itr.items()} # Calculate the mean of the results
        D = len(self._settings['main']['listOfParamDicts']) # Get the number of parameter combinations
        app_goal = self._settings['main']['applicationGoal'] # Get the application goal
        delta = 0.1 # strictness of the asymptotic regret bound.
        ei_v = {} # Initialize the expected improvement vector
        th_sc = {} # Initialize the threshold score vector
        i_actual = i + self.n_init_runs # Calculate the actual iteration number
        kappa = 2 * np.log(D * i_actual ** 2 * np.pi ** 2 / (6 * delta)) # Calibration parameter for UCB
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        list_of_param_dicts = self._settings['main']['listOfParamDicts'] # Get the list of parameter combinations
        ei_max = -np.inf
        LCB_max = -np.inf
        self.improvement = {}
        for param_dict in list_of_param_dicts: # Loop through the parameter combinations
            if not result_storage.thresholds_satisfied(self._settings, param_dict): # Check if the thresholds are satisfied
                continue # Skip the current iteration if the thresholds are not satisfied
            combination_scaled = scaler.transform(np.array(list(param_dict.values())).reshape(1, -1)) # Scale the parameter combination
            ei, imp = self.acquisition_EI(combination_scaled, model,result_storage) # Calculate the expected improvement and improvement
            LCB, CV = self.acquisition_UCB(combination_scaled, model, kappa, param_dict) # Calculate the UCB and CV
            self.improvement[tuple(float(value) for value in param_dict.values())] = imp # Store the improvement in the dictionary
            ei_v[tuple(param_dict.values())] = ei # Store the expected improvement in the dictionary
            if ei > ei_max: # Check if the expected improvement is greater than the maximum expected improvement
                x_next = param_dict # Set the next parameter combination to the current parameter combination
                ei_max = ei # Update the maximum expected improvement
            if LCB > LCB_max: # Check if the UCB is greater than the maximum UCB
                LCB_max = LCB # Update the maximum UCB
        common_keys = set(mean_results_itr.keys()) & set(self.mean_results_itr_2.keys()) # Get the common keys between the mean results and the mean results from GP LCB
        mean_results_itr_2_filtered = {key: self.mean_results_itr_2[key] for key in common_keys} # Filter the mean results from GP LCB based on the common keys
        if len(mean_results_itr_2_filtered) > 0: # Check if the length of the filtered mean results from GP LCB is greater than 0
            if LCB_max == -np.inf or LCB_max == +np.inf: # Check if the maximum UCB is negative infinity or positive infinity
                if np.max(self.sub_opptimality_vector) == 0: # Check if the maximum suboptimality is 0
                    worst_regeret = 1 # Set the worst regret to 1
                else:
                    worst_regeret = np.max(self.sub_opptimality_vector) # Set the worst regret to the maximum suboptimality
            else:
                worst_regeret = np.min(list(mean_results_itr_2_filtered.values())) - (-LCB_max) # Calculate the worst regret based on the minimum of the mean results from GP LCB and the maximum UCB
        else: # If the length of the filtered mean results from GP LCB is 0
            if len(self.sub_opptimality_vector) == 0: # Check if the length of the suboptimality vector is 0
                worst_regeret = 1 # Set the worst regret to 1
            else:  # If the length of the suboptimality vector is not 0
                if np.max(self.sub_opptimality_vector) == 0: # Check if the maximum suboptimality is 0
                    worst_regeret = 1 # Set the worst regret to 1
                else:
                    worst_regeret = np.max(self.sub_opptimality_vector) # Set the worst regret to the maximum suboptimality
        # Check if worst_regeret is an array
        if isinstance(worst_regeret, (np.ndarray, list)):
            worst_regeret = float(worst_regeret[0])  # Convert the first element to a float

        # Append the normalized value to sub_opptimality_vector
        self.sub_opptimality_vector.append(worst_regeret) # Append the worst regret to the suboptimality vector
        if cumulative_worst_regret == {}:  # If the dictionary is empty
            cumulative_worst_regret[i_actual - 1] = worst_regeret # Set the cumulative worst regret to the worst regret as the first element
        else: # If the dictionary is not empty
            cumulative_worst_regret[i_actual - 1] = cumulative_worst_regret[i_actual - 2] + worst_regeret  # Update the cumulative worst regret
        grouped_results_itr_c = result_storage.table.groupby(self.anon_var_list) # Group the current results
        result_dict_itr_c = {key: group['goal'].tolist() for key, group in grouped_results_itr_c}  # Create a dictionary with the current results
        max_value_result = max(result_dict_itr_c.values(), key=lambda x: x[0])[0] # Get the maximum value of the results
        min_value_result = min(result_dict_itr_c.values(), key=lambda x: x[0])[0] # Get the minimum value of the results
        self.max_min_value_dict[i_actual] = {'max': max_value_result, 'min': min_value_result} # Store the maximum and minimum values of the results
        result_list = [] # Initialize the result list
        for key in result_dict_itr_c.keys(): # Loop through the keys of the current results
            result_dict = {self.anon_var_list[i]: key[i] for i in range(len(self.anon_var_list))} # Create a dictionary with the anonymous variable list
            result_list.append(result_dict) # Append the result dictionary to the result list
        try:
            # Check if the variable is available
            if x_next:
                x_next = x_next # nothing to do
            else:
                x_next = Utilities.pick_random_params(self._settings) # Pick a random parameter combination
        except NameError: # If the variable is not available due to an empty dictionary or list
            x_next = Utilities.pick_random_params(self._settings) # Pick a random parameter combination
        if ei_max < 1e-04: # Check if the maximum expected improvement is less than the threshold suggesting likely to stuck in local minima
            choice_count = i % 2 # Calculate the choice count which decides how to select the next parameter combination
            filtered_dict = {key: value for key, value in result_dict_itr.items() if len(value) < max_run} # Filter the results based on the maximum number of runs
            median_results_itr = {key: statistics.median(val) for key, val in filtered_dict.items()} # Calculate the median of the filtered results
            filtered_improvment = {key: self.improvement[key] for key in self.improvement if key in median_results_itr} # Filter the improvement based on the median results
            if filtered_improvment: # Check if the filtered improvement is not empty
                max_value_imp = max(filtered_improvment.values()) # Get the maximum improvement
                param_with_max_imp = [key for key, value in filtered_improvment.items() if value == max_value_imp] # Get the parameter combination with the maximum improvement
                param_with_max_imp_tuple = param_with_max_imp[0] # Get the parameter combination with the maximum improvement
                x_next = dict(zip(self.anon_var_list, param_with_max_imp_tuple)) # Set the next parameter combination to the parameter combination with the maximum improvement
            else:
                x_next = Utilities.pick_random_params(self._settings) # Pick a random parameter combination
            if choice_count == 0 and 'thresholds' in app_goal and app_goal['thresholds'] is not None: # Check if the choice count is 0 and the thresholds are not None
                for param_dict in list_of_param_dicts: # Loop through the parameter combinations
                    th_sc[tuple(param_dict.values())] = result_storage.threshold_score(param_dict) # Calculate the threshold score
                if all(value is None for value in th_sc.values()): # Check if all the threshold scores are None
                    x_next = Utilities.pick_random_params(self._settings) # Pick a random parameter combination
                else: # If all the threshold scores are not None
                    filtered_dict_all = {key: value for key, value in result_dict_itr_c.items() if len(value) < max_run} # Filter the results based on the maximum number of runs
                    filtered_th_sc = {key: th_sc[key] for key in th_sc if key in filtered_dict_all} # Filter the threshold scores based on the filtered results
                    if len(filtered_th_sc) == 0: # Check if the length of the filtered threshold scores is 0
                        x_next = Utilities.pick_random_params(self._settings) # Pick a random parameter combination
                    else:   # If the length of the filtered threshold scores is not 0
                        max_th_sc = max(filtered_th_sc.items(), key=lambda item: item[1])[0]  # Get the parameter combination with the maximum threshold score
                        if len(self.th_sc_param) == 0: # Check if the length of the threshold score parameter is 0
                            self.th_sc_param[i_actual] = max_th_sc # Set the threshold score parameter to the parameter combination with the maximum threshold score
                        else: # If the length of the threshold score parameter is not 0
                            grouped_results = result_storage.table.groupby(self.base_params) # Group the results based on the base parameters
                            result_dict_all = {key: group['goal'].tolist() for key, group in grouped_results} # Create a dictionary with the results
                            mean_gl = {key: sum(values) / len(values) for key, values in result_dict_all.items()} # Calculate the mean of the results
                            # Create a set of keys from filtered_th_sc for faster lookup
                            filtered_keys = set(filtered_th_sc.keys())
                            # Filter th_sc_param based on the values not in filtered_th_sc keys
                            self.th_sc_param = {key: value for key, value in self.th_sc_param.items() if value in filtered_keys}
                            for th_param in self.th_sc_param.values(): # Loop through the threshold score parameter values
                                penalty = (mean_gl[th_param] - best_goal_so_far) / best_goal_so_far # Calculate the penalty
                                if 'reliability' in app_goal['optimizationTargets']: # Check if reliability is in the optimization targets
                                    filtered_th_sc[th_param] = filtered_th_sc[th_param] + penalty # Update the filtered threshold score by adding the penalty
                                else:
                                    filtered_th_sc[th_param] = filtered_th_sc[th_param] - penalty # Update the filtered threshold score by subtracting the penalty
                            max_th_sc = max(filtered_th_sc.items(), key=lambda item: item[1])[0] # Get the parameter combination with the maximum threshold score
                            if max_th_sc not in self.th_sc_param.values(): # Check if the parameter combination with the maximum threshold score is not in the threshold score parameter values
                                self.th_sc_param[i_actual] = max_th_sc # Set the threshold score parameter to the parameter combination with the maximum threshold score
                        if max_th_sc[1] == 0: # Check if the threshold score is 0
                            x_next = Utilities.pick_random_params(self._settings) # Pick a random parameter combination
                        else: # If the threshold score is not 0
                            x_next = dict(zip(self.anon_var_list, max_th_sc)) # Set the next parameter combination to the parameter combination with the maximum threshold score
        return x_next

    def GP_LCB(self, model, scaler, result_storage, i, best_goal_so_far, cumulative_worst_regret, max_run): # GP-LCB - Gaussian Process Lower Confidence Bound
        grouped_results_itr = result_storage.current_constained_table.groupby(self.anon_var_list) # Group the current results
        result_dict_itr = {key: group['goal'].tolist() for key, group in grouped_results_itr} # Create a dictionary with the current results
        mean_results_itr = {key: statistics.mean(val) for key, val in result_dict_itr.items()} # Calculate the mean of the current results
        D = len(self._settings['main']['listOfParamDicts']) # Get the number of parameter combinations
        delta = 0.1  # strictness of the asymptotic regret bound.
        th_sc = {} # Initialize the threshold score vector
        x_next = None # Initialize the next parameter combination
        LCB_vector = {} # Initialize the UCB vector
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        i_actual = i + self.n_init_runs # Calculate the actual iteration number
        beta_t = 2 * np.log(D * i_actual ** 2 * np.pi ** 2 / (6 * delta))   # Calibration parameter for LCB
        list_of_param_dicts = self._settings['main']['listOfParamDicts'] # Get the list of parameter combinations
        LCB_max = -np.inf # Initialize the maximum UCB
        CV_max = np.inf   # Initialize the maximum CV
        for param_dict in list_of_param_dicts: # Loop through the parameter combinations
            param_list = list(param_dict.values()) # Get the parameter list
            if not result_storage.thresholds_satisfied(self._settings, param_dict):
                continue # Skip the current iteration if the thresholds are not satisfied
            combination_scaled = scaler.transform(np.array(list(param_dict.values())).reshape(1, -1)) # Scale the parameter combination
            LCB, CV = self.acquisition_UCB(combination_scaled, model, beta_t, param_dict) # Calculate the LCB and CV
            LCB_vector[tuple(param_list)] = LCB # Store the LCB in the LCB vector
            if LCB > LCB_max: # Check if the LCB is greater than the maximum LCB
                x_next = param_dict # Set the next parameter combination to the current parameter combination
                LCB_max = LCB # Update the maximum LCB
                CV_max = CV # Update the maximum CV
        if len(mean_results_itr) > 0: # Check if the length of the mean results is greater than 0
            if LCB_max == -np.inf or LCB_max == +np.inf: # Check if the maximum LCB is negative infinity or positive infinity
                if np.max(self.sub_opptimality_vector) == 0: # Check if the maximum suboptimality is 0
                    worst_regeret = 1 # Set the worst regret to 1
                else: # If the maximum suboptimality is not 0
                    worst_regeret = np.max(self.sub_opptimality_vector) # Set the worst regret to the maximum suboptimality among the past suboptimalities
            else:   # If the maximum LCB is not negative infinity or positive infinity
                worst_regeret = np.min(list(mean_results_itr.values())) - (-LCB_max) # Calculate the worst regret based on the minimum of the mean results and the maximum LCB
        else: # If the length of the mean results is 0
            if len(self.sub_opptimality_vector) == 0: # Check if the length of the suboptimality vector is 0
                worst_regeret = 1 # Set the worst regret to 1
            else: # If the length of the suboptimality vector is not 0
                if np.max(self.sub_opptimality_vector) == 0: # Check if the maximum suboptimality is 0
                    worst_regeret = 1 # Set the worst regret to 1
                else: # If the maximum suboptimality is not 0
                    worst_regeret = np.max(self.sub_opptimality_vector) # Set the worst regret to the maximum suboptimality
        # Check if worst_regeret is an array or list
        if isinstance(worst_regeret, (np.ndarray, list)):
            worst_regeret = float(worst_regeret[0])  # Convert the first element to a float
        # Append the normalized value to sub_opptimality_vector
        self.sub_opptimality_vector.append(worst_regeret) # Append the worst regret to the suboptimality vector
        if cumulative_worst_regret == {}:  # If the dictionary is empty
            cumulative_worst_regret[i_actual - 1] = worst_regeret # Set the cumulative worst regret to the worst regret as the first element
        else: # If the dictionary is not empty
            cumulative_worst_regret[i_actual - 1] = cumulative_worst_regret[i_actual - 2] + worst_regeret # Update the cumulative worst regret
        grouped_results_itr_c = result_storage.table.groupby(self.anon_var_list) # Group the current results
        result_dict_itr_c = {key: group['goal'].tolist() for key, group in grouped_results_itr_c} # Create a dictionary with the current results
        max_value_result = max(result_dict_itr_c.values(), key=lambda x: x[0])[0] # Get the maximum value of the results
        min_value_result = min(result_dict_itr_c.values(), key=lambda x: x[0])[0] # Get the minimum value of the results
        self.max_min_value_dict[i_actual] = {'max': max_value_result, 'min': min_value_result} # Store the maximum and minimum values of the results
        result_list = [] # Initialize the result list
        for key in result_dict_itr_c.keys(): # Loop through the keys of the current results
            result_dict = {self.anon_var_list[i]: key[i] for i in range(len(self.anon_var_list))} # Create a dictionary with the anonymous variable list
            result_list.append(result_dict) # Append the result dictionary to the result list
        if CV_max < 0.1: # Check if the maximum CV is less than 0.1
            app_goal = self._settings['main']['applicationGoal'] # Get the application goal
            if 'thresholds' in app_goal and app_goal['thresholds'] is not None: # Check if the thresholds are not None
                choice_count = i % 2 # Calculate the choice count which decides how to select the next parameter combination
                grouped_results = result_storage.table.groupby(self.base_params) # Group the results based on the base parameters
                result_dict_all = {key: group['goal'].tolist() for key, group in grouped_results} # Create a dictionary with the results
                result_dict_lengths = {key: len(value) for key, value in result_dict_all.items()} # Calculate the length of the results
                Exhausted_params = [key for key, value in result_dict_lengths.items() if value >= max_run] # Get the exhausted parameters
                if choice_count == 1: # Check if the choice count is 1
                    UCB_vector_without_exhausted = {key: value for key, value in LCB_vector.items() if key not in Exhausted_params} # Filter the UCB vector based on the exhausted parameters
                    if UCB_vector_without_exhausted == {}:  # If all params are exhausted
                        x_next = Utilities.pick_random_params(self._settings) # Pick a random parameter combination
                    else: # If not all parameters are exhausted
                        x_next_tuple = max(UCB_vector_without_exhausted.items(), key=lambda item: item[1])[0] # Get the parameter combination with the maximum UCB
                        x_next = dict(zip(self.Param_ranges.keys(), x_next_tuple)) # Set the next parameter combination to the parameter combination with the maximum UCB
                else: # If the choice count is not 1
                    for param_dict in list_of_param_dicts: # Loop through the parameter combinations
                        th_sc[tuple(param_dict.values())] = result_storage.threshold_score(param_dict) # Calculate the threshold score
                    if all(value is None for value in th_sc.values()): # Check if all the threshold scores are None
                        x_next = Utilities.pick_random_params(self._settings) # Pick a random parameter combination
                    else: # If all the threshold scores are not None
                        filtered_dict_all = {key: value for key, value in result_dict_itr_c.items() if len(value) < max_run} # Filter the results based on the maximum number of runs
                        filtered_th_sc = {key: th_sc[key] for key in th_sc if key in filtered_dict_all} # Filter the threshold scores based on the filtered results
                        if len(filtered_th_sc) == 0: # Check if the length of the filtered threshold scores is 0
                            x_next = Utilities.pick_random_params(self._settings) # Pick a random parameter combination
                        else: # If the length of the filtered threshold scores is not 0
                            max_th_sc = max(filtered_th_sc.items(), key=lambda item: item[1])[0] # Get the parameter combination with the maximum threshold score
                            if len(self.th_sc_param) == 0: # Check if the length of the threshold score parameter is 0
                                self.th_sc_param[i_actual] = max_th_sc # Set the threshold score parameter to the parameter combination with the maximum threshold score
                            else: # If the length of the threshold score parameter is not 0
                                grouped_results = result_storage.table.groupby(self.base_params) # Group the results based on the base parameters
                                result_dict_all = {key: group['goal'].tolist() for key, group in grouped_results} # Create a dictionary with the results
                                mean_gl = {key: sum(values) / len(values) for key, values in result_dict_all.items()} # Calculate the mean of the results
                                # Create a set of keys from filtered_th_sc for faster lookup
                                filtered_keys = set(filtered_th_sc.keys())
                                # Filter th_sc_param based on the values not in filtered_th_sc keys
                                self.th_sc_param = {key: value for key, value in self.th_sc_param.items() if
                                               value in filtered_keys}
                                for th_param in self.th_sc_param.values(): # Loop through the threshold score parameter values
                                    penalty = (mean_gl[th_param] - best_goal_so_far) / best_goal_so_far # Calculate the penalty
                                    if 'reliability' in app_goal['optimizationTargets']: # Check if reliability is in the optimization targets
                                        filtered_th_sc[th_param] = filtered_th_sc[th_param] + penalty # Update the filtered threshold score by adding the penalty
                                    else: # If reliability is not in the optimization targets
                                        filtered_th_sc[th_param] = filtered_th_sc[th_param] - penalty # Update the filtered threshold score by subtracting the penalty
                                max_th_sc = max(filtered_th_sc.items(), key=lambda item: item[1])[0] # Get the parameter combination with the maximum threshold score
                                if max_th_sc not in self.th_sc_param.values(): # Check if the parameter combination with the maximum threshold score is not in the threshold score parameter values
                                    self.th_sc_param[i_actual] = max_th_sc # Set the threshold score parameter to the parameter combination with the maximum threshold score
                        if max_th_sc[1] == 0: # Check if the threshold score is 0
                            x_next = Utilities.pick_random_params(self._settings) # Pick a random parameter combination
                        else: # If the threshold score is not 0
                            x_next = dict(zip(self.anon_var_list, max_th_sc)) # Set the next parameter combination to the parameter combination with the maximum threshold score
            else:
                grouped_results = result_storage.table.groupby(self.base_params) # Group the results based on the base parameters
                result_dict_all = {key: group['goal'].tolist() for key, group in grouped_results} # Create a dictionary with the results
                result_dict_lengths = {key: len(value) for key, value in result_dict_all.items()} # Calculate the length of the results
                Exhausted_params = [key for key, value in result_dict_lengths.items() if value >= max_run] # Get the exhausted parameters
                UCB_vector_without_exhausted = {key: value for key, value in LCB_vector.items() if key not in Exhausted_params} # Filter the UCB vector based on the exhausted parameters
                if UCB_vector_without_exhausted == {}:  # If all params are exhausted
                    x_next = Utilities.pick_random_params(self._settings) # Pick a random parameter combination
                else: # If not all parameters are exhausted
                    x_next_tuple = max(UCB_vector_without_exhausted.items(), key=lambda item: item[1])[0] # Get the parameter combination with the maximum UCB
                    x_next = dict(zip(self.Param_ranges.keys(), x_next_tuple)) # Set the next parameter combination to the parameter combination with the maximum UCB
        if x_next is None: # Check if the next parameter combination is None
            x_next = Utilities.pick_random_params(self._settings) # Pick a random parameter combination
        return x_next
    def acquisition_EI(self, x, model,result_storage):
        """
        :param x: scaled version of the parameter set
        :param model: Latest Gaussian Process model
        :param result_storage: The ResultStorage which contains the results of the tests so far.
        :return: The expected improvement and the improvement
        """
        mu, sigma = model.predict(np.array(x).reshape(1, -1), return_std=True) # Predict the mean and standard deviation of the model
        group_results = result_storage.current_constained_table.groupby(self.anon_var_list) # Group the current results
        result_dict = {key: group['goal'].tolist() for key, group in group_results} # Create a dictionary with the current results
        median_results = {key: statistics.median(val) for key, val in result_dict.items()} # Calculate the median of the results
        mean_results = {key: statistics.mean(val) for key, val in result_dict.items()} # Calculate the mean of the results
        if len(median_results) == 0: # Check if the length of the median results is 0
            mu_min = 100000000 # Set the minimum mean to a large value
        else: # If the length of the median results is not 0
            median_min = np.min(list(median_results.values())) # Get the minimum of the median results
            mean_min = np.min(list(mean_results.values())) # Get the minimum of the mean results
            if median_min < mean_min: # Check if the minimum of the median results is less than the minimum of the mean results
                mu_min = median_min # Set the minimum mean to the minimum of the median results
            else: # If the minimum of the median results is not less than the minimum of the mean results
                mu_min = mean_min # Set the minimum mean to the minimum of the mean results
        with np.errstate(divide='ignore'): # Ignore the divide by zero error
            imp = (mu_min - mu) / sigma # Calculate the improvement
            Z = norm.cdf(imp) # Calculate the cumulative distribution function
            EI = (mu_min - mu) * Z + sigma * norm.pdf(imp) # Calculate the expected improvement
            EI[sigma == 0.0] = 0.0 # Set the expected improvement to 0 if the standard deviation is 0
        return EI[0], imp

    def acquisition_UCB(self, x, model, beta_t,param_dict):
        """
        Calculates the GP LCB and the coefficient of variation for a given parameter set.
        :param x: scaled version of the parameter set
        :param model: Latest Gaussian Process model
        :param beta_t: The calibration parameter for the LCB
        :param param_dict: The parameter set for which to calculate the GP LCB and the coefficient of variation
        :return: The GP LCB and the coefficient of variation
        """
        mu, sigma = model.predict(np.array(x).reshape(1, -1), return_std=True) # Predict the mean and standard deviation of the model
        GP_LCB = mu - np.sqrt(beta_t)*sigma # Calculate the GP LCB
        self.mean_results_itr_2[tuple(float(value) for value in param_dict.values())] = mu # Store the mean results from GP LCB
        CV = abs(sigma / mu) # Calculate the coefficient of variation
        return -GP_LCB, CV

    def thresholds_satisfied(self,settings, result_storage, param_dict):
        """
        Checks whether the thresholds in the application requirements are met.
        :param settings: Reference to the global settings object.
        :param result_storage: Result storage on which to evaluate whether thresholds are met.
        :param param_dict: The parameter set for which to evaluate whether thresholds are met.
        :return: True if thresholds are satisfied, False otherwise.

        """
        app_goal = settings['main']['applicationGoal'] # Get the application goal
        if 'thresholds' in app_goal and app_goal['thresholds'] is not None: # Check if the thresholds are not None
            current_param_results = result_storage.get_filtered_subset(param_dict) # Get the filtered subset of the results
            if 'reliability' in app_goal['thresholds']: # Check if reliability is in the thresholds
                if len(current_param_results) >= 2: # Check if the length of the current parameter results is greater than or equal to 2
                    current_r = current_param_results['reliability'].median() # Calculate the median of the reliability
                else: # If not
                    current_r = result_storage.evaluate_point_on_fit(param_dict, result_storage.rel_fit) # Evaluate the point on the reliability fit
                if current_r < app_goal['thresholds']['reliability']:  # Check if the current reliability is less than the threshold reliability
                    return False
            if 'latency' in app_goal['thresholds']: # Check if latency is in the thresholds
                if len(current_param_results) >= 2: # Check if the length of the current parameter results is greater than or equal to 2
                    current_l = current_param_results['latency'].median() # Calculate the median of the latency
                else:   # If not
                    current_l = result_storage.evaluate_point_on_fit(param_dict, result_storage.lat_fit) # Evaluate the point on the latency fit
                if current_l > app_goal['thresholds']['latency']: # Check if the current latency is greater than the threshold latency
                    return False
            if 'energy' in app_goal['thresholds']: # Check if energy is in the thresholds
                if len(current_param_results) >= 2: # Check if the length of the current parameter results is greater than or equal to 2
                    current_e = current_param_results['energy'].median() # Calculate the median of the energy
                else: # If not
                    current_e = result_storage.evaluate_point_on_fit(param_dict, result_storage.nrg_fit) # Evaluate the point on the energy fit
                if current_e > app_goal['thresholds']['energy']: # Check if the current energy is greater than the threshold energy
                    return False
        return True


    def brute_force_on_argument(self,settings, result_storage, list_of_param_dicts_to_search_in=None, max_runs=6):
        """
        Search goal minimum using brute force on argument supplied table
        :param settings: A reference to the global settings object.
        :param result_storage: The ResultStorage to work on.
        :param list_of_param_dicts_to_search_in: List of parameter dictionaries to explore. Will be all combinations if
                                                 not specified.
        :param max_runs: The maximum number of runs for a parameter set to be considered exhausted.
        :return: (minimum, corresponding_params) of the lowest goal function point
                 None if thresholds are nowhere satisfied
        """
        anon_name = settings['main']['anonymousNames'] # Get the anonymous names
        anon_var_list = list(anon_name.values()) # Get the anonymous variable list
        grouped_results_itr = result_storage.current_constained_table.groupby(anon_var_list) # Group the current results
        result_dict_itr = {key: group['goal'].tolist() for key, group in grouped_results_itr} # Create a dictionary with the current results
        filtered_dict = {key: value for key, value in result_dict_itr.items() if len(value) >= max_runs} # Filter the results based on the maximum number of runs gives the exhausted parameter set
        if list_of_param_dicts_to_search_in is None: # Check if the list of parameter dictionaries to search in is None
            list_of_param_dicts_to_search_in = settings['main']['listOfParamDicts'] # Get the list of parameter dictionaries
        current_minimum = float('inf') # Initialize the current minimum to infinity
        current_minimum_params = None # Initialize the current minimum parameters to None
        for param_dict in list_of_param_dicts_to_search_in: # Loop through the parameter dictionaries to search in
            current_goal_val = result_storage.evaluate_point_on_fit(param_dict, result_storage.goal_fit) # Evaluate the point on the goal fit
            if tuple(param_dict.values()) in filtered_dict: # for ignoring the exhausted parameter set
                continue
            if current_goal_val < current_minimum: # Check if the current goal value is less than the current minimum
                if not self.thresholds_satisfied(settings, result_storage, param_dict): # Check if the thresholds are not satisfied
                    continue # Skip the current iteration if the thresholds are not satisfied
                current_minimum = current_goal_val # Update the current minimum
                current_minimum_params = param_dict # Update the current minimum parameters
        return current_minimum, current_minimum_params

    def GEL(self, result_storage): # GEL - Greedy for Exploration
        """
        Search goal minimum using brute force on application wide currently used goal fit.
        :param result_storage: The ResultStorage which contains the results of the tests so far.
        :return: The parameters which lead to the lowest fit value.
        """
        min_goal_value, brute_result = self.brute_force_on_argument(self._settings, result_storage) # Get the minimum goal value and the corresponding parameter combination
        if brute_result is None:
            brute_result = Utilities.pick_random_params(self._settings) # Pick a random parameter combination
        return brute_result

    def GUC(self, result_storage): # GUC - Greedy for Uncertainty in parameter space
        """Search next point to test by being greedy for uncertainty in parameter space.
        :param result_storage: The ResultStorage which contains the results of the tests so far.
        :return: The parameters which lead to the highest uncertainty in the parameter space."""
        tests_at_helper = self.create_tests_at_helper(result_storage) # Return the parameter set as tuple and the count of this parameter set in the result_storage table
        uncertainty = np.zeros(len(self._settings['main']['listOfParamTuples'])) # Initialize the uncertainty vector
        param_ranges = self._settings['main']['parameterRanges'] # Get the parameter ranges
        for uncertainty_list_index, current_tuple in enumerate(self._settings['main']['listOfParamTuples']): # Loop through the parameter tuples
            current_uncertainty = -2 * self.tests_at(tests_at_helper, current_tuple) # Calculate the current uncertainty
            for tuple_index, anon_name in enumerate(self._settings['main']['parameterNames'].keys()):   # Loop through the parameter names
                # tuple_index... index of the parameter in current_tuple => for each of them test if it's value is the
                # lowest of that parameter range.. if yes.. do not test outside of parameter space!
                current_param_index = param_ranges[anon_name].index(current_tuple[tuple_index]) # Get the index of the current parameter
                if current_param_index > 0: # Check if the current parameter index is greater than 0
                    current_modified_tuple = current_tuple[0:tuple_index] \
                                             + (param_ranges[anon_name][current_param_index - 1],) \
                                             + current_tuple[tuple_index + 1:len(current_tuple)] # Modify the current tuple by decreasing the parameter index by 1 and updating the tuple
                    current_uncertainty -= self.tests_at(tests_at_helper, current_modified_tuple) # Update the current uncertainty
                if current_param_index < len(param_ranges[anon_name]) - 1: # Check if the current parameter index is less than the length of the parameter ranges
                    current_modified_tuple = current_tuple[0:tuple_index] \
                                             + (param_ranges[anon_name][current_param_index + 1],) \
                                             + current_tuple[tuple_index + 1:len(current_tuple)] # Modify the current tuple by increasing the parameter index by 1 and updating the tuple
                    current_uncertainty -= self.tests_at(tests_at_helper, current_modified_tuple) # Update the current uncertainty
            uncertainty[uncertainty_list_index] = current_uncertainty # Update the uncertainty vector
        # Create a list with all combinations having the same maximum uncertainty value:
        current_maximum = float('-inf') # Initialize the current maximum to negative infinity
        candidate_list = []
        for uncertainty_list_index, current_tuple in enumerate(self._settings['main']['listOfParamTuples']): # Loop through the parameter tuples
            if uncertainty[uncertainty_list_index] > current_maximum: # Check if the current uncertainty is greater than the current maximum
                current_maximum = uncertainty[uncertainty_list_index] # Update the current maximum
                candidate_list = [current_tuple] # Update the candidate list
            elif uncertainty[uncertainty_list_index] == current_maximum: # Check if the current uncertainty is equal to the current maximum
                candidate_list.append(current_tuple) # Append the current tuple to the candidate list
            else: # If the current uncertainty is less than the current maximum
                pass  # smaller.. do nothing..
        # Choose one of the points with maximum uncertainty:
        if len(candidate_list) > 1: # Check if the length of the candidate list is greater than 1
            # Check the list for the best one according to the current fit states:
            param_dict_list = [Utilities.tuple_to_param_dict(x) for x in candidate_list]
            minimum_value, params = self.brute_force_on_argument(self._settings, result_storage, param_dict_list) # Get the minimum value and the corresponding parameter combination
            if params is None:
                print(f'GoalEvaluator: Warning, uncertain area contains only elements which do not satisfy the '
                      f'specified constraints. Selecting Random one (of the uncertain area)!')
                params = random.sample(param_dict_list, 1)[0] # Select a random parameter combination from the list
        else:
            # There is just one element! Return that!
            params = Utilities.tuple_to_param_dict(candidate_list[0])
        return params

    def GER(self): # GER - Greedy for Exploration
        "Nothing to do as NTS as it is starigt forward to explore the parameter space"
        pass

    def RL(self): # RL - Reinforcement Learning
        "Nothing to do as the model type care of it"
        pass

    def create_tests_at_helper(self, result_storage):
        """Creates a dictionary containing one element per set of same parameter combinations in the result_storage
           table. Every element contains the count of this parameter set in the result_storage table.

           This is a dynamic programming solution, much faster than creating this info every time it is needed.
           :return: Dictionary with parameter set tuples as indices and corresponding counts as integers.
           """
        tests_at_helper = dict()
        for index, row in result_storage.table.iterrows():
            current_tuple = tuple(row[anon_name] for anon_name in self._settings['main']['parameterNames'].keys())
            if current_tuple in tests_at_helper.keys():
                tests_at_helper[current_tuple] += 1
            else:
                tests_at_helper[current_tuple] = 1
        return tests_at_helper

    def tests_at(self, tests_at_helper, params_tuple):
        """
        :param tests_at_helper: The helper array from create_tests_at_helper().
        :param params_tuple: The params to search for as tuple. Pay attention to order!
        :return: Returns the number of tests for params_tuple in tests_at_helper, or 0 if there were none yet.
        """
        if params_tuple in tests_at_helper.keys():
            return tests_at_helper[params_tuple]
        else:
            return 0



