import copy
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.gaussian_process.kernels import Matern
from sklearn.preprocessing import StandardScaler


import Utilities
from joblib import parallel_backend
import statistics


class ResultStorage_LR:
    """This class holds result data in a pandas table."""

    def __init__(self, settings):
        self._settings = settings
        self._columns = ['id']
        self.base_params = list(self._settings['main']['parameterNames'].keys())
        self._columns.extend(self.base_params)
        self.additional_columns_for_fitting = set(settings['main']['fitColumns'])
        self.additional_columns_for_fitting = self.additional_columns_for_fitting - set(self.base_params)
        self.additional_columns_for_fitting = sorted(list(self.additional_columns_for_fitting))
        self._columns.extend(self.additional_columns_for_fitting)
        self._columns.append('reliability')
        self._columns.append('latency')
        self._columns.append('energy')
        self._columns.append('goal')
        # attention: keep add_single_test function in sync with columns definition!
        print(f'ResultStorage: Columns are {self._columns}')
        self.table = pd.DataFrame(columns=self._columns)
        self.constrained_table = pd.DataFrame(columns=self._columns)
        self.current_constained_table = pd.DataFrame(columns=self._columns)
        kernel = Matern(length_scale=1.0, nu=2.7)
        optimizer = 'fmin_l_bfgs_b'
        self.rel_fit = LinearRegression()
        self.lat_fit = LinearRegression()
        self.nrg_fit = LinearRegression()
        self.goal_fit = LinearRegression()
        self.scaler = StandardScaler()
        self.scaler_l = StandardScaler()
        anon_name = self._settings['main']['anonymousNames']
        self.anon_var_list = list(anon_name.values())
        self.n_jobs = 6
        self.current_goal_r2 = None

    def update_fits(self):
        """Updates all fits according to the data currently available in this result storage instance."""
        fit_data = self.table[self._settings['main']['fitColumns']]
        self.goal_fit.fit(fit_data, self.table['goal'])
        fit_calculated_z = self.goal_fit.predict(fit_data)
        self.current_goal_r2 = r2_score(self.table['goal'], fit_calculated_z)
        app_goal = self._settings['main']['applicationGoal']
        rel_fit_needed = False
        lat_fit_needed = False
        nrg_fit_needed = False
        if 'thresholds' in app_goal and app_goal['thresholds'] is not None:
            if 'reliability' in app_goal['thresholds']:
                rel_fit_needed = True
            if 'latency' in app_goal['thresholds']:
                lat_fit_needed = True
            if 'energy' in app_goal['thresholds']:
                nrg_fit_needed = True
        if rel_fit_needed:
            self.rel_fit.fit(fit_data, self.table['reliability'])
        if nrg_fit_needed:
            self.nrg_fit.fit(fit_data, self.table['energy'])
        if lat_fit_needed:
            # Special handling for latency, as it can contain None-values:
            fit_column_copy = copy.deepcopy(self._settings['main']['fitColumns'])
            fit_column_copy.append('latency')
            latency_fit_table = self.table[fit_column_copy]
            none_filtered_table = latency_fit_table.dropna()
            if len(none_filtered_table) > 0:
                fit_data = none_filtered_table[self._settings['main']['fitColumns']]
                self.lat_fit.fit(fit_data, none_filtered_table['latency'])

    def coef_for_param(self, fit: LinearRegression, param):
        """
        :param fit: The LinearRegression object of which to take the coefficients.
        :param param: The desired param for which to find the corresponding coefficient.
        :return: The coefficient of the supplied fit LinearRegression object which corresponds to param.
        """
        desired_coef_index = self._settings['main']['fitColumns'].index(param)
        return fit.coef_[desired_coef_index]

    def check_for_information_gain(self, new_params):
        """
        Check whether the rank of the coefficient-matrix which would describe the equation system used to calculate
        the fit coefficients is as high as the number of corresponding equations (=find linear interdependencies)
        :param new_params: The parameters which are about to be added.
        :return: True if new_params would be an information gain, False otherwise.
        """
        combined = Utilities.create_rank_check_data(self._settings, self.table)
        new_data = [1]
        for column_name in self._settings['main']['fitColumns']:
            new_data.append(Utilities.evaluate_column(column_name, new_params))
        data_to_check = np.row_stack((combined, new_data))
        rank = np.linalg.matrix_rank(data_to_check)
        # print(f'rank={rank} vs len={len(data_to_check)}')
        return rank == len(data_to_check)

    def evaluate_point_on_fit(self, params, fit: LinearRegression):
        """
        :param params: The parameters for which evaluation is desired as dictionary. e.g. {a:3, b:4}
        :param fit: The fit on which to evaluate
        :return: z height evaluated on the coefficients generated by fitting
        """
        if hasattr(fit, 'intercept_'):
            multiplication_input = pd.DataFrame([params])
            evaluation = fit.predict(multiplication_input)
            return evaluation[0]
        else:
            return float('inf')

    def get_mse_score(self, fit, reality_values):
        """
        :param fit: The fit for which to calculate the MSE.
        :param reality_values: The real values corresponding to the fit.
        :return: Mean Squared Error between a fit and the corresponding reality.
        """
        if np.count_nonzero(~np.isnan(reality_values)) > 0:
            multiplication_inputs = []
            for _, param_row in self.table[self._settings['main']['fitColumns']].iterrows():
                multiplication_inputs.append(param_row.to_list())
            with parallel_backend('threading', n_jobs=self.n_jobs):
                fit_calculated_z = fit.predict(multiplication_inputs)
            return mean_squared_error(reality_values, fit_calculated_z)
        else:
            return None

    def get_filtered_subset(self, params):
        """
        :param params: The params to filter for as dict. E.g. {'a': 3, 'b': 4}
        :return: Returns the subset of test results which matches the supplied filter.
        """
        pd_subset = self.table
        for name, value in params.items():
            pd_subset = pd_subset[pd_subset[name] == value]
        return pd_subset

    def already_seen(self, params):
        """
        :param params: The params to check for as dict. E.g.: {'a': 3, 'b': 4}
        :return: Returns True if at least one test point is already in the result storage, False otherwise.
        """
        return len(self.get_filtered_subset(params)) > 0

    def get_param_dict(self, single_element):
        """
        Converts the pandas table row to a dict containing only the base params and the corresponding values.
        :param single_element: The element from e.g. pandas_result.loc[0] for which to generate the params dictionary.
        :return: All parameters as dictionary with clear names and corresponding values.
        """
        param_dict = dict()
        for param_name in list(self._settings['main']['parameterNames'].keys()):
            param_dict[param_name] = single_element[param_name]
        return param_dict

    def get_param_dict_norm(self, single_element):
        """
        Converts the pandas table row to a dict containing only the base params and the corresponding values.
        :param single_element: The element from e.g. pandas_result.loc[0] for which to generate the params dictionary.
        :return: All parameters as dictionary with clear names and corresponding values.
        """
        param_dict = dict()
        for param_name in list(self._settings['main']['parameterNames'].keys()):
            param_dict[param_name] = self._settings['main']['parameters_normalize'][param_name]['values'][single_element[param_name]]
        return param_dict

    def get_last_addition(self):
        """
        :return: The last added test.
        """
        return self.table.tail(1).to_dict(orient='records')[0]

    def get_closest(self, params):
        """
        :param params: The parameters for which to return the closest test. Currently parameters are expected to be
                       supplied anonymously and returned anonymously.
        :return: The closest test result corresponding to the supplied parameters as dict.
        """
        smallest_difference = float('inf')
        closest_row = None
        self.table = self.table.sample(frac=1).reset_index(drop=True)
        for index, current_row in self.table.iterrows():
            current_difference = Utilities.norm_euclidean(self._settings, params, self.get_param_dict(current_row))
            if current_difference < smallest_difference:
                closest_row = current_row
                smallest_difference = current_difference
        return closest_row

    def add_single_test(self, test_result, used=None, parsing = False):
        """
        This function adds a single test result to the table.
        :param test_result: A dictionary or pandas row containing a test id, a value for each param and each metric.
        """
        self.current_constained_table = pd.DataFrame(columns=self._columns)
        data = [test_result['id']]
        params_only_dict = self.get_param_dict(test_result)
        for param_name, value in params_only_dict.items():
            data.append(value)
        for column_name in self.additional_columns_for_fitting:
            data.append(Utilities.evaluate_column(column_name, params_only_dict))
        r = test_result['reliability']
        l = test_result['latency']
        e = test_result['energy']
        data.extend([r, l, e])
        if 'override_goal' in self._settings['main']:
            if self._settings['main']['override_goal'] == 'settling_time':
                goal_value = test_result['goal']
            else:
                goal_value = Utilities.evaluate_goal_function(self._settings, r, l, e)
        else:
            goal_value = Utilities.evaluate_goal_function(self._settings, r, l, e)
        if self._settings['main']['testEnvironment'] == 'ArtificialTestEnvironment':
            goal_noise = np.random.normal(0, self._settings['artificialTestEnvironment']['goalNoise'])
            goal_value += goal_noise
        data.append(goal_value)
        df = pd.DataFrame([data], columns=self._columns)
        if self.table.empty:
            self.table = df
        else:
            self.table = pd.concat([self.table, df])
        if not parsing:
            thresh_satisfied = 1
            grouped_results = self.table.groupby(self.anon_var_list)
            app_goal = self._settings['main']['applicationGoal']
            if 'thresholds' in app_goal and app_goal['thresholds'] is not None:
                if 'reliability' in app_goal['thresholds']:
                    rel_grouped_results = {key: group['reliability'].tolist() for key, group in grouped_results}
                    median_rel_group = {key: statistics.median(val) for key, val in rel_grouped_results.items()}
                    for key, value in median_rel_group.items():
                        if value > app_goal['thresholds']['reliability']:
                            num_elements = len(key)
                            column_range = self.anon_var_list[0:num_elements]
                            conditions = self.table[column_range].eq(key).all(axis=1)
                            filtered_df = self.table[conditions]
                            if self.current_constained_table.empty:
                                self.current_constained_table = filtered_df
                            else:
                                self.current_constained_table = pd.concat([self.current_constained_table, filtered_df])
                            #self.current_constained_table = pd.concat([self.current_constained_table, filtered_df])
                    # if params_tuple in median_rel_group:
                    #     if median_rel_group[params_tuple] > app_goal['thresholds']['reliability']:
                    #         self.current_constained_table = pd.concat([self.current_constained_table, df])
                    if test_result['reliability'] < app_goal['thresholds']['reliability']:
                        thresh_satisfied = 0
                if 'latency' in app_goal['thresholds']:
                    lat_grouped_results = {key: group['latency'].tolist() for key, group in grouped_results}
                    median_lat_group = {key: statistics.median(val) for key, val in lat_grouped_results.items()}
                    for key, value in median_lat_group.items():
                        if value < app_goal['thresholds']['latency']:
                            num_elements = len(key)
                            column_range = self.anon_var_list[0:num_elements]
                            conditions = self.table[column_range].eq(key).all(axis=1)
                            filtered_df = self.table[conditions]
                            if self.current_constained_table.empty:
                                self.current_constained_table = filtered_df
                            else:
                                self.current_constained_table = pd.concat([self.current_constained_table, filtered_df])
                    # if params_tuple in median_lat_group:
                    #     if median_lat_group[params_tuple] < app_goal['thresholds']['latency']:
                    #         self.current_constained_table = pd.concat([self.current_constained_table, df])
                    if test_result['latency'] > app_goal['thresholds']['latency']:
                        thresh_satisfied = 0
                if 'energy' in app_goal['thresholds']:
                    nrg_grouped_results = {key: group['energy'].tolist() for key, group in grouped_results}
                    median_nrg_group = {key: statistics.median(val) for key, val in nrg_grouped_results.items()}
                    for key, value in median_nrg_group.items():
                        if value < app_goal['thresholds']['energy']:
                            num_elements = len(key)
                            column_range = self.anon_var_list[0:num_elements]
                            conditions = self.table[column_range].eq(key).all(axis=1)
                            filtered_df = self.table[conditions]
                            if self.current_constained_table.empty:
                                self.current_constained_table = filtered_df
                            else:
                                self.current_constained_table = pd.concat([self.current_constained_table, filtered_df])
                    # if params_tuple in median_nrg_group:
                    #     if median_nrg_group[params_tuple] < app_goal['thresholds']['energy']:
                    #         self.current_constained_table = pd.concat([self.current_constained_table, df])
                    if test_result['energy'] > app_goal['thresholds']['energy']:
                        thresh_satisfied = 0
            else:
                self.current_constained_table = self.table
            if thresh_satisfied == 1:
                if self.constrained_table.empty:
                    self.constrained_table = df
                else:
                    self.constrained_table = pd.concat([self.constrained_table, df])


    def load_initial_data(self, path):
        print(f'ResultStorage: Loading initial data from {path}')
        list_of_results = []
        with open(path, 'r') as json_file:
            data = json.load(json_file)
        for test_result in data:
            single_job_data= []
            for key, value in test_result.items():
                if isinstance(value, dict):
                    single_job_data.extend(value.values())
                else:
                    single_job_data.append(value)
                print(f'{key}: {value}')
            r = test_result['metrics']['reliability']
            l = test_result['metrics']['latency']
            e = test_result['metrics']['energy']
            goal_value = Utilities.evaluate_goal_function(self._settings, r, l, e)
            single_job_data.append(goal_value)
            df = pd.DataFrame([single_job_data], columns=self._columns)
            self.table = pd.concat([self.table, df])
            thresh_satisfied = 1
            app_goal = self._settings['main']['applicationGoal']
            if 'thresholds' in app_goal and app_goal['thresholds'] is not None:
                if 'reliability' in app_goal['thresholds']:
                    if test_result['metrics']['reliability'] < app_goal['thresholds']['reliability']:
                        thresh_satisfied = 0
                if 'latency' in app_goal['thresholds']:
                    if test_result['metrics']['latency'] > app_goal['thresholds']['latency']:
                        thresh_satisfied = 0
                if 'energy' in app_goal['thresholds']:
                    if test_result['metrics']['energy'] > app_goal['thresholds']['energy']:
                        thresh_satisfied = 0
            if thresh_satisfied == 1:
                self.constrained_table = pd.concat([self.constrained_table, df])
        print(f'ResultStorage: Initial data loaded. Table now contains {len(self.table)} entries.')






    def delete_single_test(self, test_id):
        """
        Removes a single individual test from the table.
        :param test_id: The test id to delete.
        """
        self.table = self.table[self.table['id'] != test_id]

    def brute_force_reality(self):
        """Search goal minima using brute force on all available real data.
           Note: This returns a list, as a single value is most likely an outlier!"""
        LIST_LEN = 100
        minima_list = []  # tuple of (minimum, param_dict)
        current_worst_minimum = float('inf')
        for _, param_row in self.table[['goal'].extend(self.base_params)].iterrows():
            if param_row['goal'] < current_worst_minimum:
                minima_list.append((param_row['goal'], self.get_param_dict(param_row)))
                minima_list.sort(key=lambda pair: pair[0])  # sort by first value!
                if len(minima_list) > LIST_LEN:
                    minima_list = minima_list[:LIST_LEN]
                current_worst_minimum = minima_list[-1][0]  # consider worst element of list next time
        return minima_list

    def print_abs_errors(self, params):
        subset = self.get_filtered_subset(params)
        for index, row in subset.iterrows():
            rel_difference = abs(row['reliability'] - self.evaluate_point_on_fit(params, self.rel_fit))
            lat_difference = abs(row['latency'] - self.evaluate_point_on_fit(params, self.lat_fit))
            nrg_difference = abs(row['energy'] - self.evaluate_point_on_fit(params, self.nrg_fit))
            goal_difference = abs(row['goal'] - self.evaluate_point_on_fit(params, self.goal_fit))
            print(f"{row['id']}: {rel_difference}, {lat_difference}, {nrg_difference}, {goal_difference}")

    def plot_state(self, xy_params_anon, fixed_filter=dict()):
        from matplotlib import pyplot as plt
        from PlotUtilities import plot_state
        clear_param_names = [self._settings['main']['parameterNames'][anon_name] for anon_name in xy_params_anon]
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.set(title='Goal Function', xlabel=clear_param_names[0], ylabel=clear_param_names[1])
        # fixed filter is an empty dict here this time!
        color_legend = plot_state(self._settings, self, xy_params_anon, fixed_filter, False, ax_goal=ax)
        if color_legend is not None:
            plt.legend()
        plt.tight_layout()
        plt.show()

    def thresholds_satisfied(self, settings, param_dict):
        """
        Checks whether the thresholds in the application requirements are met.
        :param settings: Reference to the global settings object.
        :param result_storage: Result storage on which to evaluate whether thresholds are met.
        :param param_dict: The parameter set for which to evaluate whether thresholds are met.
        :return: True if thresholds are satisfied, False otherwise.

        """
        app_goal = settings['main']['applicationGoal']
        if 'thresholds' in app_goal and app_goal['thresholds'] is not None:
            current_param_results = self.get_filtered_subset(param_dict)
            if 'reliability' in app_goal['thresholds']:
                if len(current_param_results) >= 3:
                    current_r = current_param_results['reliability'].median()
                else:
                    current_r = self.evaluate_point_on_fit(param_dict, self.rel_fit)
                if current_r < app_goal['thresholds']['reliability']:
                    return False
            if 'latency' in app_goal['thresholds']:
                if len(current_param_results) >= 3:
                    current_l = current_param_results['latency'].median()
                else:
                    current_l = self.evaluate_point_on_fit(param_dict, self.lat_fit)
                if current_l > app_goal['thresholds']['latency']:
                    return False
            if 'energy' in app_goal['thresholds']:
                if len(current_param_results) >= 3:
                    current_e = current_param_results['energy'].median()
                else:
                    current_e = self.evaluate_point_on_fit(param_dict, self.nrg_fit)
                if current_e > app_goal['thresholds']['energy']:
                    return False
        return True

    def instant_thresh_check(self, results):
        settings = self._settings
        app_goal = settings['main']['applicationGoal']
        if 'thresholds' in app_goal and app_goal['thresholds'] is not None:
            if 'reliability' in app_goal['thresholds']:
                if results['reliability'] < app_goal['thresholds']['reliability']:
                    return False
            if 'latency' in app_goal['thresholds']:
                if results['latency'] > app_goal['thresholds']['latency']:
                    return False
            if 'energy' in app_goal['thresholds']:
                if results['energy'] > app_goal['thresholds']['energy']:
                    return False
        return True

    def threshold_score(self, param_dict):
        settings = self._settings
        base_params = list(settings['main']['parameterNames'].keys())
        table = self.table
        grouped_results = table.groupby(base_params)
        result_dict_all = {key: group['goal'].tolist() for key, group in grouped_results}
        latency_dict = {key: group['latency'].tolist() for key, group in grouped_results}
        energy_dict = {key: group['energy'].tolist() for key, group in grouped_results}
        reliability_dict = {key: group['reliability'].tolist() for key, group in grouped_results}
        median_lat = {key: statistics.median(values) for key, values in latency_dict.items()}
        median_rel = {key: statistics.median(values) for key, values in reliability_dict.items()}
        median_eng = {key: statistics.median(values) for key, values in energy_dict.items()}
        median_gl = {key: statistics.median(values) for key, values in result_dict_all.items()}
        app_goal = settings['main']['applicationGoal']
        mean_of_all_lat = sum(median_lat.values()) / len(median_lat)
        mean_of_all_rel = sum(median_rel.values()) / len(median_rel)
        mean_of_all_eng = sum(median_eng.values()) / len(median_eng)
        mean_of_all_gl = sum(median_gl.values()) / len(median_gl)
        choice_dict = {'latency': 0, 'reliability': 0, 'energy': 0}
        if 'thresholds' in app_goal and app_goal['thresholds'] is not None:
            if 'reliability' in app_goal['thresholds']:
                rel_thresh = app_goal['thresholds']['reliability']
                if tuple(param_dict.values()) in median_rel:
                    current_r = median_rel[tuple(param_dict.values())]
                else:
                    #combination_scaled = self.scaler.fit_transform(np.array(list(param_dict.values())).reshape(1, -1))
                    #current_r = self.rel_fit.predict(combination_scaled)
                    current_r = self.evaluate_point_on_fit(param_dict, self.rel_fit)
                if current_r < rel_thresh:
                    rel_score =1 - (rel_thresh - current_r) / rel_thresh
                    rel_sat = table[table['reliability'] > rel_thresh]
                    choice_dict['reliability'] = len(rel_sat)
                else:
                    rel_score = 0
            if 'latency' in app_goal['thresholds']:
                lat_thresh = app_goal['thresholds']['latency']
                if tuple(param_dict.values()) in median_lat:
                    current_l = median_lat[tuple(param_dict.values())]
                else:
                    # combination_scaled = self.scaler.fit_transform(np.array(list(param_dict.values())).reshape(1, -1))
                    # current_l = self.lat_fit.predict(combination_scaled)
                    current_l = self.evaluate_point_on_fit(param_dict, self.lat_fit)
                if current_l > lat_thresh:
                    lat_score = 1 - (current_l - lat_thresh) / lat_thresh
                    lat_sat = table[table['latency'] < lat_thresh]
                    choice_dict['latency'] = len(lat_sat)
                else:
                    lat_score = 0
            if 'energy' in app_goal['thresholds']:
                eng_thresh = app_goal['thresholds']['energy']
                if tuple(param_dict.values()) in median_eng:
                    current_e = median_eng[tuple(param_dict.values())]
                else:
                    # combination_scaled = self.scaler.fit_transform(np.array(list(param_dict.values())).reshape(1, -1))
                    # current_e = self.nrg_fit.predict(combination_scaled)
                    current_e = self.evaluate_point_on_fit(param_dict, self.nrg_fit)
                if current_e > eng_thresh:
                    eng_score = 1 - (current_e - eng_thresh) / eng_thresh
                    eng_sat = table[table['energy'] < eng_thresh]
                    choice_dict['energy'] = len(eng_sat)
                else:
                    eng_score = 0
            if all(value == 0 for value in choice_dict.values()):
                return 0
            else:
                choosed_metr = max(choice_dict, key=choice_dict.get)
                if choosed_metr == 'latency':
                    return lat_score
                elif choosed_metr == 'reliability':
                    return rel_score
                elif choosed_metr == 'energy':
                    return eng_score
        return 0




