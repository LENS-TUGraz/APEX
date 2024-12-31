import copy
import itertools
import random
import numpy as np
from scipy.optimize import minimize
import statistics

import Utilities
from ResultsStorage import ResultStorage

class GoalEvaluator:
    def __init__(self, settings_ref, result_storage_ref):
        self._settings = settings_ref
        self.result_storage: ResultStorage = result_storage_ref
        self.last_minima_variance = None
        self.brute_time = 0.0
        self.calc_time = 0.0
        self.base_params = list(self._settings['main']['parameterNames'].keys())

    @staticmethod
    def brute_force_on_argument(settings, result_storage, list_of_param_dicts_to_search_in=None):
        """
        Search goal minimum using brute force on argument supplied table
        :param settings: A reference to the global settings object.
        :param result_storage: The ResultStorage to work on.
        :param list_of_param_dicts_to_search_in: List of parameter dictionaries to explore. Will be all combinations if
                                                 not specified.
        :return: (minimum, corresponding_params) of the lowest goal function point
                 None if thresholds are nowhere satisfied
        """
        if list_of_param_dicts_to_search_in is None:
            list_of_param_dicts_to_search_in = settings['main']['listOfParamDicts']
        current_minimum = float('inf')
        current_minimum_params = None
        for param_dict in list_of_param_dicts_to_search_in:
            current_goal_val = result_storage.evaluate_point_on_fit(param_dict, result_storage.goal_fit)
            if current_goal_val < current_minimum:
                if not GoalEvaluator.thresholds_satisfied(settings, result_storage, param_dict):
                    continue
                current_minimum = current_goal_val
                current_minimum_params = param_dict
        return current_minimum, current_minimum_params

    @staticmethod
    def thresholds_satisfied(settings, result_storage, param_dict):
        """
        Checks whether the thresholds in the application requirements are met.
        :param settings: Reference to the global settings object.
        :param result_storage: Result storage on which to evaluate whether thresholds are met.
        :param param_dict: The parameter set for which to evaluate whether thresholds are met.
        :return: True if thresholds are satisfied, False otherwise.

        """
        app_goal = settings['main']['applicationGoal']
        if 'thresholds' in app_goal and app_goal['thresholds'] is not None:
            current_param_results = result_storage.get_filtered_subset(param_dict)
            if 'reliability' in app_goal['thresholds']:
                if len(current_param_results) >= 3:
                    current_r = current_param_results['reliability'].median()
                else:
                    current_r = result_storage.evaluate_point_on_fit(param_dict, result_storage.rel_fit)
                if current_r < app_goal['thresholds']['reliability']:
                    return False
            if 'latency' in app_goal['thresholds']:
                if len(current_param_results) >= 3:
                    current_l = current_param_results['latency'].median()
                else:
                    current_l = result_storage.evaluate_point_on_fit(param_dict, result_storage.lat_fit)
                if current_l > app_goal['thresholds']['latency']:
                    return False
            if 'energy' in app_goal['thresholds']:
                if len(current_param_results) >= 3:
                    current_e = current_param_results['energy'].median()
                else:
                    current_e = result_storage.evaluate_point_on_fit(param_dict, result_storage.nrg_fit)
                if current_e > app_goal['thresholds']['energy']:
                    return False
        return True
    @staticmethod
    def thresholds_satisfied_single(settings, result_storage, test_results):
        """
        Checks whether the thresholds in the application requirements are met.
        :param settings: Reference to the global settings object.
        :param result_storage: Result storage on which to evaluate whether thresholds are met.
        :param param_dict: The parameter set for which to evaluate whether thresholds are met.
        :return: True if thresholds are satisfied, False otherwise.

        """
        app_goal = settings['main']['applicationGoal']
        if 'thresholds' in app_goal and app_goal['thresholds'] is not None:
            if 'reliability' in app_goal['thresholds']:
                current_r = test_results['reliability']
                if current_r < app_goal['thresholds']['reliability']:
                    return False
            if 'latency' in app_goal['thresholds']:
                current_l = test_results['latency']
                if current_l > app_goal['thresholds']['latency']:
                    return False
            if 'energy' in app_goal['thresholds']:
                current_e = test_results['energy']
                if current_e > app_goal['thresholds']['energy']:
                    return False
        return True

    @staticmethod
    def thresholds_satisfied_const(settings, result_storage, param_dict, current_param_results):
        """
        Checks whether the thresholds in the application requirements are met.
        :param settings: Reference to the global settings object.
        :param result_storage: Result storage on which to evaluate whether thresholds are met.
        :param param_dict: The parameter set for which to evaluate whether thresholds are met.
        :current_param_results: set of results that does not violate the constraints
        :return: True if thresholds are satisfied, False otherwise.


        """
        base_params = list(settings['main']['parameterNames'].keys())
        param_values = tuple([param_dict[param] for param in base_params if param in param_dict])
        grouped_results_const = current_param_results.groupby(base_params)

        app_goal = settings['main']['applicationGoal']
        if 'thresholds' in app_goal and app_goal['thresholds'] is not None:

            if 'reliability' in app_goal['thresholds']:
                r_dict_const = {key: group['reliability'].tolist() for key, group in grouped_results_const}
                median_r_const = {key: statistics.median(val) for key, val in r_dict_const.items()}
                if param_values in r_dict_const.keys():
                    if len(current_param_results) >= 1:
                        current_r = median_r_const[param_values]
                else:
                    current_r = float('-inf')
                if current_r < app_goal['thresholds']['reliability']:
                    return False
            if 'latency' in app_goal['thresholds']:
                l_dict_const = {key: group['latency'].tolist() for key, group in grouped_results_const}
                median_l_const = {key: statistics.median(val) for key, val in l_dict_const.items()}
                if param_values in l_dict_const.keys():
                    if len(current_param_results) >= 1:
                        current_l = median_l_const[param_values]
                else:
                    current_l = float('-inf')
                if current_l > app_goal['thresholds']['latency']:
                    return False
            if 'energy' in app_goal['thresholds']:
                e_dict_const = {key: group['energy'].tolist() for key, group in grouped_results_const}
                median_e_const = {key: statistics.median(val) for key, val in e_dict_const.items()}
                if param_values in e_dict_const.keys():
                    if len(current_param_results) >= 1:
                        current_e = median_e_const[param_values]
                    else:
                        current_e = result_storage.evaluate_point_on_fit(param_dict, result_storage.nrg_fit)
                else:
                    current_e = float('-inf')
                if current_e > app_goal['thresholds']['energy']:
                    return False
        return True

    @staticmethod
    def get_float_neighbourhood(settings, float_params: dict):
        list_of_value_lists = []
        for anon_param_name, float_value in float_params.items():
            # First: test the borders:
            current_range_max = settings['main']['parameterMaxs'][anon_param_name]
            current_range_min = settings['main']['parameterMins'][anon_param_name]
            if float_value >= current_range_max:
                list_of_value_lists.append([current_range_max])
                continue
            if float_value <= current_range_min:
                list_of_value_lists.append([current_range_min])
                continue
            # In every other case there must be a bigger and a lower neighbor for the list_of_value_lists to append..
            # parameterRanges are sorted... perform binary search:
            current_param_range = settings['main']['parameterRanges'][anon_param_name]
            first_bigger_index = Utilities.binary_search_first_bigger(current_param_range, float_value)
            lower_neighbor = current_param_range[first_bigger_index - 1]
            bigger_neighbor = current_param_range[first_bigger_index]
            list_of_value_lists.append([lower_neighbor, bigger_neighbor])  # append lower and bigger as list
        list_of_param_dicts = []
        for current_params in itertools.product(*list_of_value_lists):
            list_of_param_dicts.append(Utilities.tuple_to_param_dict(current_params))
        return list_of_param_dicts

    @staticmethod
    def scipy_compat_goal_function(base_params, result_storage: ResultStorage):
        bp_sum = np.sum(base_params)
        if np.isnan(bp_sum):
            raise Exception('Input was none.. Most likely because of start in constrained area..')
            # See: https://stackoverflow.com/questions/45966102/scipy-selects-nan-as-inputs-while-minimizing
        base_param_dict = Utilities.tuple_to_param_dict(base_params)
        return result_storage.evaluate_point_on_fit(base_param_dict, result_storage.goal_fit)

    @staticmethod
    def scipy_compat_reliability_constraint(base_params, result_storage: ResultStorage, threshold_value):
        base_param_dict = Utilities.tuple_to_param_dict(base_params)
        # Optimizer aims for non-negative results, example with two params:
        # c(a, b): u0 + u1 * a + u2 * b + u3 * aa + u4 * bb         > r
        # c(a, b): u0 - e + u1 * a + u2 * b + u3 * aa + u4 * bb - r > 0
        # c(a, b): evaluate_point_on_fit() - r                      > 0
        return result_storage.evaluate_point_on_fit(base_param_dict, result_storage.rel_fit) - threshold_value

    @staticmethod
    def scipy_compat_latency_constraint(base_params, result_storage: ResultStorage, threshold_value):
        base_param_dict = Utilities.tuple_to_param_dict(base_params)
        # Optimizer aims for non-negative results, example with two params:
        # c(a, b): u0 + u1 * a + u2 * b + u3 * aa + u4 * bb        <= l
        # c(a, b): u0 - l + u1 * a + u2 * b + u3 * aa + u4 * bb    <= 0
        # c(a, b): -(-l + u0 + u1 * a + u2 * b + u3 * aa + u4 * bb) > 0
        # c(a, b): -(-l + evaluate_point_on_fit())                  > 0
        # c(a, b): l - evaluate_point_on_fit()                      > 0
        return threshold_value - result_storage.evaluate_point_on_fit(base_param_dict, result_storage.lat_fit)

    @staticmethod
    def scipy_compat_energy_constraint(base_params, result_storage: ResultStorage, threshold_value):
        base_param_dict = Utilities.tuple_to_param_dict(base_params)
        # Optimizer aims for non-negative results, example with two params:
        # c(a, b): u0 + u1 * a + u2 * b + u3 * aa + u4 * bb        <= e
        # c(a, b): u0 - e + u1 * a + u2 * b + u3 * aa + u4 * bb    <= 0
        # c(a, b): -(-e + u0 + u1 * a + u2 * b + u3 * aa + u4 * bb) > 0
        # c(a, b): -(-e + evaluate_point_on_fit())                  > 0
        # c(a, b): e - evaluate_point_on_fit()                      > 0
        return threshold_value - result_storage.evaluate_point_on_fit(base_param_dict, result_storage.nrg_fit)

    @staticmethod
    def scipy_optimize_goal_minimum(settings, result_storage: ResultStorage):
        """
        Searches goal minimum, by using scipy.optimize on the goal value fit and testing all direct neighbors.
        Problem: This will not work for concave fits, as gradients might make the optimizer run in the wrong direction!
                 (=Finding a local minimum at the bounds, instead of the global one!)
        :param settings: A reference to the global settings object.
        :param result_storage: The ResultStorage to work on.
        :return: (minimum, corresponding_params) of the lowest goal function point
                 None if thresholds are nowhere satisfied
        """
        app_goal = settings['main']['applicationGoal']
        constraints_list = []
        if 'thresholds' in app_goal and app_goal['thresholds'] is not None:
            if 'reliability' in app_goal['thresholds']:
                threshold_value = app_goal['thresholds']['reliability']
                current_constraint = {
                    'type': 'ineq',
                    'fun': GoalEvaluator.scipy_compat_reliability_constraint,
                    'args': (result_storage, threshold_value)
                }
                constraints_list.append(current_constraint)
            if 'latency' in app_goal['thresholds']:
                threshold_value = app_goal['thresholds']['latency']
                current_constraint = {
                    'type': 'ineq',
                    'fun': GoalEvaluator.scipy_compat_latency_constraint,
                    'args': (result_storage, threshold_value)
                }
                constraints_list.append(current_constraint)
            if 'energy' in app_goal['thresholds']:
                threshold_value = app_goal['thresholds']['energy']
                current_constraint = {
                    'type': 'ineq',
                    'fun': GoalEvaluator.scipy_compat_energy_constraint,
                    'args': (result_storage, threshold_value)
                }
                constraints_list.append(current_constraint)
        bounds_list = []
        for anon_name in settings['main']['parameterNames'].keys():
            current_min = settings['main']['parameterMins'][anon_name]
            current_max = settings['main']['parameterMaxs'][anon_name]
            bounds_list.append((current_min, current_max))
        list_of_param_dicts = []
        for current_init_guess in settings['main']['listOfCornersTuples']:
            init_guess = np.array(current_init_guess)
            try:
                res = minimize(GoalEvaluator.scipy_compat_goal_function, init_guess, args=(result_storage,),
                               constraints=constraints_list, bounds=bounds_list)
                if not res.success:
                    print(f'GoalEvaluator: Warning: Optimizer did not work! Reason:{res.message}')
                float_params = Utilities.tuple_to_param_dict(res.x)
                # print(f'GoalEvaluator: Calc results:{float_params} for init of {current_init_guess}')
                list_of_param_dicts.extend(GoalEvaluator.get_float_neighbourhood(settings, float_params))
            except Exception as e:
                print(f'GoalEvaluator: Optimizer Exception: {str(e)}')
        unique_list = [dict(y) for y in set(tuple(x.items()) for x in list_of_param_dicts)]
        print(f'GoalEvaluator: Searching on {unique_list}')
        return GoalEvaluator.brute_force_on_argument(settings, result_storage, unique_list)

    def greedy(self, finalize=False, intermediate=False ):
        """
        Search goal minimum using brute force on application wide currently used goal fit.
        :return: The parameters which lead to the lowest fit value.
        """
        thresh_best = {}
        min_goal_value, brute_result = self.brute_force_on_argument(self._settings, self.result_storage)
        #  brute_result = brute_result_min_goal_value[0]
        all_param_comb = self._settings['main']['listOfParamDicts']
        base_params = list(self._settings['main']['parameterNames'].keys())
        if not finalize:
            if self._settings['main']['distributed_exploration']:
                brute_result = Utilities.distr_exploration(self._settings, self.result_storage, brute_result)
        # calc_result = self.scipy_optimize_goal_minimum(self._settings, self.result_storage)[1]
        if finalize or intermediate:
            if self._settings['main']['final_as_per_experiment']:
                min_goal_value, brute_result = self.best_as_per_experiment(base_params)
            if brute_result is None:
                thresh_req = self._settings['main']['applicationGoal']['thresholds']
                if 'reliability' in thresh_req:
                    grouped_results = self.result_storage.table.groupby(base_params)
                    r_grouped = {key: group['reliability'].tolist() for key, group in grouped_results}
                    median_r = {key: statistics.median(val) for key, val in r_grouped.items()}
                    thresh_best['reliability'] = max(median_r.values())
                if 'energy' in thresh_req:
                    grouped_results = self.result_storage.table.groupby(base_params)
                    e_grouped = {key: group['energy'].tolist() for key, group in grouped_results}
                    median_e = {key: statistics.median(val) for key, val in e_grouped.items()}
                    thresh_best['energy'] = min(median_e.values())
                if 'latency' in thresh_req:
                    grouped_results = self.result_storage.table.groupby(base_params)
                    l_grouped = {key: group['latency'].tolist() for key, group in grouped_results}
                    median_l = {key: statistics.median(val) for key, val in l_grouped.items()}
                    thresh_best['latency'] = min(median_l.values())
                if intermediate:
                    return brute_result, min_goal_value
                return brute_result, min_goal_value, thresh_best
            if intermediate:
                return brute_result, min_goal_value
            return brute_result, min_goal_value, thresh_best
        return brute_result

    def robust_greedy(self, finalize=False, intermediate=False):
        """Search goal minimum using something like RANSAC"""
        minima_list = []
        for id_to_leave_out in self.result_storage.table['id']:
            local_result_storage = copy.deepcopy(self.result_storage)
            local_result_storage.delete_single_test(id_to_leave_out)
            local_result_storage.update_fits()
            data = Utilities.create_rank_check_data(self._settings, local_result_storage.table)
            data_rank = np.linalg.matrix_rank(data)
            # only process fits which have enough points to correctly fit the desired function:
            if data_rank >= self._settings['main']['fitFreedomDegrees'] \
                    or self._settings['main']['disableRobustGreedyRankCheck']:
                if finalize or intermediate:
                    if self._settings['main']['final_as_per_experiment']:
                        found_minimum = self.best_as_per_experiment(self.base_params, local_result_storage)
                    else:
                        found_minimum = self.brute_force_on_argument(self._settings, local_result_storage)
                else:
                    found_minimum = self.brute_force_on_argument(self._settings, local_result_storage)
                if found_minimum[1] is not None:  # is none if goal function was inf everywhere!
                    minima_list.append(found_minimum)  # list of tuples!
        if len(minima_list) == 0:
            print('GoalEvaluator: Warning, minima list for robust greedy could not be computed!')
            return None
        minima_list.sort(key=lambda pair: pair[0])  # sort by first value!
        median_value, median_params = minima_list[int(np.floor(len(minima_list) / 2))]
        minima_values = [minima_list[i][0] for i in range(len(minima_list))]
        self.last_minima_variance = np.var(minima_values)
        if finalize or intermediate:
            return median_params, median_value
        return median_params

    def create_tests_at_helper(self):
        """Creates a dictionary containing one element per set of same parameter combinations in the result_storage
           table. Every element contains the count of this parameter set in the result_storage table.

           This is a dynamic programming solution, much faster than creating this info every time it is needed.
           :return: Dictionary with parameter set tuples as indices and corresponding counts as integers.
           """
        tests_at_helper = dict()
        for index, row in self.result_storage.table.iterrows():
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

    def greedy_uncert(self):
        """Search next point to test by being greedy for uncertainty in parameter space."""
        tests_at_helper = self.create_tests_at_helper()
        uncertainty = np.zeros(len(self._settings['main']['listOfParamTuples']))
        param_ranges = self._settings['main']['parameterRanges']
        for uncertainty_list_index, current_tuple in enumerate(self._settings['main']['listOfParamTuples']):
            current_uncertainty = -2 * self.tests_at(tests_at_helper, current_tuple)
            for tuple_index, anon_name in enumerate(self._settings['main']['parameterNames'].keys()):
                # tuple_index... index of the parameter in current_tuple => for each of them test if it's value is the
                # lowest of that parameter range.. if yes.. do not test outside of parameter space!
                current_param_index = param_ranges[anon_name].index(current_tuple[tuple_index])
                if current_param_index > 0:
                    current_modified_tuple = current_tuple[0:tuple_index] \
                                             + (param_ranges[anon_name][current_param_index - 1],) \
                                             + current_tuple[tuple_index + 1:len(current_tuple)]
                    current_uncertainty -= self.tests_at(tests_at_helper, current_modified_tuple)
                if current_param_index < len(param_ranges[anon_name]) - 1:
                    current_modified_tuple = current_tuple[0:tuple_index] \
                                             + (param_ranges[anon_name][current_param_index + 1],) \
                                             + current_tuple[tuple_index + 1:len(current_tuple)]
                    current_uncertainty -= self.tests_at(tests_at_helper, current_modified_tuple)
            uncertainty[uncertainty_list_index] = current_uncertainty
            # print(f'{current_tuple}: {current_uncertainty}')
        # Create a list with all combinations having the same maximum uncertainty value:
        current_maximum = float('-inf')
        candidate_list = []
        for uncertainty_list_index, current_tuple in enumerate(self._settings['main']['listOfParamTuples']):
            if uncertainty[uncertainty_list_index] > current_maximum:
                current_maximum = uncertainty[uncertainty_list_index]
                candidate_list = [current_tuple]
            elif uncertainty[uncertainty_list_index] == current_maximum:
                candidate_list.append(current_tuple)
            else:
                pass  # smaller.. do nothing..
        # Choose one of the points with maximum uncertainty:
        if len(candidate_list) > 1:
            # Check the list for the best one according to the current fit states:
            param_dict_list = [Utilities.tuple_to_param_dict(x) for x in candidate_list]
            minimum_value, params = self.brute_force_on_argument(self._settings, self.result_storage, param_dict_list)
            if params is None:
                print(f'GoalEvaluator: Warning, uncertain area contains only elements which do not satisfy the '
                      f'specified constraints. Selecting Random one (of the uncertain area)!')
                params = random.sample(param_dict_list, 1)[0]
        else:
            # There is just one element! Return that!
            params = Utilities.tuple_to_param_dict(candidate_list[0])
        return params

    def monte_carlo(self):
        """Search goal minimum using a monte carlo approach"""
        total_evals = 20000
        smallest_goal_value = float('inf')
        corresponding_params = None
        for run_nr in range(total_evals):
            current_params = Utilities.pick_random_params(self._settings)
            current_goal_value = self.result_storage.evaluate_point_on_fit(current_params, self.result_storage.goal_fit)
            if current_goal_value < smallest_goal_value:
                smallest_goal_value = current_goal_value
                corresponding_params = current_params
        return corresponding_params

    def random(self):
        """Return a random parameter set"""
        return Utilities.pick_random_params(self._settings)

    def best_as_per_experiment(self, base_params, local_results_storage = None):
        results_table = self.result_storage.table
        if self._settings['main']['nextPointAlgo'] is 'robust_greedy':
            results_table = self.local_results_storage.table
        grouped_results = results_table.groupby(base_params)
        result_dict = {key: group['goal'].tolist() for key, group in grouped_results}
        median_results = {key: float(statistics.median(val)) for key, val in result_dict.items()}
        current_minimum = float('inf')
        current_minimum_params = None
        for param_tuple, current_goal_val in median_results.items():
            param_dict = dict(zip(base_params, tuple(map(int, param_tuple))))
            if current_goal_val < current_minimum:
                if not GoalEvaluator.thresholds_satisfied(self._settings, self.result_storage, param_dict):
                    continue
                current_minimum = current_goal_val
                current_minimum_params = param_dict
        return current_minimum, current_minimum_params
