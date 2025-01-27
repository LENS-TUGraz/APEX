import itertools
import random

import numpy as np
import pandas as pd
import math
import scipy
from scipy.stats import binom
from math import comb
from scipy.optimize import curve_fit
from scipy.stats import qmc

from DCubeTestEnvironment import DCubeTestEnvironment
from RecordedTestEnvironment import RecordedTestEnvironment


def create_instance(class_name, *args):
    """
    Creates an instance of class_name and supplies all additional args to the constructor.
    :param class_name: The class for which to create an instance.
    :param args: All additional arguments to create_instance are just forwarded to the constructor of the created class.
    :return: An instance of the created class.
    """
    class_ = globals()[class_name]
    return class_(*args)

def get_min_test_needed(settings):
    """
    Calculate the minimum number of samples required to estimate a given percentile
    with a specified confidence level.

    Args:
        settings (dict): A dictionary containing 'constraint_percentile' and 'constraint_confidence'.

    Returns:
        int: Minimum number of samples required.
    """
    constraint_percentile = settings['main']['constraint_percentile']
    constraint_confidence = settings['main']['constraint_confidence']

    if not (0 < constraint_percentile < 100):
        raise ValueError("Percentile must be between 0 and 100.")
    if not (0 < constraint_confidence < 100):
        raise ValueError("Confidence must be between 0 and 100.")

    # Convert to probabilities
    p = constraint_percentile / 100
    c = constraint_confidence / 100

    # Calculate minimum number of samples
    N = math.ceil(math.log(1 - c) / math.log(p))

    return N

def reformat_settings(settings):
    """
    Parses hierarchical parameter settings into simpler to handle flat dictionaries of the same size.
    Also calculates some intermediate settings values out of the base user settings.
    :param settings: All settings from the settings file.
    """
    # Parameter "reformatting":
    all_info = settings['main']['parameters']
    # if  settings['main']['parameters_normalize_required']:
    #     all_info = settings['main']['parameters_normalize']
    name_map = dict()
    range_map = dict()
    range_map_norm = dict()
    anonymous_map = dict()
    settings['main']['parameterMeans'] = dict()
    settings['main']['parameterVars'] = dict()
    settings['main']['parameterMins'] = dict()
    settings['main']['parameterMaxs'] = dict()
    for anonymous_name, info in all_info.items():
        name_map[anonymous_name] = info['name']
        anonymous_map[info['name']] = anonymous_name
        if 'values' in info.keys():
            range_map[anonymous_name] = sorted(info['values'])
        else:
            range_map[anonymous_name] = range(info['min'], info['max'])
        settings['main']['parameterMeans'][anonymous_name] = np.mean(range_map[anonymous_name])
        settings['main']['parameterVars'][anonymous_name] = np.var(range_map[anonymous_name])
        settings['main']['parameterMins'][anonymous_name] = int(np.min(range_map[anonymous_name]))
        settings['main']['parameterMaxs'][anonymous_name] = int(np.max(range_map[anonymous_name]))

    settings['main']['parameterNames'] = name_map  # map to be used for anonymous => clear
    settings['main']['anonymousNames'] = anonymous_map  # map to be used for clear => anonymous
    settings['main']['parameterRanges'] = range_map
    list_of_value_lists = []
    list_of_value_lists_norm = []
    for current_value_list in settings['main']['parameterRanges'].values():
        list_of_value_lists.append(current_value_list)
    settings['main']['listOfParamTuples'] = []
    settings['main']['listOfParamDicts'] = []
    settings['main']['listOfParamTuples_norm'] = []
    settings['main']['listOfParamDicts_norm'] = []
    for current_params in itertools.product(*list_of_value_lists):
        settings['main']['listOfParamTuples'].append(current_params)
        settings['main']['listOfParamDicts'].append(tuple_to_param_dict(current_params))
    settings['main']['listOfCornersTuples'] = []
    settings['main']['listOfCornersDict'] = []
    for current_params in itertools.product(*zip(settings['main']['parameterMins'].values(),
                                                 settings['main']['parameterMaxs'].values())):
        settings['main']['listOfCornersTuples'].append(current_params)
        settings['main']['listOfCornersDict'].append(tuple_to_param_dict(current_params))
    total_min_params = dict()
    total_max_params = dict()
    for anon_name in settings['main']['parameterNames'].keys():
        total_min_params[anon_name] = settings['main']['parameterMins'][anon_name]
        total_max_params[anon_name] = settings['main']['parameterMaxs'][anon_name]
    settings['main']['maxEuclideanDistance'] = norm_euclidean(settings, total_min_params, total_max_params)
    # Additional settings generation and sanity checks:
    fit_string = ''.join(settings['main']['fitFunction'].split())  # remove whitespaces
    settings['main']['fitColumns'] = sorted(fit_string.split('+'))
    settings['main']['fitColumns_normalize'] = sorted(fit_string.split('+'))
    settings['main']['fitFreedomDegrees'] = len(settings['main']['fitColumns']) + 1  # +1 for the intercept_
    if settings['main']['nextPointAlgo'] == 'robust_greedy':
        settings['main']['additionalInitTests'] = 1  # robust_greedy needs one more because it leaves one out!
    else:
        settings['main']['additionalInitTests'] = 0
    settings['main']['totalInitTests'] = settings['main']['n_init_test']
    settings['main']['totalParameterCombinations'] = 1
    for param_range in settings['main']['parameterRanges'].values():
        settings['main']['totalParameterCombinations'] *= len(param_range)
    combinations = settings['main']['totalParameterCombinations']
    constraint_percentile = settings['main']['constraint_percentile']
    constraint_confidence = settings['main']['constraint_confidence']
    min_test_per_param = min_number_samples(constraint_percentile, constraint_confidence)
    settings['main']['totalTests'] = min_test_per_param[0] * combinations
    print(f'Main: Working on a total of {combinations} possible parameter combinations.')
    settings['main']['mainLoopRuns'] = settings['main']['termination_criteria']['Max_number_of_testbed_trials'] - settings['main']['totalInitTests']
    if settings['main']['mainLoopRuns'] <= 0:
        print('Main: Provided settings do not make sense! The number of total tests to perform (nrTestsTotal) is '
              'smaller than or equal to the number of tests performed during initialization!')
        exit(-1)
    if combinations < settings['main']['totalInitTests']:
        print('Main: Provided settings do not make sense! The number of random init tests is larger thant the '
              'number of parameter combinations available. This would result in an endless loop!')
    settings['main']['disableRobustGreedyRankCheck'] = False
    for anon_name, corresponding_range in settings['main']['parameterRanges'].items():
        # search for anon_name in fitColumns and count columns which contain this anon_name:
        current_anon_cnt = 0
        for fit_part in settings['main']['fitColumns']:
            if anon_name in fit_part:
                current_anon_cnt += 1
        if len(corresponding_range) <= current_anon_cnt:  # <= because of intercept_, we need 1 more thant cnt...
            clear_name = settings['main']['parameterNames'][anon_name]
            print(f'Main: Warning, the parameter space of {clear_name} is too small to provide enough information '
                  f'for the specified fit function. Your fits might make little sense this way.')
            settings['main']['disableRobustGreedyRankCheck'] = True
    settings['main']['parameter_values'] = {}
    # Iterate through the parameters
    for key, param in settings['main']['parameters'].items():
        if 'min' in param and 'max' in param:
            # Generate list of values from min to max (exclusive)
            settings['main']['parameter_values'][key] = list(range(param['min'], param['max']))
        elif 'values' in param:
            # Use the specified values directly
            settings['main']['parameter_values'][key] = param['values']
        else:
            # In case no valid configuration is found
            settings['main']['parameter_values'][key] = []
    settings['main']['init_samples'] = []
    if settings['main']['init_sampling'] != 'random':
        if settings['main']['init_sampling'] == 'lhs':
            num_parameters, discrete_values = extract_parameter_details(settings['main']['listOfParamTuples'])
            settings['main']['init_samples'] = generate_lhs_samples(settings['main']['n_init_test'], discrete_values)
        elif settings['main']['init_sampling'] == 'sobel':
            num_parameters, discrete_values = extract_parameter_details(settings['main']['listOfParamTuples'])
            settings['main']['init_samples']= generate_sobol_samples(settings['main']['n_init_test'], discrete_values)



def binary_search_first_bigger(haystack, needle):
    """
    Binary search for first element bigger than needle in haystack. Basically from:
    https://www.geeksforgeeks.org/first-strictly-greater-element-in-a-sorted-array-in-java/
    :param haystack: The list to search in.
    :param needle: The value with which to compare.
    :return: Index of first element in haystack with bigger value than needle
    """
    start_index = 0
    end_index = len(haystack) - 1
    if start_index == end_index:
        raise Exception('Haystack with only one element supplied! This element might be bigger or smaller than needle!')
    desired_index = None
    while start_index <= end_index:
        middle_index = (start_index + end_index) // 2
        if haystack[middle_index] <= needle:
            # Move to right side if target is greater:
            start_index = middle_index + 1
        else:
            # Move left side:
            end_index = middle_index - 1
            desired_index = middle_index
    return desired_index


def pick_random_params(settings):
    """
    :param settings: A reference to the global settings object to access the parameter ranges stored there.
    :return: A dictionary containing a randomly picked parameter set.
    """
    params = dict()
    for param_name, range_ in settings['main']['parameterRanges'].items():
        params[param_name] = random.sample(range_, 1)[0]
    return params


def evaluate_goal_function(settings, r, l, e):
    """
    Evaluates the goal function on the supplied metrics.
    :param settings: A reference to the global settings object to access the goals stored there.
    :param r: Reliability
    :param l: Latency
    :param e: Energy
    :return: Value of the goal function for specified reliability, latency and energy.
    """
    optimum_targets = settings['main']['applicationGoal']['optimizationTargets']
    goal_value = 0
    if 'reliability' in optimum_targets:
        goal_value -= r * optimum_targets['reliability']  # Negation done here and not in weights!
    if 'latency' in optimum_targets:
        goal_value += l * optimum_targets['latency']
    if 'energy' in optimum_targets:
        goal_value += e * optimum_targets['energy']
    return goal_value


def evaluate_column(column_name, params):
    """
    Evaluates the value of a user/application parameter to a fit function's parameter.
    E.g. For a fit function of a + a^2 this is called on every part and either returns a or a^2 dependant on
    column_name.
    :param column_name: The column's name, e.g. a or b or aa or ab or aab
    :param params: Dictionary containing the necessary input data as e.g. {a: 3, b: 4}
    :return: e.g. a^(x)*b^(y)*...
    """
    if 'reliability' in params.keys():
        raise Exception(f"This function must not be supplied with a whole pandas row! It's iterating over all entries!")
    for single_digit in column_name:
        if single_digit not in params.keys():
            raise Exception(f'Fit contains a parameter name which is not present in the supplied params!\n'
                            f'column_name:{column_name}, supplied params:{params}')
    result = 1
    for param_name, value in params.items():
        power_count = column_name.count(param_name)  # count e.g. the number of a in the column name..
        result *= pow(value, power_count)
    return result


def get_clear_fit_function(settings):
    """Transforms the anonymous param named fit function to a clear one."""
    fit_string = settings['main']['fitFunction']
    fit_string = ''.join(fit_string.split())  # remove whitespaces
    additive_parts = fit_string.split('+')
    total_clear_string = ''
    for total_multiplicative_part in additive_parts:
        clear_total_multiplicative = ''
        for multiplicative_part in total_multiplicative_part:
            clear_part = settings['main']['parameterNames'][multiplicative_part]
            clear_total_multiplicative += f'{clear_part} * '
        clear_total_multiplicative = clear_total_multiplicative[0:-3]
        total_clear_string += f'{clear_total_multiplicative} + \n'
    total_clear_string = total_clear_string[0:-4]
    return f'intercept +\n{total_clear_string}'


def norm_euclidean(settings, param_set_a, param_set_b):
    """
    Calculates a normalized euclidean distance between two parameter sets a and b. Normalization is implemented by
    dividing the difference by the variance of the corresponding parameter range.
    :param settings: The settings object which knows corresponding the parameter ranges.
    :param param_set_a: Set of parameters (e.g. {'a': 3, 'b':4})
    :param param_set_b: Set of parameters (e.g. {'a': 7, 'b':8})
    :return: An normalized euclidean distance between the two parameter sets.
    """
    current_sq_diff_sum = 0
    for anon_param, value_a in param_set_a.items():
        # Check if any input is None
        if param_set_a is None or param_set_b is None:
            return 10000  # Return 10000 if any input is None just a big number to indicate none
        if len(settings['main']['parameterRanges'][anon_param]) > 1:  # var would be 0 otherwise..
            current_param_mean = settings['main']['parameterMeans'][anon_param]
            current_param_var = settings['main']['parameterVars'][anon_param]
            z_score_a = (value_a - current_param_mean) / current_param_var
            z_score_b = (param_set_b[anon_param] - current_param_mean) / current_param_var
            current_sq_diff_sum += pow((z_score_a - z_score_b), 2)
    return np.sqrt(current_sq_diff_sum)


def tuple_to_param_dict(tuple_input):
    """
    Transforms a tuple containing only param values to a dict with anonymous param names as keys.
    :param tuple_input: The input tuple.
    :return: A dict with anonymous params as keys and the tuple input as values.
    """
    params_dict = dict()
    for param_number in range(len(tuple_input)):
        anon_param_name = chr(ord('a') + param_number)
        params_dict[anon_param_name] = tuple_input[param_number]
    return params_dict


def create_rank_check_data(settings, table: pd.DataFrame):
    """Adds a column of ones for the intercept to the data which is going to be rank checked."""
    ones_column = np.ones(len(table), dtype=int).transpose()
    table_data = np.array(table[settings['main']['fitColumns']], dtype=np.int64)
    combined = np.column_stack((ones_column, table_data))
    return combined


def to_clear_name_dict(settings, anon_name_dict):
    """
    Transforms a dictionary with anon param names to clear names.
    :param settings: Reference to the global settings object
    :param anon_name_dict: Dictionary, eg. {'a': 3, 'b':4}
    :return: Dictionary with clear names, eg. {'n_tx_max': 3, 'tx_power':4}
    """
    clear_dict = dict()
    for key, value in anon_name_dict.items():
        clear_dict[settings['main']['parameterNames'][key]] = value
    return clear_dict


def to_anon_name_dict(settings, clear_name_dict):
    """
    Transforms a dictionary with clear param names to anonymous names.
    :param settings: Reference to the global settings object
    :param clear_name_dict: Dictionary with clear names, eg. {'n_tx_max': 3, 'tx_power':4}
    :return: Dictionary, eg. {'a': 3, 'b':4}
    """
    anon_name_dict = dict()
    for key, value in clear_name_dict.items():
        anon_name_dict[settings['main']['anonymousNames'][key]] = value
    return anon_name_dict


def minima_variance_to_confidence(result_storage, minima_variance):
    """
    :param result_storage: A result storage object to get the goal value range from
    :param minima_variance: A minima variance value of robust greedy.
    :return: A confidence as percentage corresponding to the minima variance value of robust greedy.
    """
    goal_values = result_storage.table['goal']
    goal_min = goal_values.min()
    goal_max = goal_values.max()
    goal_range = abs(goal_max - goal_min)
    confidence = (1 - np.sqrt(minima_variance) / goal_range) * 100.0
    confidence = max(0.0, confidence)  # limit to 0 if way too little confidence!
    return confidence


def powerset(iterable):
    """
    From itertools documentation: https://docs.python.org/3/library/itertools.html
    powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    """
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s) + 1))


def closes_param(settings, params, all_param_comb):
    smallest_difference = float('inf')
    closest_row = None
    for current_row in all_param_comb:
        current_difference = norm_euclidean(settings, params, current_row)
        if current_difference < smallest_difference:
            closest_row = current_row
            smallest_difference = current_difference
    return closest_row

def min_number_samples(percentile,confidence,robustness=0):

    ##
    # Checking the inputs
    ##

    if confidence >= 100 or confidence <= 0:
        raise ValueError("Invalid confidence: "+repr(confidence)+". Provide a real number strictly between 0 and 100.")
    if percentile >= 100 or percentile <= 0:
        raise ValueError("Invalid percentile: "+repr(percentile)+". Provide a real number strictly between 0 and 100.")
    if not isinstance(robustness, int):
        raise ValueError("Invalid robustness: "+repr(robustness)+". Provide a positive integer.")
    if robustness < 0:
        raise ValueError("Invalid robustness: "+repr(robustness)+". Provide a positive integer.")

    ##
    # Single-sided interval
    ##

    N_single = math.ceil(math.log(1-confidence/100)/math.log(1-percentile/100))

    if robustness:

        # Make sure the first N is large enough
        N_single = max(N_single, 2*(robustness+1))

        # Increse N until the desired confidence is reached
        while True:
            # compute P( x_(1+r) <= Pp )
            bd = scipy.stats.binom(N_single,percentile/100)
            prob = 1-np.cumsum([bd.pmf(k) for k in range(robustness+1)])[-1]
            # test
            if prob >= (confidence/100):
                break
            else:
                N_single += 1

    ##
    # Double-sided interval
    ##

    # only relevant for the median - other percentiles are better estimated with single-sided intervals
    if percentile==50:

        N_double = math.ceil(1 - (math.log(1-confidence/100)/math.log(2)))

        if robustness:

            # Make sure the first N is large enough
            N_double = max(N_double, 2*(robustness+1))

            # Increse N until the desired confidence is reached
            while True:
                # compute P( x_(1+r) <= M <= x_(N-r) )
                bd = scipy.stats.binom(N_double,percentile/100)
                prob = 1-np.cumsum([2*bd.pmf(k) for k in range(robustness+1)])[-1]
                # test
                if prob >= (confidence/100):
                    break
                else:
                    N_double += 1

    else:
        # Double-sided interval is irrelevant -> same as single-sided
        N_double = N_single

    return N_single, N_double



def calculate_variance(data):
    mean = np.mean(data)
    std_deviation = np.std(data, ddof=1)
    return mean, std_deviation

def exponential_func(x, a, b, T):
    return T*(1 - a * np.exp(-b * x))

def derivative_exponential_func(x, a, b, T):
    return b * T * a * np.exp(-b * x)


def reward_calculator(settings, result_storage, test_results):
    app_goal = settings['main']['applicationGoal']
    if thresholds_satisfied_single(settings, result_storage, test_results):
        if 'optimizationTargets' in app_goal and app_goal['optimizationTargets'] is not None:
            if 'reliability' in app_goal['optimizationTargets']:
                reward = - test_results['goal']
            if 'latency' in app_goal['optimizationTargets']:
                reward = - test_results['goal']
            if 'energy' in app_goal['optimizationTargets']:
                reward = - test_results['goal']
    else:
        if 'thresholds' in app_goal and app_goal['thresholds'] is not None:
            if 'reliability' in app_goal['thresholds']:
                reward = - abs(test_results['reliability'] - app_goal['thresholds']['reliability'])
            if 'latency' in app_goal['thresholds']:
                reward = - abs(test_results['latency'] - app_goal['thresholds']['latency'])
            if 'energy' in app_goal['thresholds']:
                reward = - abs(test_results['energy'] - app_goal['thresholds']['energy'])
    return reward

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

def calculated_threshold_confidence(settings, results_storage, best_parameter):
    """
    Calculates the confidence level of the threshold satisfaction.
    :param settings: Reference to the global settings object.
    :param results_storage: The result storage object to evaluate the threshold satisfaction.
    :return: The confidence level of the threshold satisfaction.
    """
    #check whether the best parameter is empty
    if best_parameter == {}:
        return 0, None, 0
    threshold = settings['main']['applicationGoal']['thresholds']
    threshold_metric = list(threshold.keys())[0]
    threshold_value = threshold[threshold_metric]
    percentile = settings['main']['constraint_percentile']
    # extract threshold_metric from results_storage.table
    const_dict = {key: group[threshold_metric].tolist() for key, group in results_storage}  # Create a dictionary of the results
    # Extract the latest (last) tuple based on the dictionary keys
    latest_key = max(best_parameter.keys())  # Get the largest key
    latest_best_param = best_parameter[latest_key]  # Extract the tuple
    if latest_best_param is None:
        return 0, None, 0
    results_list = const_dict[latest_best_param]
    # calculate the confidence level
    N = len(results_list)
    # Step 1: Calculate the percentile value
    P_p = np.percentile(results_list, percentile)
    # Step 2: Check how many values satisfy >= threshold
    satisfying_values = [value for value in results_list if value <= threshold_value]
    l = len(satisfying_values)
    # Step 3: Calculate cumulative binomial probability
    cumulative_prob = sum(comb(N, k) * ((percentile/100)** k) * ((1 - (percentile/100)) ** (N - k)) for k in range(l))
    confidence = cumulative_prob
    best_thresh_confidence = 0
    param_with_highest_confidence = None
    for param, value in const_dict.items():
            results_list = value
            N = len(results_list)
            # Step 1: Calculate the percentile value
            P_p = np.percentile(results_list, percentile)
            # Step 2: Check how many values satisfy >= threshold
            satisfying_values = [value for value in results_list if value <= threshold_value]
            l = len(satisfying_values)
            # Step 3: Calculate cumulative binomial probability
            cumulative_prob = sum(comb(N, k) * ((percentile/100)** k) * ((1 - (percentile/100)) ** (N - k)) for k in range(l))
            if cumulative_prob > best_thresh_confidence:
                best_thresh_confidence = cumulative_prob
                param_with_highest_confidence = param
    return confidence*100, param_with_highest_confidence, best_thresh_confidence*100


def calculate_optimality_confidence(cumulative_worst_regret):
    """
    Calculates the confidence level of the optimality.
    :param cumulative_worst_regret: The cumulative worst regret.
    :return: The confidence level of the optimality.
    """
    # Calculate the confidence level
    keys = list(cumulative_worst_regret.keys())
    if cumulative_worst_regret == {}:
        values = [value[0] for value in cumulative_worst_regret.values()]
    else:
        return 0

    x_data = np.array(keys)
    y_data = np.array(values)

    # Normalize the data with the max value avoid if the max value is 0
    if max(y_data) != 0:
        y_data = y_data / max(y_data)
    else:
        y_data = y_data

    if max(x_data) != 0:
        x_data = x_data / max(x_data)
    else:
        x_data = x_data

    if len(x_data) < 10: # Need some data to get a reasonable fit minimum exponential fit requires at least 3 points
        return 0
    # Fit the data
    params, params_covariance = curve_fit(exponential_func, x_data, y_data, maxfev=100000000)
    a_fit, b_fit, T = params
    last_x_norm = x_data[-1]
    derivative = derivative_exponential_func(last_x_norm, a_fit, b_fit, T)
    angle = np.arctan(derivative)
    angle_degrees = angle * 180 / np.pi
    if angle_degrees < 0:
        angle_degrees = 0
    confidence = 100*(1 - min(angle_degrees,45)/45)
    return confidence


# Function to extract parameter details from the list of tuples
def extract_parameter_details(param_value_list):
    num_parameters = len(param_value_list[0])  # Determine the number of parameters
    discrete_values = [sorted(set(values[i] for values in param_value_list)) for i in range(num_parameters)]
    return num_parameters, discrete_values


# Map sampled indices back to parameter values
def map_indices_to_values(samples, discrete_values):
    mapped_samples = []
    for sample in samples:
        mapped_sample = [discrete_values[i][int(index)] for i, index in enumerate(sample)]
        mapped_samples.append(mapped_sample)
    return mapped_samples


# Generate parameter sets using Latin Hypercube Sampling
def generate_lhs_samples(n, discrete_values):
    num_parameters = len(discrete_values)
    sampler = qmc.LatinHypercube(d=num_parameters)
    lhs_samples = sampler.random(n)

    # Scale to the number of discrete values for each parameter
    scaled_samples = (lhs_samples * [len(values) for values in discrete_values]).astype(int)
    return map_indices_to_values(scaled_samples, discrete_values)


# Generate parameter sets using Sobol sequences
def generate_sobol_samples(n, discrete_values):
    num_parameters = len(discrete_values)
    sampler = qmc.Sobol(d=num_parameters, scramble=True)
    sobol_samples = sampler.random_base2(m=int(np.ceil(np.log2(n))))[:n]

    # Scale to the number of discrete values for each parameter
    scaled_samples = (sobol_samples * [len(values) for values in discrete_values]).astype(int)
    return map_indices_to_values(scaled_samples, discrete_values)