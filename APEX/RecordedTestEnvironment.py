import os
import json
import Utilities
#import PlotUtilities
from AbstractTestEnvironment import AbstractTestEnvironment
from ResultsStorage_LR import ResultStorage_LR


class RecordedTestEnvironment(AbstractTestEnvironment):
    # ToDo: Add a TestEnvironment which just returns the next element in the recordings to have some kind of
    #       "recording-player"... This could be a subclass of this class?
    @staticmethod
    def _append_file(source_data_sets, file_path):
        directory_name, file_name = os.path.split(file_path)
        with open(file_path, 'r') as file_handle:
            source_data_sets[file_name] = json.load(file_handle)

    def __init__(self, settings):
        super().__init__(settings)
        self._param_name_map = self._settings['main']['parameterNames']
        self.parsed = ResultStorage_LR(self._settings)
        self.already_used = ResultStorage_LR(self._settings)
        anon_name = self._settings['main']['anonymousNames']
        self.anon_var_list = list(anon_name.values())
        source_data_sets = dict()
        input_path = self._settings['recordedTestEnvironment']['inputPath']
        if type(input_path) == list:
            for path in input_path:
                RecordedTestEnvironment._append_file(source_data_sets, path)
        elif os.path.isdir(input_path):
            for file_name in os.listdir(input_path):
                RecordedTestEnvironment._append_file(source_data_sets, os.path.join(input_path, file_name))
        else:
            RecordedTestEnvironment._append_file(source_data_sets, input_path)

        for file_name in source_data_sets:
            print(f'Recording: Loading tests from {file_name}')
            current_data_list = source_data_sets[file_name]
            for data in current_data_list:
                # first, translate named parameters to anonymous ones:
                previous_params = data['params']
                anonymous_params = dict()
                for anonymous_param, named_param in self._param_name_map.items():
                    anonymous_params[anonymous_param] = previous_params[named_param]
                data_row = {'id': data['id']}
                data_row.update(anonymous_params)
                data_row.update(data['metrics'])
                self.parsed.add_single_test(data_row)
        print(f'Loaded a total of {len(self.parsed.table)} test results.')

    def execute_test(self, params):
        """
        Finds the closest test to the corresponding parameters out of all unreturned tests, adds it to the already
        returned tests, removes it from the list of file-parsed tests and returns it to the caller.
        :param params: Desired params for which to return a test result.
        :return: A single test result.
        """
        run_success = False
        while not run_success:
            closest_test = self.parsed.get_closest(params)
            extracted_values = closest_test[self.anon_var_list].to_dict()
            if params != extracted_values:
                params = Utilities.pick_random_params(self._settings)
            else:
                run_success = True
        self.parsed.delete_single_test(closest_test['id'])
        self.already_used.add_single_test(closest_test, used=True)
        print(f'Recording: {params} => {closest_test.to_dict()}')
        return closest_test

    @staticmethod
    def jump_to_final_state(main):
        """
        Fast forwards main to a state where all points were returned/considered
        :param main: A reference to the main object.
        """
        for index, row in main.test_environment.parsed.table.iterrows():
            main.result_storage.add_single_test(row)
        main.result_storage.update_fits()
        main.test_environment.parsed = None

    def plot_final_state(self, main):
        """
        Plots the final state a protocol would reach considering all tests.
        :param main: A reference to the main object.
        """
        self.jump_to_final_state(main)
        print(f'Recording: End Reliability intercept:{main.result_storage.rel_fit.intercept_}')
        print(f'Recording: End Reliability coefficients:{main.result_storage.rel_fit.coef_}')
        print(f'Recording: End Latency intercept:{main.result_storage.lat_fit.intercept_}')
        print(f'Recording: End Latency coefficients:{main.result_storage.lat_fit.coef_}')
        print(f'Recording: End Energy intercept:{main.result_storage.nrg_fit.intercept_}')
        print(f'Recording: End Energy coefficients:{main.result_storage.nrg_fit.coef_}')
        print(f'Recording: End Goal intercept:{main.result_storage.goal_fit.coef_}')
        print(f'Recording: End Goal coefficients:{main.result_storage.goal_fit.coef_}')
        PlotUtilities.end_plot_goal_function(main.settings, main.result_storage)

    @staticmethod
    def perform_optima_list_evaluation(main):
        """
        Generates a list of goal values of real tests, sorts them and compares them to the main result.
        :param main: A reference to the main object.
        """
        main_results = main.next_params
        RecordedTestEnvironment.jump_to_final_state(main)
        optimal_params_on_fit = main.goal_evaluator.greedy()
        print(f'Recording: Optimal parameters according to final fit state {optimal_params_on_fit}!')
        optima_list = main.goal_evaluator.brute_force_reality()
        place = 1
        for minimum_value, optimal_params_real in optima_list[0:10]:
            print(f'Recording: Optima List Place {place:2d} is {optimal_params_real} with {minimum_value}!')
            place += 1
        # Print how much'th:
        first_find = None
        place = 1
        for minimum_value, optimal_params_real in optima_list:
            all_params_same = True
            for anon_param, value in optimal_params_real.items():
                if main_results[anon_param] != value:
                    all_params_same = False
            if all_params_same:
                print(f'Recording: Main parameter results are on place {place} of the minima list!')
                if first_find is None:
                    first_find = place
            place += 1
        if first_find is None:
            print(f'Recording: Main parameter results were not found in the top {len(optima_list)} test results!')
