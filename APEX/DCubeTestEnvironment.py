import os

import numpy as np
import requests
import json
import logging
import yaml
from base64 import b64encode
from http.client import HTTPConnection
from time import sleep

import Utilities
from AbstractTestEnvironment import AbstractTestEnvironment
SLEEP_TICK_TIME = 5  # s
REQUESTS_TIMEOUT = 240  # s
API_URL = 'https://iti-testbed.tugraz.at/api'


class DCubeTestEnvironment(AbstractTestEnvironment):
    def __init__(self, settings):
        super().__init__(settings)
        self.hardware = self._settings['dcubeTestEnvironment']['hardware']
        if self.hardware != 'sky' and self.hardware != 'nrf':
            raise Exception('Only Tmote Sky and nRF52840 hardware are currently implemented.')
        with open('../config/dcubeKey.yaml') as f:
            self.dcube_key = yaml.full_load(f)
            self.api_key = '?key=' + self.dcube_key['apiKey']
        url = API_URL + '/' + self.api_key
        response = requests.get(url, params={'timeout': REQUESTS_TIMEOUT})
        if response.status_code != 200 or response.text != 'this is the api root':
            raise Exception('DCube API is currently not available.')

    @staticmethod
    def debug_requests_on():
        """Switches on logging of the requests module.
        https://stackoverflow.com/questions/16337511/log-all-requests-from-the-python-requests-module
        """
        HTTPConnection.debuglevel = 1
        logging.basicConfig()
        logging.getLogger().setLevel(logging.DEBUG)
        requests_log = logging.getLogger("requests.packages.urllib3")
        requests_log.setLevel(logging.DEBUG)
        requests_log.propagate = True

    @staticmethod
    def request(url, method='GET', dict_for_json=None):
        if method == 'POST':
            response_obj = requests.post(url, json=dict_for_json, params={'timeout': REQUESTS_TIMEOUT})
        else:
            response_obj = requests.get(url, json=dict_for_json, params={'timeout': REQUESTS_TIMEOUT})
        if response_obj.status_code != 200:
            raise Exception(f'D-Cube: URL was:{url}\n'
                            f'D-Cube: Status code was {response_obj.status_code}\n'
                            f'D-Cube: Text was:\n{response_obj.text}\n')
        return json.loads(response_obj.text)

    def append_to_job_list(self, params, job_id):
        """
        Appends another job to the job list in the json-style used for the master project and master thesis.
        :param params: The parameters only as anonymous params as supplied to execute_test().
        """
        storage_path = self._settings['dcubeTestEnvironment']['jobListPath']
        directory = os.path.dirname(os.path.realpath(storage_path))
        os.makedirs(directory, exist_ok=True)
        if os.path.isfile(storage_path):
            with open(self._settings['dcubeTestEnvironment']['jobListPath'], 'r') as file_handle:
                list_of_jobs = json.load(file_handle)
        else:
            list_of_jobs = []
        new_data = {
            'id': job_id,
            'values': Utilities.to_clear_name_dict(self._settings, params)
        }
        list_of_jobs.append(new_data)
        with open(self._settings['dcubeTestEnvironment']['jobListPath'], 'w') as file_handle:
            json.dump(list_of_jobs, file_handle, indent=4)
        print(f'D-Cube: {params}', flush=True)

    def append_to_recording(self, flat_dictionary, params):
        """
        Appends another result to the recording in the json-style used for the master project and master thesis.
        :param flat_dictionary: A flat dictionary with anonymous params and metrics as used within this application.
        :param params: The parameters only as anonymous params as supplied to execute_test().
        """
        storage_path = self._settings['dcubeTestEnvironment']['storagePath']
        directory = os.path.dirname(os.path.realpath(storage_path))
        os.makedirs(directory, exist_ok=True)
        if os.path.isfile(storage_path):
            with open(self._settings['dcubeTestEnvironment']['storagePath'], 'r') as file_handle:
                list_of_recorded_tests = json.load(file_handle)
        else:
            list_of_recorded_tests = []
        metrics = {
            'reliability': flat_dictionary['reliability'],
            'latency': flat_dictionary['latency'],
            'energy': flat_dictionary['energy']
        }
        new_data = {
            'id': flat_dictionary['id'],
            'params': Utilities.to_clear_name_dict(self._settings, params),
            'metrics': metrics
        }
        list_of_recorded_tests.append(new_data)
        with open(self._settings['dcubeTestEnvironment']['storagePath'], 'w') as file_handle:
            json.dump(list_of_recorded_tests, file_handle, indent=4)
        print(f'D-Cube: {params} => {metrics}', flush=True)

    def execute_test(self, params):
        print(f'D-Cube: Working on {params}', flush=True)
        #with open(self._settings['dcubeTestEnvironment']['binaryPath'], 'r') as file_handle:
        #    binary = file_handle.read()
        binary = ''
        with open(self._settings['dcubeTestEnvironment']['binaryPath'], 'r') as file_handle:
            for line in file_handle.readlines():
                binary += line
        with open(self._settings['dcubeTestEnvironment']['customPatchXmlPath'], 'r') as file_handle:
            custom_patch_xml = file_handle.read()
        duration = self._settings['dcubeTestEnvironment']['initTime']
        duration += self._settings['dcubeTestEnvironment']['testTime']
        config_overrides = {
            'start': self._settings['dcubeTestEnvironment']['initTime'],
            'delta': self._settings['dcubeTestEnvironment']['messageValidityDelta']
        }
        request_data = {
            'protocol': self._settings['dcubeTestEnvironment']['protocolId'],
            'name': self._settings['dcubeTestEnvironment']['jobName'],
            'description': self._settings['dcubeTestEnvironment']['jobDescription'],
            'duration': duration,
            'logs': True,  # Serial Capture
            'layout': self._settings['dcubeTestEnvironment']['nodeLayout'],
            'patching': True,  # Binary Patching On/Off Switch
            'periodicity': self._settings['dcubeTestEnvironment']['periodicity'],
            'message_length': self._settings['dcubeTestEnvironment']['messageLength'],
            'jamming': self._settings['dcubeTestEnvironment']['jamming'],
            'cpatch': False,  # Custom Patching On/Off Switch
            'file': b64encode(binary.encode('utf-8')).decode('ascii'),
            'config_overrides': config_overrides
        }
        while True:
            try:
                url = API_URL + '/queue/create_job' + self.api_key
                response = DCubeTestEnvironment.request(url, 'POST', request_data)
                job_id = response['id']
                # check jobListPath is available
                if 'jobListPath' in self._settings['dcubeTestEnvironment']:
                    self.append_to_job_list(params, job_id)
                sleep(duration - 10)
                sleep(2)
                test_done = False
                while not test_done:
                    sleep(SLEEP_TICK_TIME)
                    url = API_URL + '/queue/' + str(job_id) + self.api_key
                    response = DCubeTestEnvironment.request(url)
                    test_done = bool(response['evaluated'])
                return_dict = dict()
                if self.hardware == 'sky':
                    url = API_URL + '/metric/' + str(job_id) + self.api_key
                    response = DCubeTestEnvironment.request(url)
                    return_dict['id'] = job_id
                    return_dict.update(params)
                    return_dict['reliability'] = response['reliability']
                    return_dict['latency'] = response['latency']
                    return_dict['energy'] = response['energy']
                    return_dict['latency'] /= 1000
                if self.hardware == 'nrf':
                    url = API_URL + '/scenario/' + str(job_id) + self.api_key
                    scenarios = DCubeTestEnvironment.request(url)
                    if len(scenarios) == 0:
                        raise Exception(f'D-Cube: Scenarios data was empty!')
                    total_energy = float(scenarios[0]['Total Energy [J]'])
                    setup_energy = float(scenarios[0]['Energy during setup time [J]'])
                    total_sent_list = [int(element['Messages sent to source node']) for element in scenarios]
                    correct_list = [int(element['Correct messages']) for element in scenarios]
                    latency_list = [element['Latency combined [us]'] for element in scenarios]
                    if "None" in latency_list:
                        current_latency = None
                    else:
                        float_latencies = [float(latency_element) for latency_element in latency_list]
                        current_latency = np.mean(np.divide(float_latencies, 1000))
                    return_dict['id'] = job_id
                    return_dict.update(params)
                    return_dict['reliability'] = np.mean(np.divide(correct_list, total_sent_list))
                    return_dict['latency'] = current_latency
                    return_dict['energy'] = total_energy - setup_energy
                    r = return_dict['reliability']
                    l = return_dict['latency']
                    e = return_dict['energy']
                    goal_value = Utilities.evaluate_goal_function(self._settings, r, l, e)
                    return_dict['goal'] = goal_value
            except Exception as ex:
                print(f'D-Cube: Exception during request:\n{ex}', flush=True)
                continue
            self.append_to_recording(return_dict, params)
            return return_dict
