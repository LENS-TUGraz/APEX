from abc import ABC, abstractmethod


class AbstractTestEnvironment(ABC):
    def __init__(self, settings):
        self._settings = settings

    @abstractmethod
    def execute_test(self, params):
        pass
