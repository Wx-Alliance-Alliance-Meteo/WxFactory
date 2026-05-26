from abc import ABC, abstractmethod

import numpy


class StepHook(ABC):

    @abstractmethod
    def process(self, Q: numpy.ndarray, t: float) -> numpy.ndarray:
        pass
