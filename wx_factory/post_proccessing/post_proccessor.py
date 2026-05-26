from abc import ABC, abstractmethod

import numpy


class PostProcessor(ABC):

    @abstractmethod
    def process(self, Q: numpy.ndarray, t: float) -> numpy.ndarray:
        pass