from common.configuration import Configuration
from .integrator import Integrator


class Neural(Integrator):
    def __init__(self, param: Configuration):
          super().__init__(self, param, preconditioner=None)

    def __step__(self, Q, dt):
        return Q