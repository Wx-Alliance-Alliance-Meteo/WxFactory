from common.configuration import Configuration
from .integrator            import Integrator

class Euler1(Integrator):
   def __init__(self, param: Configuration, rhs, **kwargs):
      super().__init__(param, **kwargs)
      self.rhs = rhs

   def __step__(self, Q, dt):
      Q = Q + self.rhs(Q) * dt
      return Q
