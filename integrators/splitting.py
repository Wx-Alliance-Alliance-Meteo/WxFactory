from common.program_options import Configuration
from .integrator            import Integrator

class StrangSplitting(Integrator):
   def __init__(self, param: Configuration, scheme1: Integrator, scheme2: Integrator):
      super().__init__(param, preconditioner=None)
      self.scheme1 = scheme1
      self.scheme2 = scheme2

   def __step__(self, Q, dt):
      Q = self.scheme1.step(Q, dt/2)
      Q = self.scheme2.step(Q, dt)
      return Q
