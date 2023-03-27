from common.program_options import Configuration
from .integrator            import Integrator

class LieSplitting(Integrator):
   def __init__(self, param: Configuration, scheme1: Integrator, scheme2: Integrator):
      super().__init__(param, preconditioner=None)
      self.scheme1 = scheme1
      self.scheme2 = scheme2

   def __step__(self, Q, dt):
      Q1 = self.scheme1.step(Q, dt)
      Q2 = self.scheme2.step(Q1, dt)
      return Q2

class StrangSplitting(Integrator):
   def __init__(self, param: Configuration, scheme1: Integrator, scheme2: Integrator):
      super().__init__(param, preconditioner=None)
      self.scheme1 = scheme1
      self.scheme2 = scheme2

   def __step__(self, Q, dt):
      Q1 = self.scheme1.step(Q, dt/2)
      Q2 = self.scheme2.step(Q1, dt)
      Q3 = self.scheme1.step(Q2, dt/2)
      return Q3
