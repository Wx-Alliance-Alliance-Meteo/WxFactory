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

class OS22Splitting(Integrator):
    def __init__(self, param: Configuration, scheme1: Integrator, scheme2: Integrator, os_param):
      super().__init__()
      self.scheme1 = scheme1
      self.scheme2 = scheme2
      self.os_param = os_param
      self.alpha = numpy.array([[(2 * self.os_param - 1) / (2 * self.os_param - 2), 1 - self.os_param],
                             [-1 / (2 * self.os_param - 2), self.os_param]])

    def __step__(self, Q, dt):
      for numofstage in range(0,self.alpha.shape[0]):
            if self.alpha[numofstage,0] != 0:
                  Q = self.scheme1.step(Q, self.alpha[numofstage,0] * dt)
            if self.alpha[numofstage,1] != 0:
                  Q = self.scheme2.step(Q, self.alpha[numofstage,1] * dt)
      return Q

