from integrators.stepper import Stepper

class StrangSplitting(Stepper):
   def __init__(self, scheme1, scheme2):
      super().__init__()
      self.scheme1 = scheme1
      self.scheme2 = scheme2

   def __step__(self, Q, dt):
      Q = self.scheme1.step(Q, dt/2)
      Q = self.scheme2.step(Q, dt)
      return Q
