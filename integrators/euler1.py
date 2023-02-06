from integrators.stepper import Stepper

class Euler1(Stepper):
   def __init__(self, rhs):
      super().__init__()
      self.rhs = rhs

   def __step__(self, Q, dt):
      Q = Q + self.rhs(Q) * dt
      return Q
