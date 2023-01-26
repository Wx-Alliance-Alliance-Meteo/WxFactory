from Stepper.stepper import Stepper

class Tvdrk3(Stepper):
   def __init__(self, rhs):
      super().__init__()
      self.rhs = rhs

   def __step__(self, Q, dt):
      Q1 = Q + self.rhs(Q) * dt
      #Q = Q + self.rhs(Q) * dt
      Q2 = 0.75 * Q + 0.25 * Q1 + 0.25 * self.rhs(Q1) * dt
      Q = 1.0 / 3.0 * Q + 2.0 / 3.0 * Q2 + 2.0 / 3.0 * self.rhs(Q2) * dt
      return Q