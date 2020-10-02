
class MatvecCaller:

   def __init__(self, function, dt, field, rhs):
      self.function = function
      self.dt = dt
      self.field = field
      self.rhs = rhs

   def __call__(self, vec):
      return self.function(vec, self.dt, self.field, self.rhs)