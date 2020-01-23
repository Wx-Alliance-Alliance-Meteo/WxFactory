import numpy

from constants import earth_radius

class Metric:
   def __init__(self, sqrtG, H_contra_11, H_contra_12, H_contra_21, H_contra_22):
      self.sqrtG       = sqrtG
      self.H_contra_11 = H_contra_11
      self.H_contra_12 = H_contra_12
      self.H_contra_21 = H_contra_21
      self.H_contra_22 = H_contra_22

def build_metric(geom):

   delta2 = 1.0 + geom.X**2 + geom.Y**2
   delta = numpy.sqrt(delta2)

   # Initialize 2D Jacobian
   sqrtG = earth_radius**2 * (1.0 + geom.X**2) * (1.0 + geom.Y**2) / delta**3

   # Initialize 2D contravariant metric
   fact = delta2 / ( earth_radius**2 * (1.0 + geom.X**2) * (1.0 + geom.Y**2) )

   H_contra_11 = fact * (1.0 + geom.Y**2)
   H_contra_12 = fact * geom.X * geom.Y
   H_contra_21 = H_contra_12
   H_contra_22 = fact * (1.0 + geom.X**2)

   return Metric(sqrtG, H_contra_11, H_contra_12, H_contra_21, H_contra_22)
