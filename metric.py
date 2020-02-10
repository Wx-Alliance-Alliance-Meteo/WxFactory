import numpy

from definitions import earth_radius

class Metric:
   def __init__(self, sqrtG, sqrtG_itf_i, sqrtG_itf_j): #, H_contra_11, H_contra_12, H_contra_21, H_contra_22):
      self.sqrtG       = sqrtG
      self.sqrtG_itf_i = sqrtG_itf_i
      self.sqrtG_itf_j = sqrtG_itf_j
#      self.H_contra_11 = H_contra_11
#      self.H_contra_12 = H_contra_12
#      self.H_contra_21 = H_contra_21
#      self.H_contra_22 = H_contra_22

def build_metric(geom):

   delta2 = 1.0 + geom.X**2 + geom.Y**2
   delta  = numpy.sqrt(delta2)

   delta2_itf_i = 1.0 + geom.X_itf_i**2 + geom.Y_itf_i**2
   delta_itf_i  = numpy.sqrt(delta2_itf_i)

   delta2_itf_j = 1.0 + geom.X_itf_j**2 + geom.Y_itf_j**2
   delta_itf_j  = numpy.sqrt(delta2_itf_j)

   # Initialize 2D Jacobian
#   sqrtG = earth_radius**2 * (1.0 + geom.X**2) * (1.0 + geom.Y**2) / ( delta2 * delta )
   sqrtG = (1.0 + geom.X**2) * (1.0 + geom.Y**2) / ( delta2 * delta )
   sqrtG_itf_i = (1.0 + geom.X_itf_i**2) * (1.0 + geom.Y_itf_i**2) / ( delta2_itf_i * delta_itf_i )
   sqrtG_itf_j = (1.0 + geom.X_itf_j**2) * (1.0 + geom.Y_itf_j**2) / ( delta2_itf_j * delta_itf_j )

   # Initialize 2D contravariant metric
#   fact = delta2 / ( earth_radius**2 * (1.0 + geom.X**2) * (1.0 + geom.Y**2) )
#
#   H_contra_11 = fact * (1.0 + geom.Y**2)
#   H_contra_12 = fact * geom.X * geom.Y
#   H_contra_21 = H_contra_12
#   H_contra_22 = fact * (1.0 + geom.X**2)


   # reference element
   sqrtG *= geom.Δx1 * geom.Δx2 / 4.0

   sqrtG_itf_i *= geom.Δx1 * geom.Δx2 / 4.0
   sqrtG_itf_j *= geom.Δx1 * geom.Δx2 / 4.0
   
   return Metric(sqrtG, sqrtG_itf_i, sqrtG_itf_j) #, H_contra_11 , H_contra_12, H_contra_21, H_contra_22)
