import numpy
import math

from definitions import earth_radius, rotation_speed

class Metric:
   def __init__(self, geom):

      # 2D Jacobian

      self.sqrtG       = earth_radius**2 * (1.0 + geom.X**2) * (1.0 + geom.Y**2) / ( geom.delta2 * geom.delta )
      self.sqrtG_itf_i = earth_radius**2 * (1.0 + geom.X_itf_i**2) * (1.0 + geom.Y_itf_i**2) / ( geom.delta2_itf_i * geom.delta_itf_i )
      self.sqrtG_itf_j = earth_radius**2 * (1.0 + geom.X_itf_j**2) * (1.0 + geom.Y_itf_j**2) / ( geom.delta2_itf_j * geom.delta_itf_j )

      self.inv_sqrtG   = 1.0 / self.sqrtG

      self.dsqrtGdx1 = earth_radius**2 * geom.X * (1.0 + geom.X**2) * (1.0 + geom.Y**2) * (2.0 * geom.Y**2 - geom.X**2 - 1.0) / ( geom.delta**5 )
      self.dsqrtGdx2 = earth_radius**2 * geom.Y * (1.0 + geom.X**2) * (1.0 + geom.Y**2) * (2.0 * geom.X**2 - geom.Y**2 - 1.0) / ( geom.delta**5 )

      # 2D contravariant metric

      self.H_contra_11 = geom.delta2 / ( earth_radius**2 * (1.0 + geom.X**2) )
      self.H_contra_12 = geom.delta2 * geom.X * geom.Y / ( earth_radius**2 * (1.0 + geom.X**2) * (1.0 + geom.Y**2) )
      self.H_contra_21 = self.H_contra_12
      self.H_contra_22 = geom.delta2 / ( earth_radius**2 * (1.0 + geom.Y**2) )

      self.H_contra_11_itf_i = geom.delta2_itf_i / ( earth_radius**2 * (1.0 + geom.X_itf_i**2) )
      self.H_contra_12_itf_i = geom.delta2_itf_i * geom.X_itf_i * geom.Y_itf_i / ( earth_radius**2 * (1.0 + geom.X_itf_i**2) * (1.0 + geom.Y_itf_i**2) )
      self.H_contra_21_itf_i = self.H_contra_12_itf_i
      self.H_contra_22_itf_i = geom.delta2_itf_i / ( earth_radius**2 * (1.0 + geom.Y_itf_i**2) )

      self.H_contra_11_itf_j = geom.delta2_itf_j / ( earth_radius**2 * (1.0 + geom.X_itf_j**2) )
      self.H_contra_12_itf_j = geom.delta2_itf_j * geom.X_itf_j * geom.Y_itf_j / ( earth_radius**2 * (1.0 + geom.X_itf_j**2) * (1.0 + geom.Y_itf_j**2) )
      self.H_contra_21_itf_j = self.H_contra_12_itf_j
      self.H_contra_22_itf_j = geom.delta2_itf_j / ( earth_radius**2 * (1.0 + geom.Y_itf_j**2) )

      # 2D covariant metric

      fact = earth_radius**2 / geom.delta**4
      self.H_cov_11 = fact * (1 + geom.X**2)**2 * (1 + geom.Y**2)
      self.H_cov_12 = - fact * geom.X * geom.Y * (1 + geom.X**2) * (1 + geom.Y**2)
      self.H_cov_21 = self.H_cov_12
      self.H_cov_22 = fact * (1 + geom.X**2) * (1 + geom.Y**2)**2

      # Christoffel symbols

      gridrot = math.sin(geom.lat_p) - geom.X * math.cos(geom.lat_p) * math.sin(geom.angle_p) + geom.Y * math.cos(geom.lat_p) * math.cos(geom.angle_p)

      self.christoffel_1_01 = rotation_speed * geom.X * geom.Y / geom.delta2 * gridrot
      self.christoffel_1_10 = self.christoffel_1_01

      self.christoffel_1_02 = -rotation_speed * (1.0 + geom.Y**2) / geom.delta2 * gridrot
      self.christoffel_1_20 = self.christoffel_1_02

      self.christoffel_2_01 = rotation_speed * (1.0 + geom.X**2) / geom.delta2 * gridrot
      self.christoffel_2_10 = self.christoffel_2_01
      
      self.christoffel_2_02 =-rotation_speed * geom.X * geom.Y / geom.delta2 * gridrot
      self.christoffel_2_20 = self.christoffel_2_02

      self.christoffel_1_11 = 2 * geom.X * geom.Y**2 / geom.delta2
      self.christoffel_1_12 = - (geom.Y + geom.Y**3) / geom.delta2

      self.christoffel_1_21 = self.christoffel_1_12
      self.christoffel_1_22 = numpy.zeros_like(geom.X)

      self.christoffel_2_11 = numpy.zeros_like(geom.X)
      self.christoffel_2_12 = -geom.X * (1.0 + geom.X**2) / geom.delta2

      self.christoffel_2_21 = self.christoffel_2_12
      self.christoffel_2_22 = 2.0 * geom.X**2 * geom.Y / geom.delta2
