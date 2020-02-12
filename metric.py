import numpy

from definitions import earth_radius, rotation_speed

class Metric:
   def __init__(self, geom):

      delta2 = 1.0 + geom.X**2 + geom.Y**2
      delta  = numpy.sqrt(delta2)

      delta2_itf_i = 1.0 + geom.X_itf_i**2 + geom.Y_itf_i**2
      delta_itf_i  = numpy.sqrt(delta2_itf_i)

      delta2_itf_j = 1.0 + geom.X_itf_j**2 + geom.Y_itf_j**2
      delta_itf_j  = numpy.sqrt(delta2_itf_j)

      # Initialize 2D Jacobian
      self.sqrtG       = earth_radius**2 * (1.0 + geom.X**2) * (1.0 + geom.Y**2) / ( delta2 * delta )
      self.sqrtG_itf_i = earth_radius**2 * (1.0 + geom.X_itf_i**2) * (1.0 + geom.Y_itf_i**2) / ( delta2_itf_i * delta_itf_i )
      self.sqrtG_itf_j = earth_radius**2 * (1.0 + geom.X_itf_j**2) * (1.0 + geom.Y_itf_j**2) / ( delta2_itf_j * delta_itf_j )

      self.inv_sqrtG   = 1.0 / self.sqrtG

      # Initialize 2D contravariant metric
      fact       = delta2 / ( earth_radius**2 * (1.0 + geom.X**2) * (1.0 + geom.Y**2) )
      fact_itf_i = delta2_itf_i / ( earth_radius**2 * (1.0 + geom.X_itf_i**2) * (1.0 + geom.Y_itf_i**2) )
      fact_itf_j = delta2_itf_j / ( earth_radius**2 * (1.0 + geom.X_itf_j**2) * (1.0 + geom.Y_itf_j**2) )

      self.H_contra_11 = fact * (1.0 + geom.Y**2)
      self.H_contra_12 = fact * geom.X * geom.Y
      self.H_contra_21 = self.H_contra_12
      self.H_contra_22 = fact * (1.0 + geom.X**2)

      self.H_contra_11_itf_i = fact_itf_i * (1.0 + geom.Y_itf_i**2)
      self.H_contra_12_itf_i = fact_itf_i * geom.X_itf_i * geom.Y_itf_i
      self.H_contra_21_itf_i = self.H_contra_12_itf_i
      self.H_contra_22_itf_i = fact_itf_i * (1.0 + geom.X_itf_i**2)

      self.H_contra_11_itf_j = fact_itf_j * (1.0 + geom.Y_itf_j**2)
      self.H_contra_12_itf_j = fact_itf_j * geom.X_itf_j * geom.Y_itf_j
      self.H_contra_21_itf_j = self.H_contra_12_itf_j
      self.H_contra_22_itf_j = fact_itf_j * (1.0 + geom.X_itf_j**2)

      # Christoffel symbols

      if geom.cube_face <= 3:
         self.christoffel_1_01 = rotation_speed * geom.X * geom.Y**2 / delta2
         self.christoffel_1_02 =-rotation_speed * geom.Y * (1.0 + geom.Y**2) / delta2

         self.christoffel_1_10 = self.christoffel_1_01
         self.christoffel_1_20 = self.christoffel_1_02

         self.christoffel_2_01 = rotation_speed * geom.Y * (1.0 + geom.X**2) / delta2
         self.christoffel_2_02 =-rotation_speed * geom.X * geom.Y**2 / delta2

         self.christoffel_2_10 = self.christoffel_2_01
         self.christoffel_2_20 = self.christoffel_2_02

      elif geom.cube_face == 4:
         self.christoffel_1_01 = rotation_speed * geom.X * geom.Y / delta2
         self.christoffel_1_02 =-rotation_speed * (1.0 + geom.Y**2) / delta2

         self.christoffel_1_10 = self.christoffel_1_01
         self.christoffel_1_20 = self.christoffel_1_02

         self.christoffel_2_01 = rotation_speed * (1.0 + geom.X**2) / delta2
         self.christoffel_2_02 =-rotation_speed * geom.X * geom.Y / delta2

         self.christoffel_2_10 = self.christoffel_2_01
         self.christoffel_2_20 = self.christoffel_2_02

      elif geom.cube_face == 5:
         self.christoffel_1_01 =-rotation_speed * geom.X * geom.Y / delta2
         self.christoffel_1_02 = rotation_speed * (1.0 + geom.Y**2) / delta2

         self.christoffel_1_10 = self.christoffel_1_01
         self.christoffel_1_20 = self.christoffel_1_02

         self.christoffel_2_01 =-rotation_speed * (1.0 + geom.X**2) / delta2
         self.christoffel_2_02 = rotation_speed * geom.X * geom.Y / delta2

         self.christoffel_2_10 = self.christoffel_2_01
         self.christoffel_2_20 = self.christoffel_2_02


      self.christoffel_1_11 = 2 * geom.X * geom.Y**2 / delta2
      self.christoffel_1_12 = -geom.Y * (1.0 + geom.Y**2) / delta2

      self.christoffel_1_21 = self.christoffel_1_12
      self.christoffel_1_22 = numpy.zeros_like(geom.X)

      self.christoffel_2_11 = numpy.zeros_like(geom.X)
      self.christoffel_2_12 = -geom.X * (1.0 + geom.X**2) / delta2

      self.christoffel_2_21 = self.christoffel_2_12
      self.christoffel_2_22 = 2.0 * geom.X**2 * geom.Y / delta2
