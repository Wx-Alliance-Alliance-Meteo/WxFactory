import numpy
from   mpi4py import MPI

from .cubed_sphere_2d import CubedSphere2D

class Metric:
   '''Metric for a smooth, three-dimensional earthlike cubed-sphere with the shallow atmosphere approximation'''
   def __init__(self, geom: CubedSphere2D):
      # 3D Jacobian, for the cubed-sphere mapping
      # Note that with no topography, ∂z/∂η=1; the model top is included
      # inside the geometry definition, and η=x3

      self.sqrtG       = geom.earth_radius**2 * (1.0 + geom.X**2) * (1.0 + geom.Y**2) / ( geom.delta2 * geom.delta )
      self.sqrtG_itf_i = geom.earth_radius**2 * (1.0 + geom.X_itf_i**2) * (1.0 + geom.Y_itf_i**2) / ( geom.delta2_itf_i * geom.delta_itf_i )
      self.sqrtG_itf_j = geom.earth_radius**2 * (1.0 + geom.X_itf_j**2) * (1.0 + geom.Y_itf_j**2) / ( geom.delta2_itf_j * geom.delta_itf_j )

      self.inv_sqrtG   = 1.0 / self.sqrtG

      # 2D contravariant metric

      self.H_contra_11 = geom.delta2 / ( geom.earth_radius**2 * (1.0 + geom.X**2) )
      self.H_contra_12 = geom.delta2 * geom.X * geom.Y / ( geom.earth_radius**2 * (1.0 + geom.X**2) * (1.0 + geom.Y**2) )

      self.H_contra_21 = self.H_contra_12.copy()
      self.H_contra_22 = geom.delta2 / ( geom.earth_radius**2 * (1.0 + geom.Y**2) )

      # Metric at interfaces
      self.H_contra_11_itf_i = geom.delta2_itf_i / ( geom.earth_radius**2 * (1.0 + geom.X_itf_i**2) )
      self.H_contra_12_itf_i = geom.delta2_itf_i * geom.X_itf_i * geom.Y_itf_i / \
                               ( geom.earth_radius**2 * (1.0 + geom.X_itf_i**2) * (1.0 + geom.Y_itf_i**2) )

      self.H_contra_21_itf_i = self.H_contra_12_itf_i.copy()
      self.H_contra_22_itf_i = geom.delta2_itf_i / ( geom.earth_radius**2 * (1.0 + geom.Y_itf_i**2) )

      self.H_contra_11_itf_j = geom.delta2_itf_j / ( geom.earth_radius**2 * (1.0 + geom.X_itf_j**2) )
      self.H_contra_12_itf_j = geom.delta2_itf_j * geom.X_itf_j * geom.Y_itf_j / \
                               ( geom.earth_radius**2 * (1.0 + geom.X_itf_j**2) * (1.0 + geom.Y_itf_j**2) )

      self.H_contra_21_itf_j = self.H_contra_12_itf_j.copy()
      self.H_contra_22_itf_j = geom.delta2_itf_j / ( geom.earth_radius**2 * (1.0 + geom.Y_itf_j**2) )

      # 2D covariant metric

      fact = geom.earth_radius**2 / geom.delta**4
      self.H_cov_11 = fact * (1 + geom.X**2)**2 * (1 + geom.Y**2)
      self.H_cov_12 = - fact * geom.X * geom.Y * (1 + geom.X**2) * (1 + geom.Y**2)
      
      self.H_cov_21 = self.H_cov_12.copy()
      self.H_cov_22 = fact * (1 + geom.X**2) * (1 + geom.Y**2)**2

      # Christoffel symbols

      gridrot = numpy.sin(geom.lat_p) - \
                geom.X * numpy.cos(geom.lat_p) * numpy.sin(geom.angle_p) + \
                geom.Y * numpy.cos(geom.lat_p) * numpy.cos(geom.angle_p)

      self.christoffel_1_01 = geom.rotation_speed * geom.X * geom.Y / geom.delta2 * gridrot
      self.christoffel_1_10 = self.christoffel_1_01.copy()

      self.christoffel_1_02 = -geom.rotation_speed * (1.0 + geom.Y**2) / geom.delta2 * gridrot
      self.christoffel_1_20 = self.christoffel_1_02.copy()

      self.christoffel_2_01 = geom.rotation_speed * (1.0 + geom.X**2) / geom.delta2 * gridrot
      self.christoffel_2_10 = self.christoffel_2_01.copy()

      self.christoffel_2_02 =-geom.rotation_speed * geom.X * geom.Y / geom.delta2 * gridrot
      self.christoffel_2_20 = self.christoffel_2_02.copy()

      self.christoffel_1_11 = 2 * geom.X * geom.Y**2 / geom.delta2

      self.christoffel_1_12 = - (geom.Y + geom.Y**3) / geom.delta2
      self.christoffel_1_21 = self.christoffel_1_12.copy()

      self.christoffel_1_22 = 0

      self.christoffel_2_11 = 0

      self.christoffel_2_12 = -geom.X * (1.0 + geom.X**2) / geom.delta2
      self.christoffel_2_21 = self.christoffel_2_12.copy()

      self.christoffel_2_22 = 2.0 * geom.X**2 * geom.Y / geom.delta2


      # Coriolis parameter
      self.coriolis_f = 2 * geom.rotation_speed / geom.delta * \
                        ( numpy.sin(geom.lat_p) -  \
                          geom.X * numpy.cos(geom.lat_p) * numpy.sin(geom.angle_p) +  \
                          geom.Y * numpy.cos(geom.lat_p) * numpy.cos(geom.angle_p))

      # Now, apply the conversion to the reference element.  In geometric coordinates,
      # x1 and x2 run from -pi/4 to pi/4 and x3 runs from 0 to htop, but inside each elemenet
      # (from the perspective of the differentiation matrices) each coordinate runs from -1 to 1.

      self.sqrtG *= geom.delta_x1 * geom.delta_x2 / 8.
      self.sqrtG_itf_i *= geom.delta_x1 * geom.delta_x2 / 8.
      self.sqrtG_itf_j *= geom.delta_x1 * geom.delta_x2 / 8.

      self.inv_sqrtG   = 1.0 / self.sqrtG

      self.H_cov_11 *= geom.delta_x1**2 / 4.
      self.H_cov_12 *= geom.delta_x1 * geom.delta_x2 / 4.
      self.H_cov_21 *= geom.delta_x2 * geom.delta_x1 / 4.
      self.H_cov_22 *= geom.delta_x2**2 / 4.

      self.H_contra_11 *= 4. / (geom.delta_x1**2)
      self.H_contra_12 *= 4. / (geom.delta_x1 * geom.delta_x2)

      self.H_contra_21 *= 4. / (geom.delta_x1 * geom.delta_x2)
      self.H_contra_22 *= 4. / (geom.delta_x2**2)

      self.H_contra_11_itf_i *= 4. / (geom.delta_x1**2)
      self.H_contra_12_itf_i *= 4. / (geom.delta_x1 * geom.delta_x2)

      self.H_contra_21_itf_i *= 4. / (geom.delta_x1 * geom.delta_x2)
      self.H_contra_22_itf_i *= 4. / (geom.delta_x2**2)

      self.H_contra_11_itf_j *= 4. / (geom.delta_x1**2)
      self.H_contra_12_itf_j *= 4. / (geom.delta_x1 * geom.delta_x2)

      self.H_contra_21_itf_j *= 4. / (geom.delta_x1 * geom.delta_x2)
      self.H_contra_22_itf_j *= 4. / (geom.delta_x2**2)
      
      self.christoffel_1_11 *= 0.5 * geom.delta_x1
      self.christoffel_1_12 *= 0.5 * geom.delta_x1
      self.christoffel_1_21 *= 0.5 * geom.delta_x1
      self.christoffel_1_22 *= 0.5 * geom.delta_x1

      self.christoffel_2_11 *= 0.5 * geom.delta_x2
      self.christoffel_2_12 *= 0.5 * geom.delta_x2
      self.christoffel_2_21 *= 0.5 * geom.delta_x2
      self.christoffel_2_22 *= 0.5 * geom.delta_x2
