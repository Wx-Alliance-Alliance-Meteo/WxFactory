import numpy
import math
from definitions import earth_radius

def wind2contra(u, v, geom):
   ni, nj = u.shape

   delta2 = 1.0 + geom.X**2 + geom.Y**2

   # Convert winds coords to spherical basis
   lambda_dot = u / (earth_radius * geom.coslat)
   phi_dot    = v / earth_radius

   if geom.cube_face <= 3:

      u1_contra = lambda_dot

      u2_contra = geom.X * geom.Y / (1.0 + geom.Y**2) * lambda_dot \
                + delta2 / ((1.0 + geom.Y**2) * numpy.sqrt(1.0 + geom.X**2)) * phi_dot

	# North polar panel
   elif geom.cube_face == 4:

	   # Calculate new vector components
      radius = numpy.sqrt(geom.X**2 + geom.Y**2)

      u1_contra = - geom.Y / (1.0 + geom.X**2) * lambda_dot \
	             - delta2 * geom.X / ((1.0 + geom.X**2) * radius) * phi_dot

      u2_contra = geom.X / (1.0 + geom.Y**2) * lambda_dot \
	             - delta2 * geom.Y / ((1.0 + geom.Y**2) * radius) * phi_dot

	# South polar panel
   elif geom.cube_face == 5:

	   # Calculate new vector components
      radius = numpy.sqrt(geom.X**2 + geom.Y**2)

      u1_contra = geom.Y / (1.0 + geom.X**2) * lambda_dot \
	             + delta2 * geom.X / ((1.0 + geom.X**2) * radius) * phi_dot

      u2_contra = - geom.X / (1.0 + geom.Y**2) * lambda_dot \
	             + delta2 * geom.Y / ((1.0 + geom.Y**2) * radius) * phi_dot

   return u1_contra, u2_contra


def contra2wind(u1_contra, u2_contra, geom):

   ni,nj = u1_contra.shape

   delta2 = 1.0 + geom.X**2 + geom.Y**2

   if geom.cube_face <= 3:

      u = u1_contra
      v =  - geom.X * geom.Y * numpy.sqrt(1.0 + geom.X**2) / delta2 * u1_contra \
           + (1.0 + geom.Y**2) * numpy.sqrt(1.0 + geom.X**2) / delta2 * u2_contra

	# North polar panel
   elif geom.cube_face == 4:
      hyp2 = geom.X**2 + geom.Y**2
      hyp = numpy.sqrt(hyp2)

      u = - geom.Y * (1.0 + geom.X**2) / hyp2 * u1_contra \
          + geom.X * (1.0 + geom.Y**2) / hyp2 * u2_contra

      v = - geom.X * (1.0 + geom.X**2) / (delta2 * hyp) * u1_contra \
          - geom.Y * (1.0 + geom.Y**2) / (delta2 * hyp) * u2_contra

	# South polar panel
   elif geom.cube_face == 5:
      hyp2 = geom.X**2 + geom.Y**2
      hyp = numpy.sqrt(hyp2)

      u = geom.Y * (1.0 + geom.X**2) / hyp2 * u1_contra \
        - geom.X * (1.0 + geom.Y**2) / hyp2 * u2_contra

      v = geom.X * (1.0 + geom.X**2) / (delta2 * hyp) * u1_contra \
        + geom.Y * (1.0 + geom.Y**2) / (delta2 * hyp) * u2_contra

   u *= ( geom.coslat * earth_radius )
   v *= earth_radius

   return u, v
