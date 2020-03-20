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
   delta  = numpy.sqrt(delta2)

   if geom.cube_face <= 3:

      u = numpy.sqrt(geom.X**2 + 1.0) / delta * u1_contra
      v =  - geom.X * geom.Y * numpy.sqrt(1.0 + geom.X**2) / delta2 * u1_contra \
           + (1.0 + geom.Y**2) * numpy.sqrt(1.0 + geom.X**2) / delta2 * u2_contra

	# North polar panel
#   elif geom.cube_face == 4:
#      hyp2 = geom.X**2 + geom.Y**2
#      hyp = sqrt(hyp2)
#
#      u = - geom.Y * (1.0 + geom.X**2) / hyp2 * u1_contra \
#          + geom.X * (1.0 + geom.Y**2) / hyp2 * u2_contra
#
#      v = - geom.X * (1.0 + geom.X**2) / (delta2 * hyp) * u1_contra \
#          - geom.Y * (1.0 + geom.Y**2) / (delta2 * hyp) * u2_contra
#
#      lat = 0.5 * math.pi - numpy.atan(hyp)
#      u *= lat
#
	# South polar panel
#   elif geom.cube_face == 5:
#      radius = sqrt(geom.X * geom.X + geom.Y * geom.Y)
#
#      u = geom.Y * (1.0 + geom.X * geom.X) / (radius**2) * u1_contra \
#        - geom.X * (1.0 + geom.Y * geom.Y) / (radius**2) * u2_contra
#
#      v = geom.X * (1.0 + geom.X * geom.X) / (delta2 * radius) * u1_contra \
#        + geom.Y * (1.0 + geom.Y * geom.Y) / (delta2 * radius) * u2_contra
#
#      u *= geom.coslat
#
#   u = lambda_dot * earth_radius * geom.coslat
#   v = phi_dot * earth_radius
   else:
      u = numpy.zeros((ni,nj)) # TODO : debug
      v = numpy.zeros((ni,nj))

   u *= earth_radius
   v *= earth_radius

   return u, v
