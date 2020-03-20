import numpy

from definitions import earth_radius

def wind2contra(u, v, geom):
   ni, nj = u.shape

   u1_contra = numpy.zeros_like(u)
   u2_contra = numpy.zeros_like(v)

   delta2 = 1.0 + geom.X**2 + geom.Y**2

   coslat = numpy.cos(geom.lat) # TODO : dans geom

   # Convert winds coords to spherical basis
   lambda_dot = u / (earth_radius * coslat)
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
