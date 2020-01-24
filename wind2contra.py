import numpy

def wind2contra(u, v, geom):
   ni, nj = u.shape

   u1_contra = numpy.zeros_like(u)
   u2_contra = numpy.zeros_like(v)

   delta2 = 1.0 + geom.X**2 + geom.Y**2

   if geom.cube_face <= 3:
      # Convert spherical coords to geometric basis
      uu = u / numpy.cos(geom.lat)

      # Calculate new vector components
      u1_contra = uu

      u2_contra = geom.X * geom.Y / (1.0 + geom.Y**2) * uu \
                + delta2 / ((1.0 + geom.Y**2) * numpy.sqrt(1.0 + geom.X**2)) * v

	# North polar panel
   elif geom.cube_face == 4:
	   # Convert spherical coords to geometric basis
      uu = u / numpy.cos(geom.lat)

	   # Calculate new vector components
      radius = numpy.sqrt(geom.X**2 + geom.Y**2)

      u1_contra = - geom.Y / (1.0 + geom.X**2) * uu \
	             - delta2 * geom.X / ((1.0 + geom.X**2) * radius) * v

      u2_contra = geom.X / (1.0 + geom.Y**2) * uu \
	             - delta2 * geom.Y / ((1.0 + geom.Y**2) * radius) * v

      for j in range(nj):
         for i in range(ni):
            if abs(geom.X[i,j]) < 1.0e-13 and abs(geom.Y[i,j]) < 1.0e-13:
               u1_contra[i,j] = u[i,j]
               u2_contra[i,j] = v[i,j]

	# South polar panel
   elif geom.cube_face == 5:
	   # Convert spherical coords to geometric basis
      uu = u / numpy.cos(geom.lat)

	   # Calculate new vector components
      radius = numpy.sqrt(geom.X**2 + geom.Y**2)

      u1_contra = geom.Y / (1.0 + geom.X * geom.X) * uu \
	             + delta2 * geom.X / ((1.0 + geom.X * geom.X) * radius) * v

      u2_contra = - geom.X / (1.0 + geom.Y * geom.Y) * uu \
	             + delta2 * geom.Y / ((1.0 + geom.Y * geom.Y) * radius) * v

      for j in range(nj):
         for i in range(ni):
            if abs(geom.X[i,j]) < 1.0e-13 and abs(geom.Y[i,j]) < 1.0e-13:
               u1_contra[i,j] = -u[i,j]
               u2_contra[i,j] = v[i,j]

   return u1_contra, u2_contra
