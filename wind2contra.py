import math
import numpy

def wind2contra(u, v, geom):
   ni,nj,nbfaces = u.shape

   u1_contra = numpy.zeros_like(u)
   u2_contra = numpy.zeros_like(v)

   for face in range(nbfaces):
      for j in range(nj):
         for i in range(ni):
            dX = geom.X[i,j]
            dY = geom.Y[i,j]
            delta2 = 1.0 + dX * dX + dY * dY

            if (face > 3) and (abs(dX) < 1.0e-13) and (abs(dY) < 1.0e-13):
               if face == 4:
                  u1_contra[i,j,face] = u[i,j,face]
               else:
                  u1_contra[i,j,face] = -u[i,j,face]
               u2_contra[i,j,face] = v[i,j,face]

            if face <= 3:
               # Convert spherical coords to geometric basis
               uu = u[i,j,face] / math.cos(geom.lat[i,j,face])

               # Calculate new vector components
               u1_contra[i,j,face] = uu

               u2_contra[i,j,face] = dX * dY / (1.0 + dY * dY) * uu \
				                       + delta2 / ((1.0 + dY * dY) * math.sqrt(1.0 + dX * dX)) * v[i,j,face]
		      # North polar panel
            if face == 4:
			      # Convert spherical coords to geometric basis
               uu = u[i,j,face] / math.cos(geom.lat[i,j,face])

			      # Calculate new vector components
               radius = math.sqrt(dX * dX + dY * dY)

               u1_contra[i,j,face] = - dY / (1.0 + dX * dX) * uu \
				                       - delta2 * dX / ((1.0 + dX * dX) * radius) * v[i,j,face]

               u2_contra[i,j,face] = dX / (1.0 + dY * dY) * uu \
				                       - delta2 * dY / ((1.0 + dY * dY) * radius) * v[i,j,face]

		      # South polar panel
            if face == 5:
			      # Convert spherical coords to geometric basis
               uu = u[i,j,face] / math.cos(geom.lat[i,j,face])

			      # Calculate new vector components
               radius = math.sqrt(dX * dX + dY * dY)

               u1_contra[i,j,face] = dY / (1.0 + dX * dX) * uu \
				                       + delta2 * dX / ((1.0 + dX * dX) * radius) * v[i,j,face]

               u2_contra[i,j,face] = - dX / (1.0 + dY * dY) * uu \
				                       + delta2 * dY / ((1.0 + dY * dY) * radius) * v[i,j,face]

   return u1_contra, u2_contra
