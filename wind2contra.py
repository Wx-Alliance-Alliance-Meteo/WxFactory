import math
import numpy

def wind2contra(u, v, geom):
   ni,nj,nbfaces = u.shape

   u1_contra = numpy.zeros_like(u)
   u2_contra = numpy.zeros_like(v)

   for face in range(4):
      for j in range(nj):
         for i in range(ni):
            delta2 = 1 + geom.X[i,j]**2 + geom.Y[i,j]**2
            u1_contra[i,j,face] = u[i,j,face]
            u2_contra[i,j,face] = ( geom.X[i,j] * geom.Y[i,j] / (1 + geom.Y[i,j]**2) ) * u[i,j,face] \
                                + ((1 + geom.Y[i,j]**2) * math.sqrt(1 + geom.X[i,j]**2) / delta2) * v[i,j,face]

   face = 4
   s = 1
   for j in range(nj):
      for i in range(ni):
         delta2 = 1 + geom.X[i,j]**2 + geom.Y[i,j]**2
         u1_contra[i,j,face] = (-s * geom.Y[i,j] / (1 + geom.X[i,j]**2)) * u[i,j,face] \
                             + (-s * delta2 * geom.X[i,j] / ((1 + geom.X[i,j]**2) * math.sqrt(geom.X[i,j]**2 + geom.Y[i,j]**2))) * v[i,j,face]

         u2_contra[i,j,face] = ( s * geom.X[i,j] / (1 + geom.Y[i,j]**2)) * u[i,j,face] \
                             + ((-s * delta2 * geom.Y[i,j]) / ((1+geom.Y[i,j]**2) * math.sqrt(geom.X[i,j]**2 + geom.Y[i,j]**2))) * v[i,j,face]

   face = 5
   s = -1
   for j in range(nj):
      for i in range(ni):
         delta2 = 1 + geom.X[i,j]**2 + geom.Y[i,j]**2
         u1_contra[i,j,face] = (-s * geom.Y[i,j] / (1 + geom.X[i,j]**2)) * u[i,j,face] \
                             + (-s * delta2 * geom.X[i,j] / ((1 + geom.X[i,j]**2) * math.sqrt(geom.X[i,j]**2 + geom.Y[i,j]**2))) * v[i,j,face]

         u2_contra[i,j,face] = ( s * geom.X[i,j] / (1 + geom.Y[i,j]**2)) * u[i,j,face] \
                             + ((-s * delta2 * geom.Y[i,j]) / ((1+geom.Y[i,j]**2) * math.sqrt(geom.X[i,j]**2 + geom.Y[i,j]**2))) * v[i,j,face]
   return u1_contra, u2_contra
