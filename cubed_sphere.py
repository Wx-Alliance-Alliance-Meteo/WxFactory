import math
import numpy
import sphere
import quadrature
from definitions import *

class Geom:
   def __init__(self, solutionPoints, extension, x1, x2, Δx1, Δx2, X, Y, cartX, cartY, cartZ, lon, lat, X_itf_i, Y_itf_i, X_itf_j, Y_itf_j, lon_itf_i, lat_itf_i, lon_itf_j, lat_itf_j, cube_face):
      self.solutionPoints = solutionPoints
      self.extension = extension
      self.x1 = x1
      self.x2 = x2
      self.Δx1 = Δx1
      self.Δx2 = Δx2
      self.X = X
      self.Y = Y
      self.cartX = cartX
      self.cartY = cartY
      self.cartZ = cartZ
      self.lon = lon
      self.lat = lat
      self.X_itf_i = X_itf_i
      self.Y_itf_i = Y_itf_i
      self.X_itf_j = X_itf_j
      self.Y_itf_j = Y_itf_j
      self.lon_itf_i = lon_itf_i
      self.lat_itf_i = lat_itf_i
      self.lon_itf_j = lon_itf_j
      self.lat_itf_j = lat_itf_j

      self.cube_face = cube_face

      delta2 = 1.0 + X**2 + Y**2
      delta  = numpy.sqrt(delta2)

      if cube_face == 0:
         self.coslon = 1.0 / numpy.sqrt( 1.0 + X**2 )
         self.sinlon = X / numpy.sqrt( 1.0 + X**2 )

         self.coslat = numpy.sqrt( (1.0 + X**2) / delta2 )
         self.sinlat = Y / delta

      elif cube_face == 1:
         self.coslon = -X / numpy.sqrt( 1.0 + X**2 )
         self.sinlon = 1.0 / numpy.sqrt( 1.0 + X**2 )

         self.coslat = numpy.sqrt( (1.0 + X**2) / delta2 )
         self.sinlat = Y / delta

      elif cube_face == 2:
         self.coslon = -1.0 / numpy.sqrt( 1.0 + X**2 )
         self.sinlon = -X / numpy.sqrt( 1.0 + X**2 )

         self.coslat = numpy.sqrt( (1.0 + X**2) / delta2 )
         self.sinlat = Y / delta

      elif cube_face == 3:
         self.coslon = X / numpy.sqrt( 1.0 + X**2 )
         self.sinlon = -1.0 / numpy.sqrt( 1.0 + X**2 )

         self.coslat = numpy.sqrt( (1.0 + X**2) / delta2 )
         self.sinlat = Y / delta

      elif cube_face == 4:
         self.coslon = -Y / numpy.sqrt( X**2 + Y**2 )
         self.sinlon = X / numpy.sqrt( X**2 + Y**2 )

         self.coslat = numpy.sqrt( (X**2 + Y**2) / delta2 )
         self.sinlat = 1.0 / delta

      elif cube_face == 5:
         self.coslon = Y / numpy.sqrt( X**2 + Y**2 )
         self.sinlon = X / numpy.sqrt( X**2 + Y**2 )

         self.coslat = numpy.sqrt( (X**2 + Y**2) / delta2 )
         self.sinlat = -1.0 / delta


def cubed_sphere(nb_elements, nbsolpts, cube_face):

#      +---+
#      | 4 |
#  +---+---+---+---+
#  | 3 | 0 | 1 | 2 |
#  +---+---+---+---+
#      | 5 |
#      +---+

   domain_x1 = (-math.pi/4, math.pi/4)
   domain_x2 = (-math.pi/4, math.pi/4)

   nb_elements_x1 = nb_elements
   nb_elements_x2 = nb_elements

   # Gauss-Legendre solution points
   solutionPoints = quadrature.gauss_legendre(nbsolpts)
   print('Solution points : ', solutionPoints)

   # Extend the solution points to include -1 and 1
   extension = numpy.append(numpy.append([-1], solutionPoints), [1])

   scaled_points = 0.5 * (1.0 + solutionPoints)

   # Equiangular coordinates
   Δx1 = (domain_x1[1] - domain_x1[0]) / nb_elements_x1
   Δx2 = (domain_x2[1] - domain_x2[0]) / nb_elements_x2

   faces_x1 = numpy.linspace(start = domain_x1[0], stop = domain_x1[1], num = nb_elements_x1 + 1)
   faces_x2 = numpy.linspace(start = domain_x2[0], stop = domain_x2[1], num = nb_elements_x2 + 1)

   ni = nb_elements_x1 * len(solutionPoints)
   x1 = numpy.zeros(ni)
   for i in range(nb_elements_x1):
      idx = i * nbsolpts
      x1[idx : idx + nbsolpts] = faces_x1[i] + scaled_points * Δx1

   nj = nb_elements_x2 * len(solutionPoints)
   x2 = numpy.zeros(nj)
   for i in range(nb_elements_x2):
      idx = i * nbsolpts
      x2[idx : idx + nbsolpts] = faces_x2[i] + scaled_points * Δx2

   X1, X2 = numpy.meshgrid(x1, x2)

   X1_itf_i, X2_itf_i = numpy.meshgrid(faces_x1, x2)
   X1_itf_j, X2_itf_j = numpy.meshgrid(x1, faces_x2)

   # Gnomonic coordinates
   X = numpy.tan(X1)
   Y = numpy.tan(X2)

   X_itf_i = numpy.tan(X1_itf_i)
   Y_itf_i = numpy.tan(X2_itf_i)
   X_itf_j = numpy.tan(X1_itf_j)
   Y_itf_j = numpy.tan(X2_itf_j)

   # Spherical coordinates
   lon = numpy.zeros((ni,nj))
   lat = numpy.zeros((ni,nj))

   lon_itf_i = numpy.zeros_like(X1_itf_i)
   lon_itf_j = numpy.zeros_like(X1_itf_j)
   lat_itf_i = numpy.zeros_like(X2_itf_i)
   lat_itf_j = numpy.zeros_like(X2_itf_j)

   # Equatorial panel
   if cube_face < 4:
      lon[:,:] = X1 + math.pi/2.0 * cube_face
      lat[:,:] = numpy.arctan(Y * numpy.cos(X1))

      lon_itf_i[:,:] = X1_itf_i + math.pi/2.0 * cube_face
      lat_itf_i[:,:] = numpy.arctan(Y_itf_i * numpy.cos(X1_itf_i))
      lon_itf_j[:,:] = X1_itf_j + math.pi/2.0 * cube_face
      lat_itf_j[:,:] = numpy.arctan(Y_itf_j * numpy.cos(X1_itf_j))

   # North polar panel
   if cube_face == 4:
      for i in range(ni):
         for j in range(nj):
            if abs(X[i,j]) > numpy.finfo(float).eps :
               lon[i,j] = math.atan2(X[i,j], -Y[i,j])
            elif Y[i,j] <= 0.0:
               lon[i,j] = 0.0
            else:
               lon[i,j] = math.pi
      lat[:,:] = math.pi/2 - numpy.arctan(numpy.sqrt(X**2 + Y**2))

      nni, nnj = lon_itf_i.shape
      for i in range(nni):
         for j in range(nnj):
            if abs(X_itf_i[i,j]) > numpy.finfo(float).eps :
               lon_itf_i[i,j] = math.atan2(X_itf_i[i,j], -Y_itf_i[i,j])
            elif Y_itf_i[i,j] <= 0.0:
               lon_itf_i[i,j] = 0.0
            else:
               lon_itf_i[i,j] = math.pi
      lat_itf_i[:,:] = math.pi/2 - numpy.arctan(numpy.sqrt(X_itf_i**2 + Y_itf_i**2))

      nni, nnj = lon_itf_j.shape
      for i in range(nni):
         for j in range(nnj):
            if abs(X_itf_j[i,j]) > numpy.finfo(float).eps :
               lon_itf_j[i,j] = math.atan2(X_itf_j[i,j], -Y_itf_j[i,j])
            elif Y_itf_j[i,j] <= 0.0:
               lon_itf_j[i,j] = 0.0
            else:
               lon_itf_j[i,j] = math.pi
      lat_itf_j[:,:] = math.pi/2 - numpy.arctan(numpy.sqrt(X_itf_j**2 + Y_itf_j**2))

   # South polar panel
   if cube_face == 5:
      for i in range(ni):
         for j in range(nj):
            if abs(X[i,j]) > numpy.finfo(float).eps :
               lon[i,j] = math.atan2(X[i,j], Y[i,j])
            elif Y[i,j] > 0.0:
               lon[i,j] = 0.0
            else:
               lon[i,j] = math.pi
      lat[:,:] = -math.pi/2 + numpy.arctan(numpy.sqrt(X**2 + Y**2))

      nni, nnj = lon_itf_i.shape
      for i in range(nni):
         for j in range(nnj):
            if abs(X_itf_i[i,j]) > numpy.finfo(float).eps :
               lon_itf_i[i,j] = math.atan2(X_itf_i[i,j], Y_itf_i[i,j])
            elif Y_itf_i[i,j] > 0.0:
               lon_itf_i[i,j] = 0.0
            else:
               lon_itf_i[i,j] = math.pi
      lat_itf_i[:,:] = -math.pi/2 + numpy.arctan(numpy.sqrt(X_itf_i**2 + Y_itf_i**2))

      nni, nnj = lon_itf_j.shape
      for i in range(nni):
         for j in range(nnj):
            if abs(X_itf_j[i,j]) > numpy.finfo(float).eps :
               lon_itf_j[i,j] = math.atan2(X_itf_j[i,j], Y_itf_j[i,j])
            elif Y_itf_j[i,j] > 0.0:
               lon_itf_j[i,j] = 0.0
            else:
               lon_itf_j[i,j] = math.pi
      lat_itf_j[:,:] = -math.pi/2 + numpy.arctan(numpy.sqrt(X_itf_j**2 + Y_itf_j**2))

   # Map to the interval [0, 2 pi]
   lon[lon<0.0] = lon[lon<0.0] + (2.0 * math.pi)

   # Cartesian coordinates on unit sphere
   cartX, cartY, cartZ = sphere.sph2cart(lon, lat, 1.0)

   return Geom(solutionPoints, extension, X1, X2, Δx1, Δx2, X, Y, cartX, cartY, cartZ, lon, lat, X_itf_i, Y_itf_i, X_itf_j, Y_itf_j, lon_itf_i, lat_itf_i, lon_itf_j, lat_itf_j, cube_face)
