import math
import numpy
import sphere
import quadrature
from constants import *

class Geom:
  def __init__(self, solutionPoints, extension, x1, x2, Δx1, Δx2, X, Y, cartX, cartY, cartZ, lon, lat):
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

def cubed_sphere(nb_elements, degree):

   domain_x1 = (-math.pi/4, math.pi/4)
   domain_x2 = (-math.pi/4, math.pi/4)

   nb_elements_x1 = nb_elements
   nb_elements_x2 = nb_elements

   # Gauss-Legendre solution points
   solutionPoints, _ = quadrature.gauss_legendre(degree+1)
   print('Solution points : ', solutionPoints)

   nb_solpts = len(solutionPoints)

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
      idx = i * nb_solpts
      x1[idx : idx + nb_solpts] = faces_x1[i] + scaled_points * Δx1

   nj = nb_elements_x2 * len(solutionPoints)
   x2 = numpy.zeros(nj)
   for i in range(nb_elements_x2):
      idx = i * nb_solpts
      x2[idx : idx + nb_solpts] = faces_x2[i] + scaled_points * Δx2

   X1, X2 = numpy.meshgrid(x1, x2)

   # Gnomonic coordinates
   X = numpy.tan(X1)
   Y = numpy.tan(X2)

   # Cartesian coordinates on unit sphere
   cartX = numpy.zeros((ni,nj,nbfaces))
   cartY = numpy.zeros((ni,nj,nbfaces))
   cartZ = numpy.zeros((ni,nj,nbfaces))

   R = 1.0
   a = R / math.sqrt(3)
   x = a * numpy.tan(X1)
   y = a * numpy.tan(X2)
   proj = R / numpy.sqrt(a**2 + x**2 + y**2)

   cartX[:,:,0] = proj * a
   cartY[:,:,0] = proj * x
   cartZ[:,:,0] = proj * y

   cartX[:,:,1] = proj * -x
   cartY[:,:,1] = proj * a
   cartZ[:,:,1] = proj * y

   cartX[:,:,2] = proj * -a
   cartY[:,:,2] = proj * -x
   cartZ[:,:,2] = proj * y

   cartX[:,:,3] = proj * x
   cartY[:,:,3] = proj * -a
   cartZ[:,:,3] = proj * y

   cartX[:,:,4] = proj * -y
   cartY[:,:,4] = proj * x
   cartZ[:,:,4] = proj * a

   cartX[:,:,5] = proj * y
   cartY[:,:,5] = proj * x
   cartZ[:,:,5] = proj * -a

   # Spherical coordinates
   lon = numpy.zeros((ni,nj,nbfaces))
   lat = numpy.zeros((ni,nj,nbfaces))

   for pannel in range(nbfaces):
      lon[:,:,pannel] = X1 + math.pi/2.0 * (pannel - 1)
      lat[:,:,pannel] = numpy.arctan(Y * numpy.cos(X1))

   lon[:,:,-2:-1], lat[:,:,-2:-1], r = sphere.cart2sph(cartX[:,:,-2:-1], cartY[:,:,-2:-1], cartZ[:,:,-2:-1])


   lon[lon<0] = lon[lon<0] + (2.0 * math.pi)

   return Geom(solutionPoints, extension, X1, X2, Δx1, Δx2, X, Y, cartX, cartY, cartZ, lon, lat)
