import math
import numpy
import scipy.special

def gauss_legendre(n):
   """Return Gauss-Legendre nodes.

   Gauss-Legendre nodes are roots of P_n(x).
   """

   # https://en.wikipedia.org/wiki/Gaussian_quadrature#Gauss%E2%80%93Legendre_quadrature

   if n == 1:
      points = [0.0]
   elif n == 2:
      points  = [-1 / math.sqrt(3.0), 1.0 / math.sqrt(3.0)]
   elif n == 3:
      points  = [-math.sqrt(3.0 / 5.0), 0.0, math.sqrt(3.0 / 5.0)]
   elif n == 4:
      a = 2.0 / 7.0 * math.sqrt(6.0 / 5.0)
      points  = [-math.sqrt(3.0 / 7.0 + a), -math.sqrt(3.0/7.0-a), math.sqrt(3.0/7.0-a), math.sqrt(3.0/7.0+a)]
   elif n == 5:
      b = 2 * math.sqrt(10.0 / 7.0)
      points = [-math.sqrt(5.0 + b) / 3.0, -math.sqrt(5.0 - b) / 3.0, 0.0, math.sqrt(5.0 - b) / 3.0, math.sqrt(5.0 + b) / 3.0]
   else:
      points, _ = scipy.special.roots_legendre(n)

   return numpy.array(points)
