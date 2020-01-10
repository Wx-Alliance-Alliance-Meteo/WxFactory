import math
import numpy

def gauss_legendre(n):

   if n == 1:
      points  = [0.0]
      weights = [2.0]

   elif n == 2:
      points  = [-1 / math.sqrt(3.0), 1.0 / math.sqrt(3.0)]
      weights = [1.0, 1.0]

   elif n == 3:
      points  = [-math.sqrt(3.0 / 5.0), 0.0, math.sqrt(3.0 / 5.0)]
      weights = [5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0]

   elif n == 4:
      a = 2.0 / 7.0 * math.sqrt(6.0 / 5.0)
      points  = [-math.sqrt(3.0 / 7.0 + a), -math.sqrt(3.0/7.0-a), math.sqrt(3.0/7.0-a), math.sqrt(3.0/7.0+a)]
      weights = [(18.0 - math.sqrt(30.0)) / 36.0, (18.0 + math.sqrt(30.0)) / 36.0, (18.0 + math.sqrt(30.0)) / 36.0, (18.0 - math.sqrt(30.0)) / 36.0]

   elif n == 5:
      b = 2 * math.sqrt(10.0 / 7.0)
      points = [-math.sqrt(5.0 + b) / 3.0, -math.sqrt(5.0 - b) / 3.0, 0.0, math.sqrt(5.0 - b) / 3.0, math.sqrt(5.0 + b) / 3.0]
      weights = [(322.0 - 13.0 * math.sqrt(70.0)) / 900.0, (322.0 + 13.0 * math.sqrt(70.0)) / 900.0, 128.0 / 225.0, \
                 (322.0 + 13.0 * math.sqrt(70.0)) / 900.0, (322.0 - 13.0 * math.sqrt(70.0)) / 900.0]

   else:
         points, weights = numpy.polynomial.legendre.leggauss(n)

   return numpy.array(points), numpy.array(weights)
