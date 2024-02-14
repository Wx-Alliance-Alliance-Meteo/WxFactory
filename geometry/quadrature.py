import math
import numpy
import scipy.special
import sympy
from types import ModuleType
from typing import List, Tuple

from numpy.typing import NDArray

def gauss_legendre(n: int, xp: ModuleType = numpy) \
   -> Tuple[List[sympy.Float], NDArray[numpy.float64], NDArray[numpy.float64]]:
   """Computes the Gauss-Legendre quadrature points (symbolic and numerical) and weights.

   Gauss-Legendre nodes are roots of the Legendre polynomial

   Arguments:
   - `n`: Number of quadrature points
   - `xp`: [Optional] What python module to use for arrays. Default = numpy
   """

   # https://en.wikipedia.org/wiki/Gaussian_quadrature#Gauss%E2%80%93Legendre_quadrature

   n_digits = 34 # equivalent to quadruple precision
   if n <= 5:
      if n == 1:
         points_sym = [sympy.sympify('0')]
         weights  = [2.0]
      elif n == 2:
         points_sym = [sympy.sympify('-1 / sqrt(3)'),
                     sympy.sympify(' 1 / sqrt(3)')]
         weights = [1.0, 1.0]
      elif n == 3:
         points_sym = [sympy.sympify('-sqrt(3 / 5)'),
                     sympy.sympify('0'),
                     sympy.sympify(' sqrt(3 / 5)')]
         weights = [5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0]
      elif n == 4:
         points_sym = [sympy.sympify('-sqrt(2*sqrt(30)/35 + 3/7)'),
                     sympy.sympify('-sqrt(3/7 - 2*sqrt(30)/35)'),
                     sympy.sympify(' sqrt(3/7 - 2*sqrt(30)/35)'),
                     sympy.sympify(' sqrt(2*sqrt(30)/35 + 3/7)')]
         weights = [(18.0 - math.sqrt(30.0)) / 36.0, (18.0 + math.sqrt(30.0)) / 36.0,
                  (18.0 + math.sqrt(30.0)) / 36.0, (18.0 - math.sqrt(30.0)) / 36.0]
      elif n == 5:
         points_sym = [sympy.sympify('-sqrt(2*sqrt(70)/63 + 5/9)'),
                     sympy.sympify('-sqrt(5/9 - 2*sqrt(70)/63)'),
                     sympy.sympify('0'),
                     sympy.sympify(' sqrt(5/9 - 2*sqrt(70)/63)'),
                     sympy.sympify(' sqrt(2*sqrt(70)/63 + 5/9)')]
         weights = [(322.0 - 13.0 * math.sqrt(70.0)) / 900.0,
                  (322.0 + 13.0 * math.sqrt(70.0)) / 900.0,
                     128.0 / 225.0,
                  (322.0 + 13.0 * math.sqrt(70.0)) / 900.0,
                  (322.0 - 13.0 * math.sqrt(70.0)) / 900.0]
      else:
         raise ValueError(f'Invalid n = {n}')

      points_num = xp.array([a.evalf(n_digits, chop=True) for a in points_sym], dtype=float)
   else:
      points_num, weights = scipy.special.roots_legendre(n)
      points_sym = [sympy.Float(n, n_digits) for n in points_num]

   return points_sym, points_num, xp.asarray(weights)
