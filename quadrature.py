import math
import numpy
import scipy.special
import sympy
import mpmath

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

def gauss_lobatto(n):
   """Return Gauss-Lobatto quadrature nodes.

   Gauss-Lobatto nodes are roots of P'_{n-1}(x) and -1, 1.
   """

   # https://en.wikipedia.org/wiki/Gaussian_quadrature#Gauss.E2.80.93Lobatto_rules

   if n == 1:
      print("Lobatto undefined for n = 1.")
      exit(1)
   elif n == 2:
      points = [-1.0, 1.0]
   elif n == 3:
      points = [-1.0, 0.0, 1.0]
   elif n == 4:
      points = [-1.0, -math.sqrt(1.0 / 5.0), math.sqrt(1.0 / 5.0), 1.0]
   elif n == 5:
      points = [-1.0, -math.sqrt(3.0 / 7.0), 0.0, math.sqrt(3.0 / 7.0), 1.0]
   elif n == 6:
      sqrt7 = math.sqrt(7.0)
      points = [-1.0, -math.sqrt( 1.0 / 3.0 + 2.0 * sqrt7 / 21.0 ), -math.sqrt( 1.0 / 3.0 - 2.0 * sqrt7 / 21.0 ), \
            math.sqrt( 1.0 / 3.0 - 2.0 * sqrt7 / 21.0 ), math.sqrt( 1.0 / 3.0 + 2.0 * sqrt7 / 21.0 ), 1.0]
   else:
      x = sympy.var('x')
      p = legendre_poly(n-1).diff(x)
      r = find_roots(p)
      points = sorted([mpmath.mpf('-1.0'), mpmath.mpf('1.0')] + r)

   return numpy.array(points)

def legendre_poly(n):
   """Return Legendre polynomial P_n(x).

   :param n: polynomial degree
   """

   x = sympy.var('x')
   p = (1.0*x**2 - 1.0)**n

   top = p.diff(x, n)
   bot = 2**n * 1.0*sympy.factorial(n)

   return (top / bot).as_poly()

def find_roots(p):
   """Return set of roots of polynomial *p*.

   :param p: sympy polynomial

   This uses the *nroots* method of the SymPy polynomial class to give
   rough roots, and subsequently refines these roots to arbitrary
   precision using mpmath.

   Returns a sorted *set* of roots.
   """

   x = sympy.var('x')
   roots = set()

   for x0 in p.nroots():
      xi = mpmath.findroot(lambda z: p.eval(x, z), x0)
      roots.add(xi)

   return sorted(roots)
