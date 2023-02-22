import numpy
import sympy

from common.definitions import *
from .geometry          import Geometry
from .quadrature        import gauss_legendre

class Cartesian2D(Geometry):
   def __init__(self, domain_x, domain_z, nb_elements_x, nb_elements_z, nbsolpts):
      super().__init__('cartesian2d')

      # Gauss-Legendre solution points
      solutionPoints_sym, solutionPoints, glweights = gauss_legendre(nbsolpts)
      print('Solution points : ', solutionPoints)

      # Extend the solution points to include -1 and 1
      extension = numpy.append(numpy.append([-1], solutionPoints), [1])
      extension_sym = solutionPoints_sym.copy()
      extension_sym.insert(0, sympy.sympify('-1'))
      extension_sym.append(sympy.sympify('1'))

      scaled_points = 0.5 * (1.0 + solutionPoints)

      Δx1 = (domain_x[1] - domain_x[0]) / nb_elements_x
      Δx3 = (domain_z[1] - domain_z[0]) / nb_elements_z

      print(f'domain: {domain_x}, {domain_z}')
      print(f'grid size: {Δx1}, {Δx3}')

      faces_x = numpy.linspace(start = domain_x[0], stop = domain_x[1], num = nb_elements_x + 1)
      faces_z = numpy.linspace(start = domain_z[0], stop = domain_z[1], num = nb_elements_z + 1)

      x1 = numpy.zeros(nb_elements_x * len(solutionPoints))
      for i in range(nb_elements_x):
         idx = i * nbsolpts
         x1[idx : idx + nbsolpts] = faces_x[i] + scaled_points * Δx1

      x3 = numpy.zeros(nb_elements_z * len(solutionPoints))
      for i in range(nb_elements_z):
         idz = i * nbsolpts
         x3[idz : idz + nbsolpts] = faces_z[i] + scaled_points * Δx3

      X1, X3 = numpy.meshgrid(x1, x3)

      self.solutionPoints = solutionPoints
      self.solutionPoints_sym = solutionPoints_sym
      self.glweights = glweights
      self.extension = extension
      self.extension_sym = extension_sym
      self.X1 = X1
      self.X3 = X3
      self.itf_Z = faces_z
      self.Δx1 = Δx1
      self.Δx2 = Δx1
      self.Δx3 = Δx3
      self.xperiodic = False
