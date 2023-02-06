import numpy
import sympy

from common.definitions import *
from geometry.geometry  import Geometry
import geometry.quadrature as quadrature

class Cartesian2d(Geometry):
   def __init__(self, domain_x, domain_z, nb_elements_x, nb_elements_z, nbsolpts):
      super().__init__('cartesian2d')

      # Gauss-Legendre solution points
      solutionPoints_sym, solutionPoints, glweights = quadrature.gauss_legendre(nbsolpts)
      print('Solution points : ', solutionPoints)

      # Extend the solution points to include -1 and 1
      extension = numpy.append(numpy.append([-1], solutionPoints), [1])
      extension_sym = solutionPoints_sym.copy()
      extension_sym.insert(0, sympy.sympify('-1'))
      extension_sym.append(sympy.sympify('1'))

      scaled_points = 0.5 * (1.0 + solutionPoints)

      Δx = (domain_x[1] - domain_x[0]) / nb_elements_x
      Δz = (domain_z[1] - domain_z[0]) / nb_elements_z

      print(f'domain: {domain_x}, {domain_z}')
      print(f'grid size: {Δx}, {Δz}')

      faces_x = numpy.linspace(start = domain_x[0], stop = domain_x[1], num = nb_elements_x + 1)
      faces_z = numpy.linspace(start = domain_z[0], stop = domain_z[1], num = nb_elements_z + 1)

      x = numpy.zeros(nb_elements_x * len(solutionPoints))
      for i in range(nb_elements_x):
         idx = i * nbsolpts
         x[idx : idx + nbsolpts] = faces_x[i] + scaled_points * Δx

      z = numpy.zeros(nb_elements_z * len(solutionPoints))
      for i in range(nb_elements_z):
         idz = i * nbsolpts
         z[idz : idz + nbsolpts] = faces_z[i] + scaled_points * Δz

      X, Z = numpy.meshgrid(x, z)

      self.solutionPoints = solutionPoints
      self.solutionPoints_sym = solutionPoints_sym
      self.glweights = glweights
      self.extension = extension
      self.extension_sym = extension_sym
      self.X = X
      self.Z = Z
      self.itf_Z = faces_z
      self.Δx = Δx
      self.Δz = Δz
      self.xperiodic = False
