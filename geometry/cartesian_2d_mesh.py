import numpy

from common.definitions import *
from .geometry          import Geometry

# typing
from typing import Self
from common.program_options import Configuration

class Cartesian2D(Geometry):
   def __init__(self: Self, domain_x: tuple[float, float], domain_z: tuple[float, float],
                nb_elements_x: int, nb_elements_z: int, nbsolpts: int, param: Configuration):
      super().__init__(nbsolpts, 'cartesian2d', param)

      scaled_points = 0.5 * (1.0 + self.solutionPoints)

      Δx1 = (domain_x[1] - domain_x[0]) / nb_elements_x
      Δx3 = (domain_z[1] - domain_z[0]) / nb_elements_z

      print(f'domain: {domain_x}, {domain_z}')
      print(f'grid size: {Δx1}, {Δx3}')

      faces_x = param.array_module.linspace(start=domain_x[0], stop=domain_x[1], num=nb_elements_x + 1)
      faces_z = param.array_module.linspace(start=domain_z[0], stop=domain_z[1], num=nb_elements_z + 1)

      x1 = numpy.zeros(nb_elements_x * len(self.solutionPoints), like=faces_x)
      for i in range(nb_elements_x):
         idx = i * nbsolpts
         x1[idx : idx + nbsolpts] = faces_x[i] + scaled_points * Δx1

      x3 = numpy.zeros(nb_elements_z * len(self.solutionPoints), like=faces_z)
      for i in range(nb_elements_z):
         idz = i * nbsolpts
         x3[idz : idz + nbsolpts] = faces_z[i] + scaled_points * Δx3

      X1, X3 = numpy.meshgrid(x1, x3)

      self.X1 = X1
      self.X3 = X3
      self.itf_Z = faces_z
      self.Δx1 = Δx1
      self.Δx2 = Δx1
      self.Δx3 = Δx3
      self.xperiodic = False
