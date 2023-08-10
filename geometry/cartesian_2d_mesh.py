from typing import Tuple

from mpi4py import MPI
import numpy

from .geometry import Geometry

class Cartesian2D(Geometry):
   def __init__(self,
                domain_x: Tuple[float, float],
                domain_z: Tuple[float, float],
                nb_elements_x: int,
                nb_elements_z: int,
                nbsolpts: int,
                nb_elements_bottom_layer: int,
                bottom_layer_height: int):
      super().__init__(nbsolpts, 'cartesian2d')

      scaled_points = 0.5 * (1.0 + self.solutionPoints)

      if MPI.COMM_WORLD.size > 1:
         rank = MPI.COMM_WORLD.rank
         global_width = domain_x[1] - domain_x[0]
         local_width = global_width / MPI.COMM_WORLD.size
         new_domain_x = (rank * local_width + domain_x[0], (rank+1) * local_width + domain_x[0])
         domain_x = new_domain_x

      # --- Horizontal coord
      Δx1 = (domain_x[1] - domain_x[0]) / nb_elements_x
      itf_x1 = numpy.linspace(start = domain_x[0], stop = domain_x[1], num = nb_elements_x + 1)
      x1 = numpy.zeros(nb_elements_x * len(self.solutionPoints))
      for i in range(nb_elements_x):
         idx = i * nbsolpts
         x1[idx : idx + nbsolpts] = itf_x1[i] + scaled_points * Δx1

      # --- Vertical coord
      Δx3 = (domain_z[1] - domain_z[0]) / nb_elements_z
      x3 = numpy.zeros(nb_elements_z * len(self.solutionPoints))

      if nb_elements_bottom_layer > 0:
         Δbottom_layer = (bottom_layer_height - domain_z[0]) / nb_elements_bottom_layer

         Δx3 = (domain_z[1] - bottom_layer_height) / (nb_elements_z - nb_elements_bottom_layer)
         itf_x3 = numpy.linspace(start=bottom_layer_height, stop=domain_z[1],
                                 num=(nb_elements_z - nb_elements_bottom_layer) + 1)
         itf_bottom_layer = numpy.linspace(start=domain_z[0], stop=bottom_layer_height,
                                           num=nb_elements_bottom_layer + 1)

         z1 = numpy.zeros(nb_elements_bottom_layer * len(self.solutionPoints))
         z2 = numpy.zeros((nb_elements_z-nb_elements_bottom_layer) * len(self.solutionPoints))

         for i in range(nb_elements_bottom_layer):
            idz = i * nbsolpts
            z1[idz: idz + nbsolpts] = itf_bottom_layer[i] + scaled_points * Δbottom_layer

         for i in range(nb_elements_z - nb_elements_bottom_layer):
            idz = i * nbsolpts
            z2[idz: idz + nbsolpts] = itf_x3[i] + scaled_points * Δx3

         x3 = numpy.concatenate((z1,z2),axis=None)

      else:
         Δx3 = (domain_z[1] - domain_z[0]) / nb_elements_z
         Δbottom_layer = 0 # being lazy ...
         itf_x3 = numpy.linspace(start=domain_z[0], stop=domain_z[1], num=nb_elements_z + 1)
         print(f'domain: {domain_x}, {domain_z}')
         print(f'grid size: {Δx1}, {Δx3}')

         for i in range(nb_elements_z):
            idz = i * nbsolpts
            x3[idz : idz + nbsolpts] = itf_x3[i] + scaled_points * Δx3

      X1, X3 = numpy.meshgrid(x1, x3)

      self.X1 = X1
      self.X3 = X3
      self.itf_Z = itf_x3
      self.Δx1 = Δx1
      self.Δx2 = Δx1
      self.Δx3 = Δx3
      self.bottom_layer_delta = Δbottom_layer
      self.nb_elements_bottom_layer = nb_elements_bottom_layer
      self.xperiodic = False

      # We may want to make a 'terrain' class as this is not ideal spot for this...
      self.chiMask = list() # Mask for terrain
      self.chiMaskBoundary= list() # terrible way to do this i know :(
      self.terrainNormalXcmp = numpy.zeros_like(X1)  # normal component in x-direction of terrain
      self.terrainNormalZcmp = numpy.zeros_like(X1)  # normal component in z-direction of terrain
