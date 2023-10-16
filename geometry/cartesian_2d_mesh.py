from typing import Tuple

import numpy

from common.graphx   import print_mountain
from .geometry       import Geometry

class Cartesian2D(Geometry):
   def __init__(self,
                domain_x: Tuple[float, float],
                domain_z: Tuple[float, float],
                nb_elements_x: int,
                nb_elements_z: int,
                nbsolpts: int,
                nb_elements_relief_layer: int,
                relief_layer_height: int):
      super().__init__(nbsolpts, 'cartesian2d')

      scaled_points = 0.5 * (1.0 + self.solutionPoints)

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

      if nb_elements_relief_layer > 0:
         Δrelief_layer = (relief_layer_height - domain_z[0]) / nb_elements_relief_layer

         Δx3 = (domain_z[1] - relief_layer_height) / (nb_elements_z - nb_elements_relief_layer)
         itf_x3 = numpy.linspace(start=relief_layer_height, stop=domain_z[1], num=(nb_elements_z - nb_elements_relief_layer) + 1)
         itf_relief_layer = numpy.linspace(start=domain_z[0], stop=relief_layer_height, num=nb_elements_relief_layer + 1)

         z1 = numpy.zeros(nb_elements_relief_layer * len(self.solutionPoints))
         z2 = numpy.zeros((nb_elements_z-nb_elements_relief_layer) * len(self.solutionPoints))

         for i in range(nb_elements_relief_layer):
            idz = i * nbsolpts
            z1[idz: idz + nbsolpts] = itf_relief_layer[i] + scaled_points * Δrelief_layer

         for i in range(nb_elements_z - nb_elements_relief_layer):
            idz = i * nbsolpts
            z2[idz: idz + nbsolpts] = itf_x3[i] + scaled_points * Δx3

         x3 = numpy.concatenate((z1,z2),axis=None)

      else:
         Δx3 = (domain_z[1] - domain_z[0]) / nb_elements_z
         Δrelief_layer = 0 # being lazy ...
         itf_x3 = numpy.linspace(start=domain_z[0], stop=domain_z[1], num=nb_elements_z + 1)

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
      self.relief_layer_delta = Δrelief_layer
      self.nb_elements_relief_layer = nb_elements_relief_layer
      self.xperiodic = False


   def make_mountain(self, mountain_type='sine'):

      if self.nb_elements_relief_layer <= 0:
         print(f'Need to have a relief layer to make a mountain')
         return

      if mountain_type == 'sine':
         h0    = 250.   # Max mountain height
         a     = 5000.  # Overall mountain width factor
         lmbda = 4000.  # Mountain ripple width factor

         def h_func(x):
            return h0 * numpy.exp(-(x / a)**2) * numpy.cos(numpy.pi * x / lmbda)**2 + 100.0

      elif mountain_type == 'step':
         cliff_pos = 0.0
         mountain_height = 1000.0

         def h_func(x):
            return numpy.where(x > cliff_pos, mountain_height, 0.0)

      else:
         raise ValueError(f'Invalid mountain type: {mountain_type}')

      x = self.X1[0,:]
      h = h_func(x)

      # Compute normal component
      # Note: This will only work for our 2D terrain.
      delta_x = 1e-8
      ncompx = (h_func(x + delta_x) - h_func(x - delta_x)) / (2 * delta_x)  # centered differences

      end = self.nb_elements_relief_layer * self.nbsolpts

      # True where points are inside the mountain, False elsewhere
      relief_mask = self.X3[:end] <= h

      # Mask for marking mountain boundary to the top, left and right
      top_boundary   = numpy.zeros_like(relief_mask)
      left_boundary  = numpy.zeros_like(relief_mask)
      right_boundary = numpy.zeros_like(relief_mask)

      # Top of the mountain: points that are inside, with a neighbor above that is outside
      top_boundary[-1, :]  = relief_mask[-1, :]
      top_boundary[:-1, :] = numpy.where(numpy.diff(relief_mask, axis=0), True, False)

      # Sides of the mountain: points that are inside, with a left/right neighbor that is outside
      # *excluding* top points
      h_diff = numpy.diff(relief_mask.astype(int), axis=1)
      left_boundary [:, 1:]   = (h_diff == 1)   & numpy.logical_not(top_boundary[:, 1:])
      right_boundary[:,  :-1] = (h_diff == -1)  & numpy.logical_not(top_boundary[:,  :-1])

      if numpy.any(left_boundary & right_boundary):
         raise ValueError(f'We can\'t deal with mountains that are too thin, we need to have at least 2 grid points wide (except at the top)')

      # Overall boundary
      relief_boundary_mask = top_boundary | left_boundary | right_boundary
      side_boundary_mask   = left_boundary | right_boundary

      # Compute normals on the boundary (only take vertical position of mountain as a reference)
      nsq_plus1 = numpy.sqrt(ncompx ** 2 + 1)
      normals_x = ncompx / nsq_plus1
      normals_z = -1.0 / nsq_plus1

      normals_x = numpy.where(relief_boundary_mask, normals_x, 0.0)
      normals_z = numpy.where(relief_boundary_mask, normals_z, 0.0)

      # print(f'normals_z: {normals_z[:5, :30]}')

      if mountain_type == 'step':
         normals_x = numpy.select([left_boundary, right_boundary], [1.0, -1.0], normals_x)
         normals_z = numpy.where(side_boundary_mask, 0.0, normals_z)

      print_mountain(self.X1[:end, :], self.X3[:end, :], relief_mask + relief_boundary_mask * 2,
                     normals_x=normals_x, normals_z=normals_z, filename='mountain.png')

      print(f'Number of Terrain points: {numpy.count_nonzero(relief_mask)}, '
            f'Number of boundary points: {numpy.count_nonzero(relief_boundary_mask)} '
            f'({numpy.count_nonzero(left_boundary)} left, {numpy.count_nonzero(right_boundary)} right)')

      self.normals_x = normals_x
      self.normals_z = normals_z
      self.relief_mask = relief_mask
      self.relief_boundary_mask = relief_boundary_mask
      self.side_boundary_mask = side_boundary_mask
