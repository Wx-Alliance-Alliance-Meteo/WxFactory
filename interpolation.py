import numpy
from mpi4py import MPI
import time

from matrices   import lagrangeEval
from quadrature import gauss_legendre
from timer      import Timer

basis_point_sets = {}

def eval_single_field(num_elem, field, elem_interp, new_num_basis_points, old_num_basis_points, result=None):
   # Interpolate (vertically) on entire rows of elements (horizontally) at once
   interp_1 = numpy.empty((num_elem * new_num_basis_points, num_elem * old_num_basis_points), dtype = field.dtype)
   for i in range(num_elem):
      interp_1[i * new_num_basis_points:(i + 1) * new_num_basis_points, :] = \
         elem_interp @ field[i * old_num_basis_points:(i + 1) * old_num_basis_points, :]

   # Interpolate (horizontally) on columns of interpolated elements
   if result is None:
      result = numpy.empty((num_elem * new_num_basis_points, num_elem * old_num_basis_points), dtype = field.dtype)

   for i in range(num_elem):
      result[:, i * new_num_basis_points:(i + 1) * new_num_basis_points] = \
         interp_1[:, i * old_num_basis_points:(i + 1) * old_num_basis_points] @ elem_interp.T

   return result


class BilinearInterpolator:
   def __init__(self, grid):
      self.grid = grid
      self.x_pos = grid.x1
      self.y_pos = grid.x2

      self.basis_points = grid.solutionPoints
      self.num_basis_points = len(self.basis_points)
      self.n_elem_i = int(len(self.x_pos) / self.num_basis_points)

      basis_point_sets[self.num_basis_points] = self.basis_points

   def get_weights(self, point_set, x):
      ix = 0
      while point_set[ix + 1] < x:
         if ix + 2 >= len(point_set):
            break
         ix += 1

      x_small = point_set[ix]
      x_large = point_set[ix+1]

      alpha = 1.0 - (x - x_small) / (x_large - x_small)

      result       = numpy.zeros_like(point_set)
      result[ix]   = alpha
      result[ix+1] = 1.0 - alpha

      return result


   def eval_grid_fast(self, field, new_num_basis_points, old_num_basis_points):
      for i in [new_num_basis_points, old_num_basis_points]:
         if i not in basis_point_sets:
            basis_point_sets[i], _ = gauss_legendre(i)

      new_basis_points = basis_point_sets[new_num_basis_points]
      old_basis_points = basis_point_sets[old_num_basis_points]

      elem_interp = numpy.array([self.get_weights(old_basis_points, px) for px in new_basis_points])

      result = None
      if field.ndim == 2:
         result = eval_single_field(self.n_elem_i, field, elem_interp, new_num_basis_points, old_num_basis_points)
      elif field.ndim == 3:
         num_fields = field.shape[0]
         result = numpy.empty((num_fields, self.n_elem_i * new_num_basis_points, self.n_elem_i * new_num_basis_points),
                              dtype = field.dtype)
         for i, f in enumerate(field):
            eval_single_field(self.n_elem_i, f, elem_interp, new_num_basis_points, old_num_basis_points, result[i])

      return result


   def get_value_at_pos(self, x, y):
      ix = 0
      iy = 0

      while self.x_pos[ix + 1] < x:
         if ix + 2 >= len(self.x_pos):
            break
         ix += 1

      while self.y_pos[iy + 1] < y:
         if iy + 2 >= len(self.y_pos):
            break
         iy += 1

      x_small = self.x_pos[ix]
      x_large = self.x_pos[ix+1]
      y_small = self.y_pos[iy]
      y_large = self.y_pos[iy+1]

      alpha = 1.0 - (x - x_small) / (x_large - x_small)
      beta  = 1.0 - (y - y_small) / (y_large - y_small)

      val = (self.field[ix, iy]     * alpha + self.field[ix + 1, iy]     * (1.0 - alpha)) * beta  + \
            (self.field[ix, iy + 1] * alpha + self.field[ix + 1, iy + 1] * (1.0 - alpha)) * (1.0 - beta)

      return val


class LagrangeSimpleInterpolator:
   def __init__(self, grid):

      numpy.set_printoptions(precision=2)

      self.grid  = grid

      self.x_pos = grid.x1
      self.y_pos = grid.x2
      self.basis_points = grid.solutionPoints
      self.num_basis_points = len(self.basis_points)

      basis_point_sets[self.num_basis_points] = self.basis_points

      self.delta_x = grid.Δx1
      self.delta_y = grid.Δx2
      self.n_elem_i = int(len(self.x_pos) / self.num_basis_points)

      self.rank = MPI.COMM_WORLD.Get_rank()
      self.grid_eval_timer = Timer(time.time())
      self.grid_eval_fast_timer = Timer(self.grid_eval_timer.initial_time)


   def eval_grid_fast(self, field, new_num_basis_points, old_num_basis_points):
      """
      Interpolate the current grid to a new one with the same number of elements,
      but with a different number of solution points, using vectorized operations
      as much as possible.
      """

      self.grid_eval_fast_timer.start()

      for i in [new_num_basis_points, old_num_basis_points]:
         if i not in basis_point_sets:
            basis_point_sets[i], _ = gauss_legendre(i)

      new_basis_points = basis_point_sets[new_num_basis_points]
      old_basis_points = basis_point_sets[old_num_basis_points]

      elem_interp = numpy.array([lagrangeEval(old_basis_points, px) for px in new_basis_points])

      result = None
      if field.ndim == 2:
         result = eval_single_field(self.n_elem_i, field, elem_interp, new_num_basis_points, old_num_basis_points)
      elif field.ndim == 3:
         num_fields = field.shape[0]
         result = numpy.empty((num_fields, self.n_elem_i * new_num_basis_points, self.n_elem_i * new_num_basis_points),
                              dtype = field.dtype)
         for i, f in enumerate(field):
            eval_single_field(self.n_elem_i, f, elem_interp, new_num_basis_points, old_num_basis_points, result[i])

      self.grid_eval_fast_timer.stop()

      return result


   def eval_grid(self, field, new_num_basis_points):
      """
      Interpolate the current grid to a new one with the same number of elements,
      but with a different number of solution points, element by element
      """

      self.grid_eval_timer.start()

      new_grid_size = self.n_elem_i * new_num_basis_points
      new_basis_points, _ = gauss_legendre(new_num_basis_points)
      new_field = numpy.zeros((new_grid_size, new_grid_size))

      elem_interp = numpy.array([lagrangeEval(self.basis_points, px) for px in new_basis_points])
      col_interp = numpy.array([numpy.append(elem_row, [elem_row for i in range(self.n_elem_i - 1)]) for elem_row in elem_interp])

      # interpolate along vertical axis
      interp_1 = numpy.array([[
         elem_interp @ field[
                           i * self.num_basis_points:(i+1)*self.num_basis_points,
                           j * self.num_basis_points:(j+1)*self.num_basis_points]
            for j in range(self.n_elem_i)]
                for i in range(self.n_elem_i)])

      # interpolate along horizontal axis
      interp_2 = numpy.array([[
         interp_1[i,j] @ elem_interp.T
            for j in range(self.n_elem_i)]
                for i in range(self.n_elem_i)])

      # Put in global grid form (rather than element by element)
      for i in range(self.n_elem_i):
         start_i = i * new_num_basis_points
         stop_i = start_i + new_num_basis_points
         for j in range(self.n_elem_i):
            start_j = j * new_num_basis_points
            stop_j = start_j + new_num_basis_points

            new_field[start_i:stop_i, start_j:stop_j] = interp_2[i, j]

      self.grid_eval_timer.stop()

      return new_field


class LagrangeInterpolator:
   def __init__(self, grid, field, comm_dist_graph):

      print('LagrangeInterpolator is not working (yet?). You need to fix xchange stuff (MPI communication).')
      raise ValueError

      self.grid  = grid
      self.field = field

      self.x_pos = grid.x1
      self.y_pos = grid.x2
      self.basis_points = grid.solutionPoints
      self.delta_x = grid.Δx1
      self.delta_y = grid.Δx2

      rank = MPI.COMM_WORLD.Get_rank()

      num_basis_points = len(self.basis_points)

      self.n_elem_i = int(len(self.x_pos) / num_basis_points)

      left_extrapolate   = lagrangeEval(self.basis_points, -1.0)
      right_extrapolate  = lagrangeEval(self.basis_points, 1.0)
      bottom_extrapolate = left_extrapolate
      top_extrapolate    = right_extrapolate

      interface_i = numpy.zeros((self.n_elem_i + 2, 2, len(self.x_pos)))
      interface_j = numpy.zeros((self.n_elem_i + 2, 2, len(self.x_pos)))

      for i in range(self.n_elem_i):
         sliceIds = i * num_basis_points + numpy.arange(num_basis_points)
         pos = i + 1

         interface_i[pos, 0, :] = field[:, sliceIds] @ left_extrapolate
         interface_i[pos, 1, :] = field[:, sliceIds] @ right_extrapolate
         interface_j[pos, 0, :] = bottom_extrapolate @ field[sliceIds, :]
         interface_j[pos, 1, :] = top_extrapolate    @ field[sliceIds, :]


      xchange_scalars(comm_dist_graph, grid, interface_i, interface_j)

      grid_size = (num_basis_points + 1) * self.n_elem_i + 1
      self.full_grid = numpy.zeros((grid_size, grid_size))

      numpy.set_printoptions(precision=2)

      if rank == 0:
          print('field: \n{}'.format(self.field))        
          print('interface i: \n{}'.format(interface_i))
          print('full grid (1): \n{}'.format(self.full_grid))

      for i in range(self.n_elem_i):
         target_pos_i = i * (num_basis_points + 1) + 1
         target_slice_range_i = range(target_pos_i, target_pos_i + num_basis_points)

         src_pos_i = i * num_basis_points
         src_slice_range_i = range(src_pos_i, src_pos_i + num_basis_points)

         for j in range(self.n_elem_i):
            target_pos_j = j * (num_basis_points + 1) + 1
            src_pos_j = j * num_basis_points

            target_slice_range_j = range(target_pos_j, target_pos_j + num_basis_points)
            target_slice = numpy.ix_(target_slice_range_i, target_slice_range_j)

            src_slice_range_j = range(src_pos_j, src_pos_j + num_basis_points)
            src_slice = numpy.ix_(src_slice_range_i, src_slice_range_j)

            self.full_grid[target_slice] = field[src_slice]

      if rank == 0:
         print('full grid (2): \n{}'.format(self.full_grid))

      for i in range(self.n_elem_i + 1):
         target_pos_left = i * (num_basis_points + 1)

         itf_elem_pos = i + 1
         for j in range(self.n_elem_i):
            target_slice_start = j * (num_basis_points + 1) + 1
            target_slice = target_slice_start + numpy.arange(num_basis_points)

            src_slice_start = j * num_basis_points
            src_slice = src_slice_start + numpy.arange(num_basis_points)

            self.full_grid[target_slice, target_pos_left] = \
                    (interface_i[itf_elem_pos - 1, 1, src_slice] + interface_i[itf_elem_pos, 0, src_slice]) * 0.5
            self.full_grid[target_pos_left, target_slice] = \
                    (interface_j[itf_elem_pos - 1, 1, src_slice] + interface_j[itf_elem_pos, 0, src_slice]) * 0.5
         
      if rank == 0:
          print('full grid (3): \n{}'.format(self.full_grid))


      corners_i = numpy.zeros((self.n_elem_i + 2, 2, self.n_elem_i + 1))
      corners_j = numpy.zeros((self.n_elem_i + 2, 2, self.n_elem_i + 1))

      for i in range(self.n_elem_i):
         slice_i = i * (num_basis_points + 1) + 1 + numpy.arange(num_basis_points)
         pos_i = i + 1
         for j in range(self.n_elem_i + 1):
            pos_j = j 
            src_j = j * (num_basis_points + 1)

            corners_i[pos_i, 0, pos_j] = self.full_grid[src_j, slice_i] @ left_extrapolate
            corners_i[pos_i, 1, pos_j] = self.full_grid[src_j, slice_i] @ right_extrapolate
            corners_j[pos_i, 0, pos_j] = bottom_extrapolate @ self.full_grid[slice_i, src_j]
            corners_j[pos_i, 1, pos_j] = top_extrapolate    @ self.full_grid[slice_i, src_j]

      if rank == 0:
         print('corners_i: \n{}'.format(corners_i))
         print('corners_j: \n{}'.format(corners_j))

      xchange_scalars(comm_dist_graph, grid, corners_i, corners_j)

      if rank == 0:
         print('corners_i: \n{}'.format(corners_i))
         print('corners_j: \n{}'.format(corners_j))

      for i in range(self.n_elem_i + 1):
        target_pos_i = i * (num_basis_points + 1)

        for j in range(self.n_elem_i + 1):
            target_pos_j = j * (num_basis_points + 1)

            self.full_grid[target_pos_i, target_pos_j] = \
                  (corners_i[j    , 1, i] +
                   corners_i[j + 1, 0, i] +
                   corners_j[i    , 1, j] +
                   corners_j[i + 1, 0, j]) * 0.25


      if rank == 0:
         print('full grid(4): \n{}'.format(self.full_grid))


def interpolate(dest_grid, src_grid, field, comm_dist_graph):

   dest_ni, dest_nj = dest_grid.lon.shape
   dest_x_pos, dest_y_pos = dest_grid.x1, dest_grid.x2

   result = numpy.zeros((dest_ni, dest_nj))

   #interpolator = BilinearInterpolator(src_grid, field)
   interpolator = LagrangeSimpleInterpolator(src_grid)

   #int_test = LagrangeInterpolator(src_grid, field, comm_dist_graph)

   alt_result = interpolator.eval_grid(field, len(dest_grid.solutionPoints))
   alt_result_fast = interpolator.eval_grid_fast(field, len(dest_grid.solutionPoints))
   difference = alt_result - alt_result_fast

   if interpolator.rank == 0:
      print('time (entire grid):          {}'.format(interpolator.grid_eval_timer.times))
      print('time (entire grid, fast):    {}'.format(interpolator.grid_eval_fast_timer.times))
      #print('difference: \n{}'.format(difference))

   return alt_result

