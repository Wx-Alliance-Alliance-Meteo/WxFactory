import numpy
from mpi4py import MPI
import time

from matrices   import lagrangeEval
from parallel   import xchange_scalars
from quadrature import gauss_legendre
from timer      import Timer

class BilinearInterpolator:
   def __init__(self, grid, field):
      self.grid = grid
      self.field = field
      self.x_pos = grid.x1
      self.y_pos = grid.x2

   def getValueAtXY(self, x, y):
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
   def __init__(self, grid, field):

      numpy.set_printoptions(precision=2)

      self.grid  = grid
      self.field = field

      self.x_pos = grid.x1
      self.y_pos = grid.x2
      self.basis_points = grid.solutionPoints
      self.num_basis_points = len(self.basis_points)
      self.min_x, self.max_x = grid.domain_x1
      self.min_y, self.max_y = grid.domain_x2
      self.delta_x = grid.Δx1
      self.delta_y = grid.Δx2
      self.n_elem_i = int(len(self.x_pos) / self.num_basis_points)

      self.rank = MPI.COMM_WORLD.Get_rank()
      self.grid_eval_timer = Timer(time.time())


   def evalGrid(self, new_num_basis_points):

      self.grid_eval_timer.start()

      new_grid_size = self.n_elem_i * new_num_basis_points
      new_basis_points, _ = gauss_legendre(new_num_basis_points)
      new_field = numpy.zeros((new_grid_size, new_grid_size))

      elem_interp = numpy.array([lagrangeEval(self.basis_points, px) for px in new_basis_points])
      col_interp = numpy.array([numpy.append(elem_row, [elem_row for i in range(self.n_elem_i - 1)]) for elem_row in elem_interp])

      # interpolate along vertical axis
      interp_1 = numpy.array([[
         elem_interp @ self.field[
                                i * self.num_basis_points:(i+1)*self.num_basis_points,
                                j * self.num_basis_points:(j+1)*self.num_basis_points]
            for j in range(self.n_elem_i)]
                for i in range(self.n_elem_i)])

      # interpolate along horizontal axis
      interp_2 = numpy.array([[
         interp_1[i,j] @ elem_interp.T
            for j in range(self.n_elem_i)]
                for i in range(self.n_elem_i)])


      for i in range(self.n_elem_i):
         start_i = i * new_num_basis_points
         stop_i = start_i + new_num_basis_points
         for j in range(self.n_elem_i):
            start_j = j * new_num_basis_points
            stop_j = start_j + new_num_basis_points

            new_field[start_i:stop_i, start_j:stop_j] = interp_2[i, j]

      self.grid_eval_timer.stop()

#      if self.rank == 0:
#         print('elem_interp: \n{}'.format(elem_interp))
#         print('col_interp: \n{}'.format(col_interp))
#         print('interp_1: \n{}'.format(interp_1))
#         print('interp_2: \n{}'.format(interp_2))
#         print('new_field:\n{}'.format(new_field))


      return new_field


   # TODO Lagrange on both axes
   def getValueAtXY(self, x, y):

      # Determine the element point (x,y) belongs to
      elem_i  = int((x - self.min_x) / self.delta_x)
      start_i = elem_i * self.num_basis_points
      stop_i  = start_i + self.num_basis_points

      elem_j  = int((y - self.min_y) / self.delta_y)
      start_j = elem_j * self.num_basis_points
      stop_j  = start_j + self.num_basis_points

      # Find exactly between which solution points (x,y) is
      ix = start_i
      iy = start_j

      while self.x_pos[ix + 1] < x:
         if ix + 2 >= stop_i:
            break
         ix += 1

      while self.y_pos[iy + 1] < y:
         if iy + 2 >= stop_j:
            break
         iy += 1

      # Compute element boundaries within the set of solution point positions
      x_elem_small = self.min_x + elem_i * self.delta_x
      x_elem_large = self.min_x + (elem_i+1) * self.delta_x
      y_elem_small = self.min_y + elem_j * self.delta_y
      y_elem_large = self.min_y + (elem_j+1) * self.delta_y

      # Find which points to use for the linear interpolation between sets of lagrange polynomials
      if ix < start_i:
         ix += 1
      elif ix + 1 > stop_i:
         ix -= 1

      if iy < start_j:
         iy += 1
      elif iy + 1> stop_j:
         iy -= 1

      x_small = self.x_pos[ix]
      x_large = self.x_pos[ix + 1]
      y_small = self.y_pos[iy]
      y_large = self.y_pos[iy + 1]

      # Lagrange interpolation, along x, at the 2 y-sets found
      sol_pt_x = (x - x_elem_small) / (x_elem_large - x_elem_small) * 2.0 - 1.0
      sol_pt_x_interp = lagrangeEval(self.basis_points, sol_pt_x)
      vals_y_minus = self.field[start_i:stop_i, iy]
      vals_y_plus  = self.field[start_i:stop_i, iy + 1]

      xs = self.x_pos[start_i:stop_i]

      val_y_minus = numpy.dot(vals_y_minus, sol_pt_x_interp)
      val_y_plus  = numpy.dot(vals_y_plus, sol_pt_x_interp)

      # Linear interpolation between y- and y+ values
      alpha = 1.0 - (y - y_small) / (y_large - y_small)
      val_x = val_y_minus * alpha + val_y_plus * (1.0 - alpha)

      # Lagrange interpolation, along y, at the 2 x-sets found
      sol_pt_y = (y - y_elem_small) / (y_elem_large - y_elem_small) * 2.0 - 1.0
      sol_pt_y_interp = lagrangeEval(self.basis_points, sol_pt_y)
      vals_x_minus = self.field[ix, start_j:stop_j]
      vals_x_plus  = self.field[ix + 1, start_j:stop_j]

      ys = self.y_pos[start_j:stop_j]
      val_x_minus = numpy.dot(vals_x_minus, sol_pt_y_interp)
      val_x_plus  = numpy.dot(vals_x_plus,  sol_pt_y_interp)

      # TODO Should use Lagrange polynomial too
      # Linear interpolation between x- and x+ values
      beta = 1.0 - (x - x_small) / (x_large - x_small)
      val_y = val_x_minus * beta + val_x_plus * (1.0 - beta)

      # Just take the average of the 2 values
      return (val_x + val_y) * 0.5


class LagrangeInterpolator:
   def __init__(self, grid, field, comm_dist_graph):
      self.grid  = grid
      self.field = field

      self.x_pos = grid.x1
      self.y_pos = grid.x2
      self.basis_points = grid.solutionPoints
      self.min_x, self.max_x = grid.domain_x1
      self.min_y, self.max_y = grid.domain_x2
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
   interpolator = LagrangeSimpleInterpolator(src_grid, field)

   #int_test = LagrangeInterpolator(src_grid, field, comm_dist_graph)

   alt_result = interpolator.evalGrid(len(dest_grid.solutionPoints))

   point_by_point_timer = Timer(time.time())
   point_by_point_timer.start()
   for i in range(dest_ni):
      for j in range(dest_nj):
         target_x = dest_x_pos[i]
         target_y = dest_y_pos[j]

         result[i, j] = interpolator.getValueAtXY(target_x, target_y)

   point_by_point_timer.stop()

   difference = alt_result - result

   if interpolator.rank == 0:
      print('time (point by point): {}'.format(point_by_point_timer.times))
      print('time (entire grid):    {}'.format(interpolator.grid_eval_timer.times))
      #print('difference: \n{}'.format(difference))


   if False:
      #dest_interpolator = BilinearInterpolator(dest_grid, result)
      dest_interpolator = LagrangeSimpleInterpolator(dest_grid, result)

      num_samples = 101
      sample_width_x = (src_grid.domain_x1[1] - src_grid.domain_x1[0]) / num_samples
      sample_width_y = (src_grid.domain_x2[1] - src_grid.domain_x2[0]) / num_samples

      total_val = 0.0
      total_diff = 0.0
      for i in range(num_samples):
         x = (i + 0.5) * sample_width_x + src_grid.domain_x1[0]
         for j in range(num_samples):
            y = (j + 0.5) * sample_width_y + src_grid.domain_x2[0]

            src_val = interpolator.getValueAtXY(x, y)
            dest_val= dest_interpolator.getValueAtXY(x, y)

            total_val += numpy.abs(src_val)
            total_diff += numpy.abs((dest_val - src_val)/src_val)

      avg_diff = total_diff / (num_samples**2)

      print('Average relative diff: {:.2f}%, average value: {}'.format(avg_diff * 100, total_val / num_samples**2))

   return alt_result
