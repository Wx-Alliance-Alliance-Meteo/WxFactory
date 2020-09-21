import numpy

from matrices import lagrangeEval

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
         ix += 1
         if ix + 1 >= len(self.x_pos):
            break

      while self.y_pos[iy + 1] < y:
         iy += 1
         if iy + 1 >= len(self.x_pos):
            break

      x_small = self.x_pos[ix]
      x_large = self.x_pos[ix+1]
      y_small = self.y_pos[iy]
      y_large = self.y_pos[iy+1]

      alpha = 1.0 - (x - x_small) / (x_large - x_small)
      beta  = 1.0 - (y - y_small) / (y_large - y_small)

      val = (self.field[ix, iy]     * alpha + self.field[ix + 1, iy]     * (1.0 - alpha)) * beta  + \
            (self.field[ix, iy + 1] * alpha + self.field[ix + 1, iy + 1] * (1.0 - alpha)) * (1.0 - beta)

      return val


class LagrangeInterpolator:
   def __init__(self, grid, field):
      self.grid  = grid
      self.field = field

      self.x_pos = grid.x1
      self.y_pos = grid.x2
      self.basisPoints = grid.solutionPoints
      self.min_x, self.max_x = grid.domain_x1
      self.min_y, self.max_y = grid.domain_x2
      self.delta_x = grid.Δx1
      self.delta_y = grid.Δx2

   def getValueAtXY(self, x, y):
      ix = 0
      iy = 0

      while self.x_pos[ix + 1] < x:
         ix += 1
         if ix + 1 >= len(self.x_pos):
            break

      while self.y_pos[iy + 1] < y:
         iy += 1
         if iy + 1 >= len(self.x_pos):
            break

      # Find element boundaries within the set of solution point positions
      elem_i  = int((x - self.min_x) / self.delta_x)
      start_i = elem_i * len(self.basisPoints)
      stop_i  = start_i + len(self.basisPoints)

      elem_j  = int((y - self.min_y) / self.delta_y)
      start_j = elem_j * len(self.basisPoints)
      stop_j  = start_j + len(self.basisPoints)

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
      sol_pt_x_interp = lagrangeEval(self.basisPoints, sol_pt_x)
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
      sol_pt_y_interp = lagrangeEval(self.basisPoints, sol_pt_y)
      vals_x_minus = self.field[ix, start_j:stop_j]
      vals_x_plus  = self.field[ix + 1, start_j:stop_j]

      ys = self.y_pos[start_j:stop_j]
      val_x_minus = numpy.dot(vals_x_minus, sol_pt_y_interp)
      val_x_plus  = numpy.dot(vals_x_plus,  sol_pt_y_interp)

      # Linear interpolation between x- and x+ values
      beta = 1.0 - (x - x_small) / (x_large - x_small)
      val_y = val_x_minus * beta + val_x_plus * (1.0 - beta)

      # Just take the average of the 2 values
      return (val_x + val_y) * 0.5


def interpolate(dest_grid, src_grid, field):

   dest_ni, dest_nj = dest_grid.lon.shape
   dest_x_pos, dest_y_pos = dest_grid.x1, dest_grid.x2

   result = numpy.zeros((dest_ni, dest_nj))

   #interpolator = BilinearInterpolator(src_grid, field)
   interpolator = LagrangeInterpolator(src_grid, field)

   for i in range(dest_ni):
      for j in range(dest_nj):
         target_x = dest_x_pos[i]
         target_y = dest_y_pos[j]

         result[i, j] = interpolator.getValueAtXY(target_x, target_y)

   return result

