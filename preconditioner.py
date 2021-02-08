from copy import copy

from cubed_sphere  import cubed_sphere
from initialize    import initialize_sw
from interpolation import LagrangeSimpleInterpolator
from matrices      import DFR_operators
from metric        import Metric
from dgfilter      import apply_filter

from rhs_caller            import RhsCaller
from matvec_product_caller import MatvecCaller

class Preconditioner:

   def __init__(self, param, geometry, rhs_func, ptopo,
                prefix = '   ', depth = 1, filter_before = False, filter_during = False, filter_after = False):

      min_order = 2
      max_depth = 2

      if param.nbsolpts <= min_order:
         print("Can't make a preconditioner with order less than {}. (currently asking for order {})".format(
            min_order, param.nbsolpts - 1))
         raise ValueError

      self.num_iter         = 0
      self.num_precond_iter = 0
      self.ptopo            = ptopo
      self.rank             = ptopo.rank
      self.prefix           = prefix
      self.depth            = depth

      self.filter_before = filter_before
      self.filter_during = filter_during
      self.filter_after  = filter_after

      self.order = param.nbsolpts - 1
      self.big_order = param.nbsolpts
      self.num_elements = param.nb_elements_horizontal

      if self.order == 2:
         self.max_iter = 20
      elif self.order == 3:
         self.max_iter = 60
      # elif self.order == 4:
      #    self.max_iter = 100
      else:
         self.max_iter = 1000

      if self.rank == 0:
         print('Creating a preconditioner of order {}'.format(self.order))

      param_small = copy(param)
      param_small.nbsolpts = self.order
      param_small.filter_apply = True
      param_small.filter_order = 8

      self.geom    = cubed_sphere(param.nb_elements_horizontal, self.order, param.λ0, param.ϕ0, param.α0, self.ptopo)
      self.mtrx    = DFR_operators(self.geom, param_small)
      self.metric  = Metric(self.geom)
      _, self.topo = initialize_sw(self.geom, self.metric, self.mtrx, param_small)

      self.rhs = RhsCaller(rhs_func, self.geom, self.mtrx, self.metric, self.topo, self.ptopo, self.order,
                           param.nb_elements_horizontal, param.case_number, use_filter = filter_during)

      self.interpolator = LagrangeSimpleInterpolator(geometry)

      self.mat = None
      self.preconditioner = None
      if self.order > min_order and self.depth < max_depth:
         self.preconditioner = Preconditioner(
            param_small, self.geom, rhs_func, self.ptopo,
            self.prefix + '   ', self.depth + 1, filter_before = self.filter_before,
            filter_during = self.filter_during, filter_after = self.filter_after)


   def compute_matrix_caller(self, matvec_func, dt, field):
      lowres_field = self.restrict(field)
      self.mat = MatvecCaller(matvec_func, dt, lowres_field, self.rhs)
      if self.preconditioner:
         self.preconditioner.compute_matrix_caller(matvec_func, dt, lowres_field)

   def restrict(self, field):
      return self.interpolator.eval_grid_fast(field, self.order, self.big_order)

   def prolong(self, field):
      return self.interpolator.eval_grid_fast(field, self.big_order, self.order)

   def apply(self, vec, solver, A):
      big_shape = A.field.shape
      small_shape = self.mat.field.shape

      input_vec_grid = self.restrict(vec.reshape(big_shape))
      if self.filter_before:
         input_vec_grid[0] = apply_filter(input_vec_grid[0], self.mtrx, self.num_elements, self.order)
         input_vec_grid[1] = apply_filter(input_vec_grid[1], self.mtrx, self.num_elements, self.order)
         input_vec_grid[2] = apply_filter(input_vec_grid[2], self.mtrx, self.num_elements, self.order)

      input_vec = input_vec_grid.flatten()

      output_vec, _, num_iter, _ = solver.solve(
         self.mat, input_vec, preconditioner = self.preconditioner,
         prefix = self.prefix, tol = 1e-5, max_iter= self.max_iter)
      result_grid = self.prolong(output_vec.reshape(small_shape))

      if self.filter_after:
         result_grid[0] = apply_filter(result_grid[0], A.rhs.operators, self.num_elements, self.big_order)
         result_grid[1] = apply_filter(result_grid[1], A.rhs.operators, self.num_elements, self.big_order)
         result_grid[2] = apply_filter(result_grid[2], A.rhs.operators, self.num_elements, self.big_order)

      result = result_grid.flatten()

      self.num_iter += num_iter

      if self.rank == 0:
         print('{} Preconditioned in {} iterations'.format(self.prefix, num_iter))

      return result, num_iter
