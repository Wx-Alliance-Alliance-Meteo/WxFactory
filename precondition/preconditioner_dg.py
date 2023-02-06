from copy import copy
from time import time

from cubed_sphere  import cubed_sphere
from initialize    import initialize_sw
from interpolation import LagrangeSimpleInterpolator
from linsol        import fgmres
from matrices      import DFR_operators
from metric        import Metric


class DG_preconditioner:

   def __init__(self, param, geometry, ptopo, operators, rhs_func, prefix='   ', depth=1):

      min_order = 2
      max_depth = 4

      if param.nbsolpts <= min_order:
         print(f"Can't make a preconditioner with order less than {min_order}."
               f" (currently asking for order {param.nbsolpts - 1})")
         raise ValueError

      if param.equations != 'shallow_water':
         raise ValueError('Preconditioner is only implemented for the shallow water equations. '
                          'Need to make it a bit more flexible')

      self.num_iter         = 0
      self.num_precond_iter = 0
      self.remaining_uses   = 0
      self.max_num_uses     = 1
      self.precond_time     = 0.0
      self.big_param        = copy(param)
      self.ptopo            = ptopo
      self.rhs_func         = rhs_func
      self.rank             = ptopo.rank
      self.prefix           = prefix
      self.depth            = depth

      self.filter_before = True if param.precond_filter_before == 1 else False
      self.filter_during = True if param.precond_filter_during == 1 else False
      self.filter_after  = True if param.precond_filter_after == 1 else False

      self.big_order    = param.nbsolpts
      self.small_order  = self.big_order - 1
      self.num_elements = param.nb_elements_horizontal

      self.max_iter = 1000
      # if self.small_order == 2:
      #    self.max_iter = 2
      # elif self.small_order == 3:
      #    self.max_iter = 2
      # elif self.small_order == 4:
      #    self.max_iter = 4

      print(f'Creating a preconditioner of order {self.small_order}')

      self.big_param.filter_apply = True
      self.big_operators = DFR_operators(geometry, self.big_param)

      self.small_param = copy(param)
      self.small_param.nbsolpts = self.small_order
      self.small_param.filter_apply = True
      self.small_param.filter_order = 8

      self.small_geom      = cubed_sphere(param.nb_elements_horizontal, param.nb_elements_vertical, self.small_order,
                                          param.λ0, param.ϕ0, param.α0, param.ztop, self.ptopo, self.small_param)
      self.small_operators = DFR_operators(self.small_geom, self.small_param)
      self.small_metric    = Metric(self.small_geom)
      _, self.small_topo   = initialize_sw(self.small_geom, self.small_metric, self.small_operators, self.small_param)

      self.interpolator = LagrangeSimpleInterpolator(geometry)

      self.big_shape      = None
      self.big_mat        = None
      self.small_shape    = None
      self.small_rhs      = None
      self.small_mat      = None
      self.preconditioner = None
      if self.small_order > min_order and self.depth < max_depth:
         self.preconditioner = DG_preconditioner(
            self.small_param, self.small_geom, self.ptopo, self.small_operators, self.rhs_func, self.prefix + '   ',
            self.depth + 1)

   def total_num_iter(self):
      previous_level = self.preconditioner.total_num_iter() if self.preconditioner else 0
      return self.num_iter + previous_level

   def init_time_step(self, matvec_func, dt, field, matvec_handle):
      """Compute (recursively) the matrix caller that will be used to solve the smaller problem"""
      lowres_field      = self.restrict(field)
      self.big_shape    = field.shape
      self.small_shape  = lowres_field.shape

      self.small_rhs = lambda vec : self.rhs_func(
         vec, self.small_geom, self.small_operators, self.small_metric, self.small_topo,  self.ptopo,
         self.small_param.nbsolpts, self.num_elements, self.small_param.case_number, self.filter_during)

      self.small_mat = lambda vec : matvec_func(vec, dt, lowres_field, self.small_rhs)

      self.big_mat = matvec_handle

      self.remaining_uses = self.max_num_uses
      self.precond_time   = 0.0

      if self.preconditioner:
         self.preconditioner.init_time_step(matvec_func, dt, lowres_field, matvec_handle)

   def restrict(self, field):
      return self.interpolator.eval_grid_fast(field, self.small_order, self.big_order, equidistant=False)

   def prolong(self, field):
      return self.interpolator.eval_grid_fast(field, self.big_order, self.small_order, equidistant=False)

   def apply(self, vec):

      if self.remaining_uses <= 0:
         return vec

      self.remaining_uses -= 1

      start_time = time()

      input_vec_grid = self.restrict(vec.reshape(self.big_shape))

      input_vec = input_vec_grid.flatten()

      if self.preconditioner:
         self.preconditioner.remaining_uses = self.preconditioner.max_num_uses

      output_vec, _, num_iter, _ = fgmres(
         self.small_mat, input_vec, preconditioner=self.preconditioner, tol=1e-4, maxiter=self.max_iter)
      result_grid = self.prolong(output_vec.reshape(self.small_shape))

      result = result_grid.flatten()

      stop_time = time()

      self.precond_time += stop_time - start_time
      self.num_iter     += num_iter

      print(f'{self.prefix}Preconditioned in {num_iter} iterations (total {self.total_num_iter()}) '
            f'and {self.precond_time:.2f} s')

      return result

   def __call__(self, vec):
      return self.apply(vec)
