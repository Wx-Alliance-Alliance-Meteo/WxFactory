from copy import copy
from time import time

import numpy

from cubed_sphere    import cubed_sphere
from initialize      import initialize_sw
from interpolation   import interpolator
from linsol          import fgmres
from matvec          import matvec_rat
from matrices        import DFR_operators
from metric          import Metric
from multigrid       import MG_params, mg_solve, mg
from rhs_sw          import rhs_sw


def select_order(origin_order, origin_field):
   if origin_field == 'dg':
      return origin_order

   if origin_order % 2 != 0:
      raise ValueError(f'order {origin_order} is not a multiple of 2, so it\'s going to be hard to precondition.')

   return origin_order // 2


class FV_preconditioner:

   def __init__(self, param, sample_field, ptopo, precond_type=1, origin_field='dg', origin_order=None, prefix=''):
      self.max_iter = 1000
      self.precond_type = precond_type

      if self.precond_type not in [1, 2]:
         raise ValueError('precond_type can only be 1 or 2')

      # print(f'Params:\n{param}')

      self.origin_order = param.nbsolpts if origin_order is None else origin_order
      self.dest_order = select_order(self.origin_order, origin_field)

      interp_method = param.dg_to_fv_interp
      ok_interps = ['l2-norm', 'lagrange']
      if not interp_method in ok_interps:
         print(f'ERROR: invalid interpolation method for DG to FV conversion ({interp_method}). Should pick one of {ok_interps}. Choosing "lagrange" as default.')
         interp_method = 'lagrange'

      interpolate_fct = interp_method if origin_field == 'dg' else 'bilinear'
      self.interpolate = interpolator(origin_field, self.origin_order, 'fv', self.dest_order, interpolate_fct)

      print(f'origin order: {self.origin_order}, dest order: {self.dest_order}')

      # Create a set of parameters for the FV formulation
      self.param = copy(param)
      self.param.discretization = 'fv'
      if origin_field == 'dg':
         self.param.nb_elements_horizontal = self.param.nb_elements_horizontal * self.dest_order
      else:
         self.param.nb_elements_horizontal = self.param.nb_elements_horizontal // 2
      self.param.nbsolpts               = 1

      if self.param.equations != 'shallow_water':
         raise ValueError('Preconditioner is only implemented for the shallow water equations. '
                          'Need to make it a bit more flexible')

      # Finite volume formulation of the problem
      self.ptopo          = ptopo
      self.dest_geom      = cubed_sphere(self.param.nb_elements_horizontal, self.param.nb_elements_vertical,
                                         self.param.nbsolpts, self.param.λ0, self.param.ϕ0, self.param.α0, self.param.ztop,
                                         self.ptopo, self.param)
      self.dest_operators = DFR_operators(self.dest_geom, self.param)
      self.dest_metric    = Metric(self.dest_geom)
      dest_field, self.dest_topo = initialize_sw(self.dest_geom, self.dest_metric, self.dest_operators, self.param)

      self.dest_field_shape   = dest_field.shape
      self.origin_field_shape = sample_field.shape
      self.rhs_function       = rhs_sw

      self.dest_matrix = None
      self.dest_rhs    = None

      self.prefix = prefix
      self.preconditioner = None

      self.mg_params = None
      self.mg_params = MG_params(self.param, ptopo)

      print(f'Origin field shape: {self.origin_field_shape}, dest field shape: {self.dest_field_shape}')

      # if self.dest_order > 3:
      #    self.preconditioner = FV_preconditioner(self.param, dest_field, ptopo, origin_field='fv', origin_order=self.dest_order, prefix=self.prefix+'  ')

      self.total_iter = 0
      self.total_time = 0.0

   def restrict(self, vec):
      return self.interpolate(vec.reshape(self.origin_field_shape))

   def prolong(self, vec):
      return self.interpolate(vec.reshape(self.dest_field_shape), reverse=True)

   def apply(self, vec):
      """
      Apply the preconditioner on the given vector
      """

      t0 = time()

      input_vec = numpy.ravel(self.restrict(vec))
     
      if self.precond_type == 1:    # Finite volume preconditioner (reference, or simple FV)
         output_vec, _, num_iter, _, _ = fgmres(
            self.dest_matrix, input_vec, preconditioner=self.preconditioner, tol=self.param.precond_tolerance, maxiter=self.max_iter)
      elif self.precond_type == 2:  # Multigrid preconditioner
         output_vec, num_iter, mg_time = mg_solve(
            input_vec, self.dt, self.mg_params, tolerance=self.param.precond_tolerance, max_num_it=1)

      self.last_solution = output_vec

      output_vec = numpy.ravel(self.prolong(output_vec))

      # Some stats
      t1 = time()
      precond_time = t1 - t0
      self.total_time += precond_time
      self.total_iter += num_iter

      print(f'{self.prefix}Preconditioned in {num_iter} iterations and {precond_time:.2f} s')

      return output_vec

   def init_time_step(self, dt, field):
      """
      Prepare the preconditioner for solving one time step of the problem.

      This implies
         - computing the latest value of the variables vector in the finite volume formulation
         - assembling the RHS function and the matrix-vector operator
         - computing the matrix-vector operator for each grid level (if using the MG preconditioner)
      """

      self.dest_field = self.restrict(field)
      self.dest_rhs = lambda vec: self.rhs_function(
         vec, self.dest_geom, self.dest_operators, self.dest_metric, self.dest_topo, self.ptopo, self.param.nbsolpts,
         self.param.nb_elements_horizontal, self.param.case_number, False)
      self.dest_matrix = lambda vec: matvec_rat(vec, dt, self.dest_field, self.dest_rhs)

      self.dt = dt
      self.last_solution = numpy.ravel(numpy.zeros_like(self.dest_field))

      if self.preconditioner:
         self.preconditioner.init_time_step(dt, self.dest_field)

      if self.mg_params:
         self.mg_params.init_time_step(self.dest_field, dt)

   def __call__(self, vec):
      return self.apply(vec)
