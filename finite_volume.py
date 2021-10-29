from copy import deepcopy
from time import time

import numpy

from cubed_sphere    import cubed_sphere
from initialize      import initialize_sw, initialize_euler
from interpolation   import interpolator
from linsol          import fgmres
from matvec          import matvec_rat
from matrices        import DFR_operators
from metric          import Metric
from multigrid       import Multigrid
from rhs_sw          import rhs_sw
from rhs_euler       import rhs_euler


class FiniteVolume:

   def __init__(self, param, sample_field, ptopo, precond_type='fv', prefix=''):
      self.param        = deepcopy(param)
      self.max_iter     = 1000
      self.precond_type = precond_type

      if self.precond_type not in ['fv', 'fv-mg']:
         raise ValueError('precond_type can only be "fv" (finite volume) or "fv-mg" (multigrid FV)')

      implemented_equations = ['shallow_water', 'Euler']
      if self.param.equations not in implemented_equations:
         raise ValueError(f'Preconditioner is only implemented for the following equations: {implemented_equations}. '
                          'Need to make it a bit more flexible')

      # print(f'Params:\n{param}')

      self.origin_order = param.nbsolpts
      self.dest_order   = self.origin_order

      ndim = 2 if self.param.equations == 'shallow_water' else 3

      self.interpolate = interpolator('dg', self.origin_order, 'fv', self.dest_order, param.dg_to_fv_interp, ndim)

      print(f'origin order: {self.origin_order}, dest order: {self.dest_order}')

      # Create a set of parameters for the FV formulation
      self.param.discretization = 'fv'
      self.param.nb_elements_horizontal = self.param.nb_elements_horizontal * self.dest_order
      self.param.nbsolpts               = 1

      if self.param.equations == 'Euler':
         self.param.nb_elements_vertical = self.param.nb_elements_vertical * self.dest_order

      # Finite volume formulation of the problem
      self.ptopo          = ptopo
      self.dest_geom      = cubed_sphere(self.param.nb_elements_horizontal, self.param.nb_elements_vertical,
                                         self.param.nbsolpts, self.param.λ0, self.param.ϕ0, self.param.α0, self.param.ztop,
                                         self.ptopo, self.param)
      self.dest_operators = DFR_operators(self.dest_geom, self.param)
      self.dest_metric    = Metric(self.dest_geom)

      if self.param.equations == 'Euler':
         dest_field, self.dest_topo = initialize_euler(self.dest_geom, self.dest_metric, self.dest_operators, self.param)
         self.rhs_function          = rhs_euler
      elif self.param.equations == 'shallow_water':
         dest_field, self.dest_topo = initialize_sw(self.dest_geom, self.dest_metric, self.dest_operators, self.param)
         self.rhs_function          = rhs_sw

      self.dest_field_shape   = dest_field.shape
      self.origin_field_shape = sample_field.shape

      self.dest_matrix = None # System mat-vec function for the FV problem
      self.dest_rhs    = None # RHS function for the FV problem

      self.prefix = prefix
      self.preconditioner = None # We could recursively precondition (with FV only) by initializing this

      self.mg_solver = None
      if self.precond_type == 'fv-mg':
         self.mg_solver = Multigrid(self.param, ptopo, 'fv')

      print(f'Origin field shape: {self.origin_field_shape}, dest field shape: {self.dest_field_shape}')

      self.total_iter = 0
      self.total_time = 0.0

   def restrict(self, vec):
      return self.interpolate(vec.reshape(self.origin_field_shape))

   def prolong(self, vec):
      return self.interpolate(vec.reshape(self.dest_field_shape), reverse=True)

   def apply(self, vec, verbose=False):
      """
      Apply the preconditioner on the given vector
      """

      t0 = time()

      input_vec = numpy.ravel(self.restrict(vec))

      if self.precond_type == 'fv':    # Finite volume preconditioner (reference, or simple FV)
         max_num_iter = self.max_iter if self.param.precond_tolerance < 1e-1 else 1
         output_vec, _, num_iter, _, residuals = fgmres(
            self.dest_matrix, input_vec, preconditioner=self.preconditioner, tol=self.param.precond_tolerance, maxiter=max_num_iter)
      elif self.precond_type == 'fv-mg':  # Multigrid preconditioner
         output_vec, _, num_iter, _, residuals = self.mg_solver.solve(input_vec, coarsest_level=self.param.coarsest_mg_order, max_num_it=1, verbose=verbose)

      self.last_solution = output_vec

      output_vec = numpy.ravel(self.prolong(output_vec))

      # Some stats
      t1 = time()
      precond_time = t1 - t0
      self.total_time += precond_time
      self.total_iter += num_iter
      
      work = residuals[-1][2] # Last iteration contains total amount of work up to then
      # print(f'FV precond, work = {work}, residuals = {residuals[:5]}')

      # print(f'{self.prefix}Preconditioned in {num_iter} iterations and {precond_time:.2f} s')

      return output_vec, work

   def init_time_step(self, field, dt):
      """
      Prepare the preconditioner for solving one time step of the problem.

      This implies
         - computing the latest value of the variables vector in the finite volume formulation
         - assembling the RHS function and the matrix-vector operator
         - computing the matrix-vector operator for each grid level (if using the MG preconditioner)
      """

      self.dest_field = self.restrict(field)
      self.dest_rhs = None
      if self.param.equations == 'Euler':
         self.dest_rhs = lambda vec: self.rhs_function(
            vec, self.dest_geom, self.dest_operators, self.dest_metric, self.dest_topo, self.ptopo, self.param.nbsolpts,
            self.param.nb_elements_horizontal, self.param.nb_elements_vertical, self.param.case_number, False)
      elif self.param.equations == 'shallow_water':
         self.dest_rhs = lambda vec: self.rhs_function(
            vec, self.dest_geom, self.dest_operators, self.dest_metric, self.dest_topo, self.ptopo, self.param.nbsolpts,
            self.param.nb_elements_horizontal, self.param.case_number, False)
      self.dest_matrix = lambda vec: matvec_rat(vec, dt, self.dest_field, self.dest_rhs)

      self.dt = dt
      self.last_solution = numpy.ravel(numpy.zeros_like(self.dest_field))

      if self.preconditioner:
         self.preconditioner.init_time_step(dt, self.dest_field)

      if self.mg_solver:
         self.mg_solver.init_time_step(self.dest_field, dt)

   def __call__(self, vec):
      return self.apply(vec)
