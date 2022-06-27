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
# from rhs_euler_fv    import rhs_euler_fv
from rhs_euler       import rhs_euler


class FiniteVolume:

   def __init__(self, param, sample_field, ptopo, precond_type='fv', prefix=''):
      self.param        = deepcopy(param)
      self.max_iter     = 100
      self.precond_type = precond_type

      if self.precond_type not in ['fv', 'fv-mg']:
         raise ValueError('precond_type can only be "fv" (finite volume) or "fv-mg" (multigrid FV)')

      implemented_equations = ['shallow_water', 'Euler']
      if self.param.equations not in implemented_equations:
         raise ValueError(f'Preconditioner is only implemented for the following equations: {implemented_equations}. '
                          'Need to make it a bit more flexible')

      # print(f'Params:\n{param}')

      self.origin_order = param.nbsolpts
      self.fv_order     = self.origin_order

      ndim = 2 if self.param.equations == 'shallow_water' else 3

      self.interpolate = interpolator('dg', self.origin_order, 'fv', self.fv_order, param.dg_to_fv_interp, ndim)

      print(f'origin order: {self.origin_order}, fv order: {self.fv_order}')

      # Create a set of parameters for the FV formulation
      self.param.discretization = 'fv'
      self.param.nb_elements_horizontal = self.param.nb_elements_horizontal * self.fv_order
      self.param.nbsolpts               = 1
      if ndim >= 3:
         self.param.nb_elements_vertical = self.param.nb_elements_vertical * self.fv_order

      # Finite volume formulation of the problem
      self.ptopo        = ptopo
      self.fv_geom      = cubed_sphere(self.param.nb_elements_horizontal, self.param.nb_elements_vertical,
                                       self.param.nbsolpts, self.param.λ0, self.param.ϕ0, self.param.α0, self.param.ztop,
                                       self.ptopo, self.param)
      self.fv_operators = DFR_operators(self.fv_geom, self.param)
      self.fv_metric    = Metric(self.fv_geom)

      if self.param.equations == 'Euler':
         fv_field, self.fv_topo = initialize_euler(self.fv_geom, self.fv_metric, self.fv_operators, self.param)
         # self.generic_rhs_function = rhs_euler_fv
         self.generic_rhs_function = rhs_euler
      elif self.param.equations == 'shallow_water':
         fv_field, self.fv_topo = initialize_sw(self.fv_geom, self.fv_metric, self.fv_operators, self.param)
         self.generic_rhs_function = rhs_sw

      self.fv_field_shape     = fv_field.shape
      self.origin_field_shape = sample_field.shape

      # To be understood as a linear operator by Scipy
      n = 1
      for s in self.fv_field_shape: n *= s
      self.shape = (n, n)
      self.dtype = float

      # self.fv_matrix = None # System mat-vec function for the FV problem
      # self.fv_rhs    = None # RHS function for the FV problem

      self.prefix = prefix

      self.mg_solver = None
      if self.precond_type == 'fv-mg':
         self.mg_solver = Multigrid(self.param, ptopo, 'fv')

      print(f'Origin field shape: {self.origin_field_shape}, fv field shape: {self.fv_field_shape}')

      self.total_iter = 0
      self.total_time = 0.0

   def restrict(self, vec):
      return self.interpolate(vec.reshape(self.origin_field_shape))

   def prolong(self, vec):
      return self.interpolate(vec.reshape(self.fv_field_shape), reverse=True)

   def matvec(self, vec):
      return self.apply(vec)

   def apply(self, vec, verbose=False):
      """
      Apply the preconditioner on the given vector
      """

      t0 = time()

      # print(f'preconditioning \n{vec.reshape(self.origin_field_shape)}')
      # print(f'preconditioning \n{self.restrict(vec)}')

      input_vec = numpy.ravel(self.restrict(vec))

      max_iter = self.max_iter if self.param.precond_tolerance < 1e-1 else 1
      if self.precond_type == 'fv':    # Finite volume preconditioner (reference, or simple FV)
         output_vec, _, num_iter, _, residuals = fgmres(
            self.fv_matrix, input_vec, preconditioner=None, tol=self.param.precond_tolerance, maxiter=max_iter)
      elif self.precond_type == 'fv-mg':  # Multigrid preconditioner
         # output_vec, _, num_iter, _, residuals = self.mg_solver.solve(input_vec, coarsest_level=self.param.coarsest_mg_order, max_num_it=1, verbose=verbose)
         output_vec = self.mg_solver.iterate(input_vec, coarsest_level=self.param.coarsest_mg_order)
         num_iter = 1

      # self.last_solution = output_vec

      output_vec = numpy.ravel(self.prolong(output_vec))

      # Some stats
      t1 = time()
      precond_time = t1 - t0
      self.total_time += precond_time
      self.total_iter += num_iter
      
      # print(f'{self.prefix}Preconditioned in {num_iter} iterations and {precond_time:.2f} s')

      return output_vec

   def prepare(self, dt, field):
      """
      Prepare the preconditioner for solving one time step of the problem.

      This implies
         - computing the latest value of the variables vector in the finite volume formulation
         - assembling the RHS function and the matrix-vector operator
         - computing the matrix-vector operator for each grid level (if using the MG preconditioner)
      """

      self.fv_field = self.restrict(field)
      self.fv_rhs   = None
      if self.param.equations == 'Euler':
         self.fv_rhs = lambda vec: self.generic_rhs_function(
            vec, self.fv_geom, self.fv_operators, self.fv_metric, self.fv_topo, self.ptopo, self.param.nbsolpts,
            self.param.nb_elements_horizontal, self.param.nb_elements_vertical, self.param.case_number)
      elif self.param.equations == 'shallow_water':
         self.fv_rhs = lambda vec: self.generic_rhs_function(
            vec, self.fv_geom, self.fv_operators, self.fv_metric, self.fv_topo, self.ptopo, self.param.nbsolpts,
            self.param.nb_elements_horizontal)
      self.fv_matrix = lambda vec: matvec_rat(vec, dt, self.fv_field, self.fv_rhs(self.fv_field), self.fv_rhs)

      self.dt = dt
      # self.last_solution = numpy.ravel(numpy.zeros_like(self.fv_field))

      if self.mg_solver:
         self.mg_solver.prepare(dt, self.fv_field)

   def __call__(self, vec):
      return self.apply(vec)
