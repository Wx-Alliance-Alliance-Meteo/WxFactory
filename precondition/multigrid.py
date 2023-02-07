import functools
from copy         import deepcopy
from time         import time
from typing       import Callable, Dict, Optional, Tuple, Union

import numpy
from scipy.sparse.linalg import LinearOperator

from common.definitions          import idx_2d_rho, idx_2d_rho_u, idx_2d_rho_w
from common.interpolation        import Interpolator
from geometry.cartesian_2d_mesh  import Cartesian2d
from geometry.cubed_sphere       import CubedSphere
from geometry.matrices           import DFR_operators
from init.init_state_vars        import init_state_vars
from precondition.smoother       import kiops_smoothe, exponential as exp_smoothe, rk_smoothing, rk1_smoothing
from rhs.rhs_selector            import rhs_selector
from solvers.linsol              import fgmres, global_norm
from solvers.matvec              import matvec_rat
from solvers.nonlin              import KrylovJacobian

# For type hints
from common.parallel        import Distributed_World
from common.program_options import Configuration


class MultigridLevel:
   """
   Class that contains all the parameters and operators describing one level of the multrigrid algorithm
   """

   # Type hints
   restrict:         Callable[[numpy.ndarray], numpy.ndarray]
   prolong:          Callable[[numpy.ndarray], numpy.ndarray]
   pre_smoothe:      Callable[[LinearOperator, numpy.ndarray, numpy.ndarray], numpy.ndarray]
   post_smoothe:     Callable[[LinearOperator, numpy.ndarray, numpy.ndarray], numpy.ndarray]
   matrix_operator:  Callable[[numpy.ndarray], numpy.ndarray]
   pseudo_dt:        float
   jacobian:         KrylovJacobian

   def __init__(self, param: Configuration, ptopo: Distributed_World, discretization: str, nb_elem_horiz: int,
                nb_elem_vert: int, source_order: int, target_order: int, ndim: int):

      p = deepcopy(param)
      p.nb_elements_horizontal = nb_elem_horiz
      p.nb_elements_vertical   = nb_elem_vert
      p.nbsolpts = source_order if discretization == 'dg' else 1
      p.discretization = discretization

      self.param = p
      self.ptopo = ptopo

      self.num_pre_smoothe = self.param.num_pre_smoothe
      self.num_post_smoothe = self.param.num_post_smoothe

      self.ndim = ndim

      if self.param.verbose_precond:
         print(
            f'Grid level! nb_elem_horiz = {nb_elem_horiz}, nb_elem_vert = {nb_elem_vert} '
            f' discr = {discretization}, source order = {source_order},'
            f' target order = {target_order}, nbsolpts = {p.nbsolpts}'
            f' num_mg_levels: {p.num_mg_levels}'
            f' exp radius: {p.exp_smoothe_spectral_radius}'
            # f' work ratio = {self.work_ratio}'
            )

      # Initialize problem for this level
      if p.grid_type == 'cubed_sphere':
         self.geometry = CubedSphere(p.nb_elements_horizontal, p.nb_elements_vertical, p.nbsolpts, p.λ0, p.ϕ0, p.α0,
                                     p.ztop, ptopo, p)
      elif p.grid_type == 'cartesian2d':
         self.geometry = Cartesian2d((p.x0, p.x1), (p.z0, p.z1), p.nb_elements_horizontal, p.nb_elements_vertical,
                                     p.nbsolpts)

      operators = DFR_operators(self.geometry, p.filter_apply, p.filter_order, p.filter_cutoff)

      field, topo, self.metric = init_state_vars(self.geometry, operators, self.param)
      self.rhs_operator, _, _ = rhs_selector(self.geometry, operators, self.metric, topo, ptopo, self.param)
      if self.param.verbose_precond: print(f'field shape: {field.shape}')
      self.shape = field.shape

      # Now set up the various operators: RHS, smoother, restriction, prolongation
      # matrix-vector product is done at every step, so not here

      self.work_ratio = (source_order / param.initial_nbsolpts) ** self.ndim
      self.smoother_work_unit = 3.0

      if target_order > 0:
         interp_method         = 'bilinear' if discretization == 'fv' else 'lagrange'
         self.interpolator     = Interpolator(discretization, source_order, discretization, target_order, interp_method,
                                              self.param.grid_type, self.ndim, verbose=self.param.verbose_precond)
         self.restrict         = lambda vec, op=self.interpolator, sh=field.shape: op(vec.reshape(sh))
         self.restricted_shape = self.restrict(field).shape
         self.prolong          = lambda vec, op=self.interpolator, sh=self.restricted_shape: \
                                    op(vec.reshape(sh), reverse=True)
      else:
         self.restrict = lambda x: x
         self.prolong  = lambda x: x

      self.pre_smoothe = lambda A, b, x: x
      if param.mg_smoother == 'kiops':
         self.pre_smoothe = functools.partial(kiops_smoothe, real_dt=param.dt, dt_factor=param.kiops_dt_factor)
      elif param.mg_smoother == 'exp':
         # self.smoothe[level] = functools.partial(exp_smoothe, target_spectral_radius=4, global_dt=param.dt, niter=6)
         self.pre_smoothe = functools.partial(exp_smoothe,
                                       target_spectral_radius=self.param.exp_smoothe_spectral_radius,
                                       global_dt=param.dt,
                                       niter=self.param.exp_smoothe_nb_iter)
         if self.param.verbose_precond: print(f'spectral radius for level = {self.param.exp_smoothe_spectral_radius}')

      self.post_smoothe = self.pre_smoothe

   def prepare(self, dt: float, field: numpy.ndarray, prev_field:Optional[numpy.ndarray] = None) \
         -> Tuple[numpy.ndarray, Optional[numpy.ndarray]]:
      """ Initialize structures and data that will be used for preconditioning during the ongoing time step """

      if self.param.mg_smoother in ['erk1', 'erk3']:
         cfl       = self.param.pseudo_cfl
         factor    = 1.0 / (self.ndim * (2 * self.param.nbsolpts + 1))
         delta_min = abs(1.- self.geometry.solutionPoints[-1]) * min(self.geometry.Δx, self.geometry.Δz)
         speed_max = numpy.maximum( abs( 343. +  field[idx_2d_rho_u,:,:] /  field[idx_2d_rho,:,:] ),
                                    abs( 343. +  field[idx_2d_rho_w,:,:] /  field[idx_2d_rho,:,:]) )

         self.pseudo_dt = numpy.amin(delta_min * factor / speed_max) * cfl / dt
         if self.param.verbose_precond:
            print(f'pseudo_dt = {self.pseudo_dt}')

         if self.param.mg_smoother == 'erk1':
            self.pre_smoothe  = functools.partial(rk1_smoothing, h=self.pseudo_dt)
         elif self.param.mg_smoother == 'erk3':
            self.pre_smoothe  = functools.partial(rk_smoothing, h=self.pseudo_dt)

         self.post_smoothe = self.pre_smoothe

      ##########################################
      # Matvec function of the system to solve
      if self.param.time_integrator in ['ros2', 'rosexp2', 'partrosexp2', 'strang_epi2_ros2', 'strang_ros2_epi2']:
         self.matrix_operator = functools.partial(matvec_rat, dt=dt, Q=field, rhs=self.rhs_operator(field),
                                                  rhs_handle=self.rhs_operator)

      elif self.param.time_integrator == 'crank_nicolson':
         cn_fun = CrankNicolsonFunFactory(field, dt, self.rhs_operator)

         # self.cn_fun = lambda Q_plus: (Q_plus - self.fv_field) / dt - 0.5 * ( self.fv_rhs_fun(Q_plus) + self.fv_rhs_fun(self.fv_field) )
         # self.cn_fun = cn_fun
         self.jacobian = KrylovJacobian(numpy.ravel(field), numpy.ravel(cn_fun(field)), cn_fun, fgmres_restart=10,
                                        fgmres_maxiter=1, fgmres_precond=None)
         self.matrix_operator = self.jacobian.op

      elif self.param.time_integrator == 'bdf2':
         if prev_field is None:
            raise ValueError(f'Need to specify Q_prev when using BDF2')
         nonlin_fun = Bdf2FunFactory(field, prev_field, dt, self.rhs_operator)
         self.jacobian = KrylovJacobian(numpy.ravel(field), numpy.ravel(nonlin_fun(field)), nonlin_fun,
               fgmres_restart=10, fgmres_maxiter=1, fgmres_precond=None)
         self.matrix_operator = self.jacobian.op

      else:
         raise ValueError(f'Multigrid method not made to work with integrator "{self.param.time_integrator}" yet')

      restricted_field      = self.restrict(field)
      restricted_prev_field = self.restrict(prev_field) if prev_field is not None else None

      return restricted_field, restricted_prev_field

class Multigrid:
   levels: Dict[int, MultigridLevel]
   initial_interpolate: Callable[[numpy.ndarray], numpy.ndarray]
   def __init__(self, param, ptopo, discretization, fv_only=False) -> None:

      param = deepcopy(param)

      # Detect problem dimension
      self.ndim = 2
      if param.equations == 'euler' and param.grid_type == 'cubed_sphere': self.ndim = 3

      # Adjust some parameters as needed
      if discretization == 'fv':
         param.nb_elements_horizontal *= param.nbsolpts
         param.nbsolpts = 1
         param.discretization = 'fv'
         if fv_only:
            param.num_mg_levels     = 1
            param.num_pre_smoothe   = 0
            param.num_post_smoothe  = 0
            param.mg_solve_coarsest = True
      else:
         param.discretization = 'dg'

      self.use_solver = param.mg_solve_coarsest
      self.verbose = param.verbose_precond

      # Determine number of multigrid levels
      if discretization == 'fv':
         self.max_num_levels = int(numpy.log2(param.initial_nbsolpts)) + 1
         self.max_num_fv_elems = 2**(self.max_num_levels - 1)
         self.extra_fv_step = (self.max_num_fv_elems != param.initial_nbsolpts)
      elif discretization == 'dg':
         # Subtract 1 because we include level 0
         self.max_num_levels = param.initial_nbsolpts
      else:
         raise ValueError(f'Unrecognized discretization "{discretization}"')

      param.num_mg_levels = min(param.num_mg_levels, self.max_num_levels)
      print(f'num_mg_levels: {param.num_mg_levels} (max {self.max_num_levels})')

      # Determine correct matrix-vector product function to use
      self.matvec = matvec_rat

      # Determine level-specific parameters for each level (order, num elements)
      if discretization == 'fv':
         self.orders = [self.max_num_fv_elems // (2**i) for i in range(self.max_num_levels + 1)]
         if fv_only: self.orders.insert(0, param.initial_nbsolpts)
         # if self.extra_fv_step:
         #    print(f'There is an extra FV step, from order {param.initial_nbsolpts} to {self.orders[0]}')
         #    self.orders = [param.initial_nbsolpts] + self.orders
         self.elem_counts_hori = [param.nb_elements_horizontal * order // param.initial_nbsolpts for order in self.orders]
         if self.ndim == 3 or param.grid_type == 'cartesian2d':
            self.elem_counts_vert =  [param.nb_elements_vertical * order for order in self.orders]
         else:
            self.elem_counts_vert =  [param.nb_elements_vertical for _ in self.orders]
      elif discretization == 'dg':
         self.orders = [param.initial_nbsolpts - i for i in range(self.max_num_levels + 1)]
         self.elem_counts_hori = [param.nb_elements_horizontal for _ in range(len(self.orders))]
         self.elem_counts_vert = [param.nb_elements_vertical for _ in range(len(self.orders))]
      else:
         raise ValueError(f'Unknown discretization: {discretization}')

      if self.verbose:
         print(f'orders: {self.orders}, h elem counts: {self.elem_counts_hori}, v elem counts: {self.elem_counts_vert}')

      def extended_list(list, target_len):
         diff_len = target_len - len(list)
         new_list = deepcopy(list)
         if diff_len > 0:
            new_list.extend([list[-1] for _ in range(diff_len)])
         # new_list.reverse()
         return new_list

      spectral_radii = extended_list(param.exp_smoothe_spectral_radii, len(self.orders))
      exp_nb_iters   = extended_list(param.exp_smoothe_nb_iters, len(self.orders))
      if self.verbose:
         print(f'spectral radii = {spectral_radii}, num iterations: {exp_nb_iters}')

      # Create config set for each level (whether they will be used or not, in case we want to change that at runtime)
      self.levels = {}
      for i_level in range(self.max_num_levels):
         order                = self.orders[i_level]
         new_order            = self.orders[i_level + 1]
         nb_elem_hori         = self.elem_counts_hori[i_level]
         nb_elem_vert         = self.elem_counts_vert[i_level]
         if self.verbose:
            print(f'Initializing level {i_level}, {discretization}, order {order}->{new_order}, elems {nb_elem_hori}x{nb_elem_vert}')
         param.exp_smoothe_spectral_radius = spectral_radii[i_level]
         param.exp_smoothe_nb_iter = exp_nb_iters[i_level]
         self.levels[i_level] = MultigridLevel(param, ptopo, discretization, nb_elem_hori, nb_elem_vert, order, new_order, self.ndim)

      # Default "0th step" conversion function
      self.initial_interpolate = lambda x : x
      self.get_solution_back   = self.initial_interpolate
      if discretization == 'fv':
         if fv_only:
            self.initial_interpolator = Interpolator(                                                                \
               'dg', param.initial_nbsolpts, 'fv', param.initial_nbsolpts, param.dg_to_fv_interp, param.grid_type,   \
               self.ndim, verbose=self.verbose)
         else:
            self.initial_interpolator = Interpolator(                                                                \
               'dg', param.initial_nbsolpts, 'fv', self.max_num_fv_elems, param.dg_to_fv_interp, param.grid_type,    \
               self.ndim, verbose=self.verbose)

         self.big_shape = self.levels[0].shape

         # Adjust shape of initial interpolation if we are going to do multigrid (rather than just FV)
         if not fv_only:
            int_shape = self.initial_interpolator.elem_interp.shape
            dims = [1, 2, 3] if self.ndim == 3 else [1, 2]
            self.big_shape = list(self.big_shape)
            for d in dims: self.big_shape[d] = self.big_shape[d] * int_shape[1] // int_shape[0]
            self.big_shape = tuple(self.big_shape)
         
         self.initial_interpolate = lambda vec, op=self.initial_interpolator, sh=self.big_shape: op(vec.reshape(sh))
         self.get_solution_back   = lambda vec, op=self.initial_interpolator, sh=self.levels[0].shape: op(vec.reshape(sh), reverse=True)

   def prepare(self, dt: float, field: numpy.ndarray, prev_field:Optional[numpy.ndarray] = None):
      """
      Compute the matrix-vector operator for every grid level. Also compute the pseudo time step size for each level.
      """
      next_field = self.initial_interpolate(field)
      next_prev_field = self.initial_interpolate(prev_field) if prev_field is not None else None
      for i_level in range(self.max_num_levels):
         next_field, next_prev_field = self.levels[i_level].prepare(dt, next_field, next_prev_field)

   def __call__(self, vec: numpy.ndarray, x0:Optional[numpy.ndarray] = None, verbose:Optional[bool] = None):
      if verbose is None: verbose = self.verbose
      return self.apply(vec, x0=x0, verbose=verbose)

   def apply(self, vec, x0=None, verbose=None):
      if verbose is None: verbose = self.verbose
      param = self.levels[0].param

      restricted_vec = numpy.ravel(self.initial_interpolate(vec))
      result = self.iterate(restricted_vec, x0=x0, num_levels=param.num_mg_levels, verbose=verbose)
      prolonged_result = self.get_solution_back(result)
      return numpy.ravel(prolonged_result)

   def compare_res(self, A, b, x, old_res = 0.0):
      res_vec = b
      if x is not None:
         res_vec = b - A(x).flatten()
      abs_res = global_norm(res_vec)
      rel_res = 0.0
      if old_res != 0.0:
         rel_res = (old_res - abs_res) / old_res

      return abs_res, rel_res

   def iterate(self, b: numpy.ndarray, x0:Optional[numpy.ndarray] = None, level:Optional[int] = None, num_levels:int = 1, gamma: int = 1, verbose:Optional[bool] = None) -> numpy.ndarray:
      """
      Do one pass of the multigrid algorithm.

      Arguments:
      b          -- Right-hand side of the system we want to solve
      x0         -- An estimate of the solution. Might be anything
      level      -- The level (depth) at which we are within the multiple grids. 0 is the coarsest grid
      num_levels -- How many levels we will do
      gamma      -- (optional) How many passes to do at the next grid level
      """

      if verbose is None: verbose = self.verbose
      if level is None:   level = 0

      level_work = 0.0

      x = x0  # Initial guess can be None

      lvl_param = self.levels[level]
      A               = lvl_param.matrix_operator
      pre_smoothe     = lvl_param.pre_smoothe
      post_smoothe    = lvl_param.post_smoothe
      restrict        = lvl_param.restrict
      prolong         = lvl_param.prolong

      initial_res = 0.0
      if verbose:
         initial_res, _ = self.compare_res(A, b, x0)
         print(f'Level {level:2d} residual:      {initial_res:.3e}')

      # Pre smoothing
      sm_res = initial_res
      for i in range(lvl_param.num_pre_smoothe):
         x = pre_smoothe(A, b, x)
         if verbose:
            sm_res, rel = self.compare_res(A, b, x, initial_res)
            print(f'..Presmooth  {level:2d}.{i+1:02d} res: {sm_res:.3e} (rel {rel:7.3f})')

      # Avoid having "None" in the solution from now on
      if x is None:
         x = numpy.zeros_like(b)

      # level_work += lvl_param.smoother_work_unit * lvl_param.num_pre_smoothe * lvl_param.work_ratio

      corr_res = sm_res
      if level < num_levels - 1:

         # Residual at the current grid level, and its corresponding correction
         residual   = numpy.ravel(restrict(b - A(x)))
         correction = None
         for _ in range(gamma):
            correction = self.iterate(residual, correction, level + 1, num_levels=num_levels, gamma=gamma, verbose=verbose)  # MG pass on next lower level
            # level_work += work

         before_res = 0.0
         if verbose:
            print(f'Back to level {level:2d}')
            before_res, _ = self.compare_res(A, b, x, sm_res)

         x = x + numpy.ravel(prolong(correction))

         if verbose:
            corr_res, rel = self.compare_res(A, b, x, before_res)
            print(f'..Correction res:       {corr_res:.3e} (rel {rel:7.3f})')

      elif self.use_solver:
         before_res = 0.0
         if verbose: before_res = global_norm(b - A(x).flatten())
         t0 = time()
         x, _, num_iter, _, _ = fgmres(A, b, x0=x, tol=lvl_param.param.precond_tolerance, verbose=False)
         t1 = time()
         if verbose:
            corr_res, rel = self.compare_res(A, b, x, before_res)
            print(f'..Solved res:           {corr_res:.3e} (rel {rel:7.3f}) in {num_iter} iterations and {t1 - t0:.2f}s')
         # level_work += num_iter * lvl_param.work_ratio
      
      # Post smoothing
      for i in range(lvl_param.num_post_smoothe):
         x = post_smoothe(A, b, x)
         if verbose:
            sm_res, rel = self.compare_res(A, b, x, corr_res)
            print(f'..Postsmooth {level:2d}.{i+1:02d} res: {sm_res:.3e} (rel {rel:7.3f})')

      # level_work += lvl_param.num_post_smoothe * lvl_param.smoother_work_unit * lvl_param.work_ratio

      if verbose:
         final_res, rel = self.compare_res(A, b, x, initial_res)
         print(f'Retour {level} residual:      {final_res:.3e} (total relative reduction of {rel:7.3f})')

         if level == num_levels - 1:
            print('End multigrid')

      return x #, level_work

   ####################
   # Solve
   ####################
   def solve(self, b, x0=None, field=None, dt=None, coarsest_level=1, tolerance=1e-7, gamma=1, max_num_it=100, verbose=False):
      """
      Solve the system defined by [self] using the multigrid method.

      Mandatory arguments:
      b         -- The right-hand side of the linear system to solve.

      Optional arguments:
      x0         -- Initial guess for the system solution
      field      -- Current solution vector (before time-stepping). If present, we compute the
                  matrix operators to be used at each grid level
      dt         -- The time step size (in seconds). *Must* be included if [field] is present
      tolerance  -- Size of the residual below which we consider the system solved
      gamma      -- Number of solves at each grid level. You might want to keep it at 1 (the default)
      max_num_it -- Max number of iterations to do. If we reach it, we return no matter what
                  the residual is

      Returns:
      1. The solution
      2. The latest computed residual
      3. Number of iterations performed
      4. Convergence status flag (0 if converge, -1 if not)
      5. List of residuals at every iteration
      """
      t_start = time()
      total_work = 0.0

      norm_b = global_norm(b)
      tol_relative = tolerance * norm_b

      norm_r = norm_b

      # Early return if rhs is zero
      if norm_b < 1e-15:
         return numpy.zeros_like(b), 0.0, 0, 0, [(0.0, time() - t_start, 0)]

      # Init system for this time step, if not done already
      if field is not None:
         if dt is None: dt = 1.0
         self.init_time_step(field, dt)

      residuals = [(norm_r / norm_b, time() - t_start, 0)]

      x = x0  # Initial guess (can be None)

      level     = self.max_level
      A         = self.levels[level].matrix_operator
      num_it    = 0
      for it in range(max_num_it):
         # print(f'Calling iterate, b = \n{b.reshape(self.levels[level].shape)[:5]}')

         x, work = self.iterate(b, x, level, coarsest_level=coarsest_level, gamma=gamma, verbose=verbose)
         first_zero = False
         num_it += 1
         total_work += work

         # Check tolerance exit condition
         if it < max_num_it - 1:
            residual = b - A(x)
            total_work += 1.0
            norm_r = global_norm(residual)
            # print(f'norm_r/b = {norm_r/norm_b:.2e}')
            residuals.append((norm_r / norm_b, time() - t_start, total_work))
            if norm_r < tol_relative:
               return x, norm_r / norm_b, num_it, 0, residuals
            elif norm_r > 2.0 * norm_b:
               return x, norm_r / norm_b, num_it, -1, residuals
         else:
            residuals.append((0.0, time() - t_start, total_work))

      flag = 0
      if num_it >= max_num_it: flag = -1
      return x, norm_r / norm_b, num_it, flag, residuals

class CrankNicolsonFunFactory:
   def __init__(self, Q, dt, rhs_handle):
      self.Q = Q
      self.dt = dt
      self.rhs_handle = rhs_handle

   def __call__(self, Q_plus):
      # print(f'In CN_fun, input shape {Q_plus.shape}, supposed field shape {self.Q.shape}')
      input_vec  = Q_plus.reshape(self.Q.shape)
      q_plus_rhs = self.rhs_handle(input_vec)
      q_rhs      = self.rhs_handle(self.Q)
      result = (input_vec - self.Q) / self.dt - 0.5 * ( q_plus_rhs + q_rhs )
      return numpy.ravel(result)

class Bdf2FunFactory:
   def __init__(self, Q, Q_prev, dt, rhs_handle):
      self.Q      = Q
      self.Q_prev = Q_prev
      self.dt     = dt
      self.rhs_handle = rhs_handle

   def __call__(self, Q_plus):
      input_vec = Q_plus.reshape(self.Q.shape)
      q_plus_rhs = self.rhs_handle(input_vec)
      # q_rhs = self.fv_rhs_fun(self.fv_field)
      result = (input_vec - 4./3. * self.Q + 1./3. * self.Q_prev) / self.dt - 2./3. * q_plus_rhs
      return numpy.ravel(result)
