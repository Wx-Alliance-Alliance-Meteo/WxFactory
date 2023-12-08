import functools
from copy         import deepcopy
import sys
from time         import time
from typing       import Callable, List, Optional

from mpi4py import MPI
import numpy

from common.definitions    import idx_2d_rho, idx_2d_rho_u, idx_2d_rho_w, idx_2d_rho_theta, \
                                  idx_rho, idx_rho_u1, idx_rho_u2, idx_rho_w, idx_rho_theta, \
                                  cpd, cvd, heat_capacity_ratio, p0, Rd
from common.interpolation  import Interpolator
from common.parallel        import DistributedWorld
from common.program_options import Configuration
from geometry              import Cartesian2D, CubedSphere, DFROperators
from init.init_state_vars  import init_state_vars
from precondition.smoother import KiopsSmoother, ExponentialSmoother, RK1Smoother, RK3Smoother, ARK3Smoother
from rhs.rhs_selector      import RhsBundle
from solvers               import fgmres, global_norm, KrylovJacobian, matvec_rat, MatvecOp

MatvecOperator = Callable[[numpy.ndarray], numpy.ndarray]

class MultigridLevel:
   """
   Class that contains all the parameters and operators describing one level of the multrigrid algorithm
   """

   # Type hints
   restrict:         Callable[[numpy.ndarray], numpy.ndarray]
   prolong:          Callable[[numpy.ndarray], numpy.ndarray]
   pre_smoothe:      Callable[[MatvecOperator, numpy.ndarray, numpy.ndarray], numpy.ndarray]
   post_smoothe:     Callable[[MatvecOperator, numpy.ndarray, numpy.ndarray], numpy.ndarray]
   matrix_operator:  MatvecOperator
   pseudo_dt:        float
   jacobian:         KrylovJacobian

   def __init__(self, param: Configuration, ptopo: DistributedWorld, discretization: str, nb_elem_horiz: int,
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

      verbose = self.param.verbose_precond if MPI.COMM_WORLD.rank == 0 else 0

      if verbose > 0:
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
         self.geometry = Cartesian2D((p.x0, p.x1), (p.z0, p.z1), p.nb_elements_horizontal, p.nb_elements_vertical,
                                     p.nbsolpts, p.nb_elements_relief_layer, p.relief_layer_height)

      operators = DFROperators(self.geometry, p)

      field, topo, self.metric = init_state_vars(self.geometry, operators, self.param)
      self.rhs = RhsBundle(self.geometry, operators, self.metric, topo, ptopo, self.param, field.shape)
      if verbose > 0: print(f'field shape: {field.shape}')
      self.shape = field.shape
      self.dtype = field.dtype
      self.size  = field.size

      # Now set up the various operators: RHS, smoother, restriction, prolongation
      # matrix-vector product is done at every step, so not here

      self.work_ratio = (source_order / param.initial_nbsolpts) ** self.ndim
      self.smoother_work_unit = 3.0

      if target_order > 0:
         interp_method         = 'bilinear' if discretization == 'fv' else 'lagrange'
         self.interpolator     = Interpolator(discretization, source_order, discretization, target_order, interp_method,
                                              self.param.grid_type, self.ndim, verbose=verbose)
         self.restrict         = lambda vec, op=self.interpolator, sh=field.shape: op(vec.reshape(sh))
         self.restricted_shape = self.restrict(field).shape
         self.prolong          = lambda vec, op=self.interpolator, sh=self.restricted_shape: \
                                    op(vec.reshape(sh), reverse=True)
      else:
         self.restrict = lambda x: x
         self.prolong  = lambda x: x

      self.pre_smoothe = lambda A, b, x: x
      if param.mg_smoother == 'kiops':
         self.pre_smoothe = KiopsSmoother(param.dt, param.kiops_dt_factor)
      elif param.mg_smoother == 'exp':
         self.pre_smoothe = ExponentialSmoother(self.param.exp_smoothe_nb_iter, self.param.exp_smoothe_spectral_radius,
                                                param.dt)
         if verbose > 0: print(f'spectral radius for level = {self.param.exp_smoothe_spectral_radius}')

      self.post_smoothe = self.pre_smoothe
      self.verbose = verbose

   def prepare(self, dt: float, field: numpy.ndarray, prev_field:Optional[numpy.ndarray] = None) \
         -> tuple[numpy.ndarray, Optional[numpy.ndarray]]:
      """ Initialize structures and data that will be used for preconditioning during the ongoing time step """

      if self.param.mg_smoother in ['erk1', 'erk3', 'ark3']:
         cfl    = self.param.pseudo_cfl
         # factor = 1.0 / (self.ndim * (2 * self.param.nbsolpts + 1))
         factor = 1.0 / (2 * (2 * self.param.nbsolpts + 1))

         min_geo = min(self.geometry.Δx1, min(self.geometry.Δx2, self.geometry.Δx3))
         if isinstance(self.geometry, Cartesian2D):
            delta_min = abs(1.- self.geometry.solutionPoints[-1]) * min_geo
            pressure = p0 * numpy.exp((cpd/cvd) * numpy.log((Rd/p0)*field[idx_2d_rho_theta]))
            sound_speed = numpy.sqrt(heat_capacity_ratio * pressure / field[idx_2d_rho])

            speed_x = sound_speed +  abs(field[idx_2d_rho_u] /  field[idx_2d_rho])
            speed_z = sound_speed +  abs(field[idx_2d_rho_w] /  field[idx_2d_rho])
            speed_max = numpy.maximum(speed_x, speed_z)

         elif isinstance(self.geometry, CubedSphere):
            pressure = p0 * numpy.exp((cpd/cvd) * numpy.log((Rd/p0)*field[idx_rho_theta]))
            sound_speed = numpy.sqrt(heat_capacity_ratio * pressure / field[idx_rho])
            delta_min = abs(1.- self.geometry.solutionPoints[-1])

            sound_x =  numpy.sqrt(self.metric.H_contra_11) * sound_speed
            sound_y =  numpy.sqrt(self.metric.H_contra_22) * sound_speed
            sound_z =  numpy.sqrt(self.metric.H_contra_33) * sound_speed

            speed_x = abs(field[idx_rho_u1] / field[idx_rho])
            speed_y = abs(field[idx_rho_u2] / field[idx_rho])
            speed_z = abs(field[idx_rho_w]  / field[idx_rho])

            speed_max = numpy.maximum(numpy.maximum(sound_x + speed_x, sound_y + speed_y), sound_z + speed_z)

         else:
            raise ValueError(f'Unrecognized type for Geometry object')

         tile_shape = (field.shape[0],)
         for _ in range(field.ndim - 1): tile_shape += (1,)
         speed_max = numpy.ravel(numpy.tile(speed_max, tile_shape))
         # speed_max = numpy.amax(speed_max)

         # self.pseudo_dt = numpy.amin(delta_min * factor / speed_max) * cfl / dt
         self.pseudo_dt = (delta_min * factor / speed_max) * cfl / dt

         # if self.verbose > 1:
         #    sp_z = numpy.mean(speed_z)
         #    so_z = numpy.mean(sound_z)
         #    sp_z_max = numpy.amax(speed_z)
         #    so_z_max = numpy.amax(sound_z)
         #    min_dt = numpy.amin(self.pseudo_dt)
         #    max_dt = numpy.amax(self.pseudo_dt)
         #    avg_dt = numpy.mean(self.pseudo_dt)
         #    print(
         #          # f'factor {abs(1.- self.geometry.solutionPoints[-1])},'
         #          f' h_33 {numpy.mean(numpy.sqrt(self.metric.H_contra_33)):.4f}'
         #          f', min {min_geo:.3f}'
         #          f', delta {delta_min:.3f}'
         #          f', pseudo_dt = {avg_dt:.2e} ({min_dt:.2e} - {max_dt:.2e})'
         #          f', speed max = {numpy.amax(speed_max):.2e} (avg {numpy.mean(speed_max):.2e})'
         #          f', sound_speed = {numpy.mean(sound_speed):.2e} ({numpy.max(sound_speed):.2e})'
         #          f', speed_z / sound_z = {sp_z / so_z :.3f} ({sp_z_max / so_z_max :.3f})')
         #    sys.stdout.flush()
         #    # raise ValueError

         # if self.verbose > 1:
         #    print(f'pseudo_dt = {self.pseudo_dt}')

         if self.param.mg_smoother == 'erk1':
            self.pre_smoothe = RK1Smoother(self.pseudo_dt)
         elif self.param.mg_smoother == 'erk3':
            self.pre_smoothe = RK3Smoother(self.pseudo_dt)
         elif self.param.mg_smoother == 'ark3':
            self.pre_smoothe = ARK3Smoother(self.pseudo_dt, field, dt, self.rhs)

         self.post_smoothe = self.pre_smoothe

      ##########################################
      # Matvec function of the system to solve
      if self.param.time_integrator in ['ros2', 'rosexp2', 'partrosexp2', 'strang_epi2_ros2', 'strang_ros2_epi2']:
         self.matrix_operator = functools.partial(matvec_rat, dt=dt, Q=field, rhs=self.rhs.full(field),
                                                  rhs_handle=self.rhs.full)

      elif self.param.time_integrator == 'crank_nicolson':
         cn_fun = CrankNicolsonFunFactory(field, dt, self.rhs.full)

         # self.cn_fun = lambda Q_plus: (Q_plus - self.fv_field) / dt - 0.5 * ( self.fv_rhs_fun(Q_plus) + 
         #                              self.fv_rhs_fun(self.fv_field) )
         # self.cn_fun = cn_fun
         self.jacobian = KrylovJacobian(numpy.ravel(field), numpy.ravel(cn_fun(field)), cn_fun, fgmres_restart=10,
                                        fgmres_maxiter=1, fgmres_precond=None)
         self.matrix_operator = self.jacobian.op

      elif self.param.time_integrator == 'bdf2':
         if prev_field is None:
            raise ValueError(f'Need to specify Q_prev when using BDF2')
         nonlin_fun = Bdf2FunFactory(field, prev_field, dt, self.rhs.full)
         self.jacobian = KrylovJacobian(numpy.ravel(field), numpy.ravel(nonlin_fun(field)), nonlin_fun,
               fgmres_restart=10, fgmres_maxiter=1, fgmres_precond=None)
         self.matrix_operator = self.jacobian.op

      else:
         raise ValueError(f'Multigrid method not made to work with integrator "{self.param.time_integrator}" yet')

      restricted_field      = self.restrict(field)
      restricted_prev_field = self.restrict(prev_field) if prev_field is not None else None

      return restricted_field, restricted_prev_field

class Multigrid(MatvecOp):
   levels: dict[int, MultigridLevel]
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
      self.verbose = param.verbose_precond if MPI.COMM_WORLD.rank == 0 else 0

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
      if self.verbose: print(f'num_mg_levels: {param.num_mg_levels} (max {self.max_num_levels})')

      # Determine level-specific parameters for each level (order, num elements)
      if discretization == 'fv':
         self.orders = [self.max_num_fv_elems // (2**i) for i in range(self.max_num_levels + 1)]
         if fv_only: self.orders.insert(0, param.initial_nbsolpts)
         # if self.extra_fv_step:
         #    print(f'There is an extra FV step, from order {param.initial_nbsolpts} to {self.orders[0]}')
         #    self.orders = [param.initial_nbsolpts] + self.orders
         self.elem_counts_hori = [param.nb_elements_horizontal * order // param.initial_nbsolpts 
                                  for order in self.orders]
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

      def extended_list(old_list: List, target_len: int):
         diff_len = target_len - len(old_list)
         new_list = deepcopy(old_list)
         if diff_len > 0:
            new_list.extend([old_list[-1] for _ in range(diff_len)])
         elif diff_len < 0:
            for _ in range(-diff_len):
               new_list.pop(-1)
         # new_list.reverse()
         return new_list

      self.spectral_radii = extended_list(param.exp_smoothe_spectral_radii, len(self.orders) - 1)
      self.exp_nb_iters   = extended_list(param.exp_smoothe_nb_iters, len(self.orders) - 1)
      if self.verbose:
         print(f'spectral radii = {self.spectral_radii}, num iterations: {self.exp_nb_iters}')

      # Create config set for each level (whether they will be used or not, in case we want to change that at runtime)
      self.levels = {}
      for i_level in range(self.max_num_levels):
         order                = self.orders[i_level]
         new_order            = self.orders[i_level + 1]
         nb_elem_hori         = self.elem_counts_hori[i_level]
         nb_elem_vert         = self.elem_counts_vert[i_level]
         if self.verbose:
            print(f'Initializing level {i_level}, {discretization}, order {order}->{new_order},'
                  f' elems {nb_elem_hori}x{nb_elem_vert}')
         param.exp_smoothe_spectral_radius = self.spectral_radii[i_level]
         param.exp_smoothe_nb_iter = self.exp_nb_iters[i_level]
         self.levels[i_level] = MultigridLevel(param, ptopo, discretization, nb_elem_hori, nb_elem_vert, order,
                                               new_order, self.ndim)

      super().__init__(self.apply, self.levels[0].dtype, self.levels[0].shape)

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
         self.get_solution_back   = lambda vec, op=self.initial_interpolator, sh=self.levels[0].shape: \
                                       op(vec.reshape(sh), reverse=True)

   def prepare(self, dt: float, field: numpy.ndarray, prev_field:Optional[numpy.ndarray] = None):
      """
      Compute the matrix-vector operator for every grid level. Also compute the pseudo time step size for each level.
      """

      # if MPI.COMM_WORLD.rank == 0: print(f'original field: \n{field[0]}')
      next_field = self.initial_interpolate(field)
      # if MPI.COMM_WORLD.rank == 0: print(f'FV field: \n{next_field[0]}')
      next_prev_field = self.initial_interpolate(prev_field) if prev_field is not None else None
      for i_level in range(self.max_num_levels):
         next_field, next_prev_field = self.levels[i_level].prepare(dt, next_field, next_prev_field)
         # if MPI.COMM_WORLD.rank == 0: print(f'FV field {i_level}: \n{next_field[0]}')
      # raise ValueError

   def __call__(self, vec: numpy.ndarray, x0:Optional[numpy.ndarray] = None, verbose:Optional[int] = None):
      if verbose is None: verbose = self.verbose
      return self.apply(vec, x0=x0, verbose=verbose)

   def apply(self, vec, x0=None, verbose: Optional[int] = None):
      if verbose is None: verbose = self.verbose
      param = self.levels[0].param

      restricted_vec = numpy.ravel(self.initial_interpolate(vec))
      result = self.iterate(restricted_vec, x0=x0, num_levels=param.num_mg_levels, verbose=(verbose>1))
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

   def iterate(self,
               b: numpy.ndarray,
               x0: Optional[numpy.ndarray] = None,
               level: Optional[int] = None,
               num_levels: int = 1,
               gamma: int = 1,
               verbose: Optional[bool] = None) \
                  -> numpy.ndarray:
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
            # MG pass on next lower level
            correction = self.iterate(residual, correction, level + 1, num_levels=num_levels, gamma=gamma,
                                      verbose=verbose)
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
         x, _, _, num_iter, _, _ = fgmres(A, b, x0=x, tol=lvl_param.param.precond_tolerance, restart=100, verbose=False)
         t1 = time()
         if verbose:
            corr_res, rel = self.compare_res(A, b, x, before_res)
            print(f'..Solved res:           {corr_res:.3e} (rel {rel:7.3f}) in {num_iter} iterations'
                  f' and {t1 - t0:.2f}s')
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
