import functools
import numpy
import scipy

from copy import deepcopy
from time import time

from Common.definitions       import idx_h, idx_hu1, idx_hu2, gravity, idx_rho_w
from Common.interpolation     import Interpolator
from Grid.cartesian_2d_mesh   import Cartesian2d
from Grid.cubed_sphere        import CubedSphere
from Grid.matrices            import DFR_operators
from Init.init_state_vars     import init_state_vars
from Rhs.rhs_euler            import rhs_euler
from Rhs.rhs_sw               import rhs_sw
from Solver.kiops             import kiops
from Solver.linsol            import fgmres, global_norm
from Stepper.matvec           import matvec_rat


class MultigridLevel:
   def __init__(self, param, ptopo, discretization, nb_elem_horiz, nb_elem_vert, source_order, target_order, ndim):

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

      print(
         f'Grid level! nb_elem_horiz = {nb_elem_horiz}, nb_elem_vert = {nb_elem_vert} '
         f' discr = {discretization}, source order = {source_order}, target order = {target_order}, nbsolpts = {p.nbsolpts}'
         f' num_mg_levels: {p.num_mg_levels}'
         # f' work ratio = {self.work_ratio}'
         )

      # Initialize problem for this level
      if p.grid_type == 'cubed_sphere':
         self.geometry = CubedSphere(p.nb_elements_horizontal, p.nb_elements_vertical, p.nbsolpts, p.λ0, p.ϕ0, p.α0, p.ztop, ptopo, p)
      elif p.grid_type == 'cartesian2d':
         self.geometry = Cartesian2d((p.x0, p.x1), (p.z0, p.z1), p.nb_elements_horizontal, p.nb_elements_vertical, p.nbsolpts)

      operators     = DFR_operators(self.geometry, p.filter_apply, p.filter_order, p.filter_cutoff)

      field, _, self.metric, self.rhs_operator, _, _ = init_state_vars(self.geometry, operators, ptopo, self.param)
      print(f'field shape: {field.shape}')
      self.shape = field.shape

      # Now set up the various operators: RHS, smoother, restriction, prolongation
      # matrix-vector product is done at every step, so not here

      self.work_ratio = (source_order / param.initial_nbsolpts) ** self.ndim
      self.smoother_work_unit = 3.0

      if target_order > 0:
         interp_method         = 'bilinear' if discretization == 'fv' else 'lagrange'
         self.interpolator     = Interpolator(discretization, source_order, discretization, target_order, interp_method, self.param.grid_type, self.ndim)
         self.restrict         = lambda vec, op=self.interpolator, sh=field.shape: op(vec.reshape(sh))
         self.restricted_shape = self.restrict(field).shape
         self.prolong          = lambda vec, op=self.interpolator, sh=self.restricted_shape: op(vec.reshape(sh), reverse=True)
      else:
         self.interpolator = lambda a, b=None : None
         self.restrict     = lambda a: None
         self.prolong      = lambda a: None

   def prepare(self, dt, field, matvec):
      self.matrix_operator = lambda vec, mat=matvec, dt=dt, f=field, rhs=self.rhs_operator(field), rhs_handle=self.rhs_operator : mat(numpy.ravel(vec), dt, f, rhs, rhs_handle)

      if self.param.mg_smoother == 'kiops':
         self.smoothe = functools.partial(kiops_smoothe, A=self.matrix_operator, real_dt=dt, dt_factor=self.param.kiops_dt_factor)
      else:
         raise ValueError(f'Unsupported smoother for MG: {self.param.mg_smoother}')

      return self.restrict(field)

class Multigrid:
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
            param.num_mg_levels    = 1
            param.num_pre_smoothe  = 0
            param.num_post_smoothe = 0
            param.mg_smoothe_only  = 0
      else:
         param.discretization = 'dg'

      self.use_solver = (param.mg_smoothe_only <= 0)

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

      # # Determine correct RHS function to use
      # if param.equations == 'shallow_water':
      #    self.rhs = rhs_sw
      # elif param.equations == 'euler':
      #    self.rhs = rhs_euler if discretization == 'fv' else rhs_euler
      # else:
      #    raise ValueError(f'Cannot use the multigrid solver with anything other than shallow water or Euler (got {param.equations})')

      # Determine correct matrix-vector product function to use
      self.matvec = matvec_rat

      # Determine level-specific parameters for each level (order, num elements)
      if discretization == 'fv' and not fv_only:
         self.orders = [self.max_num_fv_elems // (2**i) for i in range(self.max_num_levels + 1)]
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

      print(f'orders: {self.orders}, h elem counts: {self.elem_counts_hori}, v elem counts: {self.elem_counts_vert}')

      # Create config set for each level (whether they will be used or not, in case we want to change that at runtime)
      self.levels = {}
      for i_level in range(self.max_num_levels):
         order                = self.orders[i_level]
         new_order            = self.orders[i_level + 1]
         nb_elem_hori         = self.elem_counts_hori[i_level]
         nb_elem_vert         = self.elem_counts_vert[i_level]
         print(f'Initializing level {i_level}, {discretization}, order {order}->{new_order}, elems {nb_elem_hori}x{nb_elem_vert}')
         self.levels[i_level] = MultigridLevel(param, ptopo, discretization, nb_elem_hori, nb_elem_vert, order, new_order, self.ndim)

      # Default "0th step" conversion function
      self.initial_interpolate = lambda x : x
      self.get_solution_back   = self.initial_interpolate
      if discretization == 'fv':
         self.big_shape = self.levels[0].shape
         int_shape = self.initial_interpolator.elem_interp.shape
         if fv_only:
            self.initial_interpolator = Interpolator('dg', param.initial_nbsolpts, 'fv', param.initial_nbsolpts, param.dg_to_fv_interp, param.grid_type, self.ndim)
         else:
            self.initial_interpolator = Interpolator('dg', param.initial_nbsolpts, 'fv', self.max_num_fv_elems, param.dg_to_fv_interp, param.grid_type, self.ndim)

            dims = [1, 2, 3] if self.ndim == 3 else [1, 2]
            self.big_shape = list(self.big_shape)
            for d in dims: self.big_shape[d] = self.big_shape[d] * int_shape[1] // int_shape[0]
            self.big_shape = tuple(self.big_shape)
         
         self.initial_interpolate = lambda vec, op=self.initial_interpolator, sh=self.big_shape: op(vec.reshape(sh))
         self.get_solution_back   = lambda vec, op=self.initial_interpolator, sh=int_shape: op(vec.reshape(sh), reverse=True)

   def prepare(self, dt, field):
      """
      Compute the matrix-vector operator for every grid level. Also compute the pseudo time step size for each level.
      """
      next_field = self.initial_interpolate(field)
      for i_level in range(self.max_num_levels):
         next_field = self.levels[i_level].prepare(dt, next_field, self.matvec)

   ############
   # Call MG
   def __call__(self, vec):
      return self.apply(vec)

   def apply(self, vec):
      param = self.levels[0].param
      shape = self.levels[0].shape

      # result = self.iterate(numpy.ravel(self.initial_interpolate(vec.reshape(shape))), num_levels=param.num_mg_levels, verbose=False)
      result = self.iterate(numpy.ravel(self.initial_interpolate(vec)), num_levels=param.num_mg_levels, verbose=False)
      return numpy.ravel(self.get_solution_back(result))

###############
# The algorithm

   def iterate(self, b, x0=None, level=None, num_levels=1, gamma=1, verbose=False):
      """
      Do one pass of the multigrid algorithm.

      Arguments:
      b         -- Right-hand side of the system we want to solve
      x0        -- An estimate of the solution. Might be anything
      level     -- The level (depth) at which we are within the multiple grids. 0 is the coarsest grid
      gamma     -- (optional) How many passes to do at the next grid level
      """

      if level is None:
         level = 0

      level_work = 0.0

      x = x0  # Initial guess can be None

      lvl_param = self.levels[level]
      A               = lvl_param.matrix_operator
      smoothe         = lvl_param.smoothe
      restrict        = lvl_param.restrict
      prolong         = lvl_param.prolong

      if verbose:
         res = b.flatten()
         if x is not None: res -= A(x.flatten())
         print(f'Residual at level {level}, : {global_norm(res):.4e}')

      # Pre smoothing
      for _ in range(lvl_param.num_pre_smoothe):
         x = smoothe(x, b)
         if verbose: print(f'   Presmooth  : {global_norm(b.flatten() - (A(x.flatten()))):.4e}')

      # Avoid having "None" in the solution from now on
      if x is None:
         x = numpy.zeros_like(b)

      # level_work += lvl_param.smoother_work_unit * lvl_param.num_pre_smoothe * lvl_param.work_ratio

      if level < num_levels - 1:

         # Residual at the current grid level, and its corresponding correction
         residual   = numpy.ravel(restrict(b - A(x)))
         correction = numpy.zeros_like(residual)

         for _ in range(gamma):
            correction = self.iterate(residual, correction, level + 1, num_levels=num_levels, gamma=gamma, verbose=verbose)  # MG pass on next lower level
            # level_work += work

         x = x + numpy.ravel(prolong(correction))

      else:
         # Just directly solve the problem
         # That's no very good, we should not use FGMRES to precondition FGMRES...
         if self.use_solver:
            t0 = time()
            x, _, num_iter, _, _ = fgmres(A, b, x0=x, tol=lvl_param.param.precond_tolerance, verbose=False)
            t1 = time()
            if verbose: print(f'   Solved coarsest grid in {num_iter} iterations and {t1 - t0:.2f}s')
            # level_work += num_iter * lvl_param.work_ratio
      
      # Post smoothing
      for _ in range(lvl_param.num_post_smoothe):
         x = smoothe(x, b)
         if verbose: print(f'   Postsmooth : {global_norm(b.flatten() - (A(x.flatten()))):.4e}')

      # level_work += lvl_param.num_post_smoothe * lvl_param.smoother_work_unit * lvl_param.work_ratio

      if verbose: print(f'retour {level}: {global_norm(b.flatten() - (A(x.flatten())))}')

      if verbose is True and level == num_levels - 1:
         print('End multigrid')
         print(' ')

      return x #, level_work

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

      norm_b = global_op.norm(b)
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
            norm_r = global_op.norm(residual)
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

######################
# Smoothers
######################

def kiops_smoothe(x, b, A, real_dt, dt_factor):

   def residual(t, xx=None):
      if xx is None: return b / real_dt
      return (b - A(xx))/real_dt

   n = b.size
   vec = numpy.zeros((2, n))

   pseudo_dt = dt_factor * real_dt   # TODO : wild guess

   J = lambda v: -A(v) * pseudo_dt / real_dt

   R = residual(0, x)
   vec[1,:] = R.flatten()

   phiv, stats = kiops([1], J, vec, tol=1e-6, m_init=10, mmin=10, mmax=64, task1=False)

#      print('norm phiv', linsol.global_norm(phiv.flatten()))
#      print(f'KIOPS converged at iteration {stats[2]} (using {stats[0]} internal substeps)'
#               f' to a solution with local error {stats[4]:.2e}')
   result = phiv.flatten() * pseudo_dt
   if x is not None: result += x
   return result
