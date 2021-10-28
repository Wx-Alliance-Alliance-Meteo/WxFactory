import functools
import math
import mpi4py
import numpy

from copy import copy
from time import time

from cubed_sphere  import cubed_sphere
from initialize    import initialize_sw
from interpolation import interpolator
from linsol        import fgmres, global_norm
from matrices      import DFR_operators
from matvec        import matvec_rat
from metric        import Metric
from rhs_sw        import rhs_sw

from definitions import idx_h, idx_hu1, idx_hu2, gravity

def explicit_euler_smoothe(A, b, x, h, num_iter=1):
   for _ in range(num_iter):
      res = (b - A(x))
      # print(f'res: {global_norm(res)}')
      x = x + h * res
   return x

def runge_kutta_stable_smoothe(A, b, x, h, num_iter=1, first_zero=False):
   alpha1 = 0.145
   alpha2 = 0.395
   for _ in range(num_iter):
      first_res = b if first_zero else (b - A(x))
      first_zero = False
      s1 = x + alpha1 * h * first_res
      s2 = x + alpha2 * h * (b - A(s1))
      x = x + h * (b - A(s2))
   return x

def runge_kutta_gef_smoothe(A, b, x, h, num_iter=1):
   for _ in range(num_iter):
      x1 = x + h * (b - A(x))
      x2 = 0.75 * x + 0.25 * x1 + 0.25 * h * (b - A(x))
      x = 1.0/3.0 * x + 2.0/3.0 * x2 + 2.0/3.0 * h * (b - A(x))
   return x

class MultigridLevel:
   def __init__(self, param, ptopo, rhs, discretization, source_num_elem, target_num_elem, source_order, target_order, cfl):
      p = copy(param)
      p.nb_elements_horizontal = source_num_elem
      p.nbsolpts = source_order if discretization == 'dg' else 1
      p.discretization = discretization
      
      self.param = p
      self.cfl   = cfl
      self.work_ratio = (source_order / param.initial_nbsolpts) ** 2

      print(
         f'Grid level! nb_elem_horiz = {source_num_elem}, target {target_num_elem},'
         f' discr = {discretization}, source order = {source_order}, target order = {target_order}, nbsolpts = {p.nbsolpts}'
         f' work ratio = {self.work_ratio}'
         )

      # Initialize problem for this level
      self.geometry = cubed_sphere(p.nb_elements_horizontal, p.nb_elements_vertical, p.nbsolpts, p.λ0, p.ϕ0, p.α0, p.ztop, ptopo, p)
      operators     = DFR_operators(self.geometry, p)
      self.metric   = Metric(self.geometry)
      field, topo   = initialize_sw(self.geometry, self.metric, operators, p)

      # Now set up the various operators: RHS, smoother, restriction, prolongation
      # matrix-vector product is done at every step, so not here

      self.rhs_operator = lambda vec, rhs=rhs, p=self.param, geom=self.geometry, op=operators, met=self.metric, topo=topo, ptopo=ptopo: \
         rhs(vec, geom, op, met, topo, ptopo, p.nbsolpts, p.nb_elements_horizontal, p.case_number, False)
      # self.rhs_operator = functools.partial(rhs, geom=self.geometry, mtrx=operators, metric=self.metric, topo=topo,
      #    ptopo=ptopo, nbsolpts=p.nbsolpts, nb_elements_horizontal=p.nb_elements_horizontal, case_number=p.case_number, filter_rhs=False)

      self.smoothe = runge_kutta_stable_smoothe
      self.smoother_work_unit = 3.0

      if target_order > 0:
         interp_method     = 'bilinear' if discretization == 'fv' else 'lagrange'
         self.interpolator = interpolator(discretization, source_order, discretization, target_order, interp_method)
         self.restrict     = lambda vec, op=self.interpolator, sh=field.shape: op(vec.reshape(sh))
         restricted_shape  = self.restrict(field).shape
         self.prolong      = lambda vec, op=self.interpolator, sh=restricted_shape: op(vec.reshape(sh), reverse=True)
      else:
         self.interpolator = lambda a, b=None : None
         self.restrict     = lambda a: None
         self.prolong      = lambda a: None

      # Compute grid spacing
      X1, X2 = self.geometry.X1, self.geometry.X2
      self.dx = numpy.empty_like(X1)
      self.dy = numpy.empty_like(X2)

      self.dx[:,  0]   = X1[:, 1] - X1[:, 0]
      self.dx[:, -1]   = X1[:, -1] - X1[:, -2]
      self.dx[:, 1:-1] = (X1[:, 2:] - X1[:, :-2]) * 0.5

      self.dy[ 0, :]   = X2[ 1, :] - X2[ 0, :]
      self.dy[-1, :]   = X2[-1, :] - X2[-2, :]
      self.dy[1:-1, :] = (X2[2:, :] - X2[:-2, :]) * 0.5

      self.elem_size = numpy.minimum(self.dx.min(), self.dy.min()) # Get min size in either direction (per element)

   def compute_pseudo_dt(self, field):
      """
      Compute pseudo time step from CFL condition
      """
      # Extract the variables
      h  = field[idx_h]
      u1 = field[idx_hu1] / h
      u2 = field[idx_hu2] / h

      # Compute variables in global coordinates
      real_u1 = u1 * self.dx / 2.0
      real_u2 = u2 * self.dy / 2.0

      real_h_11 = self.metric.H_contra_11 * self.dx * self.dy / 4.0
      real_h_22 = self.metric.H_contra_22 * self.dx * self.dy / 4.0

      # Get max velocity (per element)
      v1 = numpy.absolute(real_u1) + numpy.sqrt(real_h_11 * gravity * h)
      v2 = numpy.absolute(real_u2) + numpy.sqrt(real_h_22 * gravity * h)

      vel = numpy.maximum(v1, v2)

      # The pseudo dt (per element)
      pseudo_dt = self.cfl * self.elem_size / vel

      numpy.set_printoptions(precision=1)
      # print(f'pseudo dt:\n{pseudo_dt}')
      print(f'pseudo dt max: {pseudo_dt.max()}, min: {pseudo_dt.min()}, avg: {numpy.average(pseudo_dt)} ')

      self.pseudo_dt = numpy.broadcast_to(pseudo_dt, field.shape)

      return self.pseudo_dt

   def init_time_step(self, matvec, field, dt):
      self.matrix_operator = lambda vec, mat=matvec, dt=dt, f=field, rhs=self.rhs_operator : mat(numpy.ravel(vec), dt, f, rhs)
      self.compute_pseudo_dt(field)
      return self.restrict(field)

class Multigrid:

   def __init__(self, param, ptopo, discretization) -> None:

      self.max_level = param.initial_nbsolpts
      self.min_level = 1

      if discretization == 'fv':
         self.num_levels = int(math.log2(param.initial_nbsolpts)) + 1
         if 2**(self.num_levels - 1) != param.initial_nbsolpts:
            raise ValueError('Cannot do h/fv-multigrid stuff if the order of the problem is not a power of 2 (currently {param.initial_nbsolpts})')
      elif discretization == 'dg':
         self.num_levels = param.initial_nbsolpts

      if param.equations == 'shallow_water':
         self.rhs = rhs_sw
      else:
         raise ValueError('Cannot use the multigrid solver with anything other than shallow water')

      self.matvec = matvec_rat
      self.use_solver = (param.mg_smoothe_only <= 0)
      self.num_pre_smoothing = param.num_pre_smoothing
      self.num_post_smoothing = param.num_post_smoothing
      self.cfl = param.mg_cfl

      self.levels = {}

      order    = param.initial_nbsolpts
      num_elem = param.nb_elements_horizontal

      if discretization == 'fv':
         self.next_level = lambda order: order // 2
      else:
         self.next_level = lambda order: order - 1

      for _ in range(self.num_levels):
         print(f'Initializing level {order}')
         new_order = self.next_level(order)
         new_num_elem = num_elem // 2 if discretization == 'fv' else num_elem
         self.levels[order] = MultigridLevel(param, ptopo, self.rhs, discretization, num_elem, new_num_elem, order, new_order, self.cfl)
         order, num_elem = new_order, new_num_elem

   def init_time_step(self, field, dt):
      """
      Compute the matrix-vector operator for every grid level. Also compute the pseudo time step size for each level.
      """
      next_field = field
      order = self.max_level
      while order >= self.min_level:
         next_field = self.levels[order].init_time_step(self.matvec, next_field, dt)
         order = self.next_level(order)

   def iterate(self, b, x0=None, level=-1, coarsest_level=1, gamma=1, in_first_zero=False):
      """
      Do one pass of the multigrid algorithm.

      Arguments:
      b         -- Right-hand side of the system we want to solve
      x0        -- An estimate of the solution. Might be anything
      level     -- The level (depth) at which we are within the multiple grids. 0 is the coarsest grid
      gamma     -- (optional) How many passes to do at the next grid level
      """

      if level < 0: level = self.max_level
      level_work = 0.0

      if x0 is None:
         x0 = numpy.zeros_like(b)

      lvl_param = self.levels[level]
      A         = lvl_param.matrix_operator
      smoothe   = lvl_param.smoothe
      restrict  = lvl_param.restrict
      prolong   = lvl_param.prolong
      dt        = numpy.ravel(lvl_param.pseudo_dt)

      # Pre smoothing
      x = smoothe(A, b, x0, dt, self.num_pre_smoothing, first_zero=in_first_zero)

      level_work += lvl_param.smoother_work_unit * self.num_pre_smoothing * lvl_param.work_ratio

      if level > coarsest_level:                      # Go down a level and solve that
         residual = numpy.ravel(restrict(b - A(x)))   # Compute the (restricted) residual of the current grid level system
         v = numpy.zeros_like(residual)               # A guess of the next solution (0 is pretty good, that's what we're aiming for)
         first_zero = True
         for _ in range(gamma):
            v, work = self.iterate(residual, v, self.next_level(level), coarsest_level=coarsest_level, gamma=gamma, in_first_zero=first_zero)  # MG pass on next lower level
            first_zero = False
            level_work += work
         x = x + numpy.ravel(prolong(v))              # Correction
      else:
         # Just directly solve the problem
         # That's no very good, we should not use FGMRES to precondition FGMRES...
         if self.use_solver:
            x, _, num_iter, _, _ = fgmres(A, b, x0=x, tol=1e-5)
            level_work += num_iter * lvl_param.work_ratio
      
      # Post smoothing
      x = smoothe(A, b, x, dt, self.num_post_smoothing)
      level_work += self.num_post_smoothing * lvl_param.smoother_work_unit * lvl_param.work_ratio

      return x, level_work

   def solve(self, b, x0=None, field=None, dt=None, coarsest_level=1, tolerance=1e-7, gamma=1, max_num_it=100):
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

      # Initial guess
      first_zero = (x0 is None)
      x = x0 if not first_zero else numpy.zeros_like(b)

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

      level     = self.max_level
      A         = self.levels[level].matrix_operator
      num_it    = 0
      for it in range(max_num_it):
         x, work = self.iterate(b, x, level, coarsest_level=coarsest_level, gamma=gamma, in_first_zero=first_zero)
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


def global_norm(vec):
   """Compute vector norm across all PEs"""
   local_sum = vec @ vec
   return math.sqrt( mpi4py.MPI.COMM_WORLD.allreduce(local_sum) )
