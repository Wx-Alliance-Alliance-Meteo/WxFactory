import math
import mpi4py
import numpy

from copy import copy
from time import time

from cubed_sphere  import cubed_sphere
from initialize    import initialize_sw
from interpolation import compute_dg_to_fv_small_projection, interpolator
from linsol        import fgmres, global_norm
from matrices      import DFR_operators
from matvec        import matvec_rat
from metric        import Metric
from rhs_sw        import rhs_sw

from definitions import idx_h, idx_hu1, idx_hu2, gravity

def explicit_euler_smoothe(A, b, x, h, num_iter=1):
   for i in range(num_iter):
      res = (b - A(x))
      # print(f'res: {global_norm(res)}')
      x = x + h * res
   return x

def runge_kutta_smoothe(A, b, x, h, num_iter=1):
   for i in range(num_iter):
      x1 = x + h * (b - A(x))
      x2 = 0.75 * x + 0.25 * x1 + 0.25 * h * (b - A(x))
      x = 1.0/3.0 * x + 2.0/3.0 * x2 + 2.0/3.0 * h * (b - A(x))
   return x

class MG_params:

   def __init__(self, param, ptopo) -> None:
      max_levels = int(math.log2(param.initial_nbsolpts))
      if 2**max_levels != param.initial_nbsolpts:
         raise ValueError('Cannot do multigrid stuff if the order of the problem is not a power of 2')

      num_levels = param.max_mg_level
      if num_levels < 0:
         num_levels = max_levels
      else:
         num_levels = min(int(num_levels), int(max_levels))

      if param.equations != 'shallow_water':
         raise ValueError('Cannot use the multigrid solver with anything other than shallow water')

      self.max_level = num_levels

      self.rhs = rhs_sw
      self.matvec = matvec_rat
      self.use_solver = (param.mg_smoothe_only <= 0)
      self.num_pre_smoothing = param.num_pre_smoothing
      self.num_post_smoothing = param.num_post_smoothing
      self.cfl = param.mg_cfl

      self.smoothers = {}
      self.interpolators = {}
      self.restrict_operators = {}
      self.prolong_operators = {}
      self.matrix_operators = {}
      self.rhs_operators = {}
      self.params = {}
      self.geometries = {}
      self.metrics = {}

      order = param.initial_nbsolpts
      for level in range(self.max_level, -1, -1):
         p = copy(param)
         p.nb_elements_horizontal = p.nb_elements_horizontal * p.nbsolpts
         p.nbsolpts = 1
         p.discretization = 'fv'
         if level < self.max_level:
            p.nb_elements_horizontal = self.params[level+1].nb_elements_horizontal // 2
         
         self.params[level] = p

         # Initialize problem for this level
         self.geometries[level] = cubed_sphere(p.nb_elements_horizontal, p.nb_elements_vertical, p.nbsolpts, p.λ0, p.ϕ0, p.α0, p.ztop, ptopo, p)
         operators              = DFR_operators(self.geometries[level], p)
         self.metrics[level]    = Metric(self.geometries[level])
         field, topo = initialize_sw(self.geometries[level], self.metrics[level], operators, p)

         # Now set up the various operators: RHS, smoother, restriction, prolongation
         # matrix-vector product is done at every step, so not here

         self.rhs_operators[level] = lambda vec, rhs=self.rhs, p=self.params[level], geom=self.geometries[level], op=operators, met=self.metrics[level], topo=topo, ptopo=ptopo: \
            rhs(vec, geom, op, met, topo, ptopo, p.nbsolpts, p.nb_elements_horizontal, p.case_number, False)

         self.smoothers[level] = explicit_euler_smoothe
         # self.smoothers[level] = runge_kutta_smoothe

         if level > 0:
            self.interpolators[level] = interpolator('fv', order, 'fv', order//2, 'bilinear')
            self.restrict_operators[level] = lambda vec, op=self.interpolators[level], sh=field.shape: op(vec.reshape(sh))

         if level < self.max_level:
            self.prolong_operators[level] = lambda vec, op=self.interpolators[level + 1], sh=field.shape: op(vec.reshape(sh), reverse=True)

         order = order // 2

   def compute_pseudo_dt(self, field, X, Y, h_contra_11, h_contra_22):
      """
      Compute pseudo time step from CFL condition
      """

      # Compute the size of each element
      dx = numpy.empty_like(X)
      dy = numpy.empty_like(Y)

      dx[:,  0]   = X[:, 1] - X[:, 0]
      dx[:, -1]   = X[:, -1] - X[:, -2]
      dx[:, 1:-1] = (X[:, 2:] - X[:, :-2]) * 0.5

      dy[ 0, :]   = Y[ 1, :] - Y[ 0, :]
      dy[-1, :]   = Y[-1, :] - Y[-2, :]
      dy[1:-1, :] = (Y[2:, :] - Y[:-2, :]) * 0.5

      elem_size = numpy.minimum(dx.min(), dy.min()) # Get min size in either direction (per element)

      # Extract the variables
      h  = field[idx_h]
      u1 = field[idx_hu1] / h
      u2 = field[idx_hu2] / h

      # Compute variables in global coordinates
      real_u1 = u1 * dx / 2.0
      real_u2 = u2 * dy / 2.0

      real_h_11 = h_contra_11 * dx * dy / 4.0
      real_h_22 = h_contra_22 * dx * dy / 4.0

      # Get max velocity (per element)
      v1 = numpy.absolute(real_u1) + numpy.sqrt(real_h_11 * gravity * h)
      v2 = numpy.absolute(real_u2) + numpy.sqrt(real_h_22 * gravity * h)

      vel = numpy.maximum(v1, v2)

      # The pseudo dt (per element)
      pseudo_dt = self.cfl * elem_size / vel

      numpy.set_printoptions(precision=1)
      print(f'pseudo dt:\n{pseudo_dt}')
      print(f'max: {pseudo_dt.max()}, min: {pseudo_dt.min()}, avg: {numpy.average(pseudo_dt)} ')

      return numpy.broadcast_to(pseudo_dt, field.shape)

   def init_time_step(self, field, dt):
      """
      Compute the matrix-vector operator for every grid level. Also compute the pseudo time step size for each level.
      """

      self.pseudo_dts = {}
      next_field = field
      for level in range(self.max_level, -1, -1):
         self.matrix_operators[level] = lambda vec, mat=self.matvec, dt=dt, f=next_field, rhs=self.rhs_operators[level] : mat(numpy.ravel(vec), dt, f, rhs)
         self.pseudo_dts[level] = self.compute_pseudo_dt(
            next_field, self.geometries[level].X1, self.geometries[level].X2, self.metrics[level].H_contra_11, self.metrics[level].H_contra_22)
         if level > 0:
            next_field = self.restrict_operators[level](next_field)


def mg(b, x0, level, mg_params, gamma=1):
   """
   Do one pass of the multigrid algorithm.

   Arguments:
   b         -- Right-hand side of the system we want to solve
   x0        -- An estimate of the solution. Might be anything
   level     -- The level (depth) at which we are within the multiple grids. 0 is the coarsest grid
   mg_params -- MG_params object that describes the system to solve at each grid level
   gamma     -- (optional) How many passes to do at the next grid level
   """

   A        = mg_params.matrix_operators[level]
   smoothe  = mg_params.smoothers[level]
   restrict = mg_params.restrict_operators[level]    if level > 0 else None
   prolong  = mg_params.prolong_operators[level - 1] if level > 0 else None
   dt_shaped = mg_params.pseudo_dts[level]
   dt = numpy.ravel(dt_shaped)

   # Pre smoothing
   x = smoothe(A, b, x0, dt, mg_params.num_pre_smoothing)

   if level > 0:                                   # Go down a level and solve that
      residual = numpy.ravel(restrict(b - A(x)))   # Compute the (restricted) residual of the current grid level system
      v = numpy.zeros_like(residual)               # A guess of the next solution (0 is pretty good, that's what we're aiming for)
      for i in range(gamma):
         v = mg(residual, v, level - 1, mg_params, gamma=gamma)  # MG pass on next lower level
      x = x + numpy.ravel(prolong(v))              # Correction
   else:
      # Just directly solve the problem
      # That's no good, we should not use FGMRES to precondition FGMRES...
      if mg_params.use_solver:
         x, _, _, _ = fgmres(A, b, x0=x, tol=1e-1)
   
   # Post smoothing
   x = smoothe(A, b, x, dt, mg_params.num_post_smoothing)

   return x


def mg_solve(b, mg_params, x0=None, field=None, dt=None, tolerance=1e-7, gamma=1, max_num_it=100):
   """
   Solve the system defined by mg_params using the multigrid method.

   Mandatory arguments:
   b         -- The right-hand side of the linear system to solve.
   mg_params -- The MG_params object that describes the system at all grid levels

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

   # Initial guess
   x = x0 if x0 is not None else numpy.zeros_like(b)

   norm_b = global_norm(b)
   tol_relative = tolerance * norm_b

   norm_r = -norm_b

   # Early return if rhs is zero
   if norm_b < 1e-15:
      return x, 0.0, 0, 0, [0.0]

   # Init system for this time step, if not done already
   if field is not None:
      if dt is None: dt = 1.0
      mg_params.compute_matrix_operators(field, dt)

   residuals = []
   level     = mg_params.max_level
   A         = mg_params.matrix_operators[level]
   num_it    = 0
   for it in range(max_num_it):
      x = mg(b, x, level, mg_params, gamma=gamma)
      num_it += 1

      # Check tolerance exit condition
      if it < max_num_it - 1:
         residual = b - A(x)
         norm_r = global_norm(residual)
         # print(f'norm_r = {norm_r:.2e} (rel tol {tol_relative:.2e})')
         residuals.append(norm_r / norm_b)
         if norm_r < tol_relative:
            return x, norm_r / norm_b, num_it, 0, residuals
         elif norm_r > 10.0:
            return x, norm_r / norm_b, num_it, -1, residuals

   flag = 0
   if num_it >= max_num_it: flag = -1
   return x, norm_r / norm_b, num_it, flag, residuals


def global_norm(vec):
   """Compute vector norm across all PEs"""
   local_sum = vec @ vec
   return math.sqrt( mpi4py.MPI.COMM_WORLD.allreduce(local_sum) )
