import functools
import numpy
import scipy
from copy import deepcopy

from Grid.matrices      import DFR_operators, remesh_operator
from Grid.metric        import Metric
from Grid.cubed_sphere  import CubedSphere
from Rhs.rhs_sw         import rhs_sw
from Solver.kiops       import kiops
from Solver.linsol      import global_norm
from Stepper.matvec     import matvec_rat

class Multigrid:
   def __init__(self, param, ptopo, num_levels, rhs_handle, increment=1):

      self.max_level = num_levels

      self.rhs = rhs_handle

      self.increment = increment

      self.params = {}
      self.geoms = {}
      self.metric = {}
      self.mtrx = {}
      self.restrict = {}
      self.restrict_solpt = {}
      self.restrict_correction = {}
      self.prolong = {}
      self.prolong_solpt = {}
      self.prolong_correction = {}

      self.ptopo = ptopo

      print('\nMultigrid meshes:')

      for level in range(self.max_level, 0, -increment):
         self.params[level] = deepcopy(param)
         self.params[level].nbsolpts = level

         self.geoms[level] = cubed_sphere(self.params[level].nb_elements_horizontal, self.params[level].nb_elements_vertical, self.params[level].nbsolpts, self.params[level].λ0, self.params[level].ϕ0, self.params[level].α0, self.params[level].ztop, ptopo, self.params[level])

         self.mtrx[level] = DFR_operators(self.geoms[level], self.params[level].filter_apply, self.params[level].filter_order, self.params[level].filter_cutoff)

         self.metric[level] = Metric(self.geoms[level])

      for level in range(self.max_level, 0, -increment):
         if level > 1:
            self.restrict[level] = remesh_operator(self.geoms[level].solutionPoints, self.geoms[level - increment].solutionPoints)


         if level < self.max_level:
            self.prolong[level] = remesh_operator(self.geoms[level].solutionPoints, self.geoms[level + increment].solutionPoints)


   def prepare(self, dt, Q):

      self.A = {}
      self.dt = dt

      rhs_handle = {}
      Q_restricted = {}

      ni, nj = self.geoms[self.max_level].X.shape
      Q_restricted[self.max_level] = numpy.reshape(Q, (3, ni, nj))

      nb_dims = 2
      level = self.max_level

      self.A[self.max_level] = {}

      for level in range(self.max_level, 0, -self.increment):

         rhs_handle[level] = functools.partial(rhs_sw, geom=self.geoms[level], mtrx=self.mtrx[level], metric=self.metric[level], topo=None, ptopo=self.ptopo, nbsolpts=self.params[level].nbsolpts, nb_elements_hori=self.params[level].nb_elements_horizontal)

         n = Q_restricted[level].flatten().shape[0]

         self.A[level] = scipy.sparse.linalg.LinearOperator((n,n), matvec=functools.partial(matvec_rat, dt=dt, Q=Q_restricted[level], rhs=rhs_handle[level](Q_restricted[level]), rhs_handle=rhs_handle[level]))

         if level-1 > 0:
            ni, nj = self.geoms[level - self.increment].X.shape  # TODO : éviter les reshape
            Q_restricted[level - self.increment] = numpy.reshape(self.restriction(Q_restricted[level].flatten(), level - self.increment), (3, ni, nj)) # TODO : éviter les flatten et reshape


   def __call__(self, b):
      return self.solve(b, x0=None, nb_presmooth=1, nb_postmooth=1, level=self.max_level, w_cycle=1, coarsest_level=1, verbose=False)

   def solve(self, b, x0=None, level=None, nb_presmooth=1, nb_postmooth=1, w_cycle=1, coarsest_level=1, verbose=False):

      if level is None:
         level = self.max_level
      elif level == coarsest_level:
         nb_presmooth += nb_postmooth

      if x0 is None:
         x0 = numpy.zeros_like(b)

      if verbose:
         print('Residual at level', level, ':', global_norm(b.flatten() - (self.A[level](x0.flatten()))) )

      x = x0

      for step in range(nb_presmooth):
         x = self.smoothing(self.A[level], b, x, self.dt, level)
         if verbose: print('   Presmooth : ', global_norm(b.flatten() - (self.A[level](x.flatten()))) )

      if level > coarsest_level:

         residual = self.restriction((self.A[level](x)).flatten() - b, level - self.increment)

         correction = numpy.zeros_like(residual)
         for j in range(w_cycle):
            correction = self.solve(residual, correction, level - self.increment, nb_presmooth, nb_postmooth, w_cycle, coarsest_level, verbose)

         if verbose: print('Back to level', level)

         x -= self.prolongation(correction, level)
         if verbose: print('   Correction : ', global_norm(b.flatten() - (self.A[level](x.flatten()))) )

         for step in range(nb_postmooth):
            x = self.smoothing(self.A[level], b, x, self.dt, level)
            if verbose: print('   Postsmooth : ', global_norm(b.flatten() - (self.A[level](x.flatten()))) )

      if verbose: print('retour', level, ' : ', global_norm(b.flatten() - (self.A[level](x.flatten()))) )

      if verbose is True and level == self.max_level:
         print('End multigrid')
         print(' ')
      return x


   def restriction(self, Q, target_level):
      nb_equations = 3
      level = target_level + self.increment

      ni, nj = self.geoms[level].X.shape
      Q_reshaped = numpy.reshape(Q, (nb_equations, ni, nj))

      ni_restricted, nj_restricted = self.geoms[target_level].X.shape
      Q_restricted = numpy.zeros((nb_equations, ni_restricted, nj_restricted))

      Q_interim = numpy.zeros((nb_equations, ni_restricted, nj))

      for elem in range(self.params[level].nb_elements_horizontal):
         epais = elem * self.params[level].nbsolpts + numpy.arange(self.params[level].nbsolpts)
         epais_target = elem * self.params[target_level].nbsolpts + numpy.arange(self.params[target_level].nbsolpts)
         Q_interim[:,epais_target,:] = self.restrict[level] @ Q_reshaped[:,epais,:]

      for elem in range(self.params[level].nb_elements_horizontal):
         epais = elem * self.params[level].nbsolpts + numpy.arange(self.params[level].nbsolpts)
         epais_target = elem * self.params[target_level].nbsolpts + numpy.arange(self.params[target_level].nbsolpts)
         Q_restricted[:,:,epais_target] = Q_interim[:,:,epais] @ self.restrict[level].T

      return Q_restricted.flatten()

   def prolongation(self, Q, target_level):
      nb_equations = 3
      level = target_level - self.increment

      ni, nj = self.geoms[level].X.shape
      Q_reshaped = numpy.reshape(Q, (nb_equations, ni, nj))

      nk_prolongated, ni_prolongated = self.geoms[target_level].X.shape
      Q_prolongated = numpy.zeros((nb_equations, nk_prolongated, ni_prolongated))

      Q_interim = numpy.zeros((nb_equations, nk_prolongated, ni))

      for elem in range(self.params[level].nb_elements_horizontal):
         epais = elem * self.params[level].nbsolpts + numpy.arange(self.params[level].nbsolpts)
         epais_target = elem * self.params[target_level].nbsolpts + numpy.arange(self.params[target_level].nbsolpts)
         Q_interim[:,epais_target,:] = self.prolong[level] @ Q_reshaped[:,epais,:]

      for elem in range(self.params[level].nb_elements_horizontal):
         epais = elem * self.params[level].nbsolpts + numpy.arange(self.params[level].nbsolpts)
         epais_target = elem * self.params[target_level].nbsolpts + numpy.arange(self.params[target_level].nbsolpts)
         Q_prolongated[:,:,epais_target] = Q_interim[:,:,epais] @ self.prolong[level].T

      return Q_prolongated.flatten()

   def smoothing(self, A, b, x, dt, level):

      def residual(t, xx):
         return (b - A(xx))/dt

      n = x.size
      vec = numpy.zeros((2, n))

      exp_dt = 1.1*dt   # TODO : wild guess

      J = scipy.sparse.linalg.LinearOperator((n,n), matvec=lambda v: -A(v)*exp_dt/dt)

      R = residual(0, x)
      vec[1,:] = R.flatten()

      phiv, stats = kiops([1], J, vec, tol=1e-6, m_init=10, mmin=10, mmax=64, task1=False)

#      print('norm phiv', global_norm(phiv.flatten()))
#      print(f'KIOPS converged at iteration {stats[2]} (using {stats[0]} internal substeps)'
#               f' to a solution with local error {stats[4]:.2e}')
      return x + phiv.flatten() * exp_dt
