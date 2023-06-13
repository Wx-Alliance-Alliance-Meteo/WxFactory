from collections  import deque
import math

import numpy
from mpi4py       import MPI

from common.program_options import Configuration
from .epi            import Epi
from .integrator     import Integrator, alpha_coeff
from solvers         import kiops, matvec_fun, pmex

class EpiStiff(Integrator):
   def __init__(self, param: Configuration, order: int, rhs, init_method=None, init_substeps: int = 1):
      super().__init__(param, preconditioner=None)
      self.rhs = rhs
      self.tol = param.tolerance
      self.krylov_size = 1
      self.jacobian_method = param.jacobian_method
      self.exponential_solver = param.exponential_solver

      if order < 2:
         raise ValueError('Unsupported order for EPI method')
      self.A = alpha_coeff([-i for i in range(-1, 1-order,-1)])

      m, self.n_prev = self.A.shape

      self.max_phi = order if order>2 else 1
      self.previous_Q = deque()
      self.previous_rhs = deque()
      self.dt = 0.0

      if init_method or self.n_prev == 0:
         self.init_method = init_method
      else:
         #self.init_method = Epirk4s3a(rhs, tol, krylov_size)
         self.init_method = Epi(param, 2, rhs)

      self.init_substeps = init_substeps

   def __step__(self, Q: numpy.ndarray, dt: float):
      mpirank = MPI.COMM_WORLD.Get_rank()

      # If dt changes, discard saved value and redo initialization
      if self.dt and abs(self.dt - dt) > 1e-10:
         self.previous_Q = deque()
         self.previous_rhs = deque()
      self.dt = dt

      # Initialize saved values using init_step method
      if len(self.previous_Q) < self.n_prev:
         self.previous_Q.appendleft(Q)
         self.previous_rhs.appendleft(self.rhs(Q))

         dt /= self.init_substeps
         for i in range(self.init_substeps):
            Q = self.init_method.step(Q, dt)
         return Q

      # Regular EPI step
      rhs = self.rhs(Q)

      def matvec_handle(v): return matvec_fun(v, dt, Q, rhs, self.rhs, self.jacobian_method)

      vec = numpy.zeros((self.max_phi+1, rhs.size))
      vec[1,:] = rhs.flatten()
      for i in range(self.n_prev):
         J_deltaQ = matvec_fun(self.previous_Q[i] - Q, 1., Q, rhs, self.rhs, self.jacobian_method)

         # R(y_{n-i})
         r = (self.previous_rhs[i] - rhs) - numpy.reshape(J_deltaQ, Q.shape)

         for k, alpha in enumerate(self.A[:,i]):
            # v_k = Sum_{i=1}^{n_prev} A_{k,i} R(y_{n-i})
            vec[k+3,:] += alpha * r.flatten()

      if self.exponential_solver == 'pmex':

         phiv, stats = pmex([1.], matvec_handle, vec, tol=self.tol, mmax=64, task1=False)

         if mpirank == 0:
            print(f'PMEX converged at iteration {stats[2]} (using {stats[0]} internal substeps and'
                  f' {stats[1]} rejected expm) to a solution with local error {stats[4]:.2e}')

      else:
         phiv, stats = kiops([1], matvec_handle, vec, tol=self.tol, m_init=self.krylov_size, mmin=16, mmax=64,
                             task1=False)

         self.krylov_size = math.floor(0.7 * stats[5] + 0.3 * self.krylov_size)

         if mpirank == 0:
            print(f'KIOPS converged at iteration {stats[2]} (using {stats[0]} internal substeps and'
                  f' {stats[1]} rejected expm) to a solution with local error {stats[4]:.2e}')

      # Save values for the next timestep
      if self.n_prev > 0:
         self.previous_Q.pop()
         self.previous_Q.appendleft(Q)
         self.previous_rhs.pop()
         self.previous_rhs.appendleft(rhs)

      # Update solution
      return Q + numpy.reshape(phiv, Q.shape) * dt
