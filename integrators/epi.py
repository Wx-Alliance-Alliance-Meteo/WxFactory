from collections import deque
import math
from typing      import Callable

from mpi4py      import MPI
import numpy

from common.program_options import Configuration
from .integrator            import Integrator, SolverInfo
from solvers                import kiops, matvec_fun, pmex

class Epi(Integrator):
   def __init__(self, param: Configuration, order: int, rhs: Callable, init_method=None, init_substeps: int = 1):
      super().__init__(param, preconditioner=None)
      self.rhs = rhs
      self.tol = param.tolerance
      self.krylov_size = 1
      self.jacobian_method = param.jacobian_method
      self.exponential_solver = param.exponential_solver

      if order == 2:
         self.A = numpy.array([[]])
      elif order == 3:
         self.A = numpy.array([[2/3]])
      elif order == 4:
         self.A = numpy.array([
            [-3/10, 3/40], [32/5, -11/10]
         ])
      elif order == 5:
         self.A = numpy.array([
            [-4/5, 2/5, -4/45],
            [12, -9/2, 8/9],
            [3, 0, -1/3]
         ])
      elif order == 6:
         self.A = numpy.array([
            [-49/60, 351/560, -359/1260, 367/6720],
            [92/7, -99/14, 176/63, -1/2],
            [485 / 21, -151 / 14, 23 / 9, -31 / 168]
         ])
      else:
         raise ValueError('Unsupported order for EPI method')

      k, self.n_prev = self.A.shape
      # Limit max phi to 1 for EPI 2
      if order == 2:
         k -= 1
      self.max_phi = k+1
      self.previous_Q = deque()
      self.previous_rhs = deque()
      self.dt = 0.0

      if init_method or self.n_prev == 0:
         self.init_method = init_method
      else:
         self.init_method = Epi(param, 2, rhs)

      self.init_substeps = init_substeps

   def __step__(self, Q: numpy.ndarray, dt: float):
      # If dt changes, discard saved value and redo initialization
      mpirank = MPI.COMM_WORLD.Get_rank()
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

         for k, alpha in enumerate(self.A[:,i],start=2):
            # v_k = Sum_{i=1}^{n_prev} A_{k,i} R(y_{n-i})
            vec[k,:] += alpha * r.flatten()

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

      self.solver_info = SolverInfo(total_num_it = stats[2])

      # Save values for the next timestep
      if self.n_prev > 0:
         self.previous_Q.pop()
         self.previous_Q.appendleft(Q)
         self.previous_rhs.pop()
         self.previous_rhs.appendleft(rhs)

      # Update solution
      return Q + numpy.reshape(phiv, Q.shape) * dt
