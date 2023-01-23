import numpy
import math
import scipy.sparse.linalg
import mpi4py.MPI

from collections  import deque
from itertools    import combinations
from time         import time

# from Solver.Bamphi.bamphi   import bamphi
from Output.solver_stats   import write_solver_stats
from Solver.kiops          import kiops
from Solver.linsol         import fgmres
from Solver.matvec         import matvec_fun, matvec_rat
from Solver.nonlin         import newton_krylov
from Solver.pmex           import pmex

# Computes the coefficients for stiffness resilient exponential methods based on node values c
def alpha_coeff(c):
   m = len(c)
   p = m + 2
   alpha = numpy.zeros((m, m))
   for i in range(m):
      c_no_i = [cc for (j, cc) in enumerate(c) if j != i]
      denom = c[i] ** 2 * math.prod([c[i] - cl for cl in c_no_i])
      for k in range(m):
         sp = sum([math.prod(v) for v in combinations(c_no_i, m - k - 1)])
         alpha[k, i] = (-1) ** (m - k + 1) * math.factorial(k + 2) * sp / denom

   return alpha

# Computes nodes for SRERK methods with minimal error terms
def opt_nodes(order: int):
   if order < 3:
      raise ValueError('Order should be at least 3')

   coeff = lambda p,q: (-1)**(p+q) * math.factorial(p+q+2) / (math.factorial(q) * math.factorial(q+2) * math.factorial(p-q))

   c = [];
   # Compute optimal nodes for each stage order starting at order 2
   for o in list(range(2,order-2, 2)) + [order-2]:
      p = numpy.polynomial.Polynomial([coeff(o,q) for q in range(0,o+1)])
      c.append(p.roots())

   c.append(numpy.ones(1))
   return c

class SRERK:
   # Stiffness resilient exponential Runge-Kutta methods
   # If the nodes are NOT specified, return the SRERK method of the specified order with min error terms
   # If the nodes are specified, return the SRERK method with these nodes and ignore the 'order' parameter
   def __init__(self, order: int, rhs, tol: float, exponential_solver, jacobian_method='complex', nodes = None):
      self.rhs = rhs
      self.tol = tol
      self.krylov_size = 1
      self.jacobian_method = jacobian_method
      self.exponential_solver = exponential_solver

      if nodes:
         self.c = nodes
      else:
         self.c = opt_nodes(order)
      self.n_proj = len(self.c)

      self.alpha = []
      for i in range(self.n_proj-1):
         self.alpha.append(alpha_coeff(self.c[i]))

   def step(self, Q: numpy.ndarray, dt: float):
      rhs = self.rhs(Q)
      matvec_handle = lambda v: matvec_fun(v, dt, Q, rhs, self.rhs, self.jacobian_method)

      # Initial projection
      vec = numpy.zeros((2, rhs.size))
      vec[1, :] = rhs.flatten()

      if self.exponential_solver == 'kiops':
         z, stats = kiops(self.c[0], matvec_handle, vec, tol=self.tol, m_init=self.krylov_size, mmin=16, mmax=64, task1=False)

         print(f'KIOPS converged at iteration {stats[2]} (using {stats[0]} internal substeps and {stats[1]} rejected expm)'
               f' to a solution with local error {stats[4]:.2e}')

         self.krylov_size = math.floor(0.7 * stats[5] + 0.3 * self.krylov_size)

      elif self.exponential_solver == 'pmex':

         z, stats = pmex(self.c[0], matvec_handle, vec, tol=self.tol, mmax=64, task1=False)

         print(f'PMEX converged at iteration {stats[2]} (using {stats[0]} internal substeps and {stats[1]} rejected expm)'
               f' to a solution with local error {stats[4]:.2e}')
      else:
         print('There is nothing to see here, go away!')
         exit(0)

      # Loop over all the other projections
      for i_proj in range(1, self.n_proj):

         for i in range(z.shape[0]):
            z[i, :] = Q.flatten() + dt * z[i,:]

         # Compute r(z_i)
         rz = numpy.empty_like(z)
         for i in range(z.shape[0]):
            tmp_z = numpy.reshape(z[i,:], Q.shape)
            rz[i,:] = (self.rhs(tmp_z) - rhs).flatten() - matvec_handle(tmp_z - Q)/dt

         vec = numpy.zeros((z.shape[0]+3, rhs.size))
         vec[1, :] = rhs.flatten()
         vec[3:,:] = self.alpha[i_proj-1] @ rz

         if self.exponential_solver == 'kiops':
            z, stats = kiops(self.c[i_proj], matvec_handle, vec, tol=self.tol, m_init=self.krylov_size, mmin=16, mmax=64,
                             task1=False)

            print(f'KIOPS converged at iteration {stats[2]} (using {stats[0]} internal substeps and {stats[1]} rejected expm)'
                  f' to a solution with local error {stats[4]:.2e}')

            self.krylov_size = math.floor(0.7 * stats[5] + 0.3 * self.krylov_size)

         elif self.exponential_solver == 'pmex':

            z, stats = pmex(self.c[i_proj], matvec_handle, vec, tol=self.tol, mmax=64, task1=False)

            print(f'PMEX converged at iteration {stats[2]} (using {stats[0]} internal substeps and {stats[1]} rejected expm)'
                  f' to a solution with local error {stats[4]:.2e}')
         else:
            print('There is nothing to see here, go away!')
            exit(0)

      # Update solution
      return Q + dt * numpy.reshape(z, Q.shape)

class Epi:
   def __init__(self, order: int, rhs, tol: float, exponential_solver, jacobian_method='complex', init_method=None, init_substeps: int = 1):
      self.rhs = rhs
      self.tol = tol
      self.krylov_size = 1
      self.jacobian_method = jacobian_method
      self.exponential_solver = exponential_solver

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
         self.init_method = Epi(2, rhs, tol, self.exponential_solver, self.jacobian_method)

      self.init_substeps = init_substeps

   def step(self, Q: numpy.ndarray, dt: float):
      # If dt changes, discard saved value and redo initialization
      mpirank = mpi4py.MPI.COMM_WORLD.Get_rank()
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

      matvec_handle = lambda v: matvec_fun(v, dt, Q, rhs, self.rhs, self.jacobian_method)

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

         if (mpirank == 0):
            print(f'PMEX converged at iteration {stats[2]} (using {stats[0]} internal substeps and {stats[1]} rejected expm)'
                  f' to a solution with local error {stats[4]:.2e}')

      else:
         phiv, stats = kiops([1], matvec_handle, vec, tol=self.tol, m_init=self.krylov_size, mmin=16, mmax=64, task1=False)

         self.krylov_size = math.floor(0.7 * stats[5] + 0.3 * self.krylov_size)

         if (mpirank == 0):
            print(f'KIOPS converged at iteration {stats[2]} (using {stats[0]} internal substeps and {stats[1]} rejected expm)'
                  f' to a solution with local error {stats[4]:.2e}')

      # Save values for the next timestep
      if self.n_prev > 0:
         self.previous_Q.pop()
         self.previous_Q.appendleft(Q)
         self.previous_rhs.pop()
         self.previous_rhs.appendleft(rhs)

      # Update solution
      return Q + numpy.reshape(phiv, Q.shape) * dt

class EpiStiff:
   def __init__(self, order: int, rhs, tol: float, exponential_solver, jacobian_method='complex', init_method=None, init_substeps: int = 1):
      self.rhs = rhs
      self.tol = tol
      self.krylov_size = 1
      self.jacobian_method = jacobian_method
      self.exponential_solver = exponential_solver

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
         self.init_method = Epi(2, rhs, tol, self.exponential_solver, self.jacobian_method)

      self.init_substeps = init_substeps

   def step(self, Q: numpy.ndarray, dt: float):
      mpirank = mpi4py.MPI.COMM_WORLD.Get_rank()
      
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

      matvec_handle = lambda v: matvec_fun(v, dt, Q, rhs, self.rhs, self.jacobian_method)

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

         if (mpirank == 0):
            print(f'PMEX converged at iteration {stats[2]} (using {stats[0]} internal substeps and {stats[1]} rejected expm)'
                  f' to a solution with local error {stats[4]:.2e}')

      else:
         phiv, stats = kiops([1], matvec_handle, vec, tol=self.tol, m_init=self.krylov_size, mmin=16, mmax=64, task1=False)

         self.krylov_size = math.floor(0.7 * stats[5] + 0.3 * self.krylov_size)

         if (mpirank == 0):
            print(f'KIOPS converged at iteration {stats[2]} (using {stats[0]} internal substeps and {stats[1]} rejected expm)'
                  f' to a solution with local error {stats[4]:.2e}')

      # Save values for the next timestep
      if self.n_prev > 0:
         self.previous_Q.pop()
         self.previous_Q.appendleft(Q)
         self.previous_rhs.pop()
         self.previous_rhs.appendleft(rhs)

      # Update solution
      return Q + numpy.reshape(phiv, Q.shape) * dt

class Euler1:
   def __init__(self, rhs):
      self.rhs = rhs

   def step(self, Q, dt):
      Q = Q + self.rhs(Q) * dt
      return Q

class Tvdrk3:
   def __init__(self, rhs):
      self.rhs = rhs

   def step(self, Q, dt):
      Q1 = Q + self.rhs(Q) * dt
      #Q = Q + self.rhs(Q) * dt
      Q2 = 0.75 * Q + 0.25 * Q1 + 0.25 * self.rhs(Q1) * dt
      Q = 1.0 / 3.0 * Q + 2.0 / 3.0 * Q2 + 2.0 / 3.0 * self.rhs(Q2) * dt
      return Q

class scipy_counter(object): # TODO : tempo
   def __init__(self, disp=False):
      self._disp = disp
      self.niter = 0
   def __call__(self, rk=None):
      self.niter += 1
      if self._disp:
         print(f'iter {self.niter:3d}\trk = {str(rk)}')
   def nb_iter(self):
      return self.niter

class Ros2:
   def __init__(self, rhs_handle, tol: float, preconditioner=None):
      self.rhs_handle     = rhs_handle
      self.tol            = tol
      self.preconditioner = preconditioner

   def step(self, Q: numpy.ndarray, dt: float):
      
      rhs    = self.rhs_handle(Q)
      Q_flat = Q.flatten()
      n      = Q_flat.shape[0]

      A = scipy.sparse.linalg.LinearOperator((n,n), matvec=lambda v: matvec_rat(v, dt, Q, rhs, self.rhs_handle))
      b = A(Q_flat) + rhs.flatten() * dt

      if self.preconditioner is not None:
         self.preconditioner.prepare(dt, Q)

      t0 = time()
      Qnew, local_error, num_iter, flag, residuals = fgmres(A, b, x0=Q_flat, tol=self.tol, restart=100, maxiter=None, preconditioner=self.preconditioner, verbose=False)
      t1 = time()

      write_solver_stats(num_iter, t1 - t0, flag, residuals)

      if flag == 0:
         print(f'FGMRES converged at iteration {num_iter} in {t1 - t0:4.1f} s to a solution with relative local error {local_error : .2e}')
      else:
         print(f'FGMRES stagnation/interruption at iteration {num_iter} in {t1 - t0:4.1f} s, returning a solution with relative local error {local_error: .2e}')

      return numpy.reshape(Qnew, Q.shape)

class StrangSplitting:
   def __init__(self, scheme1, scheme2):
      self.scheme1 = scheme1
      self.scheme2 = scheme2

   def step(self, Q, dt):
      Q = self.scheme1.step(Q, dt/2)
      Q = self.scheme2.step(Q, dt)
      return self.scheme1.step(Q, dt / 2)

class RosExp2: 
   def __init__(self, rhs_full, rhs_imp, tol, preconditioner):

      if mpi4py.MPI.COMM_WORLD.size > 1:
         raise ValueError(f'RosExp2 has only been tested with 1 PE. Gotta make sure it works with more than that.')

      self.rhs_full = rhs_full
      self.rhs_imp = rhs_imp
      self.tol = tol
      self.preconditioner = preconditioner

   def step(self, Q, dt):
      rhs_full = self.rhs_full(Q)
      rhs_imp = self.rhs_imp(Q)
      f_imp = rhs_imp.flatten()
      f_exp = (rhs_full - rhs_imp).flatten()

      J_full = lambda v: matvec_fun(v, dt, Q, rhs_full, self.rhs_full)
      J_imp = lambda v: matvec_fun(v, dt, Q, rhs_imp, self.rhs_imp)
      J_exp = lambda v: J_full(v) - J_imp(v)

      Q_flat = Q.flatten()
      n = len(Q_flat)

      vec = numpy.zeros((2, n))
      vec[1,:] = rhs_full.flatten()

      tic = time()
      phiv, stats = pmex([1.], J_exp, vec, tol=self.tol,task1=False)
      time_exp = time() - tic
      print(f'PMEX converged at iteration {stats[2]} (using {stats[0]} internal substeps and {stats[1]} rejected expm)')
      tic = time()
      A = scipy.sparse.linalg.LinearOperator((n,n), matvec = lambda v: v - J_imp(v) / 2)
      b = ( A(Q_flat) + phiv * dt ).flatten()

      Q_x0 = Q_flat.copy()

      inner_m = 20
      counter = scipy_counter()
      Qnew, info = scipy.sparse.linalg.gcrotmk(A, b, x0=Q_x0, tol=self.tol, M=None, callback=counter, m=inner_m)
#      Qnew, _, niter, _, _ = fgmres(A, b, x0=Q_x0, tol = self.tol, restart=20, maxiter = None, preconditioner = None, hegedus = True)

      time_imp = time() - tic

      print(f'gcrotmk converged at iteration {counter.nb_iter()*inner_m} using {counter.nb_iter()} restarts')
#      print('fgmres converged at iteration %d using %d restarts' % (niter, outer))

      print(f'Elapsed time: exponential {time_exp:.3f} secs ; implicit {time_imp:.3f} secs')

      return numpy.reshape(Qnew, Q.shape)

class PartRosExp2:
   def __init__(self, rhs_full, rhs_imp, tol, preconditioner):

      if mpi4py.MPI.COMM_WORLD.size > 1:
         raise ValueError(f'RosExp2 has only been tested with 1 PE. Gotta make sure it works with more than that.')

      self.rhs_full = rhs_full
      self.rhs_imp = rhs_imp
      self.tol = tol
      self.preconditioner = preconditioner

   def step(self, Q, dt):
      if self.preconditioner is not None:
         self.preconditioner.prepare(dt, Q)      
         # self.preconditioner.verbose = True

      rhs_full = self.rhs_full(Q)
      rhs_imp = self.rhs_imp(Q)
      f_imp = rhs_imp.flatten()
      f_exp = (rhs_full - rhs_imp).flatten()

      J_full = lambda v: matvec_fun(v, dt, Q, rhs_full, self.rhs_full)
      J_imp = lambda v: matvec_fun(v, dt, Q, rhs_imp, self.rhs_imp)
      J_exp = lambda v: J_full(v) - J_imp(v)

      Q_flat = Q.flatten()
      n = len(Q_flat)

      vec = numpy.zeros((2, n))
      vec[0,:] = 0.5 * f_imp
      vec[1,:] = f_exp.copy()

      tic = time()
      phiv, stats = pmex([1.], J_exp, vec, tol=self.tol,task1=False)
      time_exp = time() - tic
      print(f'PMEX converged at iteration {stats[2]} (using {stats[0]} internal substeps and {stats[1]} rejected expm)')
      tic = time()
      A = scipy.sparse.linalg.LinearOperator((n,n), matvec = lambda v: v - J_imp(v) / 2)
      b = ( A(Q_flat) + (phiv + 0.5 * f_imp) * dt ).flatten()

      Q_x0 = Q_flat.copy()

      inner_m = 20
      counter = scipy_counter()
#      Qnew, info = scipy.sparse.linalg.gcrotmk(A, b, x0=Q_x0, tol=self.tol, M=self.preconditioner, callback=counter, m=inner_m)
      Qnew, _, niter, _, _ = fgmres(A, b, x0=Q_x0, tol = self.tol, restart=100, maxiter = None, preconditioner = self.preconditioner, hegedus = True, verbose = False)

      time_imp = time() - tic

      # num_pre_calls = 0
      # if self.preconditioner is not None: num_pre_calls = self.preconditioner.num_apply
      # print(f'gcrotmk converged at iteration {counter.nb_iter()*inner_m} using {counter.nb_iter()} restarts (and {num_pre_calls} preconditioner calls)')

      t0 = time()
      Qnew, local_error, num_iter, flag, residuals = fgmres(A, b, x0=Q_x0, tol=self.tol, preconditioner=self.preconditioner, verbose=False)
      t1 = time()

      write_solver_stats(num_iter, t1 - t0, flag, residuals)

      if flag == 0:
         print(f'fgmres converged at iteration {num_iter} to a solution with relative local error {local_error : .2e}')
      else:
         print(f'FGMRES stagnation/interruption at iteration {num_iter}, returning a solution with relative local error {local_error: .2e}')


      print(f'Elapsed time: exponential {time_exp:.3f} secs ; implicit {time_imp:.3f} secs')



      return numpy.reshape(Qnew, Q.shape)

class Imex2:
   def __init__(self, rhs_exp, rhs_imp, tol):

      if mpi4py.MPI.COMM_WORLD.size > 1:
         raise ValueError(f'RosExp2 has only been tested with 1 PE. Gotta make sure it works with more than that.')

      self.rhs_exp = rhs_exp
      self.rhs_imp = rhs_imp
      self.tol = tol

   def step(self, Q, dt):
      rhs = Q + dt/2 * self.rhs_exp(Q)
      g = lambda v: v - dt/2 * self.rhs_imp(v) - rhs
      Y1, _, _ = newton_krylov(g, Q)

      # Update solution
      return Q + dt * (self.rhs_imp(Y1) + self.rhs_exp(Y1))

class crank_nicolson:
   def __init__(self, rhs, tol, preconditioner=None):
      self.rhs = rhs
      self.tol = tol
      self.preconditioner = preconditioner

   def CN_system(self, Q_plus, Q, dt, rhs):
      return (Q_plus - Q) / dt - 0.5 * ( rhs(Q_plus) + rhs(Q) )

   def step(self, Q, dt):
      CN_fun = lambda Q_plus: self.CN_system(Q_plus, Q, dt, self.rhs)

      maxiter = None
      if self.preconditioner is not None:
         self.preconditioner.prepare(dt, Q)
         maxiter = 800

      # Update solution
      newQ, nb_iter, residuals = nonlin.newton_krylov(CN_fun, Q, f_tol=self.tol, fgmres_restart=30, fgmres_precond=self.preconditioner, verbose=False, maxiter=maxiter)

      return numpy.reshape(newQ, Q.shape)

class bdf2:
   def __init__(self, rhs, tol, preconditioner=None, init_substeps=1):
      self.rhs = rhs
      self.tol = tol
      self.init_substeps = init_substeps
      self.preconditioner = preconditioner
      self.Qprev = None

   def step(self, Q, dt):
      if self.Qprev is None:
         # Initialize with the backward Euler method
         newQ = Q.copy()
         for s in range(self.init_substeps):
            init_dt = dt / self.init_substeps
            nonlin_fun = lambda Q_plus: (Q_plus - newQ) / init_dt - 0.5 * self.rhs(Q_plus)

            newQ, nb_iter, residuals = nonlin.newton_krylov(nonlin_fun, newQ, f_tol=self.tol)
      else:
         maxiter = None
         nonlin_fun = lambda Q_plus: (Q_plus - 4./3. * Q + 1./3. * self.Qprev) / dt - 2./3. * self.rhs(Q_plus)
         if self.preconditioner is not None:
            self.preconditioner.prepare(dt, Q, self.Qprev)
            maxiter = 800
         newQ, nb_iter, residuals = nonlin.newton_krylov(nonlin_fun, Q, f_tol=self.tol, fgmres_precond=self.preconditioner, verbose=False, maxiter=maxiter)

      self.Qprev = Q.copy()

      return numpy.reshape(newQ, Q.shape)
