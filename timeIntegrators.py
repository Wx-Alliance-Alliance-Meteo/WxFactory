import numpy
import math
import scipy.sparse.linalg
from collections import deque
from time        import time

from matvec          import matvec_fun, matvec_rat
from kiops           import kiops
from linsol          import fgmres
from multigrid       import Multigrid
from output_residual import write_output
from phi             import phi_ark
from timer           import Timer
from finite_volume   import FiniteVolume

class Epirk4s3a:
   g21 = 1/2
   g31 = 2/3

   alpha21 = 1/2
   alpha31 = 2/3

   alpha32 = 0

   p211 = 1
   p212 = 0
   p213 = 0

   p311 = 1
   p312 = 0
   p313 = 0

   b2p3 = 32
   b2p4 = -144

   b3p3 = -27/2
   b3p4 = 81

   gCoeffVec = numpy.array([g21, g31])
   KryIndex = numpy.array([1, 2])

   def __init__(self, rhs, tol, krylov_size):
      self.rhs = rhs
      self.tol = tol
      self.krylov_size = [krylov_size, krylov_size]

   def step(self, Q, dt):
      # Using a 4th Order 3-stage EPIRK time integration

      rhs = self.rhs(Q)

      hF = rhs * dt
      ni, nj, ne = Q.shape
      zeroVec = numpy.zeros(ni * nj * ne)

      matvec_handle = lambda v: matvec_fun(v, dt, Q, self.rhs(Q), self.rhs)

      # stage 1
      u_mtrx = numpy.row_stack((zeroVec, hF.flatten()))
      print(u_mtrx.shape)
      phiv, stats = kiops(self.gCoeffVec, matvec_handle, u_mtrx, tol=self.tol, m_init=self.krylov_size[0], mmin=16, mmax=64, task1=True)
      self.krylov_size[0] = math.floor(0.7 * stats[5] + 0.3 * self.krylov_size[0])
      U2 = Q + self.alpha21 * numpy.reshape(phiv[0, :], Q.shape)

      # Calculate residual r(U2)
      mv = numpy.reshape(matvec_handle(U2 - Q), Q.shape)
      hb1 = dt * self.rhs(U2) - hF - mv

      # stage 2
      U3 = Q + self.alpha31 * numpy.reshape(phiv[1, :], Q.shape)

      # Calculate residual r(U3)
      mv = numpy.reshape(matvec_handle(U3 - Q), Q.shape)
      hb2 = dt * self.rhs(U3) - hF - mv

      # stage 3
      u_mtrx = numpy.row_stack(
         (zeroVec, hF.flatten(), zeroVec, (self.b2p3 * hb1 + self.b3p3 * hb2).flatten(), (self.b2p4 * hb1 + self.b3p4 * hb2).flatten()))
      phiv, stats = kiops([1], matvec_handle, u_mtrx, tol=self.tol, m_init=self.krylov_size[1], mmin=16, mmax=64, task1=False)
      self.krylov_size[1] = math.floor(0.7 * stats[5] + 0.3 * self.krylov_size[1])

      return Q + numpy.reshape(phiv, Q.shape)

class Epi:
   def __init__(self, order, rhs, tol, krylov_size, jacobian_method='complex', init_method=None, init_substeps=1):
      self.rhs = rhs
      self.tol = tol
      self.krylov_size = krylov_size
      self.jacobian_method = jacobian_method

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
         #self.init_method = Epirk4s3a(rhs, tol, krylov_size)
         self.init_method = Epi(2, rhs, tol, krylov_size)

      self.init_substeps = init_substeps

   def step(self, Q, dt):
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

         for k, alpha in enumerate(self.A[:,i],start=2):
            # v_k = Sum_{i=1}^{n_prev} A_{k,i} R(y_{n-i})
            vec[k,:] += alpha * r.flatten()

      phiv, stats = kiops([1], matvec_handle, vec, tol=self.tol, m_init=self.krylov_size, mmin=16, mmax=64,
                          task1=False)

      print(f'KIOPS converged at iteration {stats[2]} (using {stats[0]} internal substeps)'
            f' to a solution with local error {stats[4]:.2e}')

      self.krylov_size = math.floor(0.7 * stats[5] + 0.3 * self.krylov_size)

      # Save values for the next timestep
      if self.n_prev > 0:
         self.previous_Q.pop()
         self.previous_Q.appendleft(Q)
         self.previous_rhs.pop()
         self.previous_rhs.appendleft(rhs)

      # Update solution
      return Q + numpy.reshape(phiv, Q.shape) * dt

class Tvdrk3:
   def __init__(self, rhs):
      self.rhs = rhs

   def step(self, Q, dt):
      Q1 = Q + self.rhs(Q) * dt
      Q2 = 0.75 * Q + 0.25 * Q1 + 0.25 * self.rhs(Q1) * dt
      Q = 1.0 / 3.0 * Q + 2.0 / 3.0 * Q2 + 2.0 / 3.0 * self.rhs(Q2) * dt
      return Q

class Rat2:
   def __init__(self, rhs_handle, tol, preconditioner=None):
      self.rhs_handle     = rhs_handle
      self.tol            = tol
      self.preconditioner = preconditioner

   def step(self, Q, dt):
      
      rhs    = self.rhs_handle(Q)
      Q_flat = Q.flatten()
      n      = Q_flat.shape[0]

      A = scipy.sparse.linalg.LinearOperator((n,n), matvec=lambda v: matvec_rat(v, dt, Q, rhs, self.rhs_handle))
      b = A(Q_flat) + rhs.flatten() * dt

      if self.preconditioner is not None:
         self.preconditioner.prepare(dt, Q)

      # TODO : gcro
      t0 = time()
      Qnew, local_error, nb_iter, flag, residuals = fgmres(A, b, x0=Q_flat, tol=self.tol, preconditioner=self.preconditioner, verbose=True)
      t1 = time()

      write_output(nb_iter, t1 - t0, flag, residuals)

      t2 = time()
      Qnew_noprecond, _, num_iter_noprecond, _, _ = fgmres(A, b, x0=Qnew, tol=self.tol, preconditioner=None, verbose=True)
      if num_iter_noprecond > 1:
         raise ValueError(f'Comparison failed!')
      t3 = time()

      if flag == 0:
         print(f'FGMRES converged at iteration {nb_iter} in {t1 - t0:4.1f} s to a solution with local error {local_error : .2e}')
      else:
         print(f'FGMRES stagnation/interruption at iteration {nb_iter} in {t1 - t0:4.1f} s, returning a solution with local error {local_error: .2e}')

      # gmres_sol, gmres_res, num_gmres_it, gmres_flag, res = fgmres(matvec_handle, rhs, x0=phiv, tol=self.tol, preconditioner=None, restart=20, maxiter=1200//20)

      return numpy.reshape(Qnew, Q.shape)

class ARK_epi2:
   def __init__(self, rhs, rhs_explicit, rhs_implicit, param):
      self.rhs = rhs
      self.butcher_exp = param.ark_solver_exp
      self.butcher_imp = param.ark_solver_imp
      self.tol = param.tolerance

      self.runs = []
      self.add_run(rhs_explicit, rhs_implicit)

   def add_run(self, rhs_explicit, rhs_implicit):
      self.runs.append({'rhs_explicit': rhs_explicit, 'rhs_implicit': rhs_implicit, 'timer': Timer()})

   def exec_run(self, run_params, dt, Q, rhs):
      run_params['timer'].start()
      rhs_exp = run_params['rhs_explicit']
      rhs_imp = run_params['rhs_implicit']
      J_explicit = lambda v: matvec_fun(v, dt, Q, rhs_exp(Q), rhs_exp)
      J_implicit = lambda v: matvec_fun(v, dt, Q, rhs_imp(Q), rhs_imp)

      # We only need the second phi function
      vec = numpy.row_stack((numpy.zeros_like(rhs), rhs))

      run_params['output'] = phi_ark([0, 1], J_explicit, J_implicit, vec, tol = self.tol, task1 = False,
              butcher_exp = self.butcher_exp, butcher_imp = self.butcher_imp)
      run_params['timer'].stop()

      return run_params['output']

   def step(self, Q, dt):
      rhs = self.rhs(Q).flatten()

      for r in self.runs:
         _, num_steps = self.exec_run(r, dt, Q, rhs)
         time = r['timer'].last_time()
         print(f'PHI/ARK converged using {num_steps} substeps in {time: .3f} s')

      phiv, _ = self.runs[0]['output']

      # Update solution
      return Q + numpy.reshape(phiv[:,-1], Q.shape) * dt
