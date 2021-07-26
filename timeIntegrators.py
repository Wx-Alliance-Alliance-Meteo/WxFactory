import numpy
import math
from collections import deque
from time        import time

from matvec        import matvec_fun, matvec_rat
from kiops         import kiops
from linsol        import fgmres
from multigrid     import mg_solve
from phi           import phi_ark
from timer         import Timer

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

      matvec_handle = lambda v: matvec_fun(v, dt, Q, self.rhs)

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
   def __init__(self, order, rhs, tol, krylov_size, init_method=None, init_substeps=1):
      self.rhs = rhs
      self.tol = tol
      self.krylov_size = krylov_size

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
      matvec_handle = lambda v: matvec_fun(v, dt, Q, self.rhs)

      rhs = self.rhs(Q)

      vec = numpy.zeros((self.max_phi+1, rhs.size))
      vec[1,:] = rhs.flatten()
      for i in range(self.n_prev):
         J_deltaQ = matvec_fun(self.previous_Q[i] - Q, 1., Q, self.rhs)

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
   def __init__(self, rhs, tol, solver = 'fgmres', preconditioner=None, mg_params=None, rank=-1, param=None):
      self.rhs = rhs
      self.tol = tol
      self.preconditioner = preconditioner
      self.mg_params = mg_params
      self.rank = rank
      self.use_mg = False
      if solver == 'fgmres':
         self.solver_name = 'FGMRES'
         self.max_it = 1200//20 if self.preconditioner is None else 160//20
         self.solve = lambda A, b, x0 : fgmres(A, b, x0=x0, tol=self.tol, preconditioner=self.preconditioner, restart=20, maxiter=self.max_it)

      elif solver in ['mg', 'multigrid']:
         self.solver_name = 'Multigrid'
         self.use_mg = True
         self.max_it = 80
         self.solve = lambda A, b, x0 : mg_solve(b, self.mg_params, x0=x0, tolerance=self.tol, max_num_it=self.max_it)

      if self.rank == 0:
         try:
            f = open('test_result.txt')
            file_exists = True
            f.close()
         except:
            file_exists = False

         with open('test_result.txt', 'a+') as output_file:
            if not file_exists:
               output_file.write('# order | num_elements | dt | linear solver | precond | precond_interp | precond tol | max MG lvl | MG smoothe only | # pre smoothe | # post smoothe | CFL # ::: FGMRES #it | FGMRES time | precond #it | precond time | conv. flag \n')
            if param is not None:
               output_file.write(f'{param.nbsolpts} {param.nb_elements_horizontal:3d} {int(param.dt):5d} {param.linear_solver[:10]:10s} '
                                 f'{param.use_preconditioner} {param.dg_to_fv_interp[:8]:8s} {param.precond_tolerance:9.1e} '
                                 f'{param.max_mg_level:3d} {param.mg_smoothe_only} '
                                 f'{param.num_pre_smoothing:3d} {param.num_post_smoothing:3d} {param.mg_cfl:6.3f} ::: ')
            else:
               output_file.write(f'NO PARAMS - ')

   def step(self, Q, dt):
      matvec_handle = lambda v: matvec_rat(v, dt, Q, self.rhs)

      if self.preconditioner:
         self.preconditioner.init_time_step(dt, Q)

      if self.use_mg:
         self.mg_params.init_time_step(Q, dt)

      # Transform to the shifted linear system (I/dt - J/2) x = F/dt
      rhs = self.rhs(Q).flatten() / dt

      first_guess = numpy.zeros_like(rhs)

      t0 = time()
      phiv, local_error, num_iter, flag, residuals = self.solve(matvec_handle, rhs, first_guess)
      t1 = time()

      if self.rank == 0:
         with open('test_result.txt', 'a+') as output_file:
            output_file.write(f'{num_iter:4d} {t1 - t0:6.1f} ')
            precond_it, precond_time = 0, 0.0
            if self.preconditioner is not None:
               precond_it, precond_time = self.preconditioner.total_iter, self.preconditioner.total_time
            output_file.write(f'{precond_it:5d} {precond_time:6.1f} {flag:2d} ')
            output_file.write(f'- {" ".join(f"{r:.2e}" for r in residuals)} ')
            output_file.write('\n')

      if flag == 0:
         print(f'{self.solver_name} converged at iteration {num_iter} to a solution with local error {local_error : .2e}')
      else:
         print(f'{self.solver_name} stagnation/interruption at iteration {num_iter}, returning a solution with local error {local_error: .2e}')

      # Update solution
      return Q + numpy.reshape(phiv, Q.shape) * dt

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
      J_explicit = lambda v: matvec_fun(v, dt, Q, run_params['rhs_explicit'])
      J_implicit = lambda v: matvec_fun(v, dt, Q, run_params['rhs_implicit'])

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
