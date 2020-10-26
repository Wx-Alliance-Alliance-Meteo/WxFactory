import numpy
import math
import pickle
from collections import deque


from matvec        import matvec_fun, matvec_rat
from kiops         import kiops
from linsol        import Fgmres
from phi           import phi_ark
from interpolation import LagrangeSimpleInterpolator, BilinearInterpolator
from matvec_product_caller import MatvecCaller
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
      phiv, stats = kiops(self.gCoeffVec, matvec_handle, u_mtrx, tol=self.tol, m_init=self.krylov_size[0], mmin=14, mmax=64, task1=True)
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
      phiv, stats = kiops([1], matvec_handle, u_mtrx, tol=self.tol, m_init=self.krylov_size[1], mmin=14, mmax=64, task1=False)
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

      phiv, stats = kiops([1], matvec_handle, vec, tol=self.tol, m_init=self.krylov_size, mmin=14, mmax=64,
                          task1=False)

      print('KIOPS converged at iteration %d to a solution with local error %e' % (stats[2], stats[4]))

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
   def __init__(self, rhs, tol, rank, preconditioner = None):
      self.rhs            = rhs
      self.preconditioner = preconditioner
      self.out_stat_file  = 'rat2out.txt'

      self.solver = Fgmres(tol = tol)

      self.fgmres_time = Timer(0.0)
      self.no_precond_time  = Timer(0.0)
      self.rank = rank

   def step(self, Q, dt):
      matvec_handle = MatvecCaller(matvec_rat, dt, Q, self.rhs)
      if self.preconditioner: self.preconditioner.compute_matrix_caller(matvec_rat, dt, Q)

      rhs = self.rhs(Q).flatten()
      x0 = numpy.zeros_like(rhs)

      self.fgmres_time.start()
      phiv, local_error, niter, flag = self.solver.solve(
            matvec_handle, rhs, preconditioner = self.preconditioner, x0 = x0)
      self.fgmres_time.stop()

      self.no_precond_time.start()
      phiv_no_precond, _, iter_no_precond, _ = self.solver.solve(
         matvec_handle, rhs, preconditioner = None, x0 = x0
      )
      self.no_precond_time.stop()

      if self.rank == 0:
         with open(self.out_stat_file, 'a') as out_file:
            out_file.write('{} {} -- {} {} {} -- {} {}\n'.format(
               self.preconditioner.order + 1, self.preconditioner.num_elements,
               niter, flag, self.fgmres_time.last_time(), iter_no_precond, self.no_precond_time.last_time()))

      if flag == 0:
         print('GMRES converged at iteration %d to a solution with local error %e' % (niter, local_error))
      else:
         print('GMRES stagnation at iteration %d, returning a solution with local error %e' % (niter, local_error))

      # Update solution
      return Q + numpy.reshape(phiv, Q.shape) * dt

class ARK_epi2:
   def __init__(self, rhs, rhs_explicit1, rhs_implicit1, rhs_explicit2, rhs_implicit2, tol, rank):
      self.rhs = rhs
      self.rhs_explicit1 = rhs_explicit1
      self.rhs_implicit1 = rhs_implicit1
      self.rhs_explicit2 = rhs_explicit2
      self.rhs_implicit2 = rhs_implicit2
      self.tol = tol
      self.rank = rank

      self.timer = Timer(0.0)
      self.interp_timer = Timer(0.0)
      self.out_stat_file = 'epi2out.txt'

   def step(self, Q, dt):
      rhs = self.rhs(Q).flatten()

      self.interp_timer.start()
      J_e_interp = lambda v: matvec_fun(v, dt, Q, self.rhs_explicit2)
      J_i_interp = lambda v: matvec_fun(v, dt, Q, self.rhs_implicit2)
      vec = numpy.row_stack((numpy.zeros_like(rhs), rhs))

      phiv_interp, num_steps_interp = phi_ark([0, 1], J_e_interp, J_i_interp, vec, tol = self.tol, task1 = False)
      self.interp_timer.stop()

      self.timer.start()
      J_e = lambda v: matvec_fun(v, dt, Q, self.rhs_explicit1)
      J_i = lambda v: matvec_fun(v, dt, Q, self.rhs_implicit1)

      # We only need the second phi function
      vec = numpy.row_stack((numpy.zeros_like(rhs), rhs))

      phiv, num_steps = phi_ark([0, 1], J_e, J_i, vec, tol=self.tol, task1=False)
      # phiv, num_steps = phiv_interp, num_steps_interp
      # phiv_interp, num_steps_interp = phiv, num_steps
      self.timer.stop()

      if self.rank == 0:
         print('Finished in {} / {} iterations and {:.3f} / {:.3f} seconds'.format(
            num_steps, num_steps_interp, self.timer.last_time(), self.interp_timer.last_time()))

         with open(self.out_stat_file, 'a') as out_file:
             out_file.write('{:4d} {:5d} {:5.0f} -- {:4d} {:5.0f} -- {:4d} {:5.0f}\n'.format(
               self.rhs.nb_sol_pts, self.rhs.nb_elem, dt,
               num_steps_interp, self.interp_timer.last_time(), num_steps, self.timer.last_time()))

      diff = phiv[:,-1] - phiv_interp[:,-1]
      diff_norm = numpy.linalg.norm(diff)
      sol_norm = numpy.linalg.norm(phiv)

      print('Difference: {}'.format(diff_norm/sol_norm))
      if diff_norm / sol_norm > self.tol:
         print('AHHHHH not the same answer!!! Diff = {} / {}'.format(diff_norm, sol_norm))
         with open('geom{:04d}.dat'.format(self.rank), 'wb') as file:
            pickle.dump(self.rhs.geometry, file)
         with open('diff{:04d}.dat'.format(self.rank), 'wb') as file:
            pickle.dump(diff.reshape(Q.shape), file)
         # raise ValueError

   #     print('PHI/ARK converged at iteration %d' % stats)

      # Update solution
      return Q + numpy.reshape(phiv[:,-1], Q.shape) * dt
