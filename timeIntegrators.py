import numpy
import math

from matvec import matvec_fun, matvec_rat
from kiops  import kiops
from linsol import gmres_mgs

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
            [-19/20, 2/5, -13/180],
            [66/5, -9/2, 34/45]
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
      self.previous_Q = []
      self.previous_rhs = []
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
         self.previous_Q = []
         self.previous_rhs = []
      self.dt = dt

      # Initialize saved values using init_step method
      if len(self.previous_Q) < self.n_prev:
         self.previous_Q.append(Q)
         self.previous_rhs.append(self.rhs(Q))

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
         self.previous_Q[1:] = self.previous_Q[:-1]
         self.previous_Q[0] = Q
         self.previous_rhs[1:] = self.previous_rhs[:-1]
         self.previous_rhs[0] = rhs

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
   def __init__(self, rhs, tol):
      self.rhs = rhs
      self.tol = tol

   def step(self, Q, dt):
      matvec_handle = lambda v: matvec_rat(v, dt, Q, self.rhs)

      rhs = self.rhs(Q).flatten()

      x0 = numpy.zeros_like(rhs)

      phiv, local_error, niter, flag = gmres_mgs(matvec_handle, rhs, x0, tol=self.tol)

      if flag == 0:
         print('GMRES converged at iteration %d to a solution with local error %e' % (niter, local_error))
      else:
         print('GMRES stagnation at iteration %d, returning a solution with local error %e' % (niter, local_error))

      # Update solution
      return Q + numpy.reshape(phiv, Q.shape) * dt
