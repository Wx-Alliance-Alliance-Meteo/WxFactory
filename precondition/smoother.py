from abc    import ABC, abstractmethod
import math
from typing import Callable

from mpi4py import MPI
import numpy
import scipy.linalg

from solvers import kiops, global_norm, matvec_rat, matvec_fun
from rhs.rhs_selector import RhsBundle

_MatvecOp = Callable[[numpy.ndarray], numpy.ndarray]
_Rhs      = Callable[[numpy.ndarray], numpy.ndarray]

class Smoother(ABC):
   @abstractmethod
   def __smoothe__(self, A: _MatvecOp, b: numpy.ndarray, x: numpy.ndarray) -> numpy.ndarray:
      raise ValueError("You can't call an abstract method!")

   def __call__(self, A: _MatvecOp, b: numpy.ndarray, x: numpy.ndarray) -> numpy.ndarray:
      return self.__smoothe__(A, b, x)


class ExponentialSmoother(Smoother):
   def __init__(self,
                niter: int=4,
                target_spectral_radius: float=1.0,
                global_dt: float=1.0,
                verbose: bool=False) -> None:
      super().__init__()
      self.niter = niter
      self.target_spectral_radius = target_spectral_radius
      self.global_dt = global_dt
      self.verbose = verbose

   def __smoothe__(self, A: _MatvecOp, b: numpy.ndarray, x0: numpy.ndarray) -> numpy.ndarray:
      """ Reduce the residual by integrating dx/dt = b - A x for an appropriate step size."""
      n = b.shape[0]

      x0_residual = b - A(x0) if x0 is not None else b
      x0_residual_norm = global_norm(x0_residual)

      happy_tol = 1e-10

      if self.verbose: print('x0 residual', x0_residual_norm)

      # --- Krylov projection

      p = 1

      # Preallocate matrix
      V = numpy.zeros((self.niter + 1, n + p))
      H = numpy.zeros((self.niter + 1, self.niter + 1))
      Minv = numpy.eye(self.niter)
      M = numpy.eye(self.niter)
      N = numpy.zeros((self.niter,self.niter))

      u = numpy.zeros((2, n))
      u[1,:] = x0_residual / self.global_dt

      happy   = False

      # compute the 1-norm of u
      local_nrmU = numpy.sum(abs(u[1:, :]), axis=1)
      global_normU = numpy.empty_like(local_nrmU)
      MPI.COMM_WORLD.Allreduce([local_nrmU, MPI.DOUBLE], [global_normU, MPI.DOUBLE])
      normU = numpy.amax(global_normU)

      # Normalization factors
      if normU > 0:
         ex = math.ceil(math.log2(normU))
         nu = 2**(-ex)
         mu = 2**(ex)
      else:
         nu = 1.0
         mu = 1.0

      # Flip the rest of the u matrix
      u_flip = nu * numpy.flipud(u[1:, :])

      # Compute necessary starting information
      j = 0
      H[:,:] = 0.0

      V[j, 0:n] = 0. # Initial condition

      # Update the last part of the first vector
      for k in range(p-1):
         i = p - k + 1
         V[j, n+k] = (τ_now**i) / math.factorial(i) * mu
      V[j, n+p-1] = mu

      # Normalize initial vector (this norm is nonzero)
      local_sum = V[j, 0:n] @ V[j, 0:n]
      global_sum_nrm = numpy.empty_like(local_sum)
      MPI.COMM_WORLD.Allreduce([local_sum, MPI.DOUBLE], [global_sum_nrm, MPI.DOUBLE])
      β = math.sqrt( global_sum_nrm + V[j, n:n+p] @ V[j, n:n+p] )

      # The first Krylov basis vector
      V[j, :] /= β

      # Incomplete orthogonalization process
      while j < self.niter:

         j = j + 1

         # Augmented matrix - vector product
         V[j, 0:n    ] = -A( V[j-1, 0:n] ) / self.global_dt + V[j-1, n:n+p] @ u_flip
         V[j, n:n+p-1] = V[j-1, n+1:n+p]
         V[j, -1     ] = 0.0

         #2. compute terms needed for R and T
         local_vec = V[0:j+1, 0:n] @ V[j-1:j+1, 0:n].T
         global_vec = numpy.empty_like(local_vec)
         MPI.COMM_WORLD.Allreduce([local_vec, MPI.DOUBLE], [global_vec, MPI.DOUBLE])
         global_vec += V[0:j+1, n:n+p] @ V[j-1:j+1, n:n+p].T

         #3. set values for Hessenberg matrix H
         H[0:j, j-1] = global_vec[0:j,1]

         #4. Procection of 2-step Gauss-Sidel to the orthogonal complement
         # Note: this is done in two steps. (1) matvec and (2) a lower
         # triangular solve
         # 4a. here we set the values for matrix M, Minv, N
         if j > 1:
            M[j-1, 0:j-1]    =  global_vec[0:j-1,0]
            N[0:j-1, j-1]    = -global_vec[0:j-1,0]
            Minv[j-1, 0:j-1] = -Minv[0:j-1, 0:j-1] @ global_vec[0:j-1,0]

         #4b. part 1: the mat-vec
         rhs = ( numpy.eye(j) + numpy.matmul(N[0:j, 0:j], Minv[0:j,0:j]) ) @ global_vec[0:j,1]

         #4c. part 2: the lower triangular solve
         sol = scipy.linalg.solve_triangular(M[0:j, 0:j], rhs, unit_diagonal=True, check_finite=False, overwrite_b=True)

         #5. Orthogonalize
         V[j, :] -= sol @ V[0:j, :]

         #7. compute norm estimate
         sum_sqrd = sum(global_vec[0:j,1]**2)
         if (global_vec[-1,1] < sum_sqrd):
            #use communication to compute norm estimate
            local_sum = V[j, 0:n] @ V[j, 0:n]
            global_sum_nrm = numpy.empty_like(local_sum)
            MPI.COMM_WORLD.Allreduce([local_sum, MPI.DOUBLE], [global_sum_nrm, MPI.DOUBLE])
            curr_nrm = math.sqrt( global_sum_nrm + V[j,n:n+p] @ V[j, n:n+p] )
         else:
            curr_nrm = numpy.sqrt(global_vec[-1,1] - sum_sqrd)

         # Happy breakdown
         if curr_nrm < happy_tol:
            happy = True
            if self.verbose: print(' ---> Happy breakdown in exponential smoother')
            break

         # Normalize vector and set norm to H matrix
         V[j,:] /= curr_nrm
         H[j,j-1] = curr_nrm

      # Scale the H matrix to the target spectral radius
      D_eig,_ = numpy.linalg.eig(H[0:j + 1, 0:j + 1])

      dt = self.target_spectral_radius / max(abs(D_eig))

      # To obtain the phi_1 function which is needed for error estimate
      H[0, j] = 1.0

      # Save h_j+1,j and remove it temporarily to compute the exponential of H
      nrm = H[j, j-1]
      H[j, j-1] = 0.0

      if self.verbose: print('Spectral radius of H:',max(abs(D_eig)))
      if self.verbose: print('dt:', dt )

      # Compute the exponential of the augmented matrix
      F = scipy.linalg.expm(dt * H[0:j + 1, 0:j + 1])

      # Restore the value of H_{m+1,m}
      H[j, j-1] = nrm

      phiv = β * F[:j, 0] @ V[:j, :n]  

      new_sol = phiv * dt
      if x0 is not None:
         new_sol += x0

      new_residual_norm = global_norm(b - A(new_sol))

      if new_residual_norm > x0_residual_norm:
         if self.verbose:
            print(f'Smoother failed to reduce the norm of the residual. '
                  f'Norm increased by {abs(new_residual_norm - x0_residual_norm)}')
            print('--> Rejected! Will return the old solution!')
         if x0 is not None:
            new_sol = x0.copy() # return the old solution
         else:
            new_sol = numpy.zeros_like(b)
      elif self.verbose:
         print('Smoothed residual', new_residual_norm)
         print('Smoother reduction:', abs(new_residual_norm - x0_residual_norm) )

      return new_sol


class KiopsSmoother(Smoother):
   def __init__(self, real_dt: float, dt_factor: float) -> None:
      super().__init__()
      self.real_dt = real_dt
      self.dt_factor = dt_factor

   def __smoothe__(self, A: _MatvecOp, b: numpy.ndarray, x: numpy.ndarray) -> numpy.ndarray:

      def residual(t, xx=None):
         if xx is None: return b / self.real_dt
         return (b - A(xx))/self.real_dt

      n = b.size
      vec = numpy.zeros((2, n))

      pseudo_dt = self.dt_factor * self.real_dt   # TODO : wild guess

      J = lambda v: -A(v) * pseudo_dt / self.real_dt

      R = residual(0, x)
      vec[1,:] = R.flatten()

      phiv, stats = kiops([1], J, vec, tol=1e-6, m_init=10, mmin=10, mmax=64, task1=False)

   #      print('norm phiv', global_norm(phiv.flatten()))
   #      print(f'KIOPS converged at iteration {stats[2]} (using {stats[0]} internal substeps)'
   #               f' to a solution with local error {stats[4]:.2e}')
      result = phiv.flatten() * pseudo_dt
      if x is not None: result += x
      return result


class RK1Smoother(Smoother):
   """1-stage Runge-Kutta smoother (Euler)."""
   def __init__(self, h: float) -> None:
      super().__init__()
      self.h = h

   def __smoothe__(self, A: _MatvecOp, b: numpy.ndarray, x: numpy.ndarray) -> numpy.ndarray:
      # print(f'b:\n{b}')
      # print(f'x:\n{x}')
      if x is None:
         x = self.h * b
      else:
         x += self.h * (b - A(x))

      # print(f'A: \n{A}')
      # print(f'x (after):\n{x}')

      return x


class RK3Smoother(Smoother):
   """3-stage Runge-Kutta smoother."""
   def __init__(self, h: float) -> None:
      super().__init__()
      self.h = h

   def __smoothe__(self, A: _MatvecOp, b: numpy.ndarray, x: numpy.ndarray) -> numpy.ndarray:
      alpha1 = 0.145
      alpha2 = 0.395

      if x is None:
         s1 = alpha1 * self.h * b
         s2 = alpha2 * self.h * (b - A(s1))
         x  = self.h * (b - A(s2))
      else:
         s1 = x + alpha1 * self.h * (b - A(x))
         s2 = x + alpha2 * self.h * (b - A(s1))
         x = x + self.h * (b - A(s2))

      return x

class ARKSmoother(Smoother):
   """Base additive Runge-Kutta smoother."""
   def __init__(self, h: float, field: numpy.ndarray, dt: float, rhs: RhsBundle) -> None:
      super().__init__()
      self.h = h
      self.field = field
      self.dt = dt
      self.rhs = rhs
      # self.rhs_implicit = rhs.implicit
      # self.rhs_explicit = rhs.explicit
      # self.rhs_imp_0 = rhs.implicit(field)
      # self.rhs_exp_0 = rhs.explicit(field)
      self.rhs_conv_0 = self.rhs.convective(field)
      self.rhs_full_0 = self.rhs.full(field)

   def A_conv(self, x):
      # return matvec_rat(x, self.dt, self.field, self.rhs_imp_0, self.rhs_implicit)
      return matvec_fun(x, self.dt, self.field, self.rhs_conv_0, self.rhs.convective) * (-self.dt / 2.0)

   def A_visc(self, x):
      # return matvec_rat(x, self.dt, self.field, self.rhs_exp_0, self.rhs_explicit)
      return x - (self.dt / 2.0) * (matvec_fun(x, self.dt, self.field, self.rhs_full_0, self.rhs.full) \
                                    - matvec_fun(x, self.dt, self.field, self.rhs_conv_0, self.rhs.convective))


class ARK3Smoother(ARKSmoother):
   """3-stage additive Runke-Kutta smoother."""
   def __smoothe__(self, A: _MatvecOp, b: numpy.ndarray, x: numpy.ndarray) -> numpy.ndarray:
      alpha1 = 0.145
      alpha2 = 0.395
      alpha3 = 1.0

      # beta1 = 1.0
      beta2 = 0.5
      beta3 = 0.5

      def f_c(u):
         return b - self.A_conv(u)

      def f_v(u):
         return - self.A_visc(u)

      if x is None:
         f_c0 = b
         u1 = alpha1 * self.h * f_c0
         f_c1 = f_c(u1)
         f_v1 = beta2 * f_v(u1)
         u2 = alpha2 * self.h * (f_c1 + f_v1)
         f_c2 = f_c(u2)
         f_v2 = beta3 * f_v(u2) + (1.0 - beta3) * f_v1
         u3 = alpha3 * self.h * (f_c2 + f_v2)
      else:
         u0 = x
         f_c0 = f_c(u0)
         f_v0 = f_v(u0)
         u1 = x + alpha1 * self.h * (f_c0 + f_v0)
         f_c1 = f_c(u1)
         f_v1 = beta2 * f_v(u1) + (1.0 - beta2) * f_v0
         u2 = x + alpha2 * self.h * (f_c1 + f_v1)
         f_c2 = f_c(u2)
         f_v2 = beta3 * f_v(u2) + (1.0 - beta3) * f_v1
         u3 = x + alpha3 * self.h * (f_c2 + f_v2)

      return u3
