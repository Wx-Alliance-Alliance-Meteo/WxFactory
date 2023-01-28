import math
import numpy
import mpi4py.MPI
import scipy.linalg

from solvers.kiops  import kiops
from solvers.linsol import global_norm

def exponential(A, b: numpy.ndarray, x0: numpy.ndarray, niter:int=4, target_spectral_radius:float = 1.0, global_dt:float = 1.0, verbose:bool = False):
   """ Reduce the residual by integrating dx/dt = b - A x for an appropriate step size.
   """

   n = b.shape[0]

   x0_residual = b - A(x0) if x0 is not None else b
   x0_residual_norm = global_norm(x0_residual)

   happy_tol = 1e-10
   
   if verbose: print('x0 residual', x0_residual_norm) 
   
   # --- Krylov projection

   p = 1

   # Preallocate matrix
   V = numpy.zeros((niter + 1, n + p))
   H = numpy.zeros((niter + 1, niter + 1))
   Minv = numpy.eye(niter)
   M = numpy.eye(niter)
   N = numpy.zeros((niter,niter))

   u = numpy.zeros((2, n))
   u[1,:] = x0_residual / global_dt

   happy   = False

   # compute the 1-norm of u
   local_nrmU = numpy.sum(abs(u[1:, :]), axis=1)
   global_normU = numpy.empty_like(local_nrmU)
   mpi4py.MPI.COMM_WORLD.Allreduce([local_nrmU, mpi4py.MPI.DOUBLE], [global_normU, mpi4py.MPI.DOUBLE])
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
   mpi4py.MPI.COMM_WORLD.Allreduce([local_sum, mpi4py.MPI.DOUBLE], [global_sum_nrm, mpi4py.MPI.DOUBLE])
   β = math.sqrt( global_sum_nrm + V[j, n:n+p] @ V[j, n:n+p] )

   # The first Krylov basis vector
   V[j, :] /= β

   # Incomplete orthogonalization process
   while j < niter:

      j = j + 1

      # Augmented matrix - vector product
      V[j, 0:n    ] = -A( V[j-1, 0:n] ) / global_dt + V[j-1, n:n+p] @ u_flip
      V[j, n:n+p-1] = V[j-1, n+1:n+p]
      V[j, -1     ] = 0.0

      #2. compute terms needed for R and T
      local_vec = V[0:j+1, 0:n] @ V[j-1:j+1, 0:n].T
      global_vec = numpy.empty_like(local_vec)
      mpi4py.MPI.COMM_WORLD.Allreduce([local_vec, mpi4py.MPI.DOUBLE], [global_vec, mpi4py.MPI.DOUBLE])
      global_vec += V[0:j+1, n:n+p] @ V[j-1:j+1, n:n+p].T

      #3. set values for Hessenberg matrix H
      H[0:j, j-1] = global_vec[0:j,1]

      #4. Procection of 2-step Gauss-Sidel to the orthogonal complement
      # Note: this is done in two steps. (1) matvec and (2) a lower
      # triangular solve
      # 4a. here we set the values for matrix M, Minv, N
      if (j > 1):
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
         mpi4py.MPI.COMM_WORLD.Allreduce([local_sum, mpi4py.MPI.DOUBLE], [global_sum_nrm, mpi4py.MPI.DOUBLE])
         curr_nrm = math.sqrt( global_sum_nrm + V[j,n:n+p] @ V[j, n:n+p] )
      else:
        curr_nrm = numpy.sqrt(global_vec[-1,1] - sum_sqrd)

      # Happy breakdown
      if curr_nrm < happy_tol:
         happy = True
         if verbose: print(' ---> Happy breakdown in exponential smoother')
         break
      
      # Normalize vector and set norm to H matrix
      V[j,:] /= curr_nrm
      H[j,j-1] = curr_nrm

   # Scale the H matrix to the target spectral radius
   D_eig,_ = numpy.linalg.eig(H[0:j + 1, 0:j + 1])

   dt = target_spectral_radius / max(abs(D_eig))

   # To obtain the phi_1 function which is needed for error estimate
   H[0, j] = 1.0

   # Save h_j+1,j and remove it temporarily to compute the exponential of H
   nrm = H[j, j-1]
   H[j, j-1] = 0.0

   if verbose: print('Spectral radius of H:',max(abs(D_eig)))
   if verbose: print('dt:', dt )

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
      if verbose: 
         print('Smoother failed to reduce the norm of the residual. Norm increased by', abs(new_residual_norm - x0_residual_norm) )
         print('--> Rejected! Will return the old solution!')
      if x0 is not None:
         new_sol = x0.copy() # return the old solution
      else:
         new_sol = numpy.zeros_like(b)
   elif verbose: 
      print('Smoothed residual', new_residual_norm)
      print('Smoother reduction:', abs(new_residual_norm - x0_residual_norm) )

   return new_sol


def kiops_smoothe(A, b, x, real_dt, dt_factor):

   def residual(t, xx=None):
      if xx is None: return b / real_dt
      return (b - A(xx))/real_dt

   n = b.size
   vec = numpy.zeros((2, n))

   pseudo_dt = dt_factor * real_dt   # TODO : wild guess

   J = lambda v: -A(v) * pseudo_dt / real_dt

   R = residual(0, x)
   vec[1,:] = R.flatten()

   phiv, stats = kiops([1], J, vec, tol=1e-6, m_init=10, mmin=10, mmax=64, task1=False)

#      print('norm phiv', global_norm(phiv.flatten()))
#      print(f'KIOPS converged at iteration {stats[2]} (using {stats[0]} internal substeps)'
#               f' to a solution with local error {stats[4]:.2e}')
   result = phiv.flatten() * pseudo_dt
   if x is not None: result += x
   return result


def rk_smoothing(A, b, x, h):
   # t0 = time()

   alpha1 = 0.145
   alpha2 = 0.395
   # alpha3 = 1.0

   # beta1 = 1.0
   # beta2 = 0.5
   # beta3 = 0.5

   # u0 = x
   # f0 = b - A(u0)
   # u1 = x + alpha1 * h * P(f0)
   # f1 = beta2 * (b - A(u1)) + (1.0 - beta2) * f0
   # u2 = x + alpha2 * h * P(f1)
   # f2 = beta3 * (b - A(u2)) + (1.0 - beta3) * f1
   # u3 = x + alpha3 * h * P(f2)

   # x = u3

   if x is None:
      s1 = alpha1 * h * b
      s2 = alpha2 * h * (b - A(s1))
      x  = h * (b - A(s2))
   else:
      s1 = x + alpha1 * h * (b - A(x))
      s2 = x + alpha2 * h * (b - A(s1))
      x = x + h * (b - A(s2))

   # t1 = time()
   # print(f'smoothed in {t1-t0:.3} s')

   return x
