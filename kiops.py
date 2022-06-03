import math
import numpy
import mpi4py.MPI
import scipy.linalg

import scipy.linalg

def kiops(τ_out, A, u, tol = 1e-7, m_init = 10, mmin = 10, mmax = 128, task1 = False):

   ppo, n = u.shape
   p = ppo - 1

   if p == 0:
      p = 1
      # Add extra column of zeros
      u = numpy.row_stack((u, numpy.zeros(len(u))))

   # We only allow m to vary between mmin and mmax
   m = max(mmin, min(m_init, mmax))

   # Preallocate matrix
   V = numpy.zeros((mmax + 1, n + p))
   H = numpy.zeros((mmax + 1, mmax + 1))
   Minv = numpy.eye(mmax)
   M = numpy.eye(mmax)
   N = numpy.zeros([mmax,mmax])

   step    = 0
   krystep = 0
   ireject = 0
   reject  = 0
   exps    = 0
   sgn     = numpy.sign(τ_out[-1])
   τ_now   = 0.0
   τ_end   = abs(τ_out[-1])
   happy   = False
   j       = 0
   conv    = 0.0

   numSteps = len(τ_out)

   # Initial condition
   w = numpy.zeros((numSteps, n))
   w[0, :] = u[0, :].copy()

   # compute the 1-norm of u
   local_nrmU = numpy.sum(abs(u[1:, :]), axis=1)
   global_normU = numpy.empty_like(local_nrmU)
   mpi4py.MPI.COMM_WORLD.Allreduce([local_nrmU, mpi4py.MPI.DOUBLE], [global_normU, mpi4py.MPI.DOUBLE])
   normU = numpy.amax(global_normU)

   # Normalization factors
   if ppo > 1 and normU > 0:
      ex = math.ceil(math.log2(normU))
      nu = 2**(-ex)
      mu = 2**(ex)
   else:
      nu = 1.0
      mu = 1.0

   # Flip the rest of the u matrix
   u_flip = nu * numpy.flipud(u[1:, :])

   # Compute and initial starting approximation for the step size
   τ = τ_end

   # Setting the safety factors and tolerance requirements
   if τ_end > 1:
      γ = 0.2
      γ_mmax = 0.1
   else:
      γ = 0.9
      γ_mmax = 0.6

   delta = 1.4

   # Used in the adaptive selection
   oldm = -1; oldτ = math.nan; ω = math.nan
   orderold = True; kestold = True

   l = 0

   while τ_now < τ_end:

      # Compute necessary starting information
      if j == 0:

         H[:,:] = 0.0

         V[0, 0:n] = w[l, :]

         # Update the last part of w
         for k in range(p-1):
            i = p - k + 1
            V[j, n+k] = (τ_now**i) / math.factorial(i) * mu
         V[j, n+p-1] = mu

         # Normalize initial vector (this norm is nonzero)
         local_sum = V[0, 0:n] @ V[0, 0:n]
         global_sum_nrm = numpy.empty_like(local_sum)
         mpi4py.MPI.COMM_WORLD.Allreduce([local_sum, mpi4py.MPI.DOUBLE], [global_sum_nrm, mpi4py.MPI.DOUBLE])
         β = math.sqrt( global_sum_nrm + V[j, n:n+p] @ V[j, n:n+p] )

         # The first Krylov basis vector
         V[j, :] /= β

      # Incomplete orthogonalization process
      while j < m:

         j = j + 1

         # Augmented matrix - vector product
         V[j, 0:n    ] = A( V[j-1, 0:n] ) + V[j-1, n:n+p] @ u_flip
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
         if curr_nrm < tol:
            happy = True
            break

         # Normalize vector and set norm to H matrix
         V[j,:] /= curr_nrm
         H[j,j-1] = curr_nrm

         krystep += 1

      # To obtain the phi_1 function which is needed for error estimate
      H[0, j] = 1.0

      # Save h_j+1,j and remove it temporarily to compute the exponential of H
      nrm = H[j, j-1]
      H[j, j-1] = 0.0

      # Compute the exponential of the augmented matrix
      F = scipy.linalg.expm(sgn * τ * H[0:j + 1, 0:j + 1])
      exps += 1

      # Restore the value of H_{m+1,m}
      H[j, j-1] = nrm

      if happy is True:
         # Happy breakdown wrap up
         ω     = 0.
         err   = 0.
         τ_new = min(τ_end - (τ_now + τ), τ)
         m_new = m
         happy = False

      else:

         # Local truncation error estimation
         err = abs(β * nrm * F[j-1, j])

         # Error for this step
         oldω = ω
         ω = τ_end * err / (τ * tol)

         # Estimate order
         if m == oldm and τ != oldτ and ireject >= 1:
            order = max(1, math.log(ω/oldω) / math.log(τ/oldτ))
            orderold = False
         elif orderold is True or ireject == 0:
            orderold = True
            order = j/4
         else:
            orderold = True

         # Estimate k
         if m != oldm and τ == oldτ and ireject >= 1:
            kest = max(1.1, (ω/oldω)**(1/(oldm-m)))
            kestold = False
         elif kestold is True or ireject == 0:
            kestold = True
            kest = 2
         else:
            kestold = True

         if ω > delta:
            remaining_time = τ_end - τ_now
         else:
            remaining_time = τ_end - (τ_now + τ)

         # Krylov adaptivity

         same_τ = min(remaining_time, τ)
         τ_opt  = τ * (γ / ω)**(1 / order)
         τ_opt  = min(remaining_time, max(τ/5, min(5*τ, τ_opt)))

         m_opt = math.ceil(j + math.log(ω / γ) / math.log(kest))
         m_opt = max(mmin, min(mmax, max(math.floor(3/4*m), min(m_opt, math.ceil(4/3*m)))))

         if j == mmax:
            if ω > delta:
               m_new = j
               τ_new = τ * (γ_mmax / ω)**(1 / order)
               τ_new = min(τ_end - τ_now, max(τ/5, τ_new))
            else:
               τ_new = τ_opt
               m_new = m
         else:
            m_new = m_opt
            τ_new = same_τ

      # Check error against target
      if ω <= delta:

         # Yep, got the required tolerance; update
         reject += ireject
         step   += 1

         # Udate for τ_out in the interval (τ_now, τ_now + τ)
         blownTs = 0
         nextT = τ_now + τ
         for k in range(l, numSteps):
            if abs(τ_out[k]) < abs(nextT):
               blownTs += 1

         if blownTs != 0:
            # Copy current w to w we continue with.
            w[l+blownTs, :] = w[l, :].copy()

            for k in range(blownTs):
               τPhantom = τ_out[l+k] - τ_now
               F2 = scipy.linalg.expm(sgn * τPhantom * H[0:j, :j])
               w[l+k, :] = β * F2[:j, 0] @ V[:j, :n]

            # Advance l.
            l += blownTs

         # Using the standard scheme
         w[l, :] = β * F[:j, 0] @ V[:j, :n]

         # Update τ_out
         τ_now += τ

         j = 0
         ireject = 0

         conv += err

      else:
         # Nope, try again
         ireject += 1

         # Restore the original matrix
         H[0, j] = 0.0


      oldτ = τ
      τ    = τ_new

      oldm = m
      m    = m_new


   if task1 is True:
      for k in range(numSteps):
         w[k, :] = w[k, :] / τ_out[k]

   m_ret=m

   stats = (step, reject, krystep, exps, conv, m_ret)

   return w, stats
