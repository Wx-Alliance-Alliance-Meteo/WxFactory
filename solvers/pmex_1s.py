import math
import numpy
import mpi4py.MPI
import scipy.linalg

from time import time

#note: reuse info = true is original
#change to false to get accurate portrail of pm method
#post modern arnoldi with true 1sync method
def pmex_1s(τ_out, A, u, tol = 1e-7, delta = 1.2, m_init = 1, mmax = 128, reuse_info = True, task1 = False):
   

   rank = mpi4py.MPI.COMM_WORLD.Get_rank()
   ppo, n = u.shape
   p = ppo - 1

   if p == 0:
      p = 1
      # Add extra column of zeros
      u = numpy.row_stack((u, numpy.zeros(len(u))))

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

   first_accepted = True

   if not hasattr(pmex_1s, "static_mem") or reuse_info is False:
      pmex_1s.static_mem = True
      pmex_1s.suggested_step = τ_end 
      pmex_1s.suggested_m = mmax
      m_init = 12
      m_opt  = 1
   else:
      m_init = pmex_1s.suggested_m

   # We only allow m to vary between mmin and mmax
   mmin = 1
   m = max(mmin, min(m_init, mmax))

   #if rank == 0:
   #   print("in pmex_1s, m_init = {}, m = {}".format(m_init, m))

   # Preallocate matrix
   V    = numpy.zeros((mmax + 1, n + p))
   H    = numpy.zeros((mmax + 1, mmax + 1))
   Minv = numpy.eye(mmax)
   M    = numpy.eye(mmax)
   N    = numpy.zeros([mmax,mmax])

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
   τ = min(pmex_1s.suggested_step, τ_end)

   # Setting the safety factors and tolerance requirements
   if τ_end > 1:
      γ = 0.2
      γ_mmax = 0.1
   else:
      γ = 0.9
      γ_mmax = 0.6

   # Used in the adaptive selection
   oldm = -1; oldτ = math.nan; ω = math.nan
   kestold = True
   same_τ = None
   reached_mmax    = False
   prev_normalized = False #if previous vector is normalized, skip certain parts

   l = 0

   #arrays to hold timing data
   local_dot1 = [] #local dot product for Vjv values
   ortho_sum  = [] #for orthogonalizing
   matvec_t   = [] #part 1 of applying T
   low_triang = [] #part 2 of applying T
   gsum_dots  = [] #global sum for dots

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

         #---do not normalize initial vector---
         #lagged normalization
         """
         # Normalize initial vector (this norm is nonzero)
         local_sum = V[0, 0:n] @ V[0, 0:n]
         global_sum_nrm = numpy.empty_like(local_sum)
         mpi4py.MPI.COMM_WORLD.Allreduce([local_sum, mpi4py.MPI.DOUBLE], [global_sum_nrm, mpi4py.MPI.DOUBLE])
         β = math.sqrt( global_sum_nrm + V[j, n:n+p] @ V[j, n:n+p] )

         # The first Krylov basis vector
         V[j, :] /= β
         """

      #full orthogonalization process
      while j < m:

         j = j + 1

         #1. Augmented matrix - vector product
         V[j, 0:n    ] = A( V[j-1, 0:n] ) + V[j-1, n:n+p] @ u_flip
         V[j, n:n+p-1] = V[j-1, n+1:n+p]
         V[j, -1     ] = 0.0

         #2. compute terms needed T matrix
         start_ldot = time()
         local_vec = V[0:j, 0:n] @ V[j-1:j+1, 0:n].T
         local_dot1.append(time() - start_ldot)

         global_vec = numpy.empty_like(local_vec)
         start_gsum = time()
         mpi4py.MPI.COMM_WORLD.Allreduce([local_vec, mpi4py.MPI.DOUBLE], [global_vec, mpi4py.MPI.DOUBLE])
         gsum_dots.append( time() - start_gsum )

         global_vec += V[0:j, n:n+p] @ V[j-1:j+1, n:n+p].T

         #3. compute norm of previous vector
         nrm = numpy.sqrt(global_vec[-1,0])
       
         if j == 1:
           β = nrm

         #4. if the previous vector is NOT normalized, then scale for arnoldi
         if (not prev_normalized):

            #4a. scale for arnoldi
            V[j-1:j+1,:]        /= nrm 
            global_vec[:,1]     /= nrm
            global_vec[-1,1]    /= nrm
            global_vec[0:j-1,0] /= nrm
            H[j-1,j-2]           = nrm

         #5. Projection with 2-step Gauss-Sidel to the orthogonal complement
         # Note: this is done in two steps. (1) matvec and (2) a lower
         # triangular solve
         # 5a. here we set the values for matrix M, Minv, N
         if (j > 1):
           #tmp2 = global_vec[0:j-1,0] / nrm
           M[j-1, 0:j-1]    =  global_vec[0:j-1,0]
           N[0:j-1, j-1]    = -global_vec[0:j-1,0]
           Minv[j-1, 0:j-1] = -global_vec[0:j-1,0].T @ Minv[0:j-1, 0:j-1]

         #5b. part 1: the mat-vec
         start_mvec = time()
         rhs = ( numpy.eye(j) + numpy.matmul(N[0:j, 0:j], Minv[0:j,0:j]) ) @ global_vec[:,1]
         matvec_t.append(time() - start_mvec)

         #5c. part 2: the lower triangular solve
         start_lts = time()
         sol = scipy.linalg.solve_triangular(M[0:j, 0:j], rhs, unit_diagonal=True, check_finite=False, overwrite_b=True)
         low_triang.append(time() - start_lts)

         #6. Orthogonalize
         start_ortho = time()
         V[j, :] -= sol @ V[0:j, :]
         ortho_sum.append(time() - start_ortho)

         #7. Happy breakdown
         if nrm < tol:
            happy = True
            break
         
         #8. set H matrix
         H[0:j, j-1] = sol #<--correct version 
         #H[0:j, j-1] = global_vec[0:j,1]  #old one, wrong
         prev_normalized = False

         krystep += 1


      #---end of while loop---

      #normalize final vector
      if (not reached_mmax):
         local_sum = V[m,:] @ V[m,:]
         global_sum = numpy.empty_like(local_sum)
         mpi4py.MPI.COMM_WORLD.Allreduce([local_sum, mpi4py.MPI.DOUBLE], [global_sum, mpi4py.MPI.DOUBLE])
         finalNrm = numpy.sqrt(global_sum)

         V[m,:] /= finalNrm
         H[m,m-1] = finalNrm

         """
         if rank == 0:
           #print("computing final form for vector m = {}".format(m))
           print("global_sum = {}".format(global_sum))
           print("finalNrm = {}, H[m,m-1] = {}, m = {}".format(finalNrm, H[m,m-1], m))
           print("normalizing final vec, prev_normalized = {}".format(prev_normalized))
         """

         prev_normalized = True #the previous vector is normalized; skip scale for Arnoldi parts

      # To obtain the phi_1 function which is needed for error estimate
      H[0, j] = 1.0

      # Save h_j+1,j and remove it temporarily to compute the exponential of H
      nrm = H[j, j-1]
      H[j, j-1] = 0.0

      # Compute the exponential of the augmented matrix
      F_half = scipy.linalg.expm(sgn * 0.5 * τ * H[0:j + 1, 0:j + 1])
      F = F_half @ F_half

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
         err_half = abs(β * nrm * F_half[j-1, j])
         err      = abs(β * nrm * F[j-1, j])

         # Error for this step
         oldω = ω
         ω = τ_end * err / (τ * tol)

         # Estimate order
         order = math.log(err / err_half) / math.log(2)

         # Estimate k
         if m != oldm and τ == oldτ and ireject >= 1:
            kest = max(1.1, (ω/oldω)**(1/(oldm-m)))
            kestold = False
         elif kestold is True or ireject == 0:
            kest = 2
            kestold = True
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
            reached_mmax = True
            if ω > delta:
               m_new = j
               τ_new = τ * (γ_mmax / ω)**(1 / order)
               τ_new = min(τ_end - τ_now, max(τ/5, τ_new))
            else:
               τ_new = τ_opt
               m_new = m
         else:
            if same_τ < τ:
               m_new = m # We reduced tau to avoid small step size. Then keep m constant.
            else:
               m_new = m_opt
            τ_new = same_τ

      # Check error against target
      if ω <= delta:
         # Yep, got the required tolerance; update
         reject += ireject
         step   += 1

         if first_accepted:
            pmex_1s.suggested_step = min(pmex_1s.suggested_step, τ) 
            pmex_1s.suggested_m    = min(pmex_1s.suggested_m, m_opt)
            first_accepted = False

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
         reached_mmax = False
         prev_normalized = False

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


   nn = len(ortho_sum)
   avg_ortho    = sum(ortho_sum) / nn
   avg_localsum = sum(local_dot1) / nn
   avg_matvec   = sum(matvec_t) / nn
   avg_lowtriag = sum(low_triang) / nn
   avg_gsum_dots = sum(gsum_dots) / nn

   stats = (step, reject, krystep, exps, conv, avg_ortho, avg_localsum, avg_matvec, avg_lowtriag, avg_gsum_dots)
  
   return w, stats
