import math
import numpy
import mpi4py.MPI
import scipy.linalg

from time import time 



"""

 --> full orthogonalization cwy 
    True one sync version 

   kiops(tstops, A, u; kwargs...) -> (w, stats)

Evaluate a linear combinaton of the ``φ`` functions evaluated at ``tA`` acting on
vectors from ``u``, that is

```math
  w(i) = φ_0(t[i] A) u[:, 1] + φ_1(t[i] A) u[:, 2] + φ_2(t[i] A) u[:, 3] + ...
```

The size of the Krylov subspace is changed dynamically during the integration.
The Krylov subspace is computed using the incomplete orthogonalization method.

Arguments:
  - `τ_out`    - Array of `τ_out`
  - `A`        - the matrix argument of the ``φ`` functions
  - `u`        - the matrix with rows representing the vectors to be multiplied by the ``φ`` functions

Optional arguments:
  - `tol`      - the convergence tolerance required (default: 1e-7)
  - `mmin`, `mmax` - let the Krylov size vary between mmin and mmax (default: 10, 128)
  - `m`        - an estimate of the appropriate Krylov size (default: mmin)
  - `iop`      - length of incomplete orthogonalization procedure (default: 2)
  - `task1`     - if true, divide the result by 1/T**p

Returns:
  - `w`      - the linear combination of the ``φ`` functions evaluated at ``tA`` acting on the vectors from ``u``
  - `stats[0]` - number of substeps
  - `stats[1]` - number of rejected steps
  - `stats[2]` - number of Krylov steps
  - `stats[3]` - number of matrix exponentials
  - `stats[4]` - Error estimate
  - `stats[5]` - the Krylov size of the last substep
  - `stats[6]` - number of times resorted to regular communication

`n` is the size of the original problem
`p` is the highest index of the ``φ`` functions

References:
* Gaudreault, S., Rainwater, G. and Tokman, M., 2018. KIOPS: A fast adaptive Krylov subspace solver for exponential integrators. Journal of Computational Physics. Based on the PHIPM and EXPMVP codes (http://www1.maths.leeds.ac.uk/~jitse/software.html). https://gitlab.com/stephane.gaudreault/kiops.
* Niesen, J. and Wright, W.M., 2011. A Krylov subspace method for option pricing. SSRN 1799124
* Niesen, J. and Wright, W.M., 2012. Algorithm 919: A Krylov subspace algorithm for evaluating the ``φ``-functions appearing in exponential integrators. ACM Transactions on Mathematical Software (TOMS), 38(3), p.22
"""
def cwy_1s(τ_out, A, u, tol = 1e-7, m_init = 10, mmin = 10, mmax = 128, iop = 2, task1 = False):


   rank = mpi4py.MPI.COMM_WORLD.Get_rank()

   ppo, n = u.shape
   p = ppo - 1

   if p == 0:
      p = 1
      # Add extra column of zeros
      u = numpy.row_stack((u, numpy.zeros(len(u))))

   # We only allow m to vary between mmin and mmax
   m = max(mmin, min(m_init, mmax))

   #if rank == 0:
   #   print("in cwy 1s, m_init = {}, m = {}".format(m_init, m))

   # Preallocate matrix
   V = numpy.zeros((mmax + 1, n + p))
   H = numpy.zeros((mmax + 1, mmax + 1))

   #matrices for CWY
   T = numpy.eye(mmax)

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
   normU = numpy.amax( mpi4py.MPI.COMM_WORLD.allreduce(local_nrmU) )

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
   reached_mmax    = False
   prev_normalized = False #if previous vector is normalized, skip certain parts

   l = 0

   """
   #arrays to hold timing data
   local_dot1 = [] #local dot product for Vjv values
   ortho_sum  = [] #for orthogonalizing
   matvec_t   = [] #applying T
   gsum_dots  = [] #global sum for dots
   formT      = [] #time to matvec to form T
   """

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
         #local_sum = V[0, 0:n] @ V[0, 0:n]
         #β = math.sqrt( mpi4py.MPI.COMM_WORLD.allreduce(local_sum) + V[j, n:n+p] @ V[j, n:n+p] )

      # Full CWY orthogonalization process
      while j < m:

         j = j + 1

         # Augmented matrix - vector product
         V[j, 0:n    ] = A( V[j-1, 0:n] ) + V[j-1, n:n+p] @ u_flip
         V[j, n:n+p-1] = V[j-1, n+1:n+p]
         V[j, -1     ] = 0.0

         #2. V_{j-1}^T * [v_{j-1} , v_j]
         local_vec = V[0:j, 0:n] @ V[j-1:j+1, 0:n].T

         global_vec = numpy.empty_like(local_vec)
         mpi4py.MPI.COMM_WORLD.Allreduce([local_vec, mpi4py.MPI.DOUBLE], [global_vec, mpi4py.MPI.DOUBLE])
         global_vec += V[0:j, n:n+p] @ V[j-1:j+1, n:n+p].T

         #3. norm of previous vector
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


         #6. set T matrix for cwy projection
         if (j > 1) :
            T[j-1, 0:j-1] = -global_vec[0:j-1,0].T @ T[0:j-1, 0:j-1]

         #7. orthogonalization 
         tmp     = T[0:j, 0:j] @ global_vec[:,1] 
         V[j,:] -= tmp @ V[0:j,:]

         #8. Happy breakdown
         if nrm < tol:
            happy = True
            break

         #9. set values for Hessenberg Matrix
         H[0:j,j-1] = tmp
         prev_normalized = False

         krystep += 1

      #---end of Gram-schmidt process---
      #normalize final vector
      #if we have not reached max iter
      if (not reached_mmax):
         local_sum = V[m,:] @ V[m,:]
         gsum      = numpy.empty_like(local_sum)
         mpi4py.MPI.COMM_WORLD.Allreduce([local_sum, mpi4py.MPI.DOUBLE], [gsum, mpi4py.MPI.DOUBLE])
         finalNrm = numpy.sqrt(gsum)

         V[m,:]  /= finalNrm
         H[m,m-1] = finalNrm

         prev_normalized = True #the previous vector is normalized 

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

         # residual norm estimate
         #temp_err = abs(β * nrm * F[j-2, 0] * V[j,0:n])
         #local_temp_err = temp_err @ temp_err
         #err = numpy.sqrt( mpi4py.MPI.COMM_WORLD.allreduce(local_temp_err) )


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
            reached_mmax = True
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
         #tau_now.append(τ_now)

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

   m_ret=m

   """
   nn = len(ortho_sum)
   avg_ortho    = sum(ortho_sum) / nn
   avg_localsum = sum(local_dot1) / nn
   avg_matvec   = sum(matvec_t) / nn
   avg_gsum_dots = sum(gsum_dots) / nn
   avg_formt    = sum(formT) / nn
   """

   stats = (step, reject, krystep, exps, conv, m_ret)

   return w, stats
