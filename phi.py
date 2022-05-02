import math
import numpy
import scipy.linalg

import adaptive_ark
import butcher
import linsol
from   gef_mpi       import GLOBAL_COMM

"""
   phi_ark(tstops, A, u; kwargs...) -> (w, stats)

Evaluate a linear combinaton of the ``φ`` functions evaluated at ``tA`` acting on
vectors from ``u``, that is

```math
  w(i) = φ_0(t[i] A) u[:, 1] + φ_1(t[i] A) u[:, 2] + φ_2(t[i] A) u[:, 3] + ...
```

The size of the Krylov subspace is changed dynamically during the integration.
The Krylov subspace is computed using the incomplete orthogonalization method.

Arguments:
  - `τ_out`    - Array of `τ_out`
  - `J_exp`    - non-stiff part of the matrix argument of the ``φ`` functions
  - `J_imp`    - stiff part of the matrix argument of the ``φ`` functions
  - `u`        - the matrix with rows representing the vectors to be multiplied by the ``φ`` functions

Optional arguments:
  - `tol`      - the convergence tolerance required (default: 1e-7)
  - `task1`     - if true, divide the result by 1/T**p

Returns:
  - `w`      - the linear combination of the ``φ`` functions evaluated at ``tA`` acting on the vectors from ``u``

`n` is the size of the original problem
`p` is the highest index of the ``φ`` functions
"""
def phi_ark(τ_out, J_exp, J_imp, u, tol = 1e-7, task1 = False, butcher_exp = 'ARK3(2)4L[2]SA-ERK', butcher_imp = 'ARK3(2)4L[2]SA-ESDIRK'):

   ppo, n = u.shape
   p = ppo - 1

   if p == 0:
      p = 1
      # Add extra column of zeros
      u = numpy.row_stack((u, numpy.zeros(len(u))))

   # Preallocate matrix
   V = numpy.zeros(n + p)

   τ_now   = 0.0
   numSteps = len(τ_out)

   # Initial condition
   w = numpy.zeros((n, numSteps))
   w[:, 0] = u[0, :]

   # compute the 1-norm of u
   local_nrmU = numpy.sum(abs(u[1:, :]), axis=0)
   normU = numpy.amax( GLOBAL_COMM().allreduce(local_nrmU) )

   # Normalization factors
#   if ppo > 1 and normU > 0:
#      ex = math.ceil(math.log2(normU))
#      nu = 2**(-ex)
#      mu = 2**(ex)
#   else:
#      nu = 1.0
#      mu = 1.0

   # TODO : debug
   nu = 1.0
   mu = 1.0

   # Flip the rest of the u matrix
   u_flip = nu * numpy.flipud(u[1:, :])

   l = 0
   V[0:n] = w[:, l]

   f_e = lambda vec: rhs_exp(vec, n, p, J_exp, u_flip)
   f_i = lambda vec: rhs_imp(vec, n, J_imp)

   Be = butcher.tableau(butcher_exp)
   Bi = butcher.tableau(butcher_imp)
   print(f'Solving for the φ-functions with ARK integrator : {butcher_exp} / {butcher_imp}')

   # Update the last part of w
   for k in range(p-1):
      i = p - k + 1
      V[n+k] = (τ_now**i) / math.factorial(i) * mu
   V[n+p-1] = mu

   # TODO : rtol and atol should be specified separately in config
   rtol = tol
   atol = tol
   hmin = 1e-7
   hmax = 1.0

   tvals, V, nsteps, ierr = adaptive_ark.solve(f_e, f_i, J_imp, τ_out, V, Be, Bi, rtol, atol, hmin, hmax, n, p, mu)

   if ierr != 0:
      print('Something has gone horribly wrong. Back away slowly')
      exit(1)

   w = V[:n,:] # TODO : switch the order of indices in w
   
   if task1 is True:
      for k in range(numSteps):
         w[:, k] = w[:, k] / τ_out[k]

   return w, nsteps

def rhs_exp(vec, n, p, J_exp, B):
   retval = numpy.zeros_like(vec)
   retval[0:n] = J_exp( vec[0:n] ) + vec[n:n+p] @ B
   retval[n:n+p-1] = vec[n+1:n+p]
   retval[-1] = 0.0
   return retval

def rhs_imp(vec, n, J_imp):
   retval = numpy.zeros_like(vec)
   retval[0:n] = J_imp( vec[0:n] )
   return retval

def I_minus_tauJ_imp(vec, τ, n, J_imp):
   mv = numpy.zeros_like(vec)
   mv[:n] = J_imp(vec[:n])
   return vec - τ * mv
