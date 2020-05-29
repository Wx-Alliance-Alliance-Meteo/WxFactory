import adaptive_ark
import butcher
import linsol
import math
import numpy
import mpi4py.MPI
import scipy.linalg

"""
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
  - `ishermitian` -  whether ``A`` is Hermitian (default: ishermitian(A))
  - `task1`     - if true, divide the result by 1/T**p

Returns:
  - `w`      - the linear combination of the ``φ`` functions evaluated at ``tA`` acting on the vectors from ``u``
  - `stats[1]` - number of substeps
  - `stats[2]` - number of rejected steps
  - `stats[3]` - number of Krylov steps
  - `stats[4]` - number of matrix exponentials
  - `stats[5]` - Error estimate
  - `stats[6]` - the Krylov size of the last substep

`n` is the size of the original problem
`p` is the highest index of the ``φ`` functions

References:
* Gaudreault, S., Rainwater, G. and Tokman, M., 2018. KIOPS: A fast adaptive Krylov subspace solver for exponential integrators. Journal of Computational Physics. Based on the PHIPM and EXPMVP codes (http://www1.maths.leeds.ac.uk/~jitse/software.html). https://gitlab.com/stephane.gaudreault/kiops.
* Niesen, J. and Wright, W.M., 2011. A Krylov subspace method for option pricing. SSRN 1799124
* Niesen, J. and Wright, W.M., 2012. Algorithm 919: A Krylov subspace algorithm for evaluating the ``φ``-functions appearing in exponential integrators. ACM Transactions on Mathematical Software (TOMS), 38(3), p.22
"""
def phi_ark(τ_out, J_exp, J_imp, u, tol = 1e-7, task1 = False):

   ppo, n = u.shape
   p = ppo - 1

   if p == 0:
      p = 1
      # Add extra column of zeros
      u = numpy.row_stack((u, numpy.zeros(len(u))))

   # Preallocate matrix
   res_mv = numpy.zeros(n + p)
   V = numpy.zeros(n + p)

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
   w = numpy.zeros((n, numSteps))
   w[:, 0] = u[0, :]

   # compute the 1-norm of u
   local_nrmU = numpy.sum(abs(u[1:, :]), axis=0)
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
   τ = τ_end / 200 # Wild guess

   l = 0
   V[0:n] = w[:, l]

   niter = 0

   f_e = lambda vec: rhs_exp(vec, n, p, J_exp, u_flip)
   f_i = lambda vec: rhs_imp(vec, n, J_imp)


   name_e = 'ARK3(2)4L[2]SA-ERK'
   Be = butcher.tableau(name_e)
   name_i = 'ARK3(2)4L[2]SA-ESDIRK'
   Bi = butcher.tableau(name_i)
   print('Solving for the φ-functions with ARK integrator : ', name_e, '/', name_i)

   # Update the last part of w
   for k in range(p-1):    # TODO : tranférer dans ark et mettre à jour a chaque pas de temps
      i = p - k + 1
      V[n+k] = (τ_now**i) / math.factorial(i) * mu
   V[n+p-1] = mu

   rtol = 1e-3
   atol = tol
   atol = 1e-3
   hmin = 1e-7
   hmax = 1.0

   tvals, V, nsteps, ierr = adaptive_ark.solve(f_e, f_i, J_imp, τ_out, V, Be, Bi, rtol, atol, hmin, hmax, n)

   if ierr != 0:
      print('Something has gone horribly wrong. Back away slowly')
      exit(1)

   w = V[:n,:] # TODO : switch the order of indices in w
   
   if task1 is True:
      for k in range(numSteps):
         w[:, k] = w[:, k] / τ_out[k]

   return w, niter

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
