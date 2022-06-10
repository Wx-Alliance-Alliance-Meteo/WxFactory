import math
import numpy
import scipy
import scipy.sparse.linalg
from time import time
import sys

from gef_mpi import GLOBAL_COMM

def fgmres(A, b, x0 = None, tol = 1e-5, restart = 20, maxiter = None, preconditioner = None, hegedus = False, verbose = False):
   """
   Solve the given linear system (Ax = b) for x, using the FGMRES algorithm. 

   Mandatory arguments:
   A              -- System matrix. This may be an operator that when applied to a vector [v] results in A*v
   b              -- The right-hand side of the system to solve.

   Optional arguments:
   x0             -- Initial guess for the solution. The zero vector if absent.
   tol            -- Maximum residual (|b - Ax| / |b|), below which we consider the system solved
   restart        -- Number of iterations in the inner loop
   maxiter        -- Maximum number of *outer loop* iterations. If absent, it's going to be a very large number
   preconditioner -- Operator [M^-1] that preconditions a given vector [v]. Computes the product (M^-1)*v
   hegedus        -- Whether to apply the Hegedüs trick (whatever that is)

   Returns:
   1. The result [x]
   2. The relative residual |b - Ax| / |b|
   3. The number of (inner loop) iterations performed
   4. A flag that indicates the convergence status (0 if converged, -1 if not)
   5. The list of residuals at every iteration
   """

   t_start = time()
   niter = 0

   if preconditioner is None:
      preconditioner = lambda x: x     # Set up a preconditioner that does nothing

   num_dofs = len(b)
   total_work = 0.0

   if maxiter is None:
      maxiter = num_dofs * 10 # Wild guess

   if x0 is None:
      x = numpy.zeros_like(b)
   else:
      x = x0.copy()

   # Check for early stop
   norm_b = global_norm(b)
   if norm_b == 0.0:
      return numpy.zeros_like(b), 0., 0, 0, [(0.0, time() - t_start(), 0)]

   tol_relative = tol * norm_b

   Ax0 = A(x)
   residuals = []

   # Rescale the initial approximation using the Hegedüs trick
   if hegedus:
      norm_Ax0_2 = global_dotprod(Ax0, Ax0)
      if norm_Ax0_2 != 0.:
         ksi_min = global_dotprod(b, Ax0) / norm_Ax0_2
         x = ksi_min * x0
         Ax0 = A(x)

   r      = b - Ax0
   norm_r = global_norm(r)

   residuals.append((norm_r / norm_b, time() - t_start, 0.0))

   # Get fast access to underlying BLAS routines
   [lartg] = scipy.linalg.get_lapack_funcs(['lartg'], [x])
   [dotu, scal] = scipy.linalg.get_blas_funcs(['dotu', 'scal'], [x])

   for outer in range(maxiter):

      # NOTE: We are dealing with row-major matrices, but we store the transpose of H and V.
      H = numpy.zeros((restart+1, restart+1))
      V = numpy.zeros((restart+1, num_dofs))  # row-major ordering
      Z = numpy.zeros((restart+1, num_dofs))  # row-major ordering
      Q = []  # Givens Rotations

      V[0, :] = r / norm_r

      # This is the RHS vector for the problem in the Krylov Space
      g = numpy.zeros(num_dofs)
      g[0] = norm_r
      for inner in range(restart):

         niter += 1

         # Flexible preconditioner
         Z[inner, :] = preconditioner(V[inner, :])

         w = A(Z[inner, :])

         # Classical Gram-Schmidt process
         local_sum = V[:inner+1, :] @ w
         H[inner, :inner+1] = GLOBAL_COMM().allreduce(local_sum)
         V[inner+1, :] = w - H[inner, :inner+1] @ V[:inner+1, :]

         local_sum = V[inner+1, :] @ V[inner+1, :]
         H[inner, inner+1] = math.sqrt( GLOBAL_COMM().allreduce(local_sum) )

         # Check for breakdown
         if H[inner, inner+1] != 0.0:
            V[inner+1, :] = scal(1.0 / H[inner, inner+1], V[inner + 1, :])

         # Apply previous Givens rotations to H
         if inner > 0:
            apply_givens(Q, H[inner, :], inner)

         # Calculate and apply next complex-valued Givens Rotation
         # ==> Note that if restart = num_dofs, then this is unnecessary
         # for the last inner
         #    iteration, when inner = num_dofs-1.
         if inner != num_dofs - 1:
            if H[inner, inner + 1] != 0:
               [c, s, r] = lartg(H[inner, inner], H[inner, inner + 1])
               Qblock = numpy.array([[c, s], [-numpy.conjugate(s), c]])
               Q.append(Qblock)

               # Apply Givens Rotation to g,
               #   the RHS for the linear system in the Krylov Subspace.
               g[inner:inner + 2] = Qblock @ g[inner:inner + 2]

               # Apply effect of Givens Rotation to H
               H[inner, inner] = dotu(Qblock[0, :], H[inner, inner:inner + 2])
               H[inner, inner + 1] = 0.0

         # Don't update norm_r if last inner iteration, because
         # norm_r is calculated directly after this loop ends.
         if inner < restart - 1:
            norm_r = numpy.abs(g[inner+1])
            residuals.append((norm_r / norm_b, time() - t_start, 0.0))
            # if verbose: print(f'norm_r / b = {residuals[-1][0]:.3e}')
            if norm_r < tol_relative:
               break

      # end inner loop, back to outer loop

      # Find best update to x in Krylov Space V.
      y = scipy.linalg.solve_triangular(H[0:inner + 1, 0:inner + 1].T, g[0:inner + 1])
      update = numpy.ravel(Z[:inner+1, :].T @ y.reshape(-1, 1))
      x = x + update
      r = b - A(x)

      norm_r = global_norm(r)
      residuals.append((norm_r / norm_b, time() - t_start, 0.0))
      if verbose:
         print(f'res: {norm_r/norm_b:.2e} (iter {niter})')
         sys.stdout.flush()

      # Has GMRES stagnated?
      indices = (x != 0)
      if indices.any():
         change = numpy.max(numpy.abs(update[indices] / x[indices]))
         if change < 1e-12:
            # No change, halt
            return x, norm_r / norm_b, niter, -1, residuals

      # test for convergence
      if norm_r < tol_relative:
         return x, norm_r / norm_b, niter, 0, residuals

   # end outer loop

   flag = 0
   if norm_r >= tol_relative: flag = -1

   return x, norm_r / norm_b, niter, flag, residuals

def global_norm(vec):
   """Compute vector norm across all PEs"""
   local_sum = vec @ vec
   global_sum = numpy.array([0.0])
   GLOBAL_COMM().Allreduce(numpy.array([local_sum]), global_sum)
   return math.sqrt(global_sum[0])

def apply_givens(Q, v, k):
   """Apply the first k Givens rotations in Q to v.

   Parameters
   ----------
   Q : list
      list of consecutive 2x2 Givens rotations
   v : array
      vector to apply the rotations to
   k : int
      number of rotations to apply.

   Returns
   -------
   v is changed in place

   Notes
   -----
   This routine is specialized for GMRES.  It assumes that the first Givens
   rotation is for dofs 0 and 1, the second Givens rotation is for
   dofs 1 and 2, and so on.

   """
   for j in range(k):
      v[j:j+2] = Q[j] @ v[j:j+2]
