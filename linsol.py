import math
import numpy
import mpi4py.MPI
import scipy
import scipy.sparse.linalg

def fgmres(A, b, x0 = None, tol = 1e-5, restart = 20, maxiter = None, preconditioner = None, hegedus = False):

   niter = 0

   num_dofs = len(b)

   if maxiter is None:
      maxiter = num_dofs * 10 # Wild guess

   if x0 is None:
      x = numpy.zeros_like(b)
   else:
      x = x0.copy()

   # Check for early stop
   norm_b = global_norm(b)
   if norm_b == 0.0:
      return numpy.zeros_like(b), 0., 0, 0

   tol_relative = tol * norm_b

   Ax0 = A(x)

   # Rescale the initial approximation using the Hegedüs trick
   if hegedus:
      norm_Ax0_2 = global_dotprod(Ax0, Ax0)
      if norm_Ax0_2 != 0.:
         ksi_min = global_dotprod(b, Ax0) / norm_Ax0_2
         x = ksi_min * x0
         Ax0 = A(x)

   r          = b - Ax0
   norm_r     = global_norm(r)

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
         Z[inner, :] = V[inner, :] if preconditioner is None else preconditioner(V[inner, :])

         w = A(Z[inner, :])

         # Classical Gram-Schmidt process
         local_sum = V[:inner+1, :] @ w
         H[inner, :inner+1] = mpi4py.MPI.COMM_WORLD.allreduce(local_sum)
         V[inner+1, :] = w - H[inner, :inner+1] @ V[:inner+1, :]

         local_sum = V[inner+1, :] @ V[inner+1, :]
         H[inner, inner+1] = math.sqrt( mpi4py.MPI.COMM_WORLD.allreduce(local_sum) )

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
            if norm_r < tol_relative:
               break

      # end inner loop, back to outer loop

      # Find best update to x in Krylov Space V.
      y = scipy.linalg.solve_triangular(H[0:inner + 1, 0:inner + 1].T, g[0:inner + 1])
      update = numpy.ravel(Z[:inner+1, :].T @ y.reshape(-1, 1))
      x = x + update
      r = b - A(x)

      norm_r = global_norm(r)

      # Has GMRES stagnated?
      indices = (x != 0)
      if indices.any():
         change = numpy.max(numpy.abs(update[indices] / x[indices]))
         if change < 1e-12:
            # No change, halt
            return x, norm_r, niter, -1

      # test for convergence
      if norm_r < tol_relative:
         return x, norm_r, niter, 0

   # end outer loop

   return x, norm_r / norm_b, niter, 0

def global_norm(vec):
   """Compute vector norm across all PEs"""
   local_sum = vec @ vec
   return math.sqrt( mpi4py.MPI.COMM_WORLD.allreduce(local_sum) )

def global_dotprod(vec1, vec2):
   """Compute dot product across all PEs"""
   local_sum = vec1 @ vec2
   return mpi4py.MPI.COMM_WORLD.allreduce(local_sum)

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
