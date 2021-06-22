import math
import numpy
import mpi4py.MPI
import scipy
import scipy.sparse.linalg


def ortho_1_sync(Q, R, j):
   """Orthonormalization process that only requires a single synchronization step.

   This processes row j of matrix Q so that the first j+1 rows are orthogonal, and the first j rows are orthonormal.
   It normalizes row j-1 of matrix Q during that process.

   Arguments:
      Q -- The matrix to orthogonalize. We assume that the first (j-1) columns of that matrix are already orthonormal
      R -- Data from previous iterations of the orthonormalization process (?)
      j -- Which row vector we want to make orthogonal to the previous ones
   """
   m, _ = Q.shape

   if j == 0: return Q, R, 0.0

   local_tmp = Q[:j, :].conj() @ Q[j-1:j+1, :].T


   global_tmp = numpy.empty_like(local_tmp)
   mpi4py.MPI.COMM_WORLD.Allreduce(local_tmp, global_tmp)

   # global_tmp = mpi4py.MPI.COMM_WORLD.allreduce(local_tmp) # Expensive step on multi-node execution

   T            = numpy.zeros_like(R)
   T[:j-1, j-1] = global_tmp[:j-1, 0]
   norm2        = global_tmp[j-1, 0]
   norm         = numpy.sqrt(norm2)
   R[j-1, j-1]  = norm

   R[:j, j]     = global_tmp[:j,1]
   R[:j, j]     /= norm # Only do this for Arnoldi iterations
   Q[j-1, :]    /= norm
   Q[j, :]      /= norm # Only do this for Arnoldi iterations
   T[:j-1, j-1] /= norm
   R[j-1, j]    /= norm

   T[j-1, j-1] = 1.0
   L = numpy.tril(T[:j, :j], -1)
   R[:j, j] = (numpy.eye(j) - L) @ R[:j, j]

   delta = -R[:j, j].T @ Q[:j, :]
   Q[j, :] += delta

   return Q, R, norm


def fgmres(A, b, x0 = None, tol = 1e-5, restart = 20, maxiter = None, preconditioner = None, reorth = False, hegedus = False):

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
      return numpy.zeros_like(b), 0., 0, 0, [0.0]

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

   r          = b - Ax0
   norm_r     = global_norm(r)
   error      = norm_r / norm_b

   residuals.append(error)

   # Get fast access to underlying BLAS routines
   [lartg] = scipy.linalg.get_lapack_funcs(['lartg'], [x])
   [axpy, dotu, scal] = scipy.linalg.get_blas_funcs(['axpy', 'dotu', 'scal'], [x])

   use_new_orth = True

   for outer in range(maxiter):

      # NOTE: We are dealing with row-major matrices, but we store the transpose of H and V.
      H = numpy.zeros((restart+1, restart+1))
      R = numpy.zeros((restart+2, restart+2)) # rhs of the MGS factorization (should be H.transposed?)
      V = numpy.zeros((restart+2, num_dofs))  # row-major ordering
      Z = numpy.zeros((restart+1, num_dofs))  # row-major ordering
      Q = []  # Givens Rotations

      V[0, :] = r / norm_r
      Z[0, :] = V[0, :] if preconditioner is None else preconditioner(V[0, :])
      V[1, :] = A(Z[0, :])
      V, R, v_norm = ortho_1_sync(V, R, 1)

      # This is the RHS vector for the problem in the Krylov Space
      g = numpy.zeros(num_dofs)
      g[0] = norm_r
      for inner in range(restart):

         niter += 1

         # Modified Gram-Schmidt process (1-sync version, with lagged normalization)
         Z[inner + 1, :] = V[inner + 1] if preconditioner is None else preconditioner(V[inner + 1])
         V[inner + 2, :] = A(Z[inner + 1, :] / v_norm) * v_norm
         V, R, v_norm = ortho_1_sync(V, R, inner + 2)
         H[inner, :] = R[:restart + 1, inner + 1]
         Z[inner + 1, :] /= v_norm

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
            residuals.append(norm_r / norm_b)
            if norm_r < tol_relative:
               break

      # end inner loop, back to outer loop

      # Find best update to x in Krylov Space V.
      y = scipy.linalg.solve_triangular(H[0:inner + 1, 0:inner + 1].T, g[0:inner + 1])
      update = numpy.ravel(Z[:inner+1, :].T @ y.reshape(-1, 1))
      x = x + update
      r = b - A(x)

      norm_r = global_norm(r)
      residuals.append(norm_r / norm_b)

      # Has GMRES stagnated?
      indices = (x != 0)
      if indices.any():
         change = numpy.max(numpy.abs(update[indices] / x[indices]))
         if change < 1e-12:
            # No change, halt
            return x, norm_r, niter, -1, residuals

      # test for convergence
      if norm_r < tol_relative:
         return x, norm_r, niter, 0, residuals

   # end outer loop

   return x, norm_r / norm_b, niter, 0, residuals

def global_norm(vec):
   """Compute vector norm across all PEs"""
   local_sum = vec @ vec
   # return math.sqrt( mpi4py.MPI.COMM_WORLD.allreduce(local_sum) )
   global_sum = numpy.array([0.0])
   mpi4py.MPI.COMM_WORLD.Allreduce(numpy.array([local_sum]), global_sum)
   return math.sqrt(global_sum[0])

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
