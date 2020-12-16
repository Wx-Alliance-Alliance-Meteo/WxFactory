import math
import numpy
import mpi4py.MPI
import scipy
import scipy.sparse.linalg

class Fgmres:

   def __init__(self, tol = 1e-5, restart = 20, callback = None, reorth = False, hegedus = False, prefix = ''):
      self.tol = tol
      self.restart = restart
      self.callback = callback
      self.reorth = reorth
      self.hegedus = hegedus
      self.rank = mpi4py.MPI.COMM_WORLD.Get_rank()

   def solve(self, A, b, x0 = None, preconditioner = None, max_iter = None, prefix = '', tol = None):

      # if self.rank == 0:
      #    print('{} CALLING SOLVE WITH PRECONDITIONER = {}'.format(prefix, preconditioner))

      local_sum = numpy.zeros(1)
      def global_norm(vec):
         """Compute vector norm across all PEs"""
         local_sum[0] = vec @ vec
         return math.sqrt(mpi4py.MPI.COMM_WORLD.allreduce(local_sum))

      def gram_schmidt_mod(num, mat, vec, h_mat):
         """Modified Gram-Schmidt process"""
         for k in range(num + 1):
            local_sum[0] = mat[k, :] @ vec
            h_mat[num, k] = mpi4py.MPI.COMM_WORLD.allreduce(local_sum)
            vec = axpy(mat[k, :], vec, num_dofs, -h_mat[num, k])
         return vec

      tol      = tol if tol else self.tol
      x        = x0.copy() if x0 is not None else numpy.zeros_like(b)  # Zero if nothing given
      Ax0      = A(x)
      num_dofs = len(b)
      max_iter = max_iter if max_iter else num_dofs * 10  # Wild guess if nothing given

      # Get fast access to underlying BLAS routines
      [lartg] = scipy.linalg.get_lapack_funcs(['lartg'], [x])
      [axpy, dotu, scal] = scipy.linalg.get_blas_funcs(['axpy', 'dotu', 'scal'], [x])

      # Check for early stop
      norm_b = global_norm(b)
      if norm_b == 0.0:
         return numpy.zeros_like(b), 0., 0, 0

      # Rescale the initial approximation using the Hegedüs trick
      if self.hegedus:
         norm_Ax0 = global_norm(Ax0)
         if norm_Ax0 != 0.:
            ksi_min = numpy.sum(b * Ax0) / norm_Ax0
            x = ksi_min * x0

      r          = b - Ax0
      norm_r     = global_norm(r)
      old_norm_r = norm_r
      error      = norm_r / norm_b

      # Loop init
      num_iter       = 0
      total_num_iter = 0
      use_precond    = True

      for outer in range(int(math.ceil(max_iter / self.restart))):

         # NOTE: We are dealing with row-major matrices, but we store the transpose of H and V.
         H = numpy.zeros((self.restart + 1, self.restart + 1))
         V = numpy.zeros((self.restart + 1, num_dofs))  # row-major ordering
         Z = numpy.zeros((self.restart + 1, num_dofs))  # row-major ordering
         Q = []  # Givens Rotations

         V[0, :] = r / norm_r


         # This is the RHS vector for the problem in the Krylov Space
         g = numpy.zeros(num_dofs)
         g[0] = norm_r
         for inner in range(self.restart):

            num_iter += 1

            (Z[inner, :], num_precond_iter) = preconditioner.apply(V[inner, :], self, A) \
                  if preconditioner and use_precond else (V[inner, :], 1)
            w = A(Z[inner, :])
            V[inner + 1] = gram_schmidt_mod(inner, V, w, H)

            norm_v = global_norm(V[inner + 1, :])
            H[inner, inner + 1] = norm_v

            total_num_iter += num_precond_iter
            use_precond = False

            if self.reorth is True:
               for k in range(inner + 1):
                  local_sum[0] = V[k, :] @ V[inner + 1, :]
                  corr = mpi4py.MPI.COMM_WORLD.allreduce(local_sum)
                  H[inner, k] = H[inner, k] + corr
                  V[inner + 1, :] = axpy(V[k, :], V[inner + 1, :], num_dofs, -corr)

            # Check for breakdown
            if H[inner, inner + 1] != 0.0:
               V[inner + 1, :] = scal(1.0 / H[inner, inner + 1], V[inner + 1, :])

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
            if inner < self.restart - 1:
               old_norm_r = norm_r
               norm_r = numpy.abs(g[inner + 1])

               if preconditioner:
                  print(f'{prefix} Outer = {outer}, inner = {inner}, norm_r = {norm_r:.2e}')

               if norm_r < tol:
                  break

               if (old_norm_r - norm_r) / old_norm_r < 0.1: use_precond = False

            # Allow user access to the iterates
            if self.callback is not None:
               self.callback(x)

         # end inner loop, back to outer loop
         Z[inner + 1, :] = V[inner + 1, :]

         # Find best update to x in Krylov Space V.
         y = scipy.linalg.solve_triangular(H[0:inner + 1, 0:inner + 1].T, g[0:inner + 1])
         update = numpy.ravel(Z[:inner + 1, :].T @ y.reshape(-1, 1))
         x = x + update
         r = b - A(x)

         norm_r = global_norm(r)

         # Allow user access to the iterates
         if self.callback is not None:
            self.callback(x)

         num_precond_iter = preconditioner.num_iter + preconditioner.num_precond_iter if preconditioner else 0

         # Has GMRES stagnated?
         indices = (x != 0)
         if indices.any():
            change = numpy.max(numpy.abs(update[indices] / x[indices]))
            if change < 1e-12:
               # No change, halt
               return x, norm_r, total_num_iter, -1

         # test for convergence
         if norm_r < tol:
            return x, norm_r, total_num_iter, 0

         if (old_norm_r - norm_r) / old_norm_r < 1e-10:
            print('{} Stopping because solver is stuck!'.format(prefix))
            return x, norm_r, total_num_iter, -1

         old_norm_r = norm_r

      # end outer loop

      return x, norm_r / norm_b, total_num_iter, 0


def norm(var):
   local_sum = numpy.sum(var * var)
   return math.sqrt( mpi4py.MPI.COMM_WORLD.allreduce(local_sum) )

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
