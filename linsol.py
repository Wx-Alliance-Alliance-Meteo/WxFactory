import math
import numpy
import mpi4py.MPI
import scipy
import scipy.sparse.linalg


def fgmres(A, b, preconditioner, b_precond, interpolator,
           x0 = None, tol = 1e-5, restart = 20, maxiter = None, M = None, callback = None, reorth = False):

   rank = mpi4py.MPI.COMM_WORLD.Get_rank()

   num_iter = 0
   x        = x0.copy() if x0 is not None else numpy.zeros_like(b) # Zero if nothing given
   num_dofs = len(b)
   maxiter  = maxiter if maxiter else num_dofs * 10 # Wild guess if nothing given

   # Get fast access to underlying BLAS routines
   [lartg] = scipy.linalg.get_lapack_funcs(['lartg'], [x])
   [axpy, dotu, scal] = scipy.linalg.get_blas_funcs(['axpy', 'dotu', 'scal'], [x])

   local_sum = numpy.zeros(1)
   def global_norm(vec):
      """Compute vector norm across all PEs"""
      local_sum[0] = vec @ vec
      return math.sqrt(mpi4py.MPI.COMM_WORLD.allreduce(local_sum))

   norm_b = global_norm(b)
   if norm_b == 0.0:
      return numpy.zeros_like(b), 0., num_iter, 0

   r      = b - A(x)
   norm_r = global_norm(r)
   error  = norm_r / norm_b

   if error < tol:
      return x, error, num_iter, 0

   def gram_schmidt_mod(num, mat, vec, h_mat):
      """Modified Gram-Schmidt process"""
      for k in range(num + 1):
         local_sum[0] = mat[k, :] @ vec
         h_mat[num, k] = mpi4py.MPI.COMM_WORLD.allreduce(local_sum)
         vec = axpy(mat[k, :], vec, num_dofs, -h_mat[num, k])
      return vec

   def apply_preconditioner(vec):
      # return vec

      big_shape = A.field.shape
      big_grid = A.rhs.nb_sol_pts
      small_shape = preconditioner.field.shape
      small_grid = preconditioner.rhs.nb_sol_pts

      input_vec = interpolator.evalGridFast(vec.reshape(big_shape), small_grid, big_grid).flatten()
      output_vec, _, _, _ = gmres_mgs(preconditioner, input_vec, tol = tol)
      result = interpolator.evalGridFast(output_vec.reshape(small_shape), big_grid, small_grid).flatten()

      return result


   for outer in range(maxiter):

      # NOTE: We are dealing with row-major matrices, but we store the transpose of H and V.
      H = numpy.zeros((restart + 1, restart + 1))
      V = numpy.zeros((restart + 1, num_dofs))  # row-major ordering
      Z = numpy.zeros((restart + 1, num_dofs))  # row-major ordering
      Q = []  # Givens Rotations

      V[0, :] = r / norm_r

      # This is the RHS vector for the problem in the Krylov Space
      g = numpy.zeros(num_dofs)
      g[0] = norm_r
      for inner in range(restart):

         num_iter += 1

         Z[inner,:]   = apply_preconditioner(V[inner, :])
         w            = A(Z[inner,:])
         V[inner + 1] = gram_schmidt_mod(inner, V, w, H)

         norm_v = global_norm(V[inner + 1, :])
         H[inner, inner + 1] = norm_v

         if reorth is True:
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
         if inner < restart - 1:
            norm_r = numpy.abs(g[inner + 1])

            if rank == 0:
               print('Outer = {}, inner = {}, norm_r = {}'.format(outer, inner, norm_r))

            if norm_r < tol:
               if rank == 0:
                  print('exit loop b/c error is below tol')
               break

         # Allow user access to the iterates
         if callback is not None:
            callback(x)

      Z[-1,:] = apply_preconditioner(V[-1,:])
      # end inner loop, back to outer loop

      # Find best update to x in Krylov Space V.
      y = scipy.linalg.solve_triangular(H[0:inner + 1, 0:inner + 1].T, g[0:inner + 1])
      #update = numpy.ravel(V[:inner + 1, :].T @ y.reshape(-1, 1))
      update = numpy.ravel(Z[:inner + 1, :].T @ y.reshape(-1, 1))
      x = x + update
      r = b - A(x)

      norm_r = global_norm(r)

      # Allow user access to the iterates
      if callback is not None:
         callback(x)

      # Has GMRES stagnated?
      indices = (x != 0)
      if indices.any():
         change = numpy.max(numpy.abs(update[indices] / x[indices]))
         if change < 1e-12:
            # No change, halt
            return x, norm_r, num_iter, -1

      if rank == 0:
         print('norm_r = {}'.format(norm_r))

      # test for convergence
      if norm_r < tol:
         return x, norm_r, num_iter, 0

   # end outer loop

   return x, norm_r / norm_b, num_iter, 0


def gmres_mgs(A, b, x0=None, tol=1e-5, restart=20, maxiter=None, M=None, callback=None, reorth=False):

   niter = 0

   if x0 is None:
      x = numpy.zeros_like(b)
   else:
      x = x0.copy()

   n = len(b)

   if maxiter is None:
      maxiter = n * 10 # Wild guess

   # Get fast access to underlying BLAS routines
   [lartg] = scipy.linalg.get_lapack_funcs(['lartg'], [x])
   [axpy, dotu, scal] = scipy.linalg.get_blas_funcs(['axpy', 'dotu', 'scal'], [x])

   local_sum = numpy.zeros(1)
   def norm(x):
      local_sum[0] = x @ x
      return math.sqrt( mpi4py.MPI.COMM_WORLD.allreduce(local_sum) )

   bnrm2 = norm(b)
   if bnrm2 == 0.0:
      return numpy.zeros_like(b), 0., niter, 0

   r = b - A(x)

   normr = norm(r)

   error =  normr / bnrm2

   if error < tol:
      return x, error, niter, 0

   for outer in range(maxiter):

      # NOTE: We are dealing with row-major matrices, but we store the transpose of H and V.
      H = numpy.zeros((restart+1, restart+1))
      V = numpy.zeros((restart+1, n))  # row-major ordering
      Q = []  # Givens Rotations

      V[0, :] = r / normr

      # This is the RHS vector for the problem in the Krylov Space
      g = numpy.zeros(n)
      g[0] = normr
      for inner in range(restart):
         V[inner+1, :] = A(V[inner, :])
         niter += 1

         #  Modified Gram Schmidt
         for k in range(inner+1):
            local_sum[0] = V[k, :] @ V[inner+1, :]
            H[inner, k] = mpi4py.MPI.COMM_WORLD.allreduce(local_sum)
            V[inner+1, :] = axpy(V[k, :], V[inner+1, :], n, -H[inner, k])

         normv = norm(V[inner+1, :])
         H[inner, inner+1] = normv

         if (reorth is True):
            for k in range(inner+1):
               local_sum[0] = V[k, :] @ V[inner+1, :]
               corr = mpi4py.MPI.COMM_WORLD.allreduce(local_sum)
               H[inner, k] = H[inner, k] + corr
               V[inner+1, :] = axpy(V[k, :], V[inner+1, :], n, -corr)

         # Check for breakdown
         if H[inner, inner+1] != 0.0:
            V[inner+1, :] = scal(1.0/H[inner, inner+1], V[inner+1, :])

         # Apply previous Givens rotations to H
         if inner > 0:
            apply_givens(Q, H[inner, :], inner)

         # Calculate and apply next complex-valued Givens Rotation
         # ==> Note that if restart = n, then this is unnecessary
         # for the last inner
         #    iteration, when inner = n-1.
         if inner != n-1:
            if H[inner, inner+1] != 0:
               [c, s, r] = lartg(H[inner, inner], H[inner, inner+1])
               Qblock = numpy.array([[c, s], [-numpy.conjugate(s), c]])
               Q.append(Qblock)

               # Apply Givens Rotation to g,
               #   the RHS for the linear system in the Krylov Subspace.
               g[inner:inner+2] = Qblock @ g[inner:inner+2]

               # Apply effect of Givens Rotation to H
               H[inner, inner] = dotu(Qblock[0, :], H[inner, inner:inner+2])
               H[inner, inner+1] = 0.0

         # Don't update normr if last inner iteration, because
         # normr is calculated directly after this loop ends.
         if inner < restart-1:
            normr = numpy.abs(g[inner+1])
            if normr < tol:
               break

         # Allow user access to the iterates
         if callback is not None:
            callback(x)

      # end inner loop, back to outer loop

      # Find best update to x in Krylov Space V.
      y = scipy.linalg.solve_triangular(H[0:inner+1, 0:inner+1].T, g[0:inner+1])
      update = numpy.ravel(V[:inner+1, :].T @ y.reshape(-1, 1))
      x = x + update
      r = b - A(x)

      normr = norm(r)

      # Allow user access to the iterates
      if callback is not None:
        callback(x)

      # Has GMRES stagnated?
      indices = (x != 0)
      if indices.any():
        change = numpy.max(numpy.abs(update[indices] / x[indices]))
        if change < 1e-12:
         # No change, halt
         return x, normr, niter, -1

      # test for convergence
      if normr < tol:
         return x, normr, niter, 0

   # end outer loop

   return x, normr/bnrm2, niter, 0


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
