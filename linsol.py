import math
import numpy
import mpi4py.MPI
import scipy
import scipy.sparse.linalg

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
   [axpy, dot, scal] = scipy.linalg.get_blas_funcs(['axpy', 'dot', 'scal'], [x])

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

      Q = []  # Givens Rotations
      H = numpy.zeros((restart+1, restart+1))
      V = numpy.zeros((restart+1, n))  # row-major ordering
      
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
            H[k, inner] = mpi4py.MPI.COMM_WORLD.allreduce(local_sum)
            V[inner+1, :] = axpy(V[k, :], V[inner+1, :], n, -H[k, inner])

         normv = norm(V[inner+1, :])
         H[inner, inner+1] = normv

         if (reorth is True):
            for k in range(inner+1):
               local_sum[0] = V[k, :] @ V[inner+1, :]
               corr = mpi4py.MPI.COMM_WORLD.allreduce(local_sum)
               H[k, inner] = H[k, inner] + corr
               V[inner+1, :] = axpy(V[k, :], V[inner+1, :], n, -corr)

         # Check for breakdown
         if H[inner+1, inner] != 0.0:
            V[inner+1, :] = scal(1.0/H[inner+1, inner], V[inner+1, :])

         # Apply previous Givens rotations to H
         if inner > 0:
            apply_givens(Q, H[:, inner], inner)

         # Calculate and apply next complex-valued Givens Rotation
         # ==> Note that if restart = n, then this is unnecessary
         # for the last inner
         #    iteration, when inner = n-1.
         if inner != n-1:
            if H[inner+1, inner] != 0:
               [c, s, r] = lartg(H[inner, inner], H[inner+1, inner])
               Qblock = numpy.array([[c, s], [-numpy.conjugate(s), c]])
               Q.append(Qblock)

               # Apply Givens Rotation to g,
               #   the RHS for the linear system in the Krylov Subspace.
               g[inner:inner+2] = Qblock @ g[inner:inner+2]

               # Apply effect of Givens Rotation to H
               H[inner, inner] = dot(Qblock[0, :], H[inner:inner+2,inner])
               H[inner+1, inner] = 0.0

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

      y = scipy.linalg.solve_triangular(H[0:inner+1, 0:inner+1], g[0:inner+1])

      update = numpy.ravel(y.reshape(-1, 1) @ V[:inner+1, :])
      x = x + update
      r = b - A(x)

      # Apply preconditioner
#      r = M * r
      
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
      v[j+2:j] = Q[j] @ v[j+2:j]
