from time import time
import sys
from typing import Callable, List, Optional, Tuple
from common.device import Device, default_device

from mpi4py import MPI
from numpy import ndarray

from .global_operations import global_dotprod, global_norm

__all__ = ['fgmres']

MatvecOperator = Callable[[ndarray], ndarray]

def _ortho_1_sync_igs(Q, R, T, K, j, comm, device: Device = default_device):
   """Orthonormalization process that only requires a single synchronization step.

   This processes row j of matrix Q (starting from 1) so that the first j rows are orthogonal, and the first (j-1)
   rows are orthonormal.
   It normalizes row j-1 of matrix Q during that process.

   Arguments:
      Q       -- The matrix to orthogonalize. We assume that the first (j-1) columns (or is it j-2?) of that matrix
                 are already orthonormal
      R, T, K -- Data from previous iterations of the orthonormalization process (?)
      j       -- Current number of vectors in the Kyrlov space. We orthogonalize the j^th vector (starting from 1)
                 with the previous ones.
                 _The matrices may be larger than that, so we can't rely on *.shape()_
   """
   if j < 2: return -1.0

   xp = device.xp

   local_tmp = Q[:j, :] @ Q[j-2:j, :].T     # Q multiplied by its own last 2 rows (up to j)

   device.synchronize()
   global_tmp = comm.allreduce(local_tmp) # Expensive step on multi-node execution
   small_tmp = global_tmp[:j-2, 0]

   R[:j-1, j-1]  = global_tmp[:j-1, 1]

   norm2         = global_tmp[j-2, 0] - (small_tmp @ small_tmp)
   norm          = xp.sqrt(norm2)
   R[j-2, j-2]   = norm
   R[j-2, j-1]  -= small_tmp @ R[:j-2, j-1]
   R[j-2, j-1]  /= norm

   T[:j-2, j-2]  = small_tmp / norm

   if j > 2:
      L = xp.tril(T[:j-2, :j-2].T, -1)
      L_plus = L + xp.eye(j-2)
      r3 = xp.linalg.solve(L_plus, small_tmp)

      R[:j-2, j-2] = K[:j-2, j-3] + r3
      K[:j-1, j-2] = (R[:j-1, j-1] - (R[:j-1, 1:j-1] @ r3)) / norm

      Q[j-2, :] -= Q[:j-2, :].T @ small_tmp
      Q[j-1, :] -= Q[:j-2, :].T @ R[:j-2, j-1] # Important here

   else:
      K[:j-1, j-2] = R[:j-1, j-1] / norm

   Q[j-2, :] /= norm
   Q[j-1, :] -= Q[j-2, :] * R[j-2, j-1]
   Q[j-1, :] /= norm

   return norm

def rotg(a: float, b: float) -> tuple[float, float, float]:
   anorm = abs(a)
   bnorm = abs(b)
   if bnorm == 0:
      return (1.0, 0.0, a)
   if anorm == 0:
      return (0.0, 1.0, b)
   import math
   scl = min(anorm, bnorm)
   sigma: float
   if (anorm > bnorm):
      sigma = math.copysign(1.0, a)
   else: sigma = math.copysign(1.0, b)

   r = sigma * (scl * math.sqrt((a / scl)**2 + (b / scl)**2))
   c = a / r
   s = b / r
   return (c, s, r)

def fgmres(A: MatvecOperator,
           b: ndarray,
           x0: Optional[ndarray] = None,
           tol: float = 1e-5,
           restart: int = 20,
           maxiter: Optional[int] = None,
           preconditioner: Optional[MatvecOperator] = None,
           hegedus: bool = False,
           verbose: int = 0,
           prefix: str = '',
           comm: MPI.Comm = MPI.COMM_WORLD,
           device: Device = default_device) \
            -> Tuple[ndarray, float, float, int, int, List[Tuple[float, float, float]]]:
   """
   Solve the given linear system (Ax = b) for x, using the FGMRES algorithm.

   Mandatory arguments:
   A              -- System matrix. This may be an operator that when applied to a vector [v] results in A*v. A should not be identity (or a multiplication of identity) when the preconditionner is none or identity
   b              -- The right-hand side of the system to solve. Must be a vector

   Optional arguments:
   x0             -- Initial guess for the solution. The zero vector if absent.
   tol            -- Maximum residual (|b - Ax| / |b|), below which we consider the system solved
   restart        -- Number of iterations in the inner loop
   maxiter        -- Maximum number of *outer loop* iterations. If absent, it's going to be a very large number
   preconditioner -- Operator [M^-1] that preconditions a given vector [v]. Computes the product (M^-1)*v
   hegedus        -- Whether to apply the Hegedüs trick (whatever that is)

   len(b) should be greater than restart

   Returns:
   1. The result [x]
   2. The relative residual |b - Ax| / |b|
   3. The number of (inner loop) iterations performed
   4. A flag that indicates the convergence status (0 if converged, -1 if not)
   5. The list of residuals at every iteration
   """
   if len(b) <= restart:
      raise ValueError('The b vector should be longer than the number of restart')

   xp = device.xp
   xalg = device.xalg

   t_start = time()
   niter = 0

   if preconditioner is None:
      preconditioner = lambda x: x     # Set up a preconditioner that does nothing

   num_dofs = len(b)

   if maxiter is None:
      maxiter = num_dofs * 10 # Wild guess

   if x0 is None:
      x = xp.zeros_like(b)
   else:
      x = x0.copy()

   # Check for early stop
   norm_b = global_norm(b, comm=comm)
   if norm_b == 0.0:
      return xp.zeros_like(b), 0., 0., 0, 0, [(0.0, time() - t_start, 0.0)]

   tol_relative = tol * norm_b

   Ax0 = A(x)
   residuals = []

   # Rescale the initial approximation using the Hegedüs trick
   if hegedus:
      norm_Ax0_2 = global_dotprod(Ax0, Ax0, comm=comm)
      if norm_Ax0_2 != 0.:
         ksi_min = global_dotprod(b, Ax0, comm=comm) / norm_Ax0_2
         x = ksi_min * x0
         Ax0 = A(x)

   r      = b - Ax0
   norm_r = global_norm(r, comm=comm)

   residuals.append(((norm_r / norm_b).item(), time() - t_start, 0.0))
   
   for outer in range(maxiter):
      # NOTE: We are dealing with row-major matrices, but we store the transpose of H and V.
      H = xp.zeros((restart+2, restart+2))
      R = xp.zeros((restart+2, restart+2)) # rhs of the MGS factorization (should be H.transposed?)
      T = xp.zeros((restart+2, restart+2))
      K = xp.zeros((restart+2, restart+2))
      V = xp.zeros((restart+2, num_dofs))  # row-major ordering
      Z = xp.zeros((restart+1, num_dofs))  # row-major ordering
      Q = []  # Givens Rotations

      V[0, :] = r / norm_r
      Z[0, :] = preconditioner(V[0, :])
      V[1, :] = A(Z[0, :])
      v_norm = _ortho_1_sync_igs(V, R, T, K, 2, comm, device)

      # This is the RHS vector for the problem in the Krylov Space
      g = xp.zeros(num_dofs)
      g[0] = norm_r
      for inner in range(restart):

         niter += 1

         # Modified Gram-Schmidt process (1-sync version, with lagged normalization)
         Z[inner + 1, :] = preconditioner(V[inner + 1])

         V[inner + 2, :] = A(Z[inner + 1, :] / v_norm) * v_norm
         v_norm = _ortho_1_sync_igs(V, R, T, K, inner + 3, comm, device)
         H[inner, :inner + 2] = R[:inner + 2, inner + 1]
         Z[inner + 1, :] /= v_norm

         # TODO : Start CPU here
         # Apply previous Givens rotations to H
         if inner > 0:
            _apply_givens(Q, H[inner, :], inner)

         # Calculate and apply next complex-valued Givens Rotation
         # ==> Note that if restart = num_dofs, then this is unnecessary
         # for the last inner
         #    iteration, when inner = num_dofs-1.
         if inner != num_dofs - 1:
            if H[inner, inner + 1] != 0:
               [c, s, r] = rotg(H[inner, inner], H[inner, inner + 1])

               Qblock = xp.array([[c, s], [-xp.conjugate(s), c]])
               Q.append(Qblock)

               # Apply Givens Rotation to g,
               #   the RHS for the linear system in the Krylov Subspace.
               g[inner:inner + 2] = Qblock @ g[inner:inner + 2]

               # Apply effect of Givens Rotation to H
               H[inner, inner] = xp.dot(Qblock[0, :], H[inner, inner:inner + 2])
               H[inner, inner + 1] = 0.0

         # Don't update norm_r if last inner iteration, because
         # norm_r is calculated directly after this loop ends.
         if inner < restart - 1:
            norm_r = xp.abs(g[inner+1])
            residuals.append(((norm_r / norm_b).item(), time() - t_start, 0.0))
            if verbose > 1:
               if comm.rank == 0: print(f'{prefix}norm_r / b = {residuals[-1][0]:.3e}')
               sys.stdout.flush()
            if norm_r < tol_relative:
               break

      # end inner loop, back to outer loop

      # Find best update to x in Krylov Space V.
      y = xalg.linalg.solve_triangular(H[0:inner + 1, 0:inner + 1].T, g[0:inner + 1])
      update = xp.ravel(Z[:inner+1, :].T @ y.reshape(-1, 1))
      x = x + update
      r = b - A(x)

      norm_r = global_norm(r, comm=comm)
      
      residuals.append(((norm_r / norm_b).item(), time() - t_start, 0.0))
      if verbose > 0:
         if comm.rank == 0: print(f'{prefix}res: {norm_r/norm_b:.2e} (iter {niter})')
         sys.stdout.flush()


      # TODO : End cpu calculation here
      # Has GMRES stagnated?
      indices = (x != 0)
      if indices.any():
         change = xp.max(xp.abs(update[indices] / x[indices]))
         if change < 1e-12:
            # No change, halt
            return x, norm_r, norm_b, niter, -1, residuals

      # test for convergence
      if norm_r < tol_relative:
         return x, norm_r, norm_b, niter, 0, residuals

   # end outer loop

   flag = 0
   if norm_r >= tol_relative: flag = -1
   return x, norm_r, norm_b, niter, flag, residuals

def _apply_givens(Q, v, k):

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
