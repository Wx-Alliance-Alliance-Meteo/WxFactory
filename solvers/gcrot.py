import numpy
from scipy.sparse.linalg._interface import aslinearoperator
from scipy.linalg import qr, solve, svd, qr_insert, lstsq

__all__ = ['gcrot']

def _fgmres(matvec, v0, m, tol, lpsolve=None, rpsolve=None, cs=(), outer_v=(),
            prepend_outer_v=False):

   if lpsolve is None:
      lpsolve = lambda x: x
   if rpsolve is None:
      rpsolve = lambda x: x

   vs = [v0]
   zs = []
   y = None
   res = numpy.nan

   m = m + len(outer_v)

   # Orthogonal projection coefficients
   B = numpy.zeros((len(cs), m), dtype=v0.dtype)

   # H is stored in QR factorized form
   Q = numpy.ones((1, 1), dtype=v0.dtype)
   R = numpy.zeros((1, 0), dtype=v0.dtype)

   eps = numpy.finfo(v0.dtype).eps

   breakdown = False

   niter = 0

   # FGMRES Arnoldi process
   for j in range(m):
      niter += 1

      if prepend_outer_v and j < len(outer_v):
         z, w = outer_v[j]
      elif prepend_outer_v and j == len(outer_v):
         z = rpsolve(v0)
         w = None
      elif not prepend_outer_v and j >= m - len(outer_v):
         z, w = outer_v[j - (m - len(outer_v))]
      else:
         z = rpsolve(vs[-1])
         w = None

      if w is None:
         w = lpsolve(matvec(z))
      else:
         # w is clobbered below
         w = w.copy()

      w_norm = numpy.linalg.norm(w)

      # GCROT projection: L A -> (1 - C C^H) L A
      # i.e. orthogonalize against C
      for i, c in enumerate(cs):
         alpha = c @ w
         B[i,j] = alpha
         w -= alpha*c

      # Orthogonalize against V
      hcur = numpy.zeros(j+2, dtype=Q.dtype)
      for i, v in enumerate(vs):
         alpha = v @ w
         hcur[i] = alpha
         w -= alpha*v
      hcur[i+1] = numpy.linalg.norm(w)

      with numpy.errstate(over='ignore', divide='ignore'):
         # Careful with denormals
         alpha = 1/hcur[-1]

      if numpy.isfinite(alpha):
         w *= alpha

      if not hcur[-1] > eps * w_norm:
         # w essentially in the span of previous vectors,
         # or we have nans. Bail out after updating the QR
         # solution.
         breakdown = True

      vs.append(w)
      zs.append(z)

      # Arnoldi LSQ problem

      # Add new column to H=Q@R, padding other columns with zeros
      Q2 = numpy.zeros((j+2, j+2), dtype=Q.dtype, order='F')
      Q2[:j+1,:j+1] = Q
      Q2[j+1,j+1] = 1

      R2 = numpy.zeros((j+2, j), dtype=R.dtype, order='F')
      R2[:j+1,:] = R

      Q, R = qr_insert(Q2, R2, hcur, j, which='col',
                      overwrite_qru=True, check_finite=False)

      # Transformed least squares problem
      # || Q R y - inner_res_0 * e_1 ||_2 = min!
      # Since R = [R'; 0], solution is y = inner_res_0 (R')^{-1} (Q^H)[:j,0]

      # Residual is immediately known
      res = abs(Q[0,-1])

      # Check for termination
      if res < tol or breakdown:
         break

   # -- Get the LSQ problem solution

   # The problem is triangular, but the condition number may be
   # bad (or in case of breakdown the last diagonal entry may be
   # zero), so use lstsq instead of trtrs.
   y, _, _, _, = lstsq(R[:j+1,:j+1], Q[0,:j+1].conj())

   B = B[:,:j+1]

   return Q, R, B, vs, zs, y, res, niter


def gcrot(A, b, x0=None, tol=1e-5, maxiter=1000, M=None, m=20, k=None, discard_C=False, truncate='oldest'):

   A = aslinearoperator(A)
   N = b.shape[0]

   niter = 0

   # Calculate initial residual.
   if x0 is None:
      x = numpy.zeros(N) if x0 is None else numpy.array(x0).ravel()
      r = b.copy()
   else:
      x = numpy.array(x0).ravel()
      r = b - A.matvec(x)

   if not numpy.isfinite(b).all():
      raise ValueError("RHS must contain only finite numbers")

   if truncate not in ('oldest', 'smallest'):
      raise ValueError(f'Invalid value for truncate: {truncate}')

   matvec = A.matvec
   if M is not None:
      psolve = M.matvec
   else:
      psolve = None

   # Persistent variable
   if not hasattr(gcrot, "CU"):
      CU = []
   else:
      CU = gcrot.CU

   if k is None:
      k = m

   if x0 is None:
      r = b.copy()
   else:
      r = b - matvec(x)

   b_norm = numpy.linalg.norm(b)
   if b_norm == 0:
      x = b
      return x, 0, 0, 0, 0

   if discard_C:
      CU[:] = [(None, u) for c, u in CU]

   # Reorthogonalize old vectors
   if CU:
      # Sort already existing vectors to the front
      CU.sort(key=lambda cu: cu[0] is not None)

      # Fill-in missing ones
      C = numpy.empty((A.shape[0], len(CU)), dtype=r.dtype, order='F')
      us = []
      j = 0
      while CU:
         # More memory-efficient: throw away old vectors as we go
         c, u = CU.pop(0)
         if c is None:
            c = matvec(u)
         C[:,j] = c
         j += 1
         us.append(u)

      # Orthogonalize
      Q, R, P = qr(C, overwrite_a=True, mode='economic', pivoting=True)
      del C

      # C := Q
      cs = list(Q.T)

      # U := U P R^-1,  back-substitution
      new_us = []
      for j in range(len(cs)):
         u = us[P[j]]
         for i in range(j):
            u -= us[P[i]] * R[i,j]
         if abs(R[j,j]) < 1e-12 * abs(R[0,0]):
            # discard rest of the vectors
            break
         u /= R[j,j]
         new_us.append(u)

      # Form the new CU lists
      CU[:] = list(zip(cs, new_us))[::-1]

   if CU:

      # Solve first the projection operation with respect to the CU
      # vectors. This corresponds to modifying the initial guess to
      # be
      #
      #     x' = x + U y
      #     y = argmin_y || b - A (x + U y) ||^2
      #
      # The solution is y = C^H (b - A x)
      for c, u in CU:
         yc = c @ r
         x += u * yc
         r -= c * yc

   # GCROT main iteration
   for j_outer in range(maxiter):

      beta = numpy.linalg.norm(r)

      # -- check stopping condition
      beta_tol = tol * b_norm

      if beta <= beta_tol and (j_outer > 0 or CU):
         # recompute residual to avoid rounding error
         r = b - matvec(x)
         beta = numpy.linalg.norm(r)

      if beta <= beta_tol:
         j_outer = -1
         break

      ml = m + max(k - len(CU), 0)

      cs = [c for c, u in CU]

      Q, R, B, vs, zs, y, _, gmiter = _fgmres(matvec, r/beta, ml, rpsolve=psolve, tol=tol*b_norm/beta, cs=cs)
      y *= beta

      niter += gmiter

      #
      # At this point,
      #
      #     [A U, A Z] = [C, V] G;   G =  [ I  B ]
      #                                   [ 0  H ]
      #
      # where [C, V] has orthonormal columns, and r = beta v_0. Moreover,
      #
      #     || b - A (x + Z y + U q) ||_2 = || r - C B y - V H y - C q ||_2 = min!
      #
      # from which y = argmin_y || beta e_1 - H y ||_2, and q = -B y
      #

      #
      # GCROT update
      #

      # Define new outer vectors

      # ux := (Z - U B) y
      ux = zs[0]*y[0]
      for z, yc in zip(zs[1:], y[1:]):
         ux += z * yc
      by = B @ y
      for cu, byc in zip(CU, by):
         c, u = cu
         ux -= u*byc

      # cx := V H y
      hy = Q @ (R @ y)
      cx = vs[0] * hy[0]
      for v, hyc in zip(vs[1:], hy[1:]):
         cx += v*hyc

      # Normalize cx, maintaining cx = A ux
      # This new cx is orthogonal to the previous C, by construction
      try:
         alpha = 1/numpy.linalg.norm(cx)
         if not numpy.isfinite(alpha):
            raise FloatingPointError()
      except (FloatingPointError, ZeroDivisionError):
         # Cannot update, so skip it
         continue

      cx *= alpha
      ux *= alpha

      # Update residual and solution
      gamma = cx @ r
      r -= gamma*cx
      x += gamma*ux

      # Truncate CU
      if truncate == 'oldest':
         while len(CU) >= k and CU:
            del CU[0]
      elif truncate == 'smallest':
         if len(CU) >= k and CU:
            D = solve(R[:-1,:].T, B.T).T
            W, _, _ = svd(D)

            # C := C W[:,:k-1],  U := U W[:,:k-1]
            new_CU = []
            for j, w in enumerate(W[:,:k-1].T):
               c, u = CU[0]
               c = c * w[0]
               u = u * w[0]
               for cup, wp in zip(CU[1:], w[1:]):
                  cp, up = cup
                  c += cp * wp
                  u += up * wp

               # Reorthogonalize at the same time; not necessary
               # in exact arithmetic, but floating point error
               # tends to accumulate here
               for cp, up in new_CU:
                  alpha = cp @ c
                  c -= cp * alpha
                  u -= up * alpha
               alpha = numpy.linalg.norm(c)
               c /= alpha
               u /= alpha

               new_CU.append((c, u))
            CU[:] = new_CU

      # Add new vector to CU
      CU.append((cx, ux))

   # Include the solution vector to the span
   CU.append((None, x.copy()))
   if discard_C:
      CU[:] = [(None, uz) for cz, uz in CU]

   gcrot.CU = CU

   return x, beta, niter, 0, 0
