import numpy
import numpy.linalg
import math
import sympy

class DFR_operators:
   def __init__(self, grd, param):
      self.extrap_west = lagrangeEval(grd.solutionPoints_sym, -1)
      self.extrap_east = lagrangeEval(grd.solutionPoints_sym,  1)

      self.extrap_south = lagrangeEval(grd.solutionPoints_sym, -1)
      self.extrap_north = lagrangeEval(grd.solutionPoints_sym,  1)

      self.extrap_down = lagrangeEval(grd.solutionPoints_sym, -1)
      self.extrap_up   = lagrangeEval(grd.solutionPoints_sym,  1)

      if param.filter_apply:
         self.V = vandermonde(grd.extension)
         self.invV = inv(self.V)
         N = len(grd.extension)-1
         Nc = math.floor(param.filter_cutoff * N)
         self.filter = filter_exponential(N, Nc, param.filter_order, self.V, self.invV)

      diff = diffmat(grd.extension_sym)

      if param.filter_apply:
         self.diff_ext = ( self.filter @ diff ).astype(float)
         self.diff_ext[numpy.abs(self.diff_ext) < 1e-20] = 0.
      else:
         self.diff_ext = diff

      if check_skewcentrosymmetry(self.diff_ext) is False:
         print('Something horribly wrong has happened in the creation of the differentiation matrix')
         exit(1)

      # Force matrices to be in C-contiguous order
      self.diff_solpt = numpy.ascontiguousarray( self.diff_ext[1:-1, 1:-1] )
      self.correction = numpy.ascontiguousarray( numpy.column_stack((self.diff_ext[1:-1,0], self.diff_ext[1:-1,-1])) )

      self.diff_solpt_tr = self.diff_solpt.T.copy()
      self.correction_tr = self.correction.T.copy()

      # Ordinary differentiation matrices (used only in diagnostic calculations)
      self.diff = diffmat(grd.solutionPoints)
      self.diff_tr = self.diff.T

      self.quad_weights = numpy.outer(grd.glweights, grd.glweights)


def lagrangeEval(points, newPt):
   M = len(points)
   x = sympy.symbols('x')
   l = numpy.zeros_like(points)
   if M == 1: 
      l[0] = 1 # Constant
   else:
      for i in range(M):
         l[i] = Lagrange_poly(x, M-1, i, points).evalf(subs={x: newPt}, n=20)
   return l.astype(float)


def diffmat(points):
   M = len(points)
   D = numpy.zeros((M,M))

   x = sympy.symbols('x')
   for i in range(M):
      dL = Lagrange_poly(x, M-1, i, points).diff()
      for j in range(M):
         if i != j:
            D[j,i] = dL.subs(x, points[j])
      D[i, i] = dL.subs(x, points[i])

   return D


def Lagrange_poly(x,order,i,xi):
    index = list(range(order+1))
    index.pop(i)
    return sympy.prod([(x-xi[j])/(xi[i]-xi[j]) for j in index])


def vandermonde(x):
   """
   Initialize the 1D Vandermonde matrix, \(\mathcal{V}_{ij}=P_j(x_i)\)
   """
   N = len(x)

   V = numpy.zeros((N, N), dtype=object)
   y = sympy.symbols('y')
   for j in range(N):
      for i in range(N):
         V[i, j] = sympy.legendre(j, y).evalf(subs={y: x[i]}, n=30, chop=True)

   return V


def filter_exponential(N, Nc, s, V, invV):
   """
   Create an exponential filter matrix that can be used to filter out
   high-frequency noise.

   The filter matrix \(\mathcal{F}\) is defined as \(\mathcal{F}=
   \mathcal{V}\Lambda\mathcal{V}^{-1}\) where the diagonal matrix,
   \(\Lambda\) has the entries \(\Lambda_{ii}=\sigma(i-1)\) for
   \(i=1,\ldots,n+1\) and the filter function, \(\sigma(i)\) has the form
   \[
      \sigma(i) =
         \begin{cases}
            1 & 0\le i\le n_c \\
            e^{-\alpha\left (\frac{i-n_c}{n-n_c}\right )^s} & n_c<i\le n.
      \end{cases}
   \]
   Here \(\alpha=-\log(\epsilon_M)\), where \(\epsilon_M\) is the machine
   precision in working precision, \(n\) is the order of the element,
   \(n_c\) is a cutoff, below which the low modes are left untouched and
   \(s\) (has to be even) is the order of the filter.

   Inputs:
      N : The order of the element.
      Nc : The cutoff, below which the low modes are left untouched.
      s : The order of the filter.
      V : The Vandermonde matrix, \(\mathcal{V}\).
      invV : The inverse of the Vandermonde matric, \(\mathcal{V}^{-1}\).

   Outputs:
      F: The return value is the filter matrix, \(\mathcal{F}\).
   """

   n_digit = 30

   alpha = -sympy.log(sympy.Float(numpy.finfo(float).eps, n_digit))

   F = numpy.identity(N+1, dtype=object)
   for i in range(Nc, N+1):
      t = sympy.Rational((i-Nc), (N-Nc))
      F[i,i] = sympy.exp(-alpha*t**s)

   F = V @ F @ invV

   return F


def check_skewcentrosymmetry(m):
   n,n = m.shape
   middle_row = 0

   if n % 2 == 0:
      middle_row = int(n / 2)
   else:
      middle_row = int(n / 2 + 1)

      if m[middle_row-1, middle_row-1] != 0.:
         print()
         print('When the order is odd, the central entry of a skew-centrosymmetric matrix must be zero.\nActual value is', m[middle_row-1, middle_row-1])
         return False

   for i in range(middle_row):
      for j in range(n):
         if (m[i, j] != -m[n-i-1, n-j-1]):
            print('Non skew-centrosymmetric entries detected:', (m[i, j], m[n-i-1, n-j-1]))
            return False

   return True


# Borrowed from Galois:
# https://github.com/mhostetter/galois
def inv(A):
    if not (A.ndim == 2 and A.shape[0] == A.shape[1]):
        raise numpy.linalg.LinAlgError(f"Argument `A` must be square, not {A.shape}.")
    field = type(A)
    n = A.shape[0]
    I = numpy.eye(n, dtype=A.dtype)

    # Concatenate A and I to get the matrix AI = [A | I]
    AI = numpy.concatenate((A, I), axis=-1)

    # Perform Gaussian elimination to get the reduced row echelon form AI_rre = [I | A^-1]
    AI_rre = row_reduce(AI, ncols=n)

    # The rank is the number of non-zero rows of the row reduced echelon form
    rank = numpy.sum(~numpy.all(AI_rre[:,0:n] == 0, axis=1))
    if not rank == n:
        raise numpy.linalg.LinAlgError(f"Argument `A` is singular and not invertible because it does not have full rank of {n}, but rank of {rank}.")

    A_inv = AI_rre[:,-n:]

    return A_inv


def row_reduce(A, ncols=None):
    if not A.ndim == 2:
        raise ValueError(f"Only 2-D matrices can be converted to reduced row echelon form, not {A.ndim}-D.")

    ncols = A.shape[1] if ncols is None else ncols
    A_rre = A.copy()
    p = 0  # The pivot

    for j in range(ncols):
        # Find a pivot in column `j` at or below row `p`
        idxs = numpy.nonzero(A_rre[p:,j])[0]
        if idxs.size == 0:
            continue
        i = p + idxs[0]  # Row with a pivot

        # Swap row `p` and `i`. The pivot is now located at row `p`.
        A_rre[[p,i],:] = A_rre[[i,p],:]

        # Force pivot value to be 1
        A_rre[p,:] /= A_rre[p,j]

        # Force zeros above and below the pivot
        idxs = numpy.nonzero(A_rre[:,j])[0].tolist()
        idxs.remove(p)
        A_rre[idxs,:] -= numpy.multiply.outer(A_rre[idxs,j], A_rre[p,:])

        p += 1
        if p == A_rre.shape[0]:
            break

    return A_rre
