import numpy
import math

def exponential(N, Nc, s, V, invV):
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

   alpha = -math.log(numpy.finfo(float).eps)
   
   F = numpy.identity(N+1)
   for i in range(Nc, N+1):
      F[i,i] = math.exp(-alpha*((i-Nc)/(N-Nc))**s)

   F = V @ F @ invV

   return F

def apply_filter2D(S, mtrx, nb_elements, nbsolpts):
   """
   Apply filter \(\mathcal{F}\) on variable \(\mathcal{S}\)
   """
   R = numpy.empty_like(S)

   for elem in range(nb_elements):
      epais = elem * nbsolpts + numpy.arange(nbsolpts)

      # --- Direction x1

      R[:,:,epais] = S[:,:,epais] @ mtrx.filter_tr 

   for elem in range(nb_elements):
      epais = elem * nbsolpts + numpy.arange(nbsolpts)

      # --- Direction x2

      R[:,epais,:] = mtrx.filter @ R[:,epais,:]

   return R


def apply_filter3D(S, mtrx, nb_elements_hori, nb_elements_vert, nbsolpts):
   """
   Apply filter \(\mathcal{F}\) on variable \(\mathcal{S}\)
   """
   R = numpy.empty_like(S)

   # --- Direction x1
   for elem in range(nb_elements_hori):
      epais = elem * nbsolpts + numpy.arange(nbsolpts)
      R[:, :, epais] = S[:, :, epais] @ mtrx.filter_tr 
   
   # --- Direction x2
   for elem in range(nb_elements_hori):
      epais = elem * nbsolpts + numpy.arange(nbsolpts)
      R[:, epais, :] = mtrx.filter @ R[:, epais, :]
   
   # --- Direction x3
   for slab in range(nb_elements_hori * nbsolpts):
      for elem in range(nb_elements_vert):
         epais = elem * nbsolpts + numpy.arange(nbsolpts)
         R[epais, slab, :] = mtrx.filter @ R[epais, slab, :]

   return R
