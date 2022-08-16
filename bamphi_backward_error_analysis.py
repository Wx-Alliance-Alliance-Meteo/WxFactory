from more_itertools import divide
import numpy
import scipy

def backward_error_analysis(step_state, points, polygon):
   """
   We find the smallest positive integer s that the backward error small in 
         q(s^{-1}A_tilde)^s = exp(A_tilde + delta-A_tilde)
   Specifically, we want
         || delta-A_tilde || <= tol * || A_tilde ||
         (with A_tilde shifted by _mu_ : [A_tilde - mu*I])

   This inequality is satisfied if
         max_{z in Gamma} {
            log[1 - exp(-s^{-1} z) * r(s^{-1}.z)] / [s^{-1}(z - mu)]
         }
            <=
         tol / (1 + sqrt(2))

   See Polygon for details on Gamma (the convex hull of the field of values of an approximation of A)
   """
   
   num_samples = 64

   x   = step_state.get_x() - points.avg
   tau = step_state.get_tau()
   p   = step_state.get_p()
   r   = points.num_points
   ell = step_state.get_ell()
   m   = r + p + ell
   L   = 3
   
   threshold = step_state.get_tol() / (1.0 + numpy.sqrt(2.0))

   log2e         = numpy.log2(numpy.e)
   log2threshold = numpy.log2(threshold)
   log2factorial = scipy.special.gammaln(m + 1) * log2e

   # Set iteration
   s = 0
   it = 0
   max_num_it = 1000000
   error = numpy.Inf

   # Quick iteration
   hull_points = polygon.get_convex_hull_points(num_samples)
   tmp = numpy.tile(hull_points, (r, 1)).T - numpy.tile(points.rho, (num_samples, 1))
   tmp = numpy.log2(tmp)
   tmp = numpy.sum(tmp, axis=1)

   # Compute log2( prod( x - x_i ) ) in a quick (?) way
   log2z = tmp + (ell-1) * numpy.log2(hull_points - points.avg) + p * numpy.log2(hull_points)
   nu = (hull_points * r - points.rho.sum()) / (m + 1)

   # Find the value of s that gives a correct set of errors (?)
   while (numpy.any(error > log2threshold) or numpy.any(numpy.isnan(error)) or numpy.any(numpy.isinf(error))) and it < max_num_it:
      while (numpy.any(error > log2threshold) or numpy.any(numpy.isnan(error)) or numpy.any(numpy.isinf(error))) and it < max_num_it:
         it += 1
         s  += 1
         tau_s = tau / s
         error = 2 * numpy.real(log2e * tau_s * nu[0] - log2factorial + (m-1) * numpy.log2(tau_s) + log2z[0]) # Check first error term for that specific s
      error = 2 * numpy.real(log2e * tau_s * nu - log2factorial + (m-1) * numpy.log2(tau_s) + log2z) # Check all error terms for that s

   # Thorough iteration
   s -= 1
   error = numpy.Inf
   log2z += numpy.log2(hull_points - points.avg)

   while (numpy.any(error > threshold) or numpy.any(numpy.isnan(error)) or numpy.any(numpy.isinf(error))) and it < max_num_it:
      it += 1
      s  += 1
      tau_s = tau / s
      norm_e = numpy.linalg.norm(error)

      d_s, F_s, s_d = phi_divided_differences(numpy.append(x * tau_s, numpy.zeros(L)), 0, L + ell - 1)

      norm_ds = numpy.linalg.norm(d_s)

      d_s *= numpy.exp(tau_s * points.avg)

      error_1a_1 = numpy.power(numpy.tile(tau_s * hull_points, (L+1, 1)).T, numpy.arange(L+1))
      error_1a_2 = d_s[m-2:m+L-1]
      error_1a   = error_1a_1 @ error_1a_2
      error_1b = numpy.exp(-tau_s * hull_points) 

      error_1  = numpy.log2(error_1a * error_1b)
      error_2 = m * numpy.log2(tau_s) + log2z
      error_3 = error_1 + error_2

      # error   = numpy.abs(numpy.log1p(-(2 ** error_3)) / (tau_s * (hull_points - points.avg)))
      #TODO Should use log1p(2**error_3), but numpy has a bug in that function. Just using the input is way more accurate for now
      error   = numpy.abs((-(2 ** error_3)) / (tau_s * (hull_points - points.avg)))

   if it >= max_num_it:
      raise ValueError(f'ERROR: Unable to determine a scaling strategy')

   step_state.set_s(s)
   step_state.set_divided_differences(d_s[:m].reshape((1, m)))
   
   big_f = numpy.exp(tau_s * points.avg) * (numpy.linalg.matrix_power(F_s[:m, :m], s_d))
   step_state.set_big_f(big_f)


def phi_divided_differences(x, index=0, L=0):
   """
   Compute phi_l divided differences at x

   Arguments:
   x       -- Interpolation points
   index   -- Index of the desired phi_l function
   L       -- Number of zero points in the tail of z
   """
   n = x.size + index

   x_augmented = numpy.append(numpy.zeros(index), x)

   # Compute scaling
   F = numpy.tile(x_augmented, (n, 1))
   F = numpy.tril(F - F.T)
   max_val = int(numpy.ceil(numpy.abs(F).max() / 3.5))

   # Compute F_0
   N = n + 30        # TODO why?
   divided_diff = numpy.append(1.0, numpy.cumprod((1.0 / numpy.arange(1, N+1)) / max_val)).astype(x.dtype)

   for j in range(n - L - 1, -1, -1):
      divided_diff[j + L : n - 1] += x[n - L - 1 : j : -1] * divided_diff[j + L + 1 : n]
      for k in range(N-1, n-2, -1):
         divided_diff[k] += x[j] * divided_diff[k+1]

   F[n - L : n, n - L : n] = scipy.linalg.toeplitz(divided_diff[:max(L, 1)])
   for j in range(n - L - 1, -1, -1):
      for k in range(n - j - 2, 0, -1):
         divided_diff[k] += F[k+j+1, j] * divided_diff[k+1]
      F[j, j+1:n+1] = divided_diff[1:n-j]

   numpy.fill_diagonal(F, numpy.append(numpy.exp(x[:n-L] / max_val), numpy.ones(L)))
   F = numpy.triu(F)
   divided_diff = F[0, :].copy()
   for k in range(max_val - 1):
      divided_diff = divided_diff @ F
   divided_diff = divided_diff[index : n]

   return divided_diff, F, max_val
