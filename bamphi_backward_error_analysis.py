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
   
   print(f' --- Backward error analysis')

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

      # print(f'd_s =\n{d_s}')

      # print(f'hp shape: {hull_points.shape}, arange shape: {numpy.arange(L+1).size}')
      error_1a_1 = numpy.power(numpy.tile(tau_s * hull_points, (L+1, 1)).T, numpy.arange(L+1))
      error_1a_2 = d_s[m-2:m+L-1]
      error_1a   = error_1a_1 @ error_1a_2
      # print(f'pow : \n{error_1a_1}')
      # print(f'shape 1: {error_1a_1.shape}, tmp shape2: {error_1a_2.shape}')
      error_1b = numpy.exp(-tau_s * hull_points) 

      # print(f'1a norm {numpy.linalg.norm(error_1a)}, 1b norm {numpy.linalg.norm(error_1b)}')
      error_1  = numpy.log2(error_1a * error_1b)
      # error_1 = numpy.log2(numpy.exp(-tau_s * hull_points) * numpy.power(tau_s * hull_points, numpy.arange(L+1)) @ d_s[m-2:m+L]) 
      error_2 = m * numpy.log2(tau_s) + log2z
      error_3 = error_1 + error_2
      # print(f'e1 norm {numpy.linalg.norm(error_1)}, e2 norm {numpy.linalg.norm(error_2)}')
      # print(f'error_3 norm = {numpy.linalg.norm(error_3)}')
      # numpy.set_printoptions(precision=16)
      error_power = -numpy.power(2.0,error_3)
      # print(f'pow2: {error_power}')
      log1p = numpy.log1p(error_power)
      # print(f'log1p: {log1p}')

      # error   = numpy.abs(numpy.log1p(-(2 ** error_3)) / (tau_s * (hull_points - points.avg)))
      #TODO Should use log1p(2**error_3), but numpy has a bug in that function. Just using the input is way more accurate for now
      error   = numpy.abs((-(2 ** error_3)) / (tau_s * (hull_points - points.avg)))

      # print(f'error = {error}')
      # print(f'threshold compare: {numpy.any(error > log2threshold)}, nans: {numpy.any(numpy.isnan(error))}, inf: {numpy.any(numpy.isinf(error))}, it: {it < max_num_it}')

   # print('We\'re done!')

   if it >= max_num_it:
      raise ValueError(f'ERROR: Unable to determine a scaling strategy')

   step_state.set_s(s)
   step_state.set_divided_differences(d_s[:m].reshape((1, m)))
   
   
   # print(f's_d = {s_d}')
   # e1 = numpy.exp(tau_s * points.avg)
   # e2 = (numpy.linalg.matrix_power(F_s[:m, :m], s_d))
   # print(f'e1 = \n{e1}')
   # print(f'e2 = \n{e2}')
   big_f = numpy.exp(tau_s * points.avg) * (numpy.linalg.matrix_power(F_s[:m, :m], s_d))
   step_state.set_big_f(big_f)

   # print(f's = {s}')
   # print(f'd = \n{d_s[:m]}')
   # print(f'F = \n{big_f}')


#   s = s - 1; err = inf;
#   lg2z = lg2z + log2( z - info.A.pts.mu );
#   while true

#     it = it + 1; s = s + 1; ts = info.A.his.tau( id ) / s;

#     if opts.scal_ref
#       [ d_s, F_s, s_d ] = bamphi_dd_phi( [ ts * x; zeros( L, 1 ) ], 0, L + ell - 1 );
#     else
#         d_s             = bamphi_dd_phi( [ ts * x; zeros( L, 1 ) ], 0, L + ell - 1 );
#     end
#     d_s = exp( ts * info.A.pts.mu ) * d_s;
#     err = log2( exp( - ts * z ) .* ( bsxfun( @power, ts * z, 0 : L ) * d_s( m - 1 : m + L - 1 ) ) ) + m * log2( ts ) + lg2z;
#     err = abs( log1p( - 2.^err ) ./ ( ts * ( z - info.A.pts.mu ) ) );

#     if ( any( err > thresh ) || any( isnan( err ) ) || any( isinf( err ) ) ) && ( it < max_it )
#       continue
#     end
#     break

#   end
#   if not( it < max_it ),
#     error('bamphi_bea: unable to determine a scaling strategy');
#   end

#   info.A.his.s( id ) = s;
#   info.A.his.d{ id }{ 1 } = d_s( 1 : m );
#   if opts.scal_ref
#     info.A.his.F{ id } = exp( ts * info.A.pts.mu ) * ( F_s( 1 : m, 1 : m )^s_d );
#   end


def phi_divided_differences(x, index=0, L=0):
   """
   Compute phi_l divided differences at x

   Arguments:
   x       -- Interpolation points
   index   -- Index of the desired phi_l function
   L       -- Number of zero points in the tail of z
   """
   # print(f'x:\n{x}')
   n = x.size + index

   x_augmented = numpy.append(numpy.zeros(index), x)

   #   % Compute scaling
   F = numpy.tile(x_augmented, (n, 1))
   F = numpy.tril(F - F.T)
   # print(f'F:\n{F}')
   max_val = int(numpy.ceil(numpy.abs(F).max() / 3.5))
   # print(f'max_val = {max_val}')

   #   % Compute F_0
   N = n + 30        # TODO why?
   divided_diff = numpy.append(1.0, 1.0 / (numpy.cumprod(numpy.arange(1, N+1, dtype=float) * max_val) ))
   # c = numpy.cumprod(numpy.arange(1, 11, dtype=float)) * max_val
   # # print(f'dd: {divided_diff}')
   # for i, dd in enumerate(c):
   #    print(f'{i:3d}: {dd:.2e}')

   for j in range(n - L - 1, -1, -1):
      # a = x[n-L-1 : j : -1]
      # b = divided_diff[j+L+1:n]
      # c = divided_diff[j+L:n-1]
      # print(f'j: {j}')
      # if a.size > 0:
      #    print(f'a: {a}')
      #    print(f'b: {b}')
      #    print(f'c: {c}')
      divided_diff[j + L : n - 1] += x[n - L - 1 : j : -1] * divided_diff[j + L + 1 : n]
      for k in range(N-1, n-2, -1):
         divided_diff[k] += x[j] * divided_diff[k+1]

   # print(f'dd: \n{divided_diff}')

   F[n - L : n, n - L : n] = scipy.linalg.toeplitz(divided_diff[:max(L, 1)])
   # print(f'F = \n{F}')
   # Check for probs here:
   # weird_f = F[n - L : n, n - L : n]
   for j in range(n - L - 1, -1, -1):
      for k in range(n - j - 2, 0, -1):
         # print(f'k = {k}')
         # print(f'dd[{k}] = {divided_diff[k]}')
         divided_diff[k] += F[k+j+1, j] * divided_diff[k+1]
         # print(f'dd[{k}] = {divided_diff[k]}')
         # print(f'other f: {F[k+j+1, j]}')
      F[j, j+1:n+1] = divided_diff[1:n-j]
      # print(f'weird f\n{F[j, j+1:n+1]}')

   # print(f'dd: \n{divided_diff}')
      
   numpy.fill_diagonal(F, numpy.append(numpy.exp(x[:n-L] / max_val), numpy.ones(L)))
   # F[0:n**2:n+1] = numpy.append(numpy.exp(x[:n-L] / max_val), numpy.ones(L))
   # print(f'f sub \n{F[0:n**2:n+1]}')
   F = numpy.triu(F)
   # print(f'F = \n{F[:16, :16]}')
   # raise ValueError
   divided_diff = F[0, :].copy()
   for k in range(max_val - 1):
      divided_diff = divided_diff @ F
   divided_diff = divided_diff[index : n]

   # print(f'dd: \n{divided_diff}')

   return divided_diff, F, max_val

#   N = n + 30;
#   dd = [ 1, 1 ./ cumprod( ( 1 : N ) * s ) ];
#   for j = n - L : -1 : 1
#     dd( j+L : n-1 ) = dd( j+L : n-1 ) + x( n-L : -1 : j+1 ) .* dd( j+L+1 : n );
#     for k = N : -1 : n
#       dd( k ) = dd( k ) + x( j ) * dd( k + 1 );
#     end
#   end
#   F( n-L+1 : n, n-L+1 : n ) = toeplitz( dd( 1 : max( L,1 ) ) ); % max( L,1 ) because toeplitz([]) triggers an error
#   for j = n - L : -1 : 1
#     for k = n - j : -1 : 2
#       dd( k ) = dd( k ) + F( k+j, j ) * dd( k+1 );
#     end
#     F( j,j+1 : n ) = dd( 2 : n-j+1 );
#   end
#   F( 1 : n+1 : n^2 ) = [ exp( x( 1 : n-L ) / s ), ones( 1,L ) ];
#   F = triu( F );
#   % Square F_0 into F_s
#   dd = F( 1,: );
#   for k = 1 : s - 1
#     dd = dd * F;
#   end
#   dd = dd( l+1 : n ).';
# end
