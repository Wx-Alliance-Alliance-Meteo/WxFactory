import numpy

class KrylovSystem:
   def __init__(self, A, A_tilde, f, p, options, info):

      self.CONVERGENCE_THRESHOLD = 1e-15
      self.num_arnoldi_iter = options.num_arnoldi_iter
      self.tolerance = options.tolerance

      self.V = numpy.zeros((self.num_arnoldi_iter + p + 1, f.shape[1]))
      self.H = numpy.zeros((self.num_arnoldi_iter + 1, self.num_arnoldi_iter + 1))

      self.V[0, :] = f[0, :]

      V_bar = numpy.zeros((self.V.shape[0], p), dtype=float)
      V_bar[0, -1] = 1.0

      for k in range(p):
         self.V[k + 1], V_bar[k + 1] = A_tilde(self.V[k], V_bar[k])

      self.norm_V = numpy.linalg.norm(self.V[p])
      self.V[p] /= self.norm_V

      # numpy.set_printoptions(precision=15)
      # print(f'V[:{p+1}, :7] = \n{self.V[:p+1, :7]}')
      # raise ValueError

      # print(f'v1: \n{self.V[p]}')

      for i in range(self.num_arnoldi_iter):
         # print(f'i = {i}')
         index = i + p
         # print(f'index: {index}')
         self.V[index + 1] = A(self.V[index])

         # print(f'(before) V[{index + 1}]: {self.V[index + 1]}')
         # Orthogonalize wrt previous vectors
         for k in range(i, max(i-2, -1), -1):
            inner_index = p + k
            # print(f'Inner index: {inner_index}, (k, i) = ({k}, {i})')
            self.H[k, i] = self.V[inner_index].conj() @ self.V[index + 1]
            # print(f'Got v_inner = {self.V[inner_index]}')
            # print(f'Got H = {self.H[k, i]}')
            self.V[index + 1] -= self.H[k, i] * self.V[inner_index]
            # print(f'(updated) V[{index + 1}]: {self.V[index + 1]}')

         self.H[i + 1, i] = numpy.linalg.norm(self.V[index + 1])
         # print (f'new H[:, {i}]:{self.H[:, i]}')
         # print (f'h val: {self.H[i+1, i]}')
         if abs(self.H[i+1, i]) < self.CONVERGENCE_THRESHOLD:
            break

         self.V[index + 1] /= self.H[i+1, i]
         # print(f'(after)  V[{index + 1}]: {self.V[index + 1]}')

      # raise ValueError(f'H = \n{self.H}')

   def assemble(self, A, jump, p, is_real, info):

      print(f' --- Krylov assemble')

      ret   = 1      
      its   = 0
      w     = None
      w_bar = None
      e_t   = None

      m  = self.V.shape[0] - 1
      d  = info.step_state.get_divided_differences()[jump - 1]
      x  = info.step_state.get_x()
      ts = info.step_state.get_ts()

      # ret = 1;
      # m = length( V ) - 1;
      # d = info.A.his.d{ id }{ jump };
      # x = info.A.his.x{ id };

      # Determine coefficients for Arnoldi space
      cf_out, cf_aux = self.compute_cof(ts, x, d, p)
      if is_real:
         cf_out = numpy.real(cf_out)

      # [ cf_out, cf_aux ] = cof_krylov( info.A.his.ts( id ), H, x, d, p, n_V );
      # if rly
      #   cf_out = real( cf_out );
      # end

      w_bar = cf_out[p-1::-1]
      e_t   = numpy.min([int(numpy.ceil(1.1 * m)), self.num_arnoldi_iter])
      w     = 0.0
      w1    = 0.0
      for i in range(self.V.shape[0]):
         w += cf_out[i] * self.V[i]
         if i >= p:
            w1 += cf_aux[i - p] * self.V[i]
         if numpy.abs(cf_out[i]) < self.tolerance * numpy.linalg.norm(w):
            e_t = min(e_t, i)

      # w_ = cf_out( p : -1 : 1 ); % = V_( :, 1:p ) * cf_out( 1:p );
      # e_t = min( ceil( 1.1 * m ), opts.m_r_arn );
      # w = 0; w1 = 0;
      # for j = 1 : length( V )
      #   w  = w + cf_out( j ) * V{ j };
      #   if j > p
      #     w1 = w1 + cf_aux( j - p ) * V{ j };
      #   end
      #   if ( abs( cf_out( j ) ) < opts.tol * sqrt( w' * w ) )
      #     e_t = min( e_t, j );
      #   end
      # end

      w1 = ts * d[m + 1] * (A(w1) - x[m] * w1)
      w += w1

      # print(f'w1 = {w1}')

      if is_real:
         x  = numpy.real(x)
         d  = numpy.real(d)
         w1 = numpy.real(w1)
         w  = numpy.real(w)

      c1 = numpy.linalg.norm(w1, numpy.inf)
      for i in range(2, len(x) - m):
         w1 = (ts * d[m + i] / d[m + i - 1]) * (A(w1) - x[m + i - 1] * w1)
         w += w1
         c2 = numpy.linalg.norm(w1, numpy.inf)
         if c1 + c2 < numpy.linalg.norm(w, numpy.inf) * self.tolerance:
            ret = 0
            break
         c1 = c2

      its = m + i

      # % Wrap up interpolation at Ritz's values
      # w1 = ( info.A.his.ts( id ) * d( m + 2 ) ) * ( A( w1 ) - x( m + 1 ) * w1 );
      # w = w + w1;
      # % Now take care of the extended interpolation points
      # if rly
      #   x = real( x );
      #   d = real( d );
      #   w1 = real( w1 );
      #   w = real( w ); w_ = real( w_ );
      # end
      # c1 = norm( w1,inf );
      # for j = 3 : length( x ) - m
      #   w1 = ( info.A.his.ts( id ) * d( m + j ) / d( m + j - 1 ) ) * ( A( w1 ) - x( m + j - 1 ) * w1 );
      #   w = w + w1;
      #   c2 = norm( w1,inf );
      #   % [ c1, c2, norm( w, Inf ) * opts.tol ], pause
      #   if ( ( c1 + c2 ) < norm( w, Inf ) * opts.tol )
      #     ret = 0;
      #     break;
      #   end
      #   c1 = c2;
      # end
      # its = m + j - 1;
      # varargout{ 1 } = w;
      # varargout{ 2 } = w_;
      # varargout{ 3 } = its;
      # varargout{ 4 } = e_t;
      # varargout{ 5 } = ret;
      return w, w_bar, its, e_t, ret

   def compute_cof(self, ts, x, d, p):
      """
         Compute e^ts*H efficiently
      """
      m = self.H.shape[0] - 1
      f_out = numpy.append(d[p], numpy.zeros(m))
      # print(f'ts = {ts}, H shape {self.H.shape}, x[{p}] = {x[p]}, m = {m}')
      f_aux = numpy.append(numpy.append(ts * (self.H[0, 0] - x[p]), ts * self.H[1, 0]), numpy.zeros(m - 1))

      # print(f'f_out: {f_out}')
      # print(f'f_aux: {f_aux}')

      # print(f'H:\n{self.H}')

      for j in range(1, m):
         f_out += f_aux * d[p + j]
         f_aux = ts * (self.H @ f_aux - x[p + j] * f_aux)

      f_out += f_aux * d[p + m]

      # print(f'f_out: {f_out}')
      # print(f'f_aux: {f_aux}')

      tp = numpy.ones(p)
      tp[1:] *= ts

      f_out = numpy.append(numpy.cumprod(tp) * d[:p], f_out * (self.norm_V * ts**p))
      f_aux *= self.norm_V * ts ** p

      # print(f'f_out: {f_out}')
      # print(f'f_aux: {f_aux}')

      return f_out, f_aux
