import numpy

from ..global_operations import global_norm, global_dotprod

class KrylovSystem:
   def __init__(self, A, A_tilde, f, p, options, info):

      self.CONVERGENCE_THRESHOLD = 1e-15
      self.num_arnoldi_iter = options.num_arnoldi_iter
      self.tolerance = options.tolerance

      self.V = numpy.zeros((self.num_arnoldi_iter + p + 1, f.shape[1]), dtype=f.dtype)
      self.H = numpy.zeros((self.num_arnoldi_iter + 1, self.num_arnoldi_iter + 1), dtype=f.dtype)

      self.V[0, :] = f[0, :]

      V_bar = numpy.zeros((self.V.shape[0], p), dtype=float)
      V_bar[0, -1] = 1.0

      for k in range(p):
         self.V[k + 1], V_bar[k + 1] = A_tilde(self.V[k], V_bar[k])

      self.norm_V = global_norm(self.V[p])
      self.V[p] /= self.norm_V

      for i in range(self.num_arnoldi_iter):
         index = i + p
         self.V[index + 1] = A(self.V[index])

         # Orthogonalize wrt previous vectors
         for k in range(i, max(i-2, -1), -1):
            inner_index = p + k
            self.H[k, i] = global_dotprod(self.V[inner_index].conj(), self.V[index + 1])
            self.V[index + 1] -= self.H[k, i] * self.V[inner_index]

         self.H[i + 1, i] = global_norm(self.V[index + 1])
         if abs(self.H[i+1, i]) < self.CONVERGENCE_THRESHOLD:
            break

         self.V[index + 1] /= self.H[i+1, i]

   def assemble(self, A, jump, p, is_real, info):

      ret   = 1      
      its   = 0
      w     = None
      w_bar = None
      e_t   = None

      m  = self.V.shape[0] - 1
      d  = info.step_state.get_divided_differences()[jump - 1]
      x  = info.step_state.get_x()
      ts = info.step_state.get_ts()

      # Determine coefficients for Arnoldi space
      cf_out, cf_aux = self.compute_cof(ts, x, d, p)
      if is_real:
         cf_out = numpy.real(cf_out)

      w_bar = cf_out[p-1::-1]
      e_t   = numpy.min([int(numpy.ceil(1.1 * m)), self.num_arnoldi_iter])
      w     = 0.0    # Global vec
      w1    = 0.0    # Global vec
      for i in range(self.V.shape[0]):
         w += cf_out[i] * self.V[i]
         # print(f'w shape: {w.shape}, cf_out.shape: {cf_out.shape}')
         if i >= p:
            w1 += cf_aux[i - p] * self.V[i]
         if numpy.abs(cf_out[i]) < self.tolerance * global_op.norm(w):
            e_t = min(e_t, i)

      w1 = ts * d[m + 1] * (A(w1) - x[m] * w1)
      w  = w1 + w

      if is_real:
         x  = numpy.real(x)
         d  = numpy.real(d)
         w1 = numpy.real(w1)
         w  = numpy.real(w)

      c1 = global_op.inf_norm(w1)
      for i in range(2, len(x) - m):
         w1 = (ts * d[m + i] / d[m + i - 1]) * (A(w1) - x[m + i - 1] * w1)
         w += w1
         c2 = global_op.inf_norm(w1)
         if c1 + c2 < global_op.inf_norm(w) * self.tolerance:
            ret = 0
            break
         c1 = c2

      its = m + i

      return w, w_bar, its, e_t, ret

   def compute_cof(self, ts, x, d, p):
      """
         Compute e^ts*H efficiently
      """
      m = self.H.shape[0] - 1
      f_out = numpy.append(d[p], numpy.zeros(m))
      f_aux = numpy.append(numpy.append(ts * (self.H[0, 0] - x[p]), ts * self.H[1, 0]), numpy.zeros(m - 1))

      for j in range(1, m):
         f_out += f_aux * d[p + j]
         f_aux = ts * (self.H @ f_aux - x[p + j] * f_aux)

      f_out += f_aux * d[p + m]

      tp = numpy.ones(p)
      tp[1:] *= ts

      f_out = numpy.append(numpy.cumprod(tp) * d[:p], f_out * (self.norm_V * ts**p))
      f_aux *= self.norm_V * ts ** p

      return f_out, f_aux
