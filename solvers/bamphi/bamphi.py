import numpy
import scipy.special

from .bamphi_krylov      import KrylovSystem
from .bamphi_step_state  import StepState
from ..global_operations import global_inf_norm

from numpy.random import default_rng
bamphi_rng = default_rng()

class BamphiOptions:
   def __init__(self):
      # Error options
      self.tolerance = numpy.finfo(float).eps
      # self.error  = True
      self.early_stop_norm   = lambda x: global_inf_norm(x)

      # Technical options
      self.scaling_refinement_low  = 0.6
      self.scaling_refinement_high = 0.9

      # Approximation options
      self.max_poly_degree  = 128
      self.num_arnoldi_iter = 64

class BamphiInfo:
   def __init__(self):
      self.krylov_done = False

class Points:
   def __init__(self, H):
      is_real  = numpy.all(numpy.isreal(H))
      evals    = numpy.linalg.eigvals(H)
      H_star   = numpy.conjugate(numpy.transpose(H))
      m_norm   = numpy.linalg.norm(H - H_star, 1)
      p_norm   = numpy.linalg.norm(H + H_star, 1)
      max_val  = numpy.max(numpy.abs(evals))

      rho = evals
      if m_norm < max_val * 1e-10:
         rho = numpy.real(evals)
      elif p_norm < max_val * 1e-10:
         rho = numpy.imag(evals) * 1j

      avg = numpy.mean(rho)
      rho -= avg
      inf_norm = numpy.linalg.norm(rho, numpy.inf)
      rho = inf_norm * hermite_ext(rho / inf_norm, is_real)
      rho += avg

      self.rho        = rho
      self.num_points = rho.size
      self.avg        = avg

class FieldOfValues:
   """
   Describes a rectangle that covers the numerical range (field of values) of a matrix H
   (which can be used to bound the eigenvalues of that matrix?). H itself is a Hessenberg matrix
   resulting from the Arnoldi procedure applied to a matrix A.
   """
   def __init__(self, H):
      H_star = numpy.conjugate(numpy.transpose(H))
      H_plus = H + H_star
      eig_val_p = numpy.real(numpy.linalg.eigvals(H_plus))
      min_p = eig_val_p.min() * 0.5
      max_p = eig_val_p.max() * 0.5

      H_minus = H - H_star
      eig_val_m = numpy.imag(numpy.linalg.eigvals(H_minus))
      min_m = eig_val_m.min() * 0.5
      max_m = eig_val_m.max() * 0.5

      self.skew_value = int(min_m == max_m) - int(min_p == max_p)   # TODO Do we really want an exact comparison here?
      self.points     = numpy.array([min_p, max_p, max_p, min_p])
      if min_m != 0.0 or max_m != 0.0:   # TODO Do we want exact comparison with 0.0 here?
         self.points = self.points + [min_m * 1j, min_m * 1j, max_m * 1j, max_m * 1j]

   def get_convex_hull_points(self, num_points):
      """
      Return a set of points randomly chosen along the border of the rectangle described by self.points
      """
      ids     = bamphi_rng.integers(self.points.size, size=num_points)  # [num_points] uniformly distributed integers, between zero (incl) and self.points.size (excl)
      weights = bamphi_rng.random(num_points)

      hull_points = weights * self.points[ids] + (1.0 - weights) * self.points[(ids + 1) % self.points.size]
      return hull_points


def bamphi(t, A, u):
   """
   Python translation of the matlab BAMPHI program, written by Franco Zivkovich, located at https://github.com/francozivcovich/bamphi.git

   Computes f s.t. (in matlab array notation)
      f(:,i) = \sum_{j=0}^{p-1} t(i)^j * \phi_j(t*A) * u( :,j+1 )

   Arguments:
   t  -- Vector of timesteps. Can be complex, for some reason. TODO Why?
   A  -- Operator that takes a (global) vector x and returns the A*x matrix-vector product
   u  -- Set of (global) vectors. Not sure what it does yet
   """

   num_p, num_dofs = u.shape
   new_p = num_p - 1

   # Init stuff
   # Find FOV and Ritz values (what are these?)
   n_u = 1.0
   if new_p > 0:
      # TODO make this work on the cube-sphere
      n_u = max(1, 2.0 ** numpy.ceil( numpy.log2( numpy.sqrt( numpy.linalg.norm( u @ u.T ))))) # u * u^T is pxp. Are we computing the norm of u? If so, might change the name of n_u

   u /= n_u

   times = compute_timesteps(t)

   # Initialize the solution vectors
   f = numpy.zeros((times.size, num_dofs)) # the solution (eventually). That's a set of global vectors
   f[0, :] = u[0, :]

   f_bar = numpy.zeros((times.size, new_p))
   f_bar[0, -1] = 1

   u = numpy.flip(u[1:, :], axis=0)

   options = BamphiOptions()
   info    = BamphiInfo()

   def A_tilde(vec, vec_bar):
      result = A(vec) + vec_bar @ u

      # Left-shift the content of vec_bar, setting the last (previously first) colum to zero
      vec_bar_out = numpy.zeros_like(vec_bar)
      vec_bar_out[:new_p - 1] = vec_bar[1:]

      return result, vec_bar_out

   compute_field_of_values(A, A_tilde, f, new_p, info, options)

   # Init step state
   info.step_state = StepState()
   info.is_real = numpy.all(numpy.isreal(f[0, :]))

   # Analyse timesteps
   # ...


   # Do the work
   for step_id in range(1, times.size):
      step_forward(A, A_tilde, step_id, times, new_p, options, info, f, f_bar)

   # Adjust stuff
   f *= n_u

   return f, info


def step_forward(A, A_tilde, step_id, times, p, options, info, solution, f_bar):
   """
   Step the algorithm forward. Will call itself recursively if its previous step has not been completed yet.
   TOOD: Why don't we just simply perform the steps in order?????

   Arguments:
   A              -- [in]     The system we are trying to solve (operation that computes the Ax product)
   A_tilde        -- [in]     Operator to do [???]
   step_id        -- [in]     The step we are computing
   times          -- [in]     The list of times corresponding to each step
   p              -- [in]     ??? Number of u-vectors
   options        -- [in]     Options for driving the algo
   info           -- [in,out] Information about the course of the algorithm
   solution       -- [in,out] The solution matrix, updated at every time step. Global vectors
   f_bar          -- [in,out] Some variation of the solution (TODO wtf is it?). Size of [# times x p]
   """

   tau = times[step_id] - times[step_id - 1]
   info.step_state.prepare_step(tau, p, options.tolerance, options.max_poly_degree, info.points, info.field_of_values)
   info.is_real = info.is_real and numpy.isreal(tau)

   # Krylov stuff
   previous_sol = solution[step_id - 1, :]
   previous_f_bar = f_bar[step_id - 1, :]
   w, w_bar, step_size, jump_size, scaling_ref_low = krylov_step(A, p, previous_sol, previous_f_bar, info, options)

   # Hermite-Newton step
   w, w_bar = hermite_newton_step(A, A_tilde, w, w_bar, step_size, jump_size, scaling_ref_low, info, options)

   info.is_real = info.is_real and numpy.all(numpy.isreal(w))

   solution[step_id, :] = w
   f_bar[step_id, :]    = w_bar

def krylov_step(A, p, previous_f, previous_f_bar, info, options):
   step = 1
   jump = info.step_state.get_jump()
   jump_limit      = numpy.iinfo(int).max
   scaling_ref_low = options.scaling_refinement_low

   # Early stop (no need to do any work)
   if info.krylov_done:
      return previous_f.copy(), previous_f_bar.copy(), step, jump, scaling_ref_low

   while not info.krylov_done:
      info.step_state.set_jump(jump)
      jump = min(info.step_state.get_s() - step + 1, jump)
      info.step_state.set_ts(jump * info.step_state.get_tau() / info.step_state.get_s())
      jump, jump_limit = info.step_state.update_d(jump)

      w, w_bar, its, e_t, ret = info.krylov.assemble(A, jump, p, info.is_real, info)

      info.krylov_done = True
      step += jump

      if info.step_state.get_s() > 1:
         step, jump, scaling_ref_low, info.krylov_done = refine_scaling(step, jump, jump_limit, e_t, ret, scaling_ref_low, options)
   
   return w, w_bar, step, jump, scaling_ref_low

def hermite_newton_step(A, A_tilde, w, w_bar, step_size, jump_size, scaling_ref_low, info, options):
   while step_size < info.step_state.get_s() + 1:
      # In case we reject the step
      w_prev = w
      w_bar_prev = w_bar

      info.step_state.set_jump(jump_size)
      jump_size = min(jump_size, info.step_state.get_s() - step_size + 1)

      ts = jump_size * info.step_state.get_tau() / info.step_state.get_s()
      info.step_state.set_ts(ts)
      jump_size, jump_limit = info.step_state.update_d(jump_size)

      w, w_bar, its, ret = newton_talezer(jump_size, A, A_tilde, w, w_bar, info.is_real, 0, options, info)

      step_size += jump_size

      if info.step_state.get_s() > 1:
         step_size, jump_size, scaling_ref_low, accept_step =  \
               refine_scaling(step_size, jump_size, jump_limit, its, ret, scaling_ref_low, options)
         if not accept_step:
            print('REJECT')
            w = w_prev
            w_bar = w_bar_prev

   return numpy.real_if_close(w), w_bar

def refine_scaling(step_size, jump_size, jump_limit, e_t, ret, lower, options):
   accept = True
   if ret == 0:
      # Adjust jump and increase scaling_ref_low
      if e_t < lower * options.max_poly_degree:
         jump_size += 1
      elif e_t > options.scaling_refinement_high * options.max_poly_degree and jump_size > 1:
         jump_size -= 1
      lower = min(options.scaling_refinement_low, lower * 1.05)
   elif jump_size > 1:
      # Adjust jump, step and decrease scaling_ref_low
      step_size -= jump_size
      jump_size -= 1
      lower *= 0.5
      accept = False

   jump_size = min(jump_size, jump_limit)

   return step_size, jump_size, lower, accept

def newton_talezer(jump, A, A_tilde, w1, w1_bar, is_real, e_t, options, info):
   
   ret = 1
   
   # Initialize parameters
   real_imag_factor = 0 if is_real else 1j
   m = info.step_state.get_x().shape[0] - 1
   cof_w1 = numpy.zeros(m + 1)
   cof_w2 = numpy.zeros(m + 1)
   c1 = 0
   c2 = None
   tol = info.step_state.get_tol()
   ts  = info.step_state.get_ts()

   # Separate real and imaginary parts. We want to work with real only, as much as possible
   z_real = numpy.real(info.step_state.get_x())
   z_imag = numpy.imag(info.step_state.get_x())
   d_real = numpy.real(info.step_state.get_divided_differences()[jump - 1])
   d_imag = numpy.imag(info.step_state.get_divided_differences()[jump - 1])

   cof_w1[:] = d_real + real_imag_factor * d_imag

   # First p + 1 iterations
   k = 0
   w1     *= cof_w1[k]
   w1_bar *= cof_w1[k]
   p_A     = w1.copy()
   p_A_bar = w1_bar.copy()
   while numpy.linalg.norm(w1_bar) > 1e-16:    # While it's not zero?
      w1, w1_bar = A_tilde(w1, w1_bar)
      w1     *= ts * cof_w1[k + 1] / cof_w1[k]
      w1_bar *= ts * cof_w1[k + 1] / cof_w1[k]
      p_A     += w1
      p_A_bar += w1_bar
      k += 1

   c2 = options.early_stop_norm(w1)

   x_complex = info.step_state.get_x_complex()
   x   = info.step_state.get_x()
   if not is_real or numpy.all(x_complex[:-1] == -1):
      # print(f'Newton!')
      # Everything is either real, or complex and unpaired, so we go with Newton
      for k in range(k, m):
         w1 = (ts * cof_w1[k+1] / cof_w1[k]) * (A(w1) - x[k] * w1)
         c1 = c2
         c2 = options.early_stop_norm(w1) if k >= e_t else None
         if c2 is not None and not numpy.isfinite(c2): break
         p_A += w1

         # Relative error, checking for early termination
         if k >= e_t and c1 + c2 <= tol * options.early_stop_norm(p_A):
            ret = 0 
            break

   else:
      # We have complex data points in conjugated pairs, so we go with Tal-Ezer
      # print(f'Tal-Ezer!')
      cof_w2 = d_real
      k_iter = iter(range(k, m))
      for k in k_iter:
         if x_complex[k + 1] == -1 or not is_real:
            w1 = (ts * cof_w1[k + 1] / cof_w1[k]) * (A(w1) - x[k] * w1)
            c1 = c2
            c2 = options.early_stop_norm(w1) if k >= e_t else None
            if c2 is not None and not numpy.isfinite(c2): break
            p_A = w1 + p_A

         else:
            w2 = (ts * cof_w2[k + 1] / cof_w1[k]) * (A(w1) - z_real[k] * w1)
            if x_complex[k + 1] * x_complex[k] == -1:
               p_A += w2
               c1 = c2
            else:
               p_A += w2 + w1
               c1 = options.early_stop_norm(w1) if k >= e_t else None
            c2 = options.early_stop_norm(w2) if k >= e_t else None

            w1 = (ts * cof_w1[k + 2] / cof_w2[k + 1]) * \
                 (A(w2) - z_real[k + 1] * w2 + (ts * z_imag[k + 1]**2 * cof_w2[k + 1] / cof_w1[k]) * w1)
            if k + 1 < m and (x_complex[k + 3] * x_complex[k + 2] == -1):
               c1 = c2
               c2 = options.early_stop_norm(w1) if k >= e_t else None
               if c2 is not None and not numpy.isfinite(c2): break
               p_A += w1
            _ = next(k_iter) # Skip an iteration

         # Relative error, checking for early termination
         if k >= e_t and c1 + c2 <= tol * options.early_stop_norm(p_A):
            ret = 0
            break
      
   its = k + 1

   return p_A, p_A_bar, its, ret

def compute_field_of_values(A, A_tilde, f, p, info, options):
   """
   Approximate the field of values of A with that of the Hessenberg matrix H computed from the Arnoldi
   procedure applied to A.
   """
   info.krylov = KrylovSystem(A, A_tilde, f, p, options, info)
   H = info.krylov.H[:-1, :-1]
   info.points  = Points(H)
   info.field_of_values = FieldOfValues(H)

def hermite_ext(input, conjugated):
   """
   Not entirely sure, but this seems to sort the input array according to some order
   Hermite-Leja ordering. See 
      REFERENCE: M.Caliari, P.Kandolf and F.Zivcovich, Backward error analysis of
      polynomial approximations for computing the action of the matrix exponential,
      Bit Numer. Math. 58 (2018), 907, https://doi.org/10.1007/s10543-018-0718-9 .
      Section 4
   """
   num_elem = input.size

   id_max = numpy.abs(input).argmax()
   result = numpy.array([input[id_max]])
   remaining = numpy.delete(input, id_max)

   if conjugated and numpy.iscomplex(result[0]):
      elems = numpy.argwhere(remaining == numpy.conj(result[0]))
      if elems.size > 0:
         result = numpy.append(result, remaining[elems[0]])
         remaining = numpy.delete(remaining, elems[0])

   # Create a vector w_x = [1, -sum(stuff), stuff..., prod(stuff), a bunch of zeros]
   # Not sure exactly how to express this. Basically:
   #   n = len(input)
   #   w_x [0] = 1
   #   w_x [1] = -sum[0:n](input_i)
   #   w_x [2] = -sum[0:n]{ -input_i * sum[i:n](input_j) }
   #   w_x [3] = -sum[0:n]((  -input_i * sum[i:n]{ -input_j * sum[j:n](input_k) }  ))
   #   ...
   #   w_x [n:] = 0
   #
   # Since the entries in [input] are supposed to come in conjugate pairs, all values
   # in w_x should be real. There's most likely a more accurate way to compute that
   w_x = numpy.zeros((num_elem + 1), dtype=input.dtype)
   w_x[0] = 1.0
   for i in range(result.size):
      w_x[1:i+2] -= result[i] * w_x[:i+1]
   w_x = numpy.real_if_close(w_x)

   K = numpy.zeros_like(remaining)
   M = numpy.zeros((remaining.size, num_elem), dtype=bool)
   num_iter = 0
   id_max = 0
   while remaining.size > 0:
      K[num_iter]   = remaining[0]
      ids           = numpy.argwhere(remaining == K[num_iter])
      remaining     = numpy.delete(remaining, ids)
      num_in_result = numpy.count_nonzero(result == K[num_iter])
      num_total     = num_in_result + ids.size
      M[num_iter, num_in_result : num_total + 1] = True
      id_max        = max(id_max, num_total)
      num_iter += 1

   for i in range(id_max):
      result = hermite_sort(w_x, result, i, K[M[:num_iter + 1, i]], conjugated, num_elem)

   return result

def hermite_sort(w_x, x, index, values, conjugated, num_elem):
   num_x = x.size
   num_z = values.size
   num_total = num_x + num_z

   tiled = numpy.tile(values, (num_total - index - 1, 1))
   cumulated = numpy.cumprod(tiled, axis=0)
   flipped = numpy.flip(cumulated, axis=0)
   vandermonde = numpy.append(flipped, numpy.ones((1, num_z)), axis=0).T

   ind1 = numpy.arange(num_total, index, -1)
   ind2 = numpy.arange(num_total - index, 0, -1)
   diff_dx = numpy.exp(scipy.special.gammaln(ind1) - scipy.special.gammaln(ind2))    # Differentiation operator for w_x

   while values.size > 0 and num_x < num_elem:
      num_x += 1
      num_z -= 1
      d = diff_dx[:num_x - index] * w_x[:num_x - index]
      id_max = numpy.abs(vandermonde[:, num_z:num_total - index + 1] @ d).argmax()
      x = numpy.append(x, values[id_max])
      values = numpy.delete(values, id_max)
      vandermonde = numpy.delete(vandermonde, id_max, axis=0)

      if conjugated:
         sames, = numpy.where(numpy.abs(values - x[-1].conj()) <= 1e-15)
         if len(sames) > 0:
            # We're trying to transfer a value from the "values" vector into "x", and remove the corresponding row from "vandermonde"
            num_x += 1
            num_z -= 1
            x           = numpy.append(x, values[sames])
            values      = numpy.delete(values, sames)
            vandermonde = numpy.delete(vandermonde, sames, axis=0)

            # Update w_x and move on to next iteration
            w_x[1:num_x + 1] -= 2 * numpy.real(x[-1]) * w_x[:num_x] - numpy.append(0, numpy.abs(x[-2])**2 * w_x[:num_x-1])
            continue

      # Update w_x. It should *not* have already been done in this iteration
      w_x[1:num_x + 1] -= numpy.real_if_close(x[-1]) * w_x[:num_x]

   return x

def compute_timesteps(t):
   """
   Sort the given times and prepend 0 if needed
   """
   if abs(t[0]) > 0.0:
      times = numpy.concatenate(([0.0], t))
   else:
      times = t.copy()

   times = numpy.sort(times)

   if times.size < 2:
      raise ValueError(f'Vector of times is too short! {t}')
   elif times.size == 2:
      previous_steps = numpy.array([0, 1])
   else:
      if (numpy.abs(numpy.imag(times).sum()) > 0.0):
         print(f'TODO: need to implement timesteps analysis')
         raise ValueError(f'Can only deal with real time steps for now')

   return times

def bamphi_with_octave(t, A, u, mat = None):
   from oct2py import octave
   from print_system_matrix import generate_system_matrix

   if mat is None:
      M = generate_system_matrix(A, u)
   else:
      M = mat

   octave.addpath('/home/vma000/site5/bamphi')
   return octave.bamphi_wrapper(t, M, u.T)
