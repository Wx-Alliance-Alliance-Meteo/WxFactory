import numpy

from common.definitions import idx_u1, idx_u2
from geometry           import gauss_legendre, lagrange_eval, remesh_operator

basis_point_sets = {}

def eval_single_field_2d(field, elem_interp, result=None):

   new_num_points = elem_interp.shape[0]
   old_num_points = elem_interp.shape[1]

   nx = field.shape[0] // old_num_points
   nz = field.shape[1] // old_num_points

   # Interpolate (vertically) on entire rows of elements at once
   interp_1 = numpy.empty((nx * new_num_points, nz * old_num_points), dtype=field.dtype)
   for i in range(nx):
      interp_1[i * new_num_points:(i + 1) * new_num_points, :] = \
         elem_interp @ field[i * old_num_points:(i + 1) * old_num_points, :]

   # Interpolate (horizontally) on columns of interpolated elements
   if result is None:
      result = numpy.empty((nx * new_num_points, nz * new_num_points), dtype=field.dtype)

   for i in range(nz):
      result[:, i * new_num_points:(i + 1) * new_num_points] = \
         interp_1[:, i * old_num_points:(i + 1) * old_num_points] @ elem_interp.T

   return result


def eval_single_field_3d(field, elem_interp, result=None):
   new_num_points = elem_interp.shape[0]
   old_num_points = elem_interp.shape[1]
   num_elem_vert = int(field.shape[0] // old_num_points)
   num_elem_horiz = int(field.shape[1] // old_num_points)

   interp_1 = numpy.empty((num_elem_vert * old_num_points, num_elem_horiz * new_num_points, num_elem_horiz * old_num_points), dtype=field.dtype)
   interp_2 = numpy.empty((num_elem_vert * old_num_points, num_elem_horiz * new_num_points, num_elem_horiz * new_num_points), dtype=field.dtype)
   if result is None:
      result = numpy.empty((num_elem_vert * new_num_points, num_elem_horiz * new_num_points, num_elem_horiz * new_num_points), dtype=field.dtype)

   # Interpolate along second dimension (should be the fastest) (horizontal)
   for j in range(num_elem_horiz):
      j_start_dst =  j    * new_num_points
      j_end_dst   = (j+1) * new_num_points
      j_start_src =  j    * old_num_points
      j_end_src   = (j+1) * old_num_points
      interp_1[:, j_start_dst:j_end_dst, :] = elem_interp @ field[:, j_start_src:j_end_src, :]

   # Interpolate along 3rd dimension (horizontal)
   for k in range(num_elem_horiz):
      k_start_dst =  k    * new_num_points
      k_end_dst   = (k+1) * new_num_points
      k_start_src =  k    * old_num_points
      k_end_src   = (k+1) * old_num_points
      interp_2[:, :, k_start_dst:k_end_dst] = interp_1[:, :, k_start_src:k_end_src] @ elem_interp.T

   # Interpolate along 1st dimension (possibly slowest? cause we need to go by hand along one of the other dimensions) (vertical)
   for i in range(num_elem_vert):
      for j in range(num_elem_horiz * new_num_points):
         i_start_src =  i    * old_num_points
         i_end_src   = (i+1) * old_num_points
         i_start_dst =  i    * new_num_points
         i_end_dst   = (i+1) * new_num_points
         result[i_start_dst:i_end_dst, j, :] = elem_interp @ interp_2[i_start_src:i_end_src, j, :]

   return result


def get_linear_weights(points, x):
   result = numpy.zeros_like(points)

   if len(result) == 1:
      result[0] = 1.0
      return result

   ix = 0
   while points[ix + 1] < x:
      if ix + 2 >= len(points):
         break
      ix += 1

   x_small = points[ix]
   x_large = points[ix+1]

   alpha = 1.0 - (x - x_small) / (x_large - x_small)

   result       = numpy.zeros_like(points)
   result[ix]   = alpha
   result[ix+1] = 1.0 - alpha

   return result


def lagrange_poly(index, order):
   """Compute the Lagrange basis function_[index] for a polynomial of order [order]."""
   points, _, _ = gauss_legendre(order)

   def L(x):
      """
      Compute the value of the Lagrange basis function_[index] of order [order] at point(s) [x]
      """
      return numpy.prod(
         [(x - points[i]) / (points[index] - points[i])
          for i in range(order) if i != index],
         axis=0)

   return L


def compute_dg_to_fv_small_projection(dg_order, fv_order, quad_order=1):
   width = 2.0 / fv_order
   points, quad_weights, _ = gauss_legendre(quad_order)
   result = []

   lagranges = [lagrange_poly(i, dg_order) for i in range(dg_order)]
   quad_weights *= width / 2.0 * (fv_order / 2.0)**0.5

   def compute_row(index):
      x0 = -1.0 + width * index
      x = x0 + width * (points + 1.0) / 2.0
      return [l(x) @ quad_weights for l in lagranges]

   result = numpy.array([compute_row(i) for i in range(fv_order)])

   return result


def compute_dg_l2_proj(origin_order, dest_order):
   quad_order = max(origin_order, dest_order) + 1
   points_sym, quad_weights = scipy.special.roots_legendre(quad_order)
   points = numpy.array(points_sym)

   L_src  = [lagrange_poly(i, origin_order) for i in range(origin_order)]
   L_dest = [lagrange_poly(i, dest_order) for i in range(dest_order)]

   def inner_prod(f, g):
      f_tmp = f(points)
      g_tmp = g(points)
      m = numpy.array(f_tmp * g_tmp)
      sol = m @ quad_weights
      return sol.evalf()

   mass_matrix = numpy.zeros((dest_order, dest_order))
   proj_matrix = numpy.zeros((dest_order, origin_order))

   for i in range(dest_order):
      mass_matrix[i] = numpy.array([inner_prod(L_dest[i], L) for L in L_dest])
      proj_matrix[i] = numpy.array([inner_prod(L_dest[i], L) for L in L_src])

   return numpy.linalg.inv(mass_matrix) @ proj_matrix


def get_basis_points(basis_type: str, order: int, include_boundary: bool = False) -> numpy.ndarray:
   """Get the basis points of a reference element of a certain order. The domain is [-1, 1]."""
   if basis_type == 'dg':
      points_sym, _, _ = gauss_legendre(order)
      points = numpy.array([p.evalf() for p in points_sym])
      if include_boundary: points = numpy.append(-1.0, numpy.append(points, 1.0))
   elif basis_type == 'fv':
      pts = numpy.linspace(-1.0, 1.0, order + 1)
      points = (pts[:-1] + pts[1:]) / 2.0
      if include_boundary: points = numpy.append(-1.0, numpy.append(points, 1.0))
   else:
      raise ValueError('Unsupported grid type')
   return points


class Interpolator:
   elem_interp: numpy.ndarray
   def __init__(self,
                origin_type: str,
                origin_order: int,
                dest_type: str,
                dest_order: int,
                interp_type: str,
                grid_type: str,
                ndim: int,
                include_boundary: bool = False,
                verbose: bool = False):
      origin_points = get_basis_points(origin_type, origin_order, include_boundary)
      dest_points = get_basis_points(dest_type, dest_order, include_boundary)

      self.origin_type = origin_type
      self.dest_type = dest_type
      self.grid_type = grid_type
      self.ndim = ndim

      # Base interpolation matrix
      # self.elem_interp = None
      self.reverse_interp = None
      if interp_type == 'lagrange':
         self.elem_interp    = numpy.array([lagrange_eval(origin_points, x) for x in dest_points])
         self.reverse_interp = numpy.array([lagrange_eval(dest_points, x) for x in origin_points])
      elif interp_type == 'l2-norm':
         self.elem_interp = compute_dg_to_fv_small_projection(origin_order, dest_order, quad_order=3)
      elif interp_type in ['bilinear', 'trilinear']:
         self.elem_interp = numpy.array([get_linear_weights(origin_points, x) for x in dest_points])
      elif interp_type == 'modal':
         self.elem_interp    = remesh_operator(origin_points, dest_points)
         self.reverse_interp = remesh_operator(dest_points, origin_points)
      else:
         raise ValueError('interp_type not one of available interpolation types')

      # Invert interpolation matrix to get the inverse operation (if needed)
      if self.reverse_interp is None:
         if dest_order == origin_order:
            self.reverse_interp = numpy.linalg.inv(self.elem_interp)
         else:
            self.reverse_interp = numpy.linalg.pinv(self.elem_interp)

      # Velocity interpolation matrix ([base matrix] * [some factor]), if needed
      self.velocity_ids            = []
      self.velocity_interp         = None
      self.velocity_reverse_interp = None
      if self.grid_type == 'cubed_sphere':
         self.velocity_ids = [idx_u1, idx_u2]
         self.velocity_interp = self.elem_interp.copy()
         if origin_type == 'dg' and dest_type == 'fv':
            # velocity_interp *= numpy.sqrt(origin_order)
            self.velocity_interp *= origin_order ** (1./ndim)
         elif dest_order < origin_order:
            # velocity_interp *= numpy.sqrt(dest_order / origin_order)
            self.velocity_interp *= (dest_order / origin_order) ** (1./ndim)

         if dest_order == origin_order:
            self.velocity_reverse_interp = numpy.linalg.inv(self.velocity_interp)
         else:
            self.velocity_reverse_interp = numpy.linalg.pinv(self.velocity_interp)

      # print(f'interpolator: origin type/order: {origin_type}/{origin_order}, dest type/order: {dest_type}/{dest_order}')
      if verbose:
         print(f'elem_interp:\n{self.elem_interp}')
         print(f'reverse:\n{self.reverse_interp}')
         print(f'vel interp:\n{self.velocity_interp}')
         print(f'vel reverse interp:\n{self.velocity_reverse_interp}')

   def __call__(self, fields: numpy.ndarray, reverse: bool = False):

      num_fields = fields.shape[0]

      # Select interpolation operator for each of the fields in the list
      base_interp = self.reverse_interp          if reverse else self.elem_interp
      vel_interp  = self.velocity_reverse_interp if reverse else self.velocity_interp
      interp_ops  = [vel_interp if i in self.velocity_ids else base_interp for i in range(num_fields)]

      # Select interpolation method (based on problem dimension) and create result array
      if self.ndim == 2:
         eval_fct = eval_single_field_2d
         new_size_i = fields.shape[1] * base_interp.shape[0] // base_interp.shape[1]
         new_size_j = fields.shape[2] * base_interp.shape[0] // base_interp.shape[1]
         result = numpy.empty((num_fields, new_size_i, new_size_j), dtype=fields.dtype)
      elif self.ndim == 3:
         eval_fct = eval_single_field_3d
         new_size_vert  = fields.shape[1] * base_interp.shape[0] // base_interp.shape[1]
         new_size_horiz = fields.shape[2] * base_interp.shape[0] // base_interp.shape[1]
         result = numpy.empty((num_fields, new_size_vert, new_size_horiz, new_size_horiz), dtype=fields.dtype)
      else:
         raise ValueError(f'We cannot deal with ndim = {self.ndim}')

      o_type = self.origin_type if not reverse else self.dest_type
      d_type = self.dest_type   if not reverse else self.origin_type
      # print(f'ndim: {self.ndim}, shape: {o_type} {fields.shape} -> {d_type} {result.shape} (interp shape: {base_interp.T.shape})')

      # Perform the interpolation for each field
      for i in range(num_fields):
         eval_fct(fields[i], interp_ops[i], result[i])

      return result
