import numpy
from mpi4py import MPI
from time   import time

from matrices   import lagrangeEval
from quadrature import gauss_legendre
from timer      import Timer
from definitions import idx_h, idx_hu1, idx_hu2, idx_rho, idx_rho_u1, idx_rho_u2, idx_rho_w, idx_rho_theta

basis_point_sets = {}

def eval_field_2d(field, elem_interp, result=None):

   new_num_points = elem_interp.shape[0]
   old_num_points = elem_interp.shape[1]
   num_elem = int(field.shape[0] // old_num_points)

   if result is None:
      result = numpy.empty((num_elem * new_num_points, num_elem * new_num_points), dtype=field.dtype)

   t0 = time()
   # Interpolate (vertically) on entire rows of elements at once
   interp_1 = numpy.empty((num_elem * new_num_points, num_elem * old_num_points), dtype=field.dtype)
   for i in range(num_elem):
      interp_1[i * new_num_points:(i + 1) * new_num_points, :] = \
         elem_interp @ field[i * old_num_points:(i + 1) * old_num_points, :]
      # print(f'shapes: elem interp {elem_interp.shape}, field {field[i * old_num_points:(i + 1) * old_num_points, :].shape}')

   # Interpolate (horizontally) on columns of interpolated elements
   for i in range(num_elem):
      result[:, i * new_num_points:(i + 1) * new_num_points] = \
         interp_1[:, i * old_num_points:(i + 1) * old_num_points] @ elem_interp.T
   t1 = time()

   return result, t1 - t0

def eval_field_2d_new(field, elem_interp, result=None):

   new_num_points = elem_interp.shape[0]
   old_num_points = elem_interp.shape[1]
   num_elem = int(field.shape[0] // old_num_points)

   if result is None:
      result = numpy.empty((num_elem * new_num_points, num_elem * new_num_points), dtype=field.dtype)

   t0 = time()
   for i in range(num_elem):
      i_start_dest = i * new_num_points
      i_end_dest   = (i+1) * new_num_points
      i_start_src = i * old_num_points
      i_end_src   = (i+1) * old_num_points

      for j in range(num_elem):
         j_start_dest = j * new_num_points
         j_end_dest   = (j+1) * new_num_points
         j_start_src = j * old_num_points
         j_end_src   = (j+1) * old_num_points
         result[i_start_dest:i_end_dest, j_start_dest:j_end_dest] = elem_interp @ field[i_start_src:i_end_src, j_start_src:j_end_src] @ elem_interp.T

   t1 = time()

   return result, t1 - t0

def eval_field_2d_alt(field, elem_interp, result=None):
   new_num_points = elem_interp.shape[0]
   old_num_points = elem_interp.shape[1]
   num_elem = int(field.shape[0] // old_num_points)

   new_elem_shape = (new_num_points, new_num_points)

   num_pts = old_num_points * old_num_points

   field_vec = numpy.empty((field.size//num_pts, num_pts))
   for i in range(num_elem):
      for j in range(num_elem):
         # start = i * num_elem * num_pts + j * num_pts
         # end   = i * num_elem * num_pts + (j+1) * num_pts
         elem_id = i * num_elem + j

         start_i = i * old_num_points
         end_i   = (i+1) * old_num_points
         start_j = j * old_num_points
         end_j   = (j+1) * old_num_points

         field_vec[elem_id, :] = field[start_i:end_i, start_j:end_j].ravel()

   operator = numpy.empty((new_num_points*new_num_points, old_num_points*old_num_points), dtype=elem_interp.dtype)
   for i in range(new_num_points):
      start_i = i * new_num_points
      end_i   = (i+1) * new_num_points
      for j in range(old_num_points):
         start_j = j * old_num_points
         end_j   = (j+1) * old_num_points
         operator[start_i:end_i, start_j:end_j] = elem_interp * elem_interp[i, j]

   if result is None:
      result = numpy.empty((num_elem * new_num_points, num_elem * new_num_points), dtype=field.dtype)

   t0 = time()
   # size = operator.shape[0] ** 2
   # print(f'shapes: operator {operator.shape}, field_vec {field_vec.reshape(size, field_vec.shape[1]//size).shape}')
   # tmp_result = operator @ field_vec.reshape(size, field_vec.shape[1]//size)
   tmp_result = field_vec @ operator.T
   t1 = time()

   num_pts_new = new_num_points * new_num_points
   for i in range(num_elem):
      for j in range(num_elem):
         start = i * num_elem * num_pts_new + j * num_pts_new
         end   = i * num_elem * num_pts_new + (j+1) * num_pts_new
         elem_id  =i * num_elem + j

         start_i = i * new_num_points
         end_i   = (i+1) * new_num_points
         start_j = j * new_num_points
         end_j   = (j+1) * new_num_points

         result[start_i:end_i, start_j:end_j] = tmp_result[elem_id, :].reshape(new_elem_shape)

   return result, t1 - t0


def eval_field_3d(field, elem_interp, result=None):
   new_num_points = elem_interp.shape[0]
   old_num_points = elem_interp.shape[1]
   num_elem_vert = int(field.shape[0] // old_num_points)
   num_elem_horiz = int(field.shape[1] // old_num_points)

   interp_1 = numpy.empty((num_elem_vert * old_num_points, num_elem_horiz * new_num_points, num_elem_horiz * old_num_points), dtype=field.dtype)
   interp_2 = numpy.empty((num_elem_vert * old_num_points, num_elem_horiz * new_num_points, num_elem_horiz * new_num_points), dtype=field.dtype)
   if result is None:
      result = numpy.empty((num_elem_vert * new_num_points, num_elem_horiz * new_num_points, num_elem_horiz * new_num_points), dtype=field.dtype)

   t0 = time()

   # Interpolate along second dimension (should be the fastest)
   for j in range(num_elem_horiz):
      j_start_dst =  j    * new_num_points
      j_end_dst   = (j+1) * new_num_points
      j_start_src =  j    * old_num_points
      j_end_src   = (j+1) * old_num_points
      interp_1[:, j_start_dst:j_end_dst, :] = elem_interp @ field[:, j_start_src:j_end_src, :]

   # Interpolate along 3rd dimension
   for k in range(num_elem_horiz):
      k_start_dst =  k    * new_num_points
      k_end_dst   = (k+1) * new_num_points
      k_start_src =  k    * old_num_points
      k_end_src   = (k+1) * old_num_points
      interp_2[:, :, k_start_dst:k_end_dst] = interp_1[:, :, k_start_src:k_end_src] @ elem_interp.T

   # Interpolate along 1st dimension (possibly slowest? cause we need to go by hand along one of the other dimensions)
   for i in range(num_elem_vert):
      for j in range(num_elem_horiz * new_num_points):
         i_start_src =  i    * old_num_points
         i_end_src   = (i+1) * old_num_points
         i_start_dst =  i    * new_num_points
         i_end_dst   = (i+1) * new_num_points
         result[i_start_dst:i_end_dst, j, :] = elem_interp @ interp_2[i_start_src:i_end_src, j, :]

   t1 = time()

   return result, t1 - t0


def eval_fields(fields, elem_interp, velocity_interp, method):
   result = None
   if fields.ndim == 3:
      num_fields = fields.shape[0]
      if num_fields != 3:
         raise ValueError('Can only interpolate on three 2D grids for Shallow Water (we assume shallow water, because 2D)')
      new_size = fields.shape[1] * elem_interp.shape[0] // elem_interp.shape[1]
      result = numpy.empty((num_fields, new_size, new_size), dtype=fields.dtype)
      
      if method == 2:
         _, t0 = eval_field_2d_alt(fields[idx_h], elem_interp, result[idx_h])
         _, t1 = eval_field_2d_alt(fields[idx_hu1], velocity_interp, result[idx_hu1])
         _, t2 = eval_field_2d_alt(fields[idx_hu2], velocity_interp, result[idx_hu2])
         total_t = t0 + t1 + t2
      elif method == 1:
         _, t0 = eval_field_2d_new(fields[idx_h], elem_interp, result[idx_h])
         _, t1 = eval_field_2d_new(fields[idx_hu1], velocity_interp, result[idx_hu1])
         _, t2 = eval_field_2d_new(fields[idx_hu2], velocity_interp, result[idx_hu2])
         total_t = t0 + t1 + t2
      else:
         _, t0 = eval_field_2d(fields[idx_h], elem_interp, result[idx_h])
         _, t1 = eval_field_2d(fields[idx_hu1], velocity_interp, result[idx_hu1])
         _, t2 = eval_field_2d(fields[idx_hu2], velocity_interp, result[idx_hu2])
         total_t = t0 + t1 + t2

      # print(f'Time: {total_t:7.4f}')
   elif fields.ndim == 4:
      num_fields     = fields.shape[0]
      new_size_vert  = fields.shape[1] * elem_interp.shape[0] // elem_interp.shape[1]
      new_size_horiz = fields.shape[2] * elem_interp.shape[0] // elem_interp.shape[1]
      result = numpy.empty((num_fields, new_size_vert, new_size_horiz, new_size_horiz), dtype=fields.dtype)
      total_t = 0.0
      for id in [idx_rho, idx_rho_theta, idx_rho_w]:
         _, t = eval_field_3d(fields[id], elem_interp, result[id])
         total_t += t
      for id in [idx_rho_u1, idx_rho_u2]:
         _, t = eval_field_3d(fields[id], velocity_interp, result[id])
         total_t += t
      idx_first_tracer = 5
      for id in range(idx_first_tracer, num_fields):
         _, t = eval_field_3d(fields[id], elem_interp, result[id])
         total_t += t

      # print(f'Time: {total_t:7.4f}')
   else:
      raise ValueError(f'We have a problem. ndim = {fields.ndim}')

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
    points, _ = gauss_legendre(order)

    def L(x):
        return numpy.prod(
           [(x - points[i]) / (points[index] - points[i])
            for i in range(order) if i != index],
           axis=0)

    return L


def compute_dg_to_fv_small_projection(dg_order, fv_order, quad_order=1):
    width = 2.0 / fv_order
    points, quad_weights = gauss_legendre(quad_order)
    result = []

    lagranges = [lagrange_poly(i, dg_order) for i in range(dg_order)]
    quad_weights *= width / 2.0 * (fv_order / 2.0)**0.5

    def compute_row(index):
        x0 = -1.0 + width * index
        x = x0 + width * (points + 1.0) / 2.0
        return [l(x) @ quad_weights for l in lagranges]

    result = numpy.array([compute_row(i) for i in range(fv_order)])

    return result


def get_basis_points(type: str, order: int):
   if type == 'dg':
      points, _ = gauss_legendre(order)
   elif type == 'fv':
      pts = numpy.linspace(-1.0, 1.0, order + 1)
      points = (pts[:-1] + pts[1:]) / 2.0
   else:
      raise ValueError('Unsupported grid type')
   return points

def interpolator(origin_type: str, origin_order: int, dest_type: str, dest_order: int, interp_type: str, ndim, method=0):
   origin_points = get_basis_points(origin_type, origin_order)
   dest_points = get_basis_points(dest_type, dest_order)

   # Base interpolation matrix
   elem_interp = None
   if interp_type == 'lagrange':
      elem_interp    = numpy.array([lagrangeEval(origin_points, x) for x in dest_points])
      print('Doing Lagrange!')
   elif interp_type == 'l2-norm':
      elem_interp = compute_dg_to_fv_small_projection(origin_order, dest_order, quad_order=3)
      print('Doing L2 norm!')
   elif interp_type in ['bilinear', 'trilinear']:
      elem_interp = numpy.array([get_linear_weights(origin_points, x) for x in dest_points])
   else:
      raise ValueError('interp_type not one of available interpolation types')

   # Velocity interpolation matrix ([base matrix] * [some factor])
   velocity_interp = elem_interp.copy()
   if origin_type == 'dg' and dest_type == 'fv':
      # velocity_interp *= numpy.sqrt(origin_order)
      velocity_interp *= origin_order ** (1./ndim)
   elif dest_order < origin_order:
      # velocity_interp *= numpy.sqrt(dest_order / origin_order)
      velocity_interp *= (dest_order / origin_order) ** (1./ndim)

   # Invert interpolation matrices to get the inverse operation
   if dest_order == origin_order:
      reverse_interp = numpy.linalg.inv(elem_interp)
      velocity_reverse_interp = numpy.linalg.inv(velocity_interp)
   else:
      reverse_interp = numpy.linalg.pinv(elem_interp)
      velocity_reverse_interp = numpy.linalg.pinv(velocity_interp)

   # print(f'interpolator: origin type/order: {origin_type}/{origin_order}, dest type/order: {dest_type}/{dest_order}')
   print(f'elem_interp:\n{elem_interp}')
   print(f'vel interp:\n{velocity_interp}')
   # print(f'reverse:\n{reverse_interp}')

   # Function that applies the specified operations on a field
   def interpolate(field, reverse=False):
      if reverse:
         return eval_fields(field, reverse_interp, velocity_reverse_interp, method)
      else:
         return eval_fields(field, elem_interp, velocity_interp, method)

   return interpolate
