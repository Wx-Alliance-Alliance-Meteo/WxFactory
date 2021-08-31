import numpy
from mpi4py import MPI
import time

from matrices   import lagrangeEval
from quadrature import gauss_legendre
from timer      import Timer
from definitions import idx_h, idx_hu1, idx_hu2

basis_point_sets = {}

def eval_single_field(field, elem_interp, result=None):

   new_num_points = elem_interp.shape[0]
   old_num_points = elem_interp.shape[1]
   num_elem = int(field.shape[0] // old_num_points)

   # Interpolate (vertically) on entire rows of elements at once
   interp_1 = numpy.empty((num_elem * new_num_points, num_elem * old_num_points), dtype=field.dtype)
   for i in range(num_elem):
      interp_1[i * new_num_points:(i + 1) * new_num_points, :] = \
         elem_interp @ field[i * old_num_points:(i + 1) * old_num_points, :]

   # Interpolate (horizontally) on columns of interpolated elements
   if result is None:
      result = numpy.empty((num_elem * new_num_points, num_elem * new_num_points), dtype=field.dtype)

   for i in range(num_elem):
      result[:, i * new_num_points:(i + 1) * new_num_points] = \
         interp_1[:, i * old_num_points:(i + 1) * old_num_points] @ elem_interp.T

   return result


def eval_field(field, elem_interp, velocity_interp = None):
   result = None
   if field.ndim == 2:
      result = eval_single_field(field, elem_interp)
   elif field.ndim == 3:
      num_fields = field.shape[0]
      if num_fields != 3:
         raise ValueError('Can only interpolate on grids for Shallow Water')
      if velocity_interp is None:
         raise ValueError('Need to pass a velocity interpolation matrix!')
      new_size = field.shape[1] * elem_interp.shape[0] // elem_interp.shape[1]
      result = numpy.empty((num_fields, new_size, new_size), dtype=field.dtype)
      eval_single_field(field[idx_h], elem_interp, result[idx_h])
      eval_single_field(field[idx_hu1], velocity_interp, result[idx_hu1])
      eval_single_field(field[idx_hu2], velocity_interp, result[idx_hu2])
   else:
      raise ValueError(f'We have a problem. ndim = {field.ndim}')

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

def interpolator(origin_type: str, origin_order: int, dest_type: str, dest_order: int, interp_type: str):
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
   elif interp_type == 'bilinear':
      elem_interp = numpy.array([get_linear_weights(origin_points, x) for x in dest_points])
   else:
      raise ValueError('interp_type not one of available interpolation types')

   # Velocity interpolation matrix ([base matrix] * [some factor])
   velocity_interp = elem_interp.copy()
   if origin_type == 'dg' and dest_type == 'fv':
      velocity_interp *= numpy.sqrt(origin_order)
   elif dest_order < origin_order:
      velocity_interp *= numpy.sqrt(dest_order / origin_order)

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
         return eval_field(field, reverse_interp, velocity_reverse_interp)
      else:
         return eval_field(field, elem_interp, velocity_interp)

   return interpolate
