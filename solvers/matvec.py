import math
from typing import Callable, Tuple

import jax
import numpy

class MatvecOp:
   def __init__(self, matvec: Callable[[numpy.ndarray], numpy.ndarray], dtype, shape: Tuple) -> None:
      self.matvec = matvec
      self.dtype = dtype
      self.shape = shape
      self.size = math.prod([i for i in shape])

   def __call__(self, vec: numpy.ndarray) -> numpy.ndarray:
      return self.matvec(vec)

class MatvecOpBasic(MatvecOp):
   def __init__(self, dt: float, Q: numpy.ndarray, rhs_vec: numpy.ndarray, rhs_handle: Callable) -> None:
      super().__init__(
         lambda vec: matvec_fun(vec, dt, Q, rhs_vec, rhs_handle),
         Q.dtype, Q.shape)

def matvec_fun(vec: numpy.ndarray, dt: float, Q: numpy.ndarray, rhs: numpy.ndarray, rhs_handle, method='complex') \
   -> numpy.ndarray:
   if method == 'complex':
      # Complex-step approximation
      epsilon = math.sqrt(numpy.finfo(float).eps)
      Qvec = Q + 1j * epsilon * numpy.reshape(vec, Q.shape)
      jac = dt * (rhs_handle(Qvec) / epsilon).imag
   elif method == 'fd':
      # Finite difference approximation
      epsilon = math.sqrt(numpy.finfo(numpy.float32).eps)
      Qvec = Q + epsilon * numpy.reshape(vec, Q.shape)
      jac = dt * ( rhs_handle(Qvec) - rhs) / epsilon
   elif method == 'ad':
      _, jac_jp = jax.jvp(rhs_handle, (numpy.ravel(Q),), (vec,))
      jac = numpy.array(jac_jp * 5.0)

      # epsilon = math.sqrt(numpy.finfo(numpy.float32).eps)
      # Qvec = Q + epsilon * numpy.reshape(vec, Q.shape)
      # jac_fd = dt * ( rhs_handle(Qvec) - rhs) / epsilon

      # jr = jac.reshape(jac_fd.shape)
      # diff = jac - numpy.ravel(jac_fd)
      # # diff = rhs_jp - numpy.ravel(rhs)
      # diff_norm = numpy.linalg.norm(diff) / numpy.linalg.norm(jac_fd)
      # print(f'diff norm = {diff_norm:.3e}')
      # # print(f'diff = \n{diff}')
      # print(f'jac = \n{jac.reshape(jac_fd.shape)}')
      # print(f'jac_fd = \n{jac_fd}')
      # print(f'jac / jac_fd = \n{jr / jac_fd}')

   else:
      raise ValueError(f'Unknown jacobian method {method}')

   return jac.flatten()

class MatvecOpRat(MatvecOp):
   def __init__(self, dt: float, Q: numpy.ndarray, rhs_vec: numpy.ndarray, rhs_handle: Callable) -> None:
      super().__init__(
         lambda vec: matvec_rat(vec, dt, Q, rhs_vec, rhs_handle),
         Q.dtype, Q.shape)

def matvec_rat(vec: numpy.ndarray, dt: float, Q: numpy.ndarray, rhs: numpy.ndarray, rhs_handle: Callable) -> numpy.ndarray:

   epsilon = math.sqrt(numpy.finfo(numpy.float32).eps)
   Qvec = Q + epsilon * numpy.reshape(vec, Q.shape)
   jac = dt * ( rhs_handle(Qvec) - rhs) / epsilon

   return vec - 0.5 * jac.flatten()
