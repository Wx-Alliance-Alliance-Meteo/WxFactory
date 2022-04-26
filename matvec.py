import math
import numpy

def matvec_fun(vec, dt, Q, rhs, rhs_handle, method='complex'):
   
   if method == 'complex':
      # Complex-step approximation
      epsilon = math.sqrt(numpy.finfo(float).eps)
      Qvec = Q + 1j * epsilon * numpy.reshape(vec, Q.shape)
      jac = dt * (rhs_handle(Qvec) / epsilon).imag
   else:
      # Finite difference approximation
      epsilon = math.sqrt(numpy.finfo(numpy.float32).eps)
      Qvec = Q + epsilon * numpy.reshape(vec, Q.shape)
      jac = dt * ( rhs_handle(Qvec) - rhs) / epsilon
      

   return jac.flatten()

def matvec_rat(vec, dt, Q, rhs, rhs_handle):

   epsilon = math.sqrt(numpy.finfo(numpy.float32).eps)
   Qvec = Q + epsilon * numpy.reshape(vec, Q.shape)
   jac = dt * ( rhs_handle(Qvec) - rhs) / epsilon

   return vec - 0.5 * jac.flatten()
