import math
import numpy

def matvec_fun(vec, dt, Q, rhs_handle):

   # Complex-step approximation
   epsilon = math.sqrt(numpy.finfo(float).eps)
   Qvec = Q + 1j * epsilon * numpy.reshape(vec, Q.shape)
   jac = dt * (rhs_handle(Qvec) / epsilon).imag

   return jac.flatten()

def matvec_rat(vec, dt, Q, rhs_handle):

   # Complex-step approximation
   epsilon = math.sqrt(numpy.finfo(float).eps)
   Qvec = Q + 1j * epsilon * numpy.reshape(vec, Q.shape)
   jac = (rhs_handle(Qvec) / epsilon).imag

   return vec/dt - 0.5 * jac.flatten()
