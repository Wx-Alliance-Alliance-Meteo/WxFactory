import numpy
import math

def matvec_fun(vec, dt, Q, rhs_handle):

   # Complex-step approximation
   epsilon = math.sqrt(numpy.finfo(float).eps)
   Qvec = Q + 1j * epsilon * numpy.reshape(vec, Q.shape)
   jac = dt * ( rhs_handle(Qvec) / epsilon ).imag

   return jac.flatten()
