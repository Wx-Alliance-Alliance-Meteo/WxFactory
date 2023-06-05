import numpy
from time import time
from typing import Callable

from common.program_options import Configuration
from .integrator            import Integrator
from solvers                import kiops, matvec_fun

class Lawson(Integrator):
   def __init__(self, param: Configuration, rhs_handle: Callable) -> None:
      super().__init__(param, preconditioner=None)
      self.rhs_handle      = rhs_handle
      self.tol             = param.tolerance
      self.jacobian_method = param.jacobian_method
      

   def __step__(self, Q: numpy.ndarray, dt: float):

      rhs = self.rhs_handle(Q)
      def matvec_handle(v): return matvec_fun(v, dt, Q, rhs, self.rhs_handle, self.jacobian_method)
      def F(u): return self.rhs_handle(u).flatten() - matvec_fun(u, 1., Q, rhs, self.rhs_handle, self.jacobian_method)

      print("stage 1")
      k1 = F(Q)

      print("stage 2")
      Q_flat = Q.flatten()
      vec = numpy.zeros((2, Q_flat.size))
      vec[0,:] = Q_flat
      exp1, stats = kiops([0.5, 1], matvec_handle, vec, tol=self.tol, m_init=1, mmin=10, mmax=64, task1=False)
      print(f'KIOPS (1/4) converged at iteration {stats[2]} (using {stats[0]} internal substeps and {stats[1]} rejected expm)' 
            f' to a solution with local error {stats[4]:.2e}')
      
      vec[:,:] = 0.
      vec[0,:] =  k1
      exp2, stats = kiops([0.5, 1], matvec_handle, vec, tol=self.tol, m_init=1, mmin=10, mmax=64, task1=False)
      print(f'KIOPS (2/4) converged at iteration {stats[2]} (using {stats[0]} internal substeps and {stats[1]} rejected expm)' 
            f' to a solution with local error {stats[4]:.2e}')
       
      k2 = F( numpy.reshape(exp1[0,:] + dt/2 * exp2[0,:], Q.shape) )

      print("stage 3")
      k3 = F( numpy.reshape(exp1[0,:] + dt/2.*k2, Q.shape) )

      print("stage 4")
      vec[:,:] = 0.
      vec[0,:] = k3
      exp3, stats = kiops([0.5], matvec_handle, vec, tol=self.tol, m_init=1, mmin=10, mmax=64, task1=False)
      print(f'KIOPS (3/4) converged at iteration {stats[2]} (using {stats[0]} internal substeps and {stats[1]} rejected expm)' 
            f' to a solution with local error {stats[4]:.2e}')

      k4 = F( numpy.reshape(exp1[1,:] + dt * exp3[0,:], Q.shape) )

      print("Update the solution")
      vec[:,:] = 0.
      vec[0,:] = (k2 + k3)
      exp4, stats = kiops([0.5], matvec_handle, vec, tol=self.tol, m_init=1, mmin=10, mmax=64, task1=False)
      print(f'KIOPS (4/4) converged at iteration {stats[2]} (using {stats[0]} internal substeps and {stats[1]} rejected expm)' 
            f' to a solution with local error {stats[4]:.2e}')

      Qnew = exp1[1,:] + dt/6. * ( exp2[1,:] + 2 * exp4[0,:] + k4 )

      return numpy.reshape(Qnew, Q.shape)
