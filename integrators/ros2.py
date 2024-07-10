from time import time
from typing import Callable

import numpy
from mpi4py import MPI

from common.program_options  import Configuration
from solvers                 import fgmres, MatvecOpRat, SolverInfo
from .integrator             import Integrator
from solvers                 import fgmres, gcrot, matvec_rat, SolverInfo
from scripts.eigenvalue_util import gen_matrix

dump_dir = 'DataDump(ShallowWaterBIG)'
save_Ab = True

class Ros2(Integrator):
   Q_flat: numpy.ndarray
   A: MatvecOpRat
   b: numpy.ndarray
   def __init__(self, param: Configuration, rhs_handle: Callable, preconditioner=None) -> None:
      super().__init__(param, preconditioner)
      self.param = param
      self.rhs_handle     = rhs_handle
      self.tol            = param.tolerance
      self.gmres_restart  = param.gmres_restart
      self.linear_solver  = param.linear_solver

   def __prestep__(self, Q: numpy.ndarray, dt: float) -> None:
      rhs = self.rhs_handle(Q)
      self.Q_flat = numpy.ravel(Q)
      self.A = MatvecOpRat(dt, Q, rhs, self.rhs_handle)
      self.b = self.A(self.Q_flat) + numpy.ravel(rhs) * dt
      
      if save_Ab:
         gen_matrix(self.A, jac_file_name=f'/data/users/jupyter-dam724/{dump_dir}/ros2_A_{self.param.sys_iter}_{self.num_completed_steps}')
      
         rhs = MPI.COMM_WORLD.gather(self.b)
         if MPI.COMM_WORLD.rank == 0:
            rhs_file = numpy.hstack(rhs)
            numpy.save(f'/data/users/jupyter-dam724/{dump_dir}/ros2_b_{self.param.sys_iter}_{self.num_completed_steps}', rhs_file)   
         else:
            numpy.save(f'/data/users/jupyter-dam724/{dump_dir}/ros2_b_{self.param.sys_iter}_{self.num_completed_steps}', self.b)  

   def __step__(self, Q: numpy.ndarray, dt: float):

      maxiter = 20000 // self.gmres_restart
      if self.preconditioner is not None:
         maxiter = 400 // self.gmres_restart
        
      if self.linear_solver == 'fgmres':
         t0 = time()
         Qnew, norm_r, norm_b, num_iter, flag, residuals = fgmres(
            self.A, self.b, self.num_completed_steps, self.param.sys_iter, x0=self.Q_flat, tol=self.tol, 
            restart=self.gmres_restart, maxiter=maxiter, preconditioner=self.preconditioner, verbose=self.verbose_solver)
         t1 = time()
         
         if save_Ab:
            rhs = MPI.COMM_WORLD.gather(Qnew)
            if MPI.COMM_WORLD.rank == 0:
               rhs_file = numpy.hstack(rhs)
               numpy.save(f'/data/users/jupyter-dam724/{dump_dir}/ros2_x_{self.num_completed_steps}.npy', rhs_file)  # x saved to disk
            else:
               numpy.save(f'/data/users/jupyter-dam724/{dump_dir}/ros2_x_{self.num_completed_steps}.npy', Qnew)  # x saved to disk
 
         self.solver_info = SolverInfo(flag, t1 - t0, num_iter, residuals)
  
         if MPI.COMM_WORLD.rank == 0:
            result_type = 'convergence' if flag == 0 else 'stagnation/interruption'
            print(f'FGMRES {result_type} at iteration {num_iter} in {t1 - t0:4.3f} s to a solution with'
                  f' relative residual {norm_r/norm_b : .2e}')
      else:
         t0 = time()
         Qnew, local_error, num_iter, flag, residuals = gcrot(self.A, self.b, x0=self.Q_flat, tol=self.tol)
         t1 = time()
         local_error = numpy.linalg.norm(self.b - self.A(Qnew))/numpy.linalg.norm(self.b)

         if flag == 0:
            print(f'GCROT converged at iteration {num_iter} in {t1 - t0:4.1f} s to a solution with'
                  f' relative residual norm {local_error : .2e}')
         else:
            print(f'GCROT stagnation/interruption at iteration {num_iter} in {t1 - t0:4.1f} s, returning a solution with'
                  f' relative local error {local_error: .2e}')


      self.failure_flag = flag

      return numpy.reshape(Qnew, Q.shape)
