import numpy
from time import time
from typing import Callable

from mpi4py import MPI

from common.program_options import Configuration
from .integrator            import Integrator
from solvers                import fgmres, matvec_rat, SolverInfo

from array import array

try:
   import matlab.engine
   ml = matlab.engine.start_matlab()
   path = ml.genpath('gcro')
   ml.addpath(path, nargout=0)
   print(f'Matlab started')
except:
   print(f'Unable to start matlab!!!')
   raise


def pf3():
   print(f'I am pf3!')

def ros2matvec(x):
   # print(f'ros2matvec!! x = \n{x}')
   result = __ros2matvec(x)
   ml_result = array('d', result)
   # print(f'ml_result = {ml_result}')
   return ml_result

def ros2matvec_complex(x_r, x_i):
   if x_r.ndim == 1:
      result_dual = __ros2matvec(x_r + 1j * x_i)
   elif x_r.ndim == 2:
      result_dual = numpy.empty((x_r.shape[1], x_r.shape[0]), dtype=numpy.cdouble)
      for i in range(x_r.shape[1]):
         result_dual[i] = __ros2matvec(x_r[:, i] + 1j * x_i[:, i])
   else:
      raise ValueError(f'Cannot handle {x_r.ndim} dimensions...')

   # print(f'imag: \n{numpy.imag(result_dual)}')

   return numpy.real(result_dual).copy(), numpy.imag(result_dual).copy()

def make_matvec(Q, dt):

   # from mpi4py import MPI
   from common.parallel        import DistributedWorld
   from common.program_options import Configuration
   from geometry               import DFROperators
   from init.init_state_vars   import init_state_vars
   from main_gef               import create_geometry
   from rhs.rhs_selector       import RhsBundle

   numpy.set_printoptions(precision=2)

   param = Configuration('config/gaussian_bubble.ini', False)
   ptopo = DistributedWorld() if param.grid_type == 'cubed_sphere' else None
   geom = create_geometry(param, ptopo)
   mtrx = DFROperators(geom, param.filter_apply, param.filter_order, param.filter_cutoff)
   Q2, topo, metric = init_state_vars(geom, mtrx, param)
   # rhs_handle, _, _ = rhs_selector(geom, mtrx, metric, topo, ptopo, param)
   rhs_handle = RhsBundle(geom, mtrx, metric, topo, ptopo, param).full

   Q = numpy.array(Q).reshape(Q2.shape)
   # print(f'make_matvec: Q = \n{Q}')
   # print(f'make_matvec: dt = {dt}')

   rhs = rhs_handle(Q)

   global __ros2matvec
   def __ros2matvec(v):
      return matvec_rat(v, dt, Q, rhs, rhs_handle)

   # print(f'Made a new matvec function!')

class Ros2(Integrator):
   def __init__(self, param: Configuration, rhs_handle: Callable, preconditioner=None) -> None:
      super().__init__(param, preconditioner)
      self.rhs_handle     = rhs_handle
      self.tol            = param.tolerance
      self.gmres_restart  = param.gmres_restart

   def __step__(self, Q: numpy.ndarray, dt: float):

      rhs    = self.rhs_handle(Q)
      Q_flat = Q.flatten()

      def A(v):
         return matvec_rat(v, dt, Q, rhs, self.rhs_handle)

      b = A(Q_flat) + rhs.flatten() * dt

      maxiter = 20000 // self.gmres_restart
      if self.preconditioner is not None:
         # maxiter = 200 // self.gmres_restart
         maxiter = 420 // self.gmres_restart
         maxiter = min(2, maxiter)

      t0 = time()
      Qnew, norm_r, norm_b, num_iter, flag, residuals = fgmres(
         A, b, x0=Q_flat, tol=self.tol, restart=self.gmres_restart, maxiter=maxiter, preconditioner=self.preconditioner,
         verbose=self.verbose_solver)
      t1 = time()

      self.solver_info = SolverInfo(flag, t1 - t0, num_iter, residuals)

      if MPI.COMM_WORLD.rank == 0:
         result_type = 'convergence' if flag == 0 else 'stagnation/interruption'
         print(f'FGMRES {result_type} at iteration {num_iter} in {t1 - t0:4.1f} s to a solution with'
               f' relative residual {norm_r/norm_b : .2e}')

      # print(f'Ab = {A(b)}')

      b_ml = matlab.double(b.tolist())
      x0_ml = matlab.double(Q_flat.tolist())
      t2 = time()
      gcrodr_sol, residuals, _, num_matvec, _ = ml.gcrodr(0, b_ml, 3000, 200, x0_ml, dt, self.tol, nargout = 5)
      t3 = time()
      # print(f'matlab call successful!')

      np_sol = numpy.array(gcrodr_sol).T[0]
      diff = numpy.linalg.norm(Qnew.flatten() - np_sol) / numpy.linalg.norm(Qnew.flatten())
      # print(f'Diff: {diff:.4e}')
      # print(f'Qnew: \n{Qnew}')
      # print(f'gcrodr sol: \n{np_sol}')
      # print(f'gcrodr sol: \n{gcrodr_sol}')
      norm_gc = numpy.linalg.norm(b - A(np_sol)) / numpy.linalg.norm(b)
      norm_fg = numpy.linalg.norm(b - A(Qnew)) / numpy.linalg.norm(b)
      print(f'norm GCRODR vs FGMRES: {norm_gc:.3e} vs {norm_fg:.3e}')
      print(f'GCRODR found the solution with {num_matvec} mat-vec products in {t3 - t2:4.1f}s. Diff with FGMRES: {diff:.4e}')


      self.failure_flag = flag

      return numpy.reshape(Qnew, Q.shape)
