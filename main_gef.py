#!/usr/bin/env python3

import numpy
import math
import os
import sys
from time import time

from blockstats      import blockstats
from cubed_sphere    import cubed_sphere
from dcmip           import dcmip_T11_update_winds, dcmip_T12_update_winds
from definitions     import idx_rho, idx_rho_u1, idx_rho_u2, idx_rho_w
from initialize      import initialize_sw, initialize_euler
from matrices        import DFR_operators
from metric          import Metric
from multigrid       import Multigrid
from output_residual import prepare_output
from parallel        import Distributed_World
from program_options import Configuration
from rhs_euler       import rhs_euler
from rhs_sw          import rhs_sw
from rhs_sw_explicit import rhs_sw_explicit
from rhs_sw_implicit import rhs_sw_implicit
from timeIntegrators import Epi, Epirk4s3a, Tvdrk3, Rat2, ARK_epi2

from gef_mpi import GLOBAL_COMM, split_comm

def main(args) -> int:
   step = 0

   # Read configuration file(s)
   if len(args.config) > 1:
      old_rank = GLOBAL_COMM().rank
      split_comm()
      config_id = old_rank // 6
      if config_id >= len(args.config): return -1
      my_config = os.path.abspath(args.config[config_id])
      my_dir = os.path.dirname(my_config)
      os.chdir(my_dir)
      param = Configuration(my_config)
   else:
      param = Configuration(args.config[0])

   if param.output_freq > 0:
      from output import output_init, output_netcdf, output_finalize

   # Set up distributed world
   ptopo = Distributed_World()

   # Create the mesh
   geom = cubed_sphere(param.nb_elements_horizontal, param.nb_elements_vertical, param.nbsolpts, param.λ0, param.ϕ0, param.α0, param.ztop, ptopo, param)

   # Build differentiation matrice and boundary correction
   mtrx = DFR_operators(geom, param.filter_apply, param.filter_order, param.filter_cutoff)

   # Initialize metric tensor
   metric = Metric(geom)

   # Initialize state variables
   if param.equations == "Euler":
      Q, topo = initialize_euler(geom, metric, mtrx, param)
      rhs_handle = lambda q: rhs_euler(q, geom, mtrx, metric, topo, ptopo, param.nbsolpts, param.nb_elements_horizontal,
            param.nb_elements_vertical, param.case_number)
   else: # Shallow water
      Q, topo = initialize_sw(geom, metric, mtrx, param)
      rhs_handle = lambda q: rhs_sw(q, geom, mtrx, metric, topo, ptopo, param.nbsolpts, param.nb_elements_horizontal)

   # Preconditioning
   preconditioner = None
   if param.preconditioner == 'p-mg':
      preconditioner = Multigrid(param, ptopo, discretization='dg')
   elif param.preconditioner == 'fv-mg':
      preconditioner = Multigrid(param, ptopo, discretization='fv')
   elif param.preconditioner == 'fv':
      preconditioner = Multigrid(param, ptopo, discretization='fv', fv_only=True)

   prepare_output(param)

   if param.output_freq > 0:
      output_init(geom, param)
      output_netcdf(Q, geom, metric, mtrx, topo, step, param)  # store initial conditions

   # Time stepping
   if param.time_integrator.lower()[:3] == 'epi' and param.time_integrator[3:].isdigit():
      order = int(param.time_integrator[3:])
      print(f'Running with EPI{order}')
      stepper = Epi(order, rhs_handle, param.tolerance, param.exponential_solver, jacobian_method=param.jacobian_method, init_substeps=10)
   elif param.time_integrator.lower() == 'epirk4s3a':
      stepper = Epirk4s3a(rhs_handle, param.tolerance, param.krylov_size)
   elif param.time_integrator.lower() == 'tvdrk3':
      stepper = Tvdrk3(rhs_handle)
   elif param.time_integrator.lower() == 'rat2':
      stepper = Rat2(rhs_handle, param.tolerance, preconditioner=preconditioner)
   elif  param.time_integrator.lower() == 'epi2/ark' and param.equations == "shallow_water": # TODO : Euler
      rhs_explicit = lambda q: rhs_sw_explicit(q, geom, mtrx, metric, topo, ptopo, param.nbsolpts, param.nb_elements_horizontal)

      rhs_implicit = lambda q: rhs_sw_implicit(q, geom, mtrx, metric, topo, ptopo, param.nbsolpts, param.nb_elements_horizontal)

      stepper = ARK_epi2(rhs_handle, rhs_explicit, rhs_implicit, param)
   else:
      raise ValueError(f'Time integration method {param.time_integrator} not supported')

   if param.stat_freq > 0:
      blockstats(Q, geom, topo, metric, mtrx, param, step)

   t = 0.0
   nb_steps = math.ceil(param.t_end / param.dt)

   while t < param.t_end:
      if t + param.dt > param.t_end:
         param.dt = param.t_end - t
         t = param.t_end
      else:
         t += param.dt

      step += 1
      print('\nStep', step, 'of', nb_steps)

      tic = time()
      Q = stepper.step(Q, param.dt)
      time_step = time() - tic
      print('Elapsed time for step: %0.3f secs' % time_step)
      sys.stdout.flush()

      # Overwrite winds for some DCMIP tests
      if param.case_number == 11:
         u1_contra, u2_contra, w = dcmip_T11_update_winds(geom, metric, mtrx, param, time=t)
         Q[idx_rho_u1,:,:,:] = Q[idx_rho, :, :, :] * u1_contra
         Q[idx_rho_u2,:,:,:] = Q[idx_rho, :, :, :] * u2_contra
         Q[idx_rho_w,:,:,:]  = Q[idx_rho, :, :, :] * w
      elif param.case_number == 12:
         u1_contra, u2_contra, w = dcmip_T12_update_winds(geom, metric, mtrx, param, time=t)
         Q[idx_rho_u1,:,:,:] = Q[idx_rho, :, :, :] * u1_contra
         Q[idx_rho_u2,:,:,:] = Q[idx_rho, :, :, :] * u2_contra
         Q[idx_rho_w,:,:,:]  = Q[idx_rho, :, :, :] * w

      if param.stat_freq > 0:
         if step % param.stat_freq == 0:
            blockstats(Q, geom, topo, metric, mtrx, param, step)

      # Plot solution
      if param.output_freq > 0:
         if step % param.output_freq == 0:
            print('=> Writing dynamic output for step', step)
            output_netcdf(Q, geom, metric, mtrx, topo, step, param)

   if param.output_freq > 0:
      output_finalize()

   return ptopo.rank

if __name__ == '__main__':

   import argparse
   import cProfile

   parser = argparse.ArgumentParser(description='Solve NWP problems with GEF!')
   parser.add_argument('--profile', action='store_true', help='Produce an execution profile when running')
   parser.add_argument('config', type=str, nargs='+', help='File that contains simulation parameters')

   args = parser.parse_args()

   # Start profiling
   if args.profile:
      pr = cProfile.Profile()
      pr.enable()

   numpy.set_printoptions(suppress=True, linewidth=256)
   numpy.seterr(invalid='raise')
   rank = main(args)

   if args.profile:
      pr.disable()

      out_file = f'prof_{rank:04d}.out'
      pr.dump_stats(out_file)
