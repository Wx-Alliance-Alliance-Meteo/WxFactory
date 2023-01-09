#!/usr/bin/env python3

import numpy
import math
from time import time
import mpi4py.MPI

from Common.blockstats        import blockstats
from Common.dcmip             import dcmip_T11_update_winds, dcmip_T12_update_winds
from Common.definitions       import idx_rho, idx_rho_u1, idx_rho_u2, idx_rho_w
from Common.initialize        import initialize_sw, initialize_euler
from Common.parallel          import Distributed_World
from Common.program_options   import Configuration
from Grid.cubed_sphere        import cubed_sphere
from Grid.matrices            import DFR_operators
from Grid.metric              import Metric, Metric_3d_topo
from Precondition.multigrid   import Multigrid
from Rhs.rhs_euler            import rhs_euler
from Rhs.rhs_sw               import rhs_sw
from Stepper.timeIntegrators  import Epi, EpiStiff, SRERK, Tvdrk3, Ros2, Euler1


def main(args) -> int:
   step = 0

   # Set up distributed world
   ptopo = Distributed_World()

   # Read configuration file
   param = Configuration(args.config, ptopo.rank == 0)

   if param.output_freq > 0:
      from output import output_init, output_netcdf, output_finalize

   # Create the mesh
   geom = cubed_sphere(param.nb_elements_horizontal, param.nb_elements_vertical, param.nbsolpts, param.λ0, param.ϕ0, param.α0, param.ztop, ptopo, param)

   # Build differentiation matrice and boundary correction
   mtrx = DFR_operators(geom, param.filter_apply, param.filter_order, param.filter_cutoff)

   # Initialize state variables
   if param.equations == "Euler":
      metric = Metric_3d_topo(geom, mtrx)
      Q, topo = initialize_euler(geom, metric, mtrx, param)
      # Q: dimensions [5,nk,nj,ni], order ρ, u, v, w, θ
      rhs_handle = lambda q: rhs_euler(q, geom, mtrx, metric, ptopo, param.nbsolpts, param.nb_elements_horizontal,
            param.nb_elements_vertical, param.case_number)
   else: # Shallow water
      metric = Metric(geom)
      Q, topo = initialize_sw(geom, metric, mtrx, param)
      rhs_handle = lambda q: rhs_sw(q, geom, mtrx, metric, topo, ptopo, param.nbsolpts, param.nb_elements_horizontal)

   if param.output_freq > 0:
      output_init(geom, param)
      output_netcdf(Q, geom, metric, mtrx, topo, step, param)  # store initial conditions

   # Time stepping
   if param.time_integrator.lower()[:9] == 'epi_stiff' and param.time_integrator[9:].isdigit():
      order = int(param.time_integrator[9:])
      if (ptopo.rank == 0): print(f'Running with EPI_stiff{order}')
      stepper = EpiStiff(order, rhs_handle, param.tolerance, param.exponential_solver, jacobian_method=param.jacobian_method, init_substeps=10)
   elif param.time_integrator.lower()[:3] == 'epi' and param.time_integrator[3:].isdigit():
      order = int(param.time_integrator[3:])
      if (ptopo.rank == 0): print(f'Running with EPI{order}')
      stepper = Epi(order, rhs_handle, param.tolerance, param.exponential_solver, jacobian_method=param.jacobian_method, init_substeps=10)
   elif param.time_integrator.lower()[:5] == 'srerk' and param.time_integrator[5:].isdigit():
      order = int(param.time_integrator[5:])
      if (ptopo.rank == 0): print(f'Running with SRERK{order}')
      stepper = SRERK(order, rhs_handle, param.tolerance, param.exponential_solver, jacobian_method=param.jacobian_method)
   elif param.time_integrator.lower() == 'tvdrk3':
      stepper = Tvdrk3(rhs_handle)   
   elif param.time_integrator.lower() == 'euler1':
      stepper = Euler1(rhs_handle)
      if (ptopo.rank == 0): 
         print('WARNING: Running with first-order explicit Euler timestepping.')
         print('         This is UNSTABLE and should be used only for debugging.')
   elif param.time_integrator.lower() == 'ros2':
      preconditioner = None
      if param.use_preconditioner:
         # preconditioner = DG_preconditioner(param, geom, ptopo, mtrx, rhs_sw)
         preconditioner = Multigrid(param, ptopo, param.nbsolpts, rhs_handle)

      stepper = Ros2(rhs_handle, param.tolerance, preconditioner=preconditioner)
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
      if (ptopo.rank == 0): print('\nStep', step, 'of', nb_steps)

      tic = time()
      Q = stepper.step(Q, param.dt)

      time_step = time() - tic
      if (ptopo.rank == 0): print('Elapsed time for step: %0.3f secs' % time_step)

      error_detected = numpy.array([0],dtype=numpy.int32)
      if (numpy.any(numpy.isnan(Q))):
         print('NaN detected on process %d' % ptopo.rank)
         error_detected[0] = 1
      error_detected_out = numpy.zeros_like(error_detected)
      mpi4py.MPI.COMM_WORLD.Allreduce(error_detected,error_detected_out,mpi4py.MPI.MAX)
      if (error_detected_out[0]):
         import sys
         sys.exit(1)

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
            if (ptopo.rank == 0): print('=> Writing dynamic output for step', step)
            output_netcdf(Q, geom, metric, mtrx, topo, step, param)

   if param.output_freq > 0:
      output_finalize()

   return ptopo.rank

if __name__ == '__main__':

   import argparse
   import cProfile

   parser = argparse.ArgumentParser(description='Solve NWP problems with GEF!')
   parser.add_argument('--profile', action='store_true', help='Produce an execution profile when running')
   parser.add_argument('config', type=str, help='File that contains simulation parameters')

   args = parser.parse_args()

   # Start profiling
   if args.profile:
      pr = cProfile.Profile()
      pr.enable()

   numpy.set_printoptions(suppress=True, linewidth=256)
   rank = main(args)

   if args.profile:
      pr.disable()

      out_file = f'prof_{rank:04d}.out'
      pr.dump_stats(out_file)
