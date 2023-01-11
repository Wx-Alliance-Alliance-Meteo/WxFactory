#!/usr/bin/env python3

import numpy
import math
from time import time
import mpi4py.MPI as MPI

from Common.dcmip             import dcmip_T11_update_winds, dcmip_T12_update_winds
from Common.definitions       import idx_rho, idx_rho_u1, idx_rho_u2, idx_rho_w
from Common.initialize        import initialize_sw, initialize_euler, initialize_cartesian2d
from Common.parallel          import Distributed_World
from Common.program_options   import Configuration
from Grid.cartesian_2d_mesh   import Cartesian2d
from Grid.cubed_sphere        import CubedSphere
from Grid.matrices            import DFR_operators
from Grid.metric              import Metric, Metric_3d_topo
from Output.blockstats        import blockstats
from Output.solver_stats      import prepare_solver_stats
from Precondition.multigrid   import Multigrid
from Rhs.rhs_bubble           import rhs_bubble
from Rhs.rhs_bubble_implicit  import rhs_bubble_implicit
from Rhs.rhs_euler            import rhs_euler
from Rhs.rhs_sw               import rhs_sw
from Stepper.timeIntegrators  import Epi, EpiStiff, SRERK, Tvdrk3, Ros2, Euler1


def main(args) -> int:
   step = 0

   # Read configuration file
   param = Configuration(args.config, MPI.COMM_WORLD.rank == 0)

   def state_file_name(step_number: int):
      return f'{param.output_dir}/state_vector_{MPI.COMM_WORLD.rank:03d}.{step_number:05d}.npy'

   # Set up distributed world
   ptopo = Distributed_World() if param.grid_type == 'cubed_sphere' else None

   # Create the mesh
   geom = None
   if param.grid_type == 'cubed_sphere':
      geom = CubedSphere(param.nb_elements_horizontal, param.nb_elements_vertical, param.nbsolpts, param.λ0, param.ϕ0, param.α0, param.ztop, ptopo, param)
   elif param.grid_type == 'cartesian2d':
      geom = Cartesian2d((param.x0, param.x1), (param.z0, param.z1), param.nb_elements_horizontal, param.nb_elements_vertical, param.nbsolpts)

   # Build differentiation matrice and boundary correction
   mtrx = DFR_operators(geom, param.filter_apply, param.filter_order, param.filter_cutoff)

   # Initialize state variables
   if param.equations == "euler" and param.grid_type == 'cubed_sphere':
      metric = Metric_3d_topo(geom, mtrx)
      Q, topo = initialize_euler(geom, metric, mtrx, param)
      # Q: dimensions [5,nk,nj,ni], order ρ, u, v, w, θ
      rhs_handle = lambda q: rhs_euler(q, geom, mtrx, metric, ptopo, param.nbsolpts, param.nb_elements_horizontal,
            param.nb_elements_vertical, param.case_number)

   elif param.equations == 'euler' and param.grid_type == 'cartesian2d':
      Q = initialize_cartesian2d(geom, param)
      rhs_handle = lambda q: rhs_bubble(q, geom, mtrx, param.nbsolpts, param.nb_elements_horizontal, param.nb_elements_vertical)
      rhs_implicit = lambda q: rhs_bubble_implicit(q, geom, mtrx, param.nbsolpts, param.nb_elements_horizontal, param.nb_elements_vertical)
      rhs_explicit = lambda q: rhs_handle(q) - rhs_implicit(q)

   elif param.equations == "shallow_water":
      metric = Metric(geom)
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

   # Determine starting step (if not 0)
   starting_step = param.starting_step
   if starting_step > 0:
      try:
         Q_tmp = numpy.load(state_file_name(starting_step))
         if Q_tmp.shape != Q.shape:
            print(f'ERROR reading state vector from file for step {starting_step}. The shape is wrong! ({Q_tmp.shape}, should be {Q.shape})')
            raise ValueError
         print(f'Starting simulation from step {starting_step} (rather than 0)')
      except:
         print(f'WARNING: Tried to start from timestep {starting_step}, but unable to read initial state for that step. Will start from 0 instead.')
         starting_step = 0

   step = starting_step

   # Prepare output
   if param.output_freq > 0:
      if param.grid_type == 'cubed_sphere':
         from Output.output_cubesphere import output_init, output_netcdf, output_finalize
         def output_step(Q, geom, step, param):
            output_netcdf(Q, geom, metric, mtrx, topo, step, param)
      elif param.grid_type == 'cartesian2d':
         from Output.output_cartesian import output_init, output_step, output_finalize

      output_init(geom, param)
      output_step(Q, geom, step, param)  # store initial conditions

   if param.store_solver_stats:
      prepare_solver_stats(param)

   # Save initial output
   if param.save_state_freq > 0:
      numpy.save(state_file_name(0), Q)

   # Time stepping
   if param.time_integrator.lower()[:9] == 'epi_stiff' and param.time_integrator[9:].isdigit():
      order = int(param.time_integrator[9:])
      if (MPI.COMM_WORLD.rank == 0): print(f'Running with EPI_stiff{order}')
      stepper = EpiStiff(order, rhs_handle, param.tolerance, param.exponential_solver, jacobian_method=param.jacobian_method, init_substeps=10)
   elif param.time_integrator.lower()[:3] == 'epi' and param.time_integrator[3:].isdigit():
      order = int(param.time_integrator[3:])
      if (MPI.COMM_WORLD.rank == 0): print(f'Running with EPI{order}')
      stepper = Epi(order, rhs_handle, param.tolerance, param.exponential_solver, jacobian_method=param.jacobian_method, init_substeps=10)
   elif param.time_integrator.lower()[:5] == 'srerk' and param.time_integrator[5:].isdigit():
      order = int(param.time_integrator[5:])
      if (MPI.COMM_WORLD.rank == 0): print(f'Running with SRERK{order}')
      stepper = SRERK(order, rhs_handle, param.tolerance, param.exponential_solver, jacobian_method=param.jacobian_method)
   elif param.time_integrator.lower() == 'tvdrk3':
      stepper = Tvdrk3(rhs_handle)   
   elif param.time_integrator.lower() == 'euler1':
      stepper = Euler1(rhs_handle)
      if (MPI.COMM_WORLD.rank == 0): 
         print('WARNING: Running with first-order explicit Euler timestepping.')
         print('         This is UNSTABLE and should be used only for debugging.')
   elif param.time_integrator.lower() == 'ros2':
      stepper = Ros2(rhs_handle, param.tolerance, preconditioner=preconditioner)
   else:
      raise ValueError(f'Time integration method {param.time_integrator} not supported')

   if param.stat_freq > 0:
      if param.grid_type == 'cartesian2d':
         param.stat_freq = 0
      else:
         blockstats(Q, geom, topo, metric, mtrx, param, step)

   t = param.dt * starting_step
   nb_steps = math.ceil(param.t_end / param.dt) - starting_step

   while t < param.t_end:
      if t + param.dt > param.t_end:
         param.dt = param.t_end - t
         t = param.t_end
      else:
         t += param.dt

      step += 1
      if (MPI.COMM_WORLD.rank == 0): print('\nStep', step, 'of', nb_steps + starting_step)

      tic = time()
      Q = stepper.step(Q, param.dt)

      time_step = time() - tic
      if (MPI.COMM_WORLD.rank == 0): print('Elapsed time for step: %0.3f secs' % time_step)

      # Check whether there are any NaNs in the solution
      error_detected = numpy.array([0],dtype=numpy.int32)
      if (numpy.any(numpy.isnan(Q))):
         print(f'NaN detected on process {MPI.COMM_WORLD.rank}')
         error_detected[0] = 1
      error_detected_out = numpy.zeros_like(error_detected)
      MPI.COMM_WORLD.Allreduce(error_detected, error_detected_out, MPI.MAX)
      if (error_detected_out[0]):
         raise ValueError(f'NaN')

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

      if param.save_state_freq > 0 and step % param.save_state_freq == 0:
         numpy.save(state_file_name(step), Q)

      if param.stat_freq > 0 and step % param.stat_freq == 0:
         blockstats(Q, geom, topo, metric, mtrx, param, step)

      # Plot solution
      if param.output_freq > 0:
         if step % param.output_freq == 0:
            if (MPI.COMM_WORLD.rank == 0): print(f'=> Writing dynamic output for step {step}')
            output_step(Q, geom, step, param)

   if param.output_freq > 0:
      output_finalize()

   return MPI.COMM_WORLD.rank

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
