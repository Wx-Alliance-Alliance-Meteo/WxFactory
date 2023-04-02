#!/usr/bin/env python3

""" The GEF model """

import math
from typing import Callable, Optional
import sys

from mpi4py import MPI
import numpy

from common.definitions         import idx_rho, idx_rho_u1, idx_rho_u2, idx_rho_w
from common.parallel            import DistributedWorld
from common.program_options     import Configuration
from geometry                   import Cartesian2D, CubedSphere, DFROperators, Geometry
from init.dcmip                 import dcmip_T11_update_winds, dcmip_T12_update_winds
from init.init_state_vars       import init_state_vars
from integrators                import Integrator, Epi, EpiStiff, Euler1, Imex2, PartRosExp2, Ros2, RosExp2, \
                                       StrangSplitting, Srerk, Tvdrk3
from output.output_manager      import OutputManager
from precondition.multigrid     import Multigrid
from rhs.rhs_selector           import rhs_selector

from time import time

def main(argv) -> int:
   """ This function sets up the infrastructure and performs the time loop of the model. """

   # Read configuration file
   param = Configuration(argv.config, MPI.COMM_WORLD.rank == 0)

   # Set up distributed world
   ptopo = DistributedWorld() if param.grid_type == 'cubed_sphere' else None

   # Create the mesh
   geom = create_geometry(param, ptopo)

   # Build differentiation matrice and boundary correction
   mtrx = DFROperators(geom, param.filter_apply, param.filter_order, param.filter_cutoff)

   # Initialize state variables
   Q, topo, metric = init_state_vars(geom, mtrx, param)

   if (param.expfilter_apply):
      mtrx.make_filter(param.expfilter_strength,param.expfilter_order,param.expfilter_cutoff,geom)

   # Preconditioning
   preconditioner = create_preconditioner(param, ptopo)

   output = OutputManager(param, geom, metric, mtrx, topo)

   # Determine starting step (if not 0)
   Q, starting_step = determine_starting_state(param, output, Q)

   # Get handle to the appropriate RHS functions
   rhs_handle, rhs_implicit, rhs_explicit = rhs_selector(geom, mtrx, metric, topo, ptopo, param)

   # Time stepping
   stepper = create_time_integrator(param, rhs_handle, rhs_implicit, rhs_explicit, preconditioner)
   stepper.output_manager = output

   output.step(Q, starting_step)
   sys.stdout.flush()

   t = param.dt * starting_step
   stepper.sim_time = t
   nb_steps = math.ceil(param.t_end / param.dt) - starting_step

   #start_time = Timer.start
   start_time = time()
   #print("start_time = {}".format(start_time))
   step = starting_step
   while t < param.t_end:
      if t + param.dt > param.t_end:
         param.dt = param.t_end - t
         t = param.t_end
      else:
         t += param.dt

      step += 1

      if MPI.COMM_WORLD.rank == 0: print(f'\nStep {step} of {nb_steps + starting_step}')

      Q = stepper.step(Q, param.dt)
      if (param.expfilter_apply):
         Q = mtrx.apply_filter_3d(Q,geom,metric)

      if MPI.COMM_WORLD.rank == 0: print(f'Elapsed time for step: {stepper.latest_time:.3f} secs')

      # Check whether there are any NaNs in the solution
      check_for_nan(Q)

      # Overwrite winds for some DCMIP tests
      if param.case_number == 11:
         u1_contra, u2_contra, w_wind = dcmip_T11_update_winds(geom, metric, mtrx, param, time=t)
         Q[idx_rho_u1,:,:,:] = Q[idx_rho, :, :, :] * u1_contra
         Q[idx_rho_u2,:,:,:] = Q[idx_rho, :, :, :] * u2_contra
         Q[idx_rho_w,:,:,:]  = Q[idx_rho, :, :, :] * w_wind
      elif param.case_number == 12:
         u1_contra, u2_contra, w_wind = dcmip_T12_update_winds(geom, metric, mtrx, param, time=t)
         Q[idx_rho_u1,:,:,:] = Q[idx_rho, :, :, :] * u1_contra
         Q[idx_rho_u2,:,:,:] = Q[idx_rho, :, :, :] * u2_contra
         Q[idx_rho_w,:,:,:]  = Q[idx_rho, :, :, :] * w_wind

      output.step(Q, step)
      sys.stdout.flush()

   #end_time   = stepper.latest_time
   #print("end_time = {}".format(end_time))
   total_time = time() - start_time

   output.finalize()

   #print the total times to a file
   if (MPI.COMM_WORLD.rank == 0):
      size = MPI.COMM_WORLD.Get_size()
      method      = str(param.time_integrator)
      methodOrtho = str(param.exponential_solver)
      caseNum     = str(param.case_number)
      totaltime_name = "results_tanya/runtime_"+ methodOrtho + "_n" +  str(size) + "_" + str(method) + "_c" + caseNum +".txt"
      with open(totaltime_name, 'a') as gg:
          gg.write('{} \n'.format(total_time))

   return MPI.COMM_WORLD.rank

def create_geometry(param: Configuration, ptopo: Optional[DistributedWorld]) -> Geometry:
   """ Create the appropriate geometry for the given problem """

   if param.grid_type == 'cubed_sphere':
      return CubedSphere(param.nb_elements_horizontal, param.nb_elements_vertical, param.nbsolpts, param.λ0, param.ϕ0,
                         param.α0, param.ztop, ptopo, param)
   if param.grid_type == 'cartesian2d':
      return Cartesian2D((param.x0, param.x1), (param.z0, param.z1), param.nb_elements_horizontal,
                         param.nb_elements_vertical, param.nbsolpts)

   raise ValueError(f'Invalid grid type: {param.grid_type}')

def create_preconditioner(param: Configuration, ptopo: Optional[DistributedWorld]) -> Optional[Multigrid]:
   """ Create the preconditioner required by the given params """
   if param.preconditioner == 'p-mg':
      return Multigrid(param, ptopo, discretization='dg')
   if param.preconditioner == 'fv-mg':
      return Multigrid(param, ptopo, discretization='fv')
   if param.preconditioner == 'fv':
      return Multigrid(param, ptopo, discretization='fv', fv_only=True)
   return None

def determine_starting_state(param: Configuration, output: OutputManager, Q: numpy.ndarray):
   """ Try to load the state for the given starting step and, if successful, swap it with the initial state """
   starting_step = param.starting_step
   if starting_step > 0:
      try:
         starting_state = numpy.load(output.state_file_name(starting_step))
         if starting_state.shape != Q.shape:
            print(f'ERROR reading state vector from file for step {starting_step}. '
                  f'The shape is wrong! ({starting_state.shape}, should be {Q.shape})')
            raise ValueError
         Q = starting_state

         if MPI.COMM_WORLD.rank == 0:
            print(f'Starting simulation from step {starting_step} (rather than 0)')
            if starting_step * param.dt >= param.t_end:
               print(f'WARNING: Won\'t run any steps, since we will stop at step '
                     f'{int(math.ceil(param.t_end / param.dt))}')

      except (FileNotFoundError, ValueError):
         print(f'WARNING: Tried to start from timestep {starting_step}, but unable to read initial state for that step.'
                ' Will start from 0 instead.')
         starting_step = 0

   return Q, starting_step

def create_time_integrator(param: Configuration,
                           rhs_handle: Callable,
                           rhs_implicit: Optional[Callable],
                           rhs_explicit: Optional[Callable],
                           preconditioner: Optional[Multigrid]) \
      -> Integrator:
   """ Create the appropriate time integrator object based on params """

   if param.time_integrator[:9] == 'epi_stiff' and param.time_integrator[9:].isdigit():
      order = int(param.time_integrator[9:])
      if MPI.COMM_WORLD.rank == 0: print(f'Running with EPI_stiff{order}')
      return EpiStiff(param, order, rhs_handle, init_substeps=10)
   if param.time_integrator[:3] == 'epi' and param.time_integrator[3:].isdigit():
      order = int(param.time_integrator[3:])
      if MPI.COMM_WORLD.rank == 0: print(f'Running with EPI{order}')
      return Epi(param, order, rhs_handle, init_substeps=10)
   if param.time_integrator[:5] == 'srerk' and param.time_integrator[5:].isdigit():
      order = int(param.time_integrator[5:])
      if MPI.COMM_WORLD.rank == 0: print(f'Running with SRERK{order}')
      return Srerk(param, order, rhs_handle)
   if param.time_integrator == 'tvdrk3':
      return Tvdrk3(param, rhs_handle)
   if param.time_integrator == 'euler1':
      if MPI.COMM_WORLD.rank == 0:
         print('WARNING: Running with first-order explicit Euler timestepping.')
         print('         This is UNSTABLE and should be used only for debugging.')
      return Euler1(param, rhs_handle)
   if param.time_integrator == 'ros2':
      return Ros2(param, rhs_handle, preconditioner=preconditioner)
   if param.time_integrator == 'imex2':
      return Imex2(param, rhs_explicit, rhs_implicit)
   if param.time_integrator == 'strang_epi2_ros2':
      stepper1 = Epi(param, 2, rhs_explicit)
      stepper2 = Ros2(param, rhs_implicit, preconditioner=preconditioner)
      return StrangSplitting(param, stepper1, stepper2)
   if param.time_integrator == 'strang_ros2_epi2':
      stepper1 = Ros2(param, rhs_implicit, preconditioner=preconditioner)
      stepper2 = Epi(param, 2, rhs_explicit)
      return StrangSplitting(param, stepper1, stepper2)
   if param.time_integrator == 'rosexp2':
      return RosExp2(param, rhs_handle, rhs_implicit, preconditioner=preconditioner)
   if param.time_integrator == 'partrosexp2':
      return PartRosExp2(param, rhs_handle, rhs_implicit, preconditioner=preconditioner)

   raise ValueError(f'Time integration method {param.time_integrator} not supported')

def check_for_nan(Q):
   """ Raise an exception if there are NaNs in the input """
   error_detected = numpy.array([0],dtype=numpy.int32)
   if numpy.any(numpy.isnan(Q)):
      print(f'NaN detected on process {MPI.COMM_WORLD.rank}')
      error_detected[0] = 1
   error_detected_out = numpy.zeros_like(error_detected)
   MPI.COMM_WORLD.Allreduce(error_detected, error_detected_out, MPI.MAX)
   if error_detected_out[0] > 0:
      raise ValueError(f'NaN')

if __name__ == '__main__':

   import argparse
   import cProfile

   parser = argparse.ArgumentParser(description='Solve NWP problems with GEF!')
   parser.add_argument('--profile', action='store_true', help='Produce an execution profile when running')
   parser.add_argument('config', type=str, help='File that contains simulation parameters')

   args = parser.parse_args()

   # Start profiling
   pr = None
   if args.profile:
      pr = cProfile.Profile()
      pr.enable()

   numpy.set_printoptions(suppress=True, linewidth=256)
   rank = main(args)

   if args.profile:
      pr.disable()

      out_file = f'prof_{rank:04d}.out'
      pr.dump_stats(out_file)
