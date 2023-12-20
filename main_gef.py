#!/usr/bin/env python3

""" The GEF model """

import math
from typing import Optional
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
                                       StrangSplitting, Srerk, Tvdrk3, BackwardEuler, CrankNicolson, Bdf2
from output.output_manager      import OutputManager
from output.state               import load_state
from precondition.factorization import Factorization
from precondition.multigrid     import Multigrid
from rhs.rhs_selector           import RhsBundle

def main(argv) -> int:
   """ This function sets up the infrastructure and performs the time loop of the model. """

   # Read configuration file
   param = Configuration(argv.config, MPI.COMM_WORLD.rank == 0)

   # Set up distributed world
   ptopo = DistributedWorld() if param.grid_type == 'cubed_sphere' else None

   adjust_nb_elements(param)

   # Create the mesh
   geom = create_geometry(param, ptopo)

   # Build differentiation matrice and boundary correction
   mtrx = DFROperators(geom, param)

   # Initialize state variables
   Q, topo, metric = init_state_vars(geom, mtrx, param)

   # Preconditioning
   preconditioner = create_preconditioner(param, ptopo, Q)

   output = OutputManager(param, geom, metric, mtrx, topo)

   # Determine starting step (if not 0)
   Q, starting_step = determine_starting_state(param, output, Q)

   # Get handle to the appropriate RHS functions
   rhs = RhsBundle(geom, mtrx, metric, topo, ptopo, param, Q.shape)

   # Time stepping
   stepper = create_time_integrator(param, rhs, preconditioner)
   stepper.output_manager = output

   output.step(Q, starting_step)
   sys.stdout.flush()

   t = param.dt * starting_step
   stepper.sim_time = t
   nb_steps = math.ceil(param.t_end / param.dt) - starting_step

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
      Q = mtrx.apply_filters(Q, geom, metric, param.dt)

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

      if stepper.failure_flag != 0: break

   output.finalize()

   return MPI.COMM_WORLD.rank

def adjust_nb_elements(param: Configuration):
   """ Adjust number of horizontal elements in the parameters so that it corresponds to the number *per processor* """
   if param.grid_type == 'cubed_sphere':
      allowed_pe_counts = [i**2 * 6
                           for i in range(1, param.nb_elements_horizontal // 2 + 1)
                           if (param.nb_elements_horizontal % i) == 0]
      if MPI.COMM_WORLD.size not in allowed_pe_counts:
         raise ValueError(f'Invalid number of processors for this particular problem size. '
                          f'Allowed counts are {allowed_pe_counts}')
      num_pe_per_tile = MPI.COMM_WORLD.size // 6
      num_pe_per_line = int(numpy.sqrt(num_pe_per_tile))
      param.nb_elements_horizontal = param.nb_elements_horizontal_total // num_pe_per_line
      if MPI.COMM_WORLD.rank == 0:
         print(f'Adjusting horizontal number of elements from {param.nb_elements_horizontal_total} (total) '
               f'to {param.nb_elements_horizontal} (per PE)')
         print(f'allowed_pe_counts = {allowed_pe_counts}')

def create_geometry(param: Configuration, ptopo: Optional[DistributedWorld]) -> Geometry:
   """ Create the appropriate geometry for the given problem """

   if param.grid_type == 'cubed_sphere' and ptopo is not None:
      return CubedSphere(param.nb_elements_horizontal, param.nb_elements_vertical, param.nbsolpts, param.λ0, param.ϕ0,
                         param.α0, param.ztop, ptopo, param)
   if param.grid_type == 'cartesian2d':
      return Cartesian2D((param.x0, param.x1), (param.z0, param.z1), param.nb_elements_horizontal,
                         param.nb_elements_vertical, param.nbsolpts, param.nb_elements_relief_layer,
                         param.relief_layer_height)

   raise ValueError(f'Invalid grid type: {param.grid_type}')

def create_preconditioner(param: Configuration, ptopo: Optional[DistributedWorld],
                          Q: numpy.ndarray) -> Optional[Multigrid]:
   """ Create the preconditioner required by the given params """
   if param.preconditioner == 'p-mg':
      return Multigrid(param, ptopo, discretization='dg')
   if param.preconditioner == 'fv-mg':
      return Multigrid(param, ptopo, discretization='fv')
   if param.preconditioner == 'fv':
      return Multigrid(param, ptopo, discretization='fv', fv_only=True)
   if param.preconditioner in ['lu', 'ilu']:
      return Factorization(Q.dtype, Q.shape, param)
   return None

def determine_starting_state(param: Configuration, output: OutputManager, Q: numpy.ndarray):
   """ Try to load the state for the given starting step and, if successful, swap it with the initial state """
   starting_step = param.starting_step
   if starting_step > 0:
      try:
         starting_state, _ = load_state(output.state_file_name(starting_step))
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
                           rhs: RhsBundle,
                           preconditioner: Optional[Multigrid]) \
      -> Integrator:
   """ Create the appropriate time integrator object based on params """

   # --- Exponential time integrators
   if param.time_integrator[:9] == 'epi_stiff' and param.time_integrator[9:].isdigit():
      order = int(param.time_integrator[9:])
      if MPI.COMM_WORLD.rank == 0: print(f'Running with EPI_stiff{order}')
      return EpiStiff(param, order, rhs.full, init_substeps=10)
   if param.time_integrator[:3] == 'epi' and param.time_integrator[3:].isdigit():
      order = int(param.time_integrator[3:])
      if MPI.COMM_WORLD.rank == 0: print(f'Running with EPI{order}')
      return Epi(param, order, rhs.full, init_substeps=10)
   if param.time_integrator[:5] == 'srerk' and param.time_integrator[5:].isdigit():
      order = int(param.time_integrator[5:])
      if MPI.COMM_WORLD.rank == 0: print(f'Running with SRERK{order}')
      return Srerk(param, order, rhs.full)

   # --- Explicit
   if param.time_integrator == 'euler1':
      if MPI.COMM_WORLD.rank == 0:
         print('WARNING: Running with first-order explicit Euler timestepping.')
         print('         This is UNSTABLE and should be used only for debugging.')
      return Euler1(param, rhs.full)
   if param.time_integrator == 'tvdrk3':
      return Tvdrk3(param, rhs.full)

   # --- Rosenbrock
   if param.time_integrator == 'ros2':
      return Ros2(param, rhs.full, preconditioner=preconditioner)

   # --- Rosenbrock - Exponential
   if param.time_integrator == 'rosexp2':
      return RosExp2(param, rhs.full, rhs.full, preconditioner=preconditioner)
   if param.time_integrator == 'partrosexp2':
      return PartRosExp2(param, rhs.full, rhs.implicit, preconditioner=preconditioner)

   # --- Implicit - Explicit
   if param.time_integrator == 'imex2':
      return Imex2(param, rhs.explicit, rhs.implicit)

   # --- Fully implicit
   if param.time_integrator == 'backward_euler':
      return BackwardEuler(param, rhs.full, preconditioner=preconditioner)
   if param.time_integrator == 'bdf2':
      return Bdf2(param, rhs.full, preconditioner=preconditioner)
   if param.time_integrator == 'crank_nicolson':
      return CrankNicolson(param, rhs.full, preconditioner=preconditioner)

   # --- Operator splitting
   if param.time_integrator == 'strang_epi2_ros2':
      stepper1 = Epi(param, 2, rhs.explicit)
      stepper2 = Ros2(param, rhs.implicit, preconditioner=preconditioner)
      return StrangSplitting(param, stepper1, stepper2)
   if param.time_integrator == 'strang_ros2_epi2':
      stepper1 = Ros2(param, rhs.implicit, preconditioner=preconditioner)
      stepper2 = Epi(param, 2, rhs.explicit)
      return StrangSplitting(param, stepper1, stepper2)

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
   import os.path
   import traceback

   args = None

   try:
      parser = argparse.ArgumentParser(description='Solve NWP problems with GEF!')
      parser.add_argument('--profile', action='store_true', help='Produce an execution profile when running')
      parser.add_argument('config', type=str, help='File that contains simulation parameters')
      parser.add_argument('--show-every-crash', action='store_true',
                          help='In case of an exception, show output from alllllll PEs')
      parser.add_argument('--numpy-warn-as-except', action='store_true',
                          help='Raise an exception if there is a numpy warning')

      args = parser.parse_args()

      if not os.path.exists(args.config):
         raise ValueError(f'Config file does not seem valid: {args.config}')

      # Start profiling
      pr = None
      if args.profile:
         pr = cProfile.Profile()
         pr.enable()

      numpy.set_printoptions(suppress=True, linewidth=256)

      if args.numpy_warn_as_except:
         numpy.seterr(all='raise')

      # Run the actual GEF
      rank = main(args)

      if args.profile and pr:
         pr.disable()

         out_file = f'prof_{rank:04d}.out'
         pr.dump_stats(out_file)

   except Exception as e:

      if args and args.show_every_crash:
         traceback.print_exc()
      elif MPI.COMM_WORLD.rank == 0:
         print(f'There was an error while running GEF. Only rank 0 is printing the traceback.')
         traceback.print_exc()
