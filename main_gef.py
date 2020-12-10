#!/usr/bin/env python3

import math
import os
from time import time

import numpy

from blockstats      import blockstats
from cubed_sphere    import cubed_sphere
from initialize      import initialize_sw
from matrices        import DFR_operators
from metric          import Metric
from output          import output_init, output_netcdf, output_finalize
from parallel        import Distributed_World
from program_options import Configuration
from rhs_sw          import rhs_sw
from rhs_sw_explicit import rhs_sw_explicit
from rhs_sw_implicit import rhs_sw_implicit
from timeIntegrators import Epi, Epirk4s3a, Tvdrk3, Rat2, ARK_epi2
from timer           import Timer
from preconditioner  import Preconditioner
from rhs_caller      import RhsCaller, RhsCallerLowRes

def main(args):

   step = 0

   # Read configuration file
   param = Configuration(args.config)

   # Set up distributed world
   ptopo = Distributed_World()

   # Create the mesh
   geom = cubed_sphere(param.nb_elements, param.nbsolpts, param.λ0, param.ϕ0, param.α0, ptopo)

   # Build differentiation matrice and boundary correction
   mtrx = DFR_operators(geom, param)

   # Initialize metric tensor
   metric = Metric(geom)

   # Initialize state variables
   Q, topo = initialize_sw(geom, metric, mtrx, param)

   if param.output_freq > 0:
      output_init(geom, param)
      output_netcdf(Q, geom, metric, mtrx, topo, step, param)  # store initial conditions

   # Time stepping
   rhs_handle = RhsCaller(rhs_sw, geom, mtrx, metric, topo, ptopo, param.nbsolpts, param.nb_elements,
                          param.case_number, use_filter = param.filter_apply)

   if param.time_integrator.lower()[:3] == 'epi' and param.time_integrator[3:].isdigit():
      order = int(param.time_integrator[3:])
      print(f'Running with EPI{order}')
      stepper = Epi(order, rhs_handle, param.tolerance, param.krylov_size, init_substeps=10)
   elif param.time_integrator.lower() == 'epirk4s3a':
      stepper = Epirk4s3a(rhs_handle, param.tolerance, param.krylov_size)
   elif param.time_integrator.lower() == 'tvdrk3':
      stepper = Tvdrk3(rhs_handle)
   elif param.time_integrator.lower() == 'rat2':
      preconditioner = Preconditioner(param, geom, rhs_sw, ptopo, initial_time)
      stepper = Rat2(rhs_handle, param.tolerance, ptopo.rank, preconditioner = preconditioner)
   elif  param.time_integrator.lower() =='epi2/ark':
      rhs_explicit1 = lambda q: rhs_sw_explicit(q, geom, mtrx, metric, topo, ptopo, param.nbsolpts, param.nb_elements, param.case_number, param.filter_apply)
      rhs_implicit1 = lambda q: rhs_sw_implicit(q, geom, mtrx, metric, topo, ptopo, param.nbsolpts, param.nb_elements, param.case_number, param.filter_apply)

      rhs_implicit2 = RhsCallerLowRes(rhs_sw_implicit, geom, mtrx, metric, topo, ptopo, param.nbsolpts,
                                     param.nb_elements, param.case_number, param.filter_apply, param = param)
      rhs_explicit2 = lambda q: rhs_handle(q) - rhs_implicit2(q)

      stepper = ARK_epi2(rhs_handle, rhs_explicit1, rhs_implicit1, rhs_explicit2, rhs_implicit2,
                         param, ptopo.rank)
   else:
      raise ValueError(f'Time integration method {param.time_integrator} not supported')

   if param.stat_freq > 0:
      blockstats(Q, geom, topo, metric, mtrx, param, step)

   t = 0.0
   nb_steps = math.ceil(param.t_end / param.dt)

   step_timer = Timer()
   while t < param.t_end:
      if t + param.dt > param.t_end:
         param.dt = param.t_end - t
         t = param.t_end
      else:
         t += param.dt

      step += 1
      print('\nStep', step, 'of', nb_steps)

      step_timer.start()
      Q = stepper.step(Q, param.dt)
      step_timer.stop()
      print(f'Elapsed time for step {step}: {step_timer.last_time():.3f} secs')

      if param.stat_freq > 0:
         if step % param.stat_freq == 0:
            blockstats(Q, geom, topo, metric, mtrx, param, step)

      # Plot solution
      if param.output_freq > 0:
         if step % param.output_freq == 0:
            print(f'=> Writing dynamic output for step {step}')
            output_netcdf(Q, geom, metric, mtrx, topo, step, param)

   print(f'Times: {step_timer.times}')

   #plot_times(comm_dist_graph, rhs_timers)

   if param.output_freq > 0:
      output_finalize()

   return ptopo.rank

if __name__ == '__main__':

   import argparse
   import cProfile

   parser = argparse.ArgumentParser(description='Solve CFD problems with GEF!')
   parser.add_argument('--profile', action='store_true', help='Produce an execution profile when running')
   parser.add_argument('config', type=str, help='File that contains simulation parameters')

   args = parser.parse_args()

   numpy.set_printoptions(suppress=True, linewidth=256)

   # Start profiling
   if args.profile:
      pr = cProfile.Profile()
      pr.enable()

   rank = main(args)

   # Stop profiling and save results
   if args.profile:
      pr.disable()
      profile_dir = 'profiles'
      if not os.path.exists(profile_dir):
         os.mkdir(profile_dir)
      out_file = os.path.join(profile_dir, f'prof_{rank:04d}.out')
      pr.dump_stats(out_file)
