#!/usr/bin/env python3

import sys
import math
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
from timer           import Timer, TimerGroup
from preconditioner  import Preconditioner
from rhs_caller      import RhsCaller, RhsCallerLowRes

def main():
   if len(sys.argv) == 1:
      cfg_file = 'config/gef_settings.ini'
   else:
      cfg_file = sys.argv[1]

   step = 0
   initial_time_tmp = time()

   # Read configuration file
   param = Configuration(cfg_file)

   # Set up distributed world
   ptopo = Distributed_World()

   # Give initial time to everyone to synchronize clocks
   initial_time = ptopo.comm_dist_graph.bcast(initial_time_tmp, root=0)

   # Create the mesh
   geom = cubed_sphere(param.nb_elements, param.nbsolpts, param.λ0, param.ϕ0, param.α0, ptopo)

   # Build differentiation matrice and boundary correction
   mtrx = DFR_operators(geom, param)

   # Initialize metric tensor
   metric = Metric(geom)

   # Initialize state variables
   Q, topo = initialize_sw(geom, metric, mtrx, param)

   # Q_interp = interpolate(geom_interp, geom, Q[0,:,:], comm_dist_graph)
   #plot_field(geom, Q[0,:,:])
   # plot_field_pair(geom, Q[0,:,:], geom_interp, Q_interp)
   # exit(0)

   if param.output_freq > 0:
      output_init(geom, param)
      output_netcdf(Q, geom, metric, mtrx, topo, step, param)  # store initial conditions

   # Time stepping
   rhs_timers = TimerGroup(5, initial_time)
   rhs_handle = RhsCaller(rhs_sw, geom, mtrx, metric, topo, ptopo, param.nbsolpts, param.nb_elements,
                          param.case_number, use_filter = param.filter_apply, timers = rhs_timers)

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
                                     param.nb_elements, param.case_number, param.filter_apply,
                                     timers = rhs_timers, param = param)
      rhs_explicit2 = lambda q: rhs_handle(q) - rhs_implicit2(q)

      stepper = ARK_epi2(rhs_handle, rhs_explicit1, rhs_implicit1, rhs_explicit2, rhs_implicit2,
                         param.tolerance, ptopo.rank)
   else:
      raise ValueError(f'Time integration method {param.time_integrator} not supported')

   if param.stat_freq > 0:
      blockstats(Q, geom, topo, metric, mtrx, param, step)

   t = 0.0
   nb_steps = math.ceil(param.t_end / param.dt)

   step_timer = Timer(initial_time)
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
      #print('Elapsed time for step %d: %0.3f secs' % step, time_step)
      print('Elapsed time for step {}: {:.3f} secs'.format(step, step_timer.last_time()))

      if param.stat_freq > 0:
         if step % param.stat_freq == 0:
            blockstats(Q, geom, topo, metric, mtrx, param, step)

      # Plot solution
      if param.output_freq > 0:
         if step % param.output_freq == 0:
            print('=> Writing dynamic output for step', step)
            output_netcdf(Q, geom, metric, mtrx, topo, step, param)

   print('Times: {}'.format(step_timer.times))

   #plot_times(comm_dist_graph, rhs_timers)

   if param.output_freq > 0:
      output_finalize()

   return ptopo.rank

if __name__ == '__main__':

   import cProfile

   numpy.set_printoptions(suppress=True, linewidth=256)
   pr = cProfile.Profile()
   pr.enable()
   rank = main()
   pr.disable()
   pr.dump_stats('prof_{:04d}.out'.format(rank))
