#!/usr/bin/env python3

import sys
import math
from time import time

import numpy

from blockstats      import blockstats
from config          import Configuration
from cubed_sphere    import cubed_sphere
from initialize      import initialize
from matrices        import DFR_operators
from metric          import Metric
from output          import output_init, output_netcdf, output_finalize
from parallel        import create_ptopo
from rhs_sw          import rhs_sw
from rhs_sw_explicit import rhs_sw_explicit
from rhs_sw_implicit import rhs_sw_implicit
from timeIntegrators import Epi, Epirk4s3a, Tvdrk3, Rat2, ARK_epi2

def main():
   if len(sys.argv) == 1:
      cfg_file = 'config/gef_settings.ini'
   else:
      cfg_file = sys.argv[1]

   step = 0

   # Read configuration file
   param = Configuration(cfg_file)

   # Set up distributed world
   comm_dist_graph, my_cube_face = create_ptopo()

   # Create the mesh
   geom = cubed_sphere(param.nb_elements, param.nbsolpts, param.λ0, param.ϕ0, param.α0, my_cube_face)

   # Build differentiation matrice and boundary correction
   mtrx = DFR_operators(geom)

   # Initialize metric tensor
   metric = Metric(geom)

   # Initialize state variables
   Q, topo = initialize(geom, metric, mtrx, param)

   if param.output_freq > 0:
      output_init(geom, param)
      output_netcdf(Q, geom, metric, mtrx, topo, step, param)  # store initial conditions

   # Time stepping
   rhs_handle = lambda q: rhs_sw(q, geom, mtrx, metric, topo, comm_dist_graph, param.nbsolpts, param.nb_elements, param.case_number)

   if param.time_integrator.lower()[:3] == 'epi' and param.time_integrator[3:].isdigit():
      order = int(param.time_integrator[3:])
      print(f'Running with EPI{order}')
      stepper = Epi(order, rhs_handle, param.tolerance, param.krylov_size, init_substeps=10)
   elif param.time_integrator.lower() == 'epirk4s3a':
      stepper = Epirk4s3a(rhs_handle, param.tolerance, param.krylov_size)
   elif param.time_integrator.lower() == 'tvdrk3':
      stepper = Tvdrk3(rhs_handle)
   elif param.time_integrator.lower() == 'rat2':
      stepper = Rat2(rhs_handle, param.tolerance)
   elif  param.time_integrator.lower() =='epi2/ark':
      rhs_explicit = lambda q: rhs_sw_explicit(q, geom, mtrx, metric, topo, comm_dist_graph, param.nbsolpts, param.nb_elements, param.case_number)

      rhs_implicit = lambda q: rhs_sw_implicit(q, geom, mtrx, metric, topo, comm_dist_graph, param.nbsolpts, param.nb_elements, param.case_number)

      stepper = ARK_epi2(rhs_handle, rhs_explicit, rhs_implicit, param.tolerance)
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

#   import graphx
#   graphx.plot_field(geom, Q[0,:,:] + topo.hsurf)

if __name__ == '__main__':
   numpy.set_printoptions(suppress=True, linewidth=256)
   main()
