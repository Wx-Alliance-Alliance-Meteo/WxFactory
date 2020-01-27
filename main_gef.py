#!/usr/bin/env python3

import sys
import math
import mpi4py.MPI
import numpy
import time

from blockstats   import blockstats
from constants    import *
from cubed_sphere import cubed_sphere
from graphx       import *
from initialize   import initialize
from kiops        import kiops
from matrices     import set_operators
from matvec       import matvec_fun
from metric       import build_metric
from parameters   import get_parameters
from rhs_fun      import rhs_fun

def main():
   if len(sys.argv) == 1:
      cfg_file = 'config/gef_settings.ini'
   else:
      cfg_file = sys.argv[1]

   # Read configuration file
   param = get_parameters(cfg_file)

   # Set up distributed world
   communicator = mpi4py.MPI.COMM_WORLD

   if communicator.Get_size() % 6 != 0:
      raise Exception('Number of processes should be a multiple of 6 ...')

   # TODO : changer ceci quand on aura plus qu'un PE par face
   my_cube_face = communicator.rank

   # Create the mesh
   geom = cubed_sphere(param.nb_elements, param.degree, my_cube_face)

   # Build differentiation matrices and DFR boundary correction
   mtrx = set_operators(geom)

   # Initialize metric tensor
   metric = build_metric(geom)

   # Initialize state variables
   Q, hsurf = initialize(geom, param.case_number, param.α)

#   if param.plot_freq  > 0:
#      plot_sphere(geom)
#      plot_field(geom, geom.lon)
#      plot_field(geom, geom.lat)
#      plot_field(geom, Q[:,:,0] + hsurf)

   # Time stepping
   t           = 0.0
   step        = 0
   krylov_size = param.krylov_size

   rhs_handle = lambda q: rhs_fun(q, geom, mtrx, metric, param.degree+1, param.nb_elements, param.nb_elements, param.α)

   nb_steps = math.ceil(param.t_end / param.dt)

   blockstats(Q, step)

   while t < param.t_end:

      if t + param.dt > param.t_end:
         param.dt = param.t_end - t
         t = param.t_end
      else:
         t += param.dt

      step += 1
      print('\nStep',  step, 'of', nb_steps)

      if (param.time_integrator).lower() == 'epi2' or ( (param.time_integrator).lower() == 'epi3' and step == 1 ):

         # Using EPI2 time integration
         tic = time.time()

         matvec_handle = lambda v: matvec_fun(v, param.dt, Q, rhs_handle)

         rhs = rhs_handle(Q)

         # We only need the second phi function
         vec = numpy.column_stack((numpy.zeros(len(rhs.flatten())), rhs.flatten()))

         phiv, stats = kiops([1], matvec_handle, vec, tol=param.tolerance, m_init=krylov_size, mmin=14, mmax=64, task1=False)

         print('KIOPS converged at iteration %d to a solution with local error %e' % (stats[2], stats[4]))

         krylov_size = math.floor(0.7 * stats[5] + 0.3 * krylov_size)

         if (param.time_integrator).lower() == 'epi3':
            # Save values for next timestep with EPI3
            previous_Q   = Q
            previous_rhs = rhs

         # Update solution
         Q = Q + numpy.reshape(phiv, Q.shape) * param.dt

         time_epi2 = time.time() - tic
         print('Elapsed time for EPI2: %0.3f secs' % time_epi2)

      elif (param.time_integrator).lower() == "epi3" and step > 1:

         # Using EPI3 time integration
         tic = time.time()

         matvec_handle = lambda v: matvec_fun(v, param.dt, Q, rhs_handle)

         rhs = rhs_handle(Q)

         J_deltaQ = matvec_fun(previous_Q - Q, 1., Q, rhs_handle)

         residual = (previous_rhs - rhs) - numpy.reshape(J_deltaQ, Q.shape)

         # We need the second and third phi functions (φ_1, φ_2)
         vec = numpy.column_stack((numpy.zeros(len(rhs.flatten())), rhs.flatten(), 2.0/3.0 * residual.flatten()))

         phiv, stats = kiops([1], matvec_handle, vec, tol=param.tolerance, m_init=krylov_size, mmin=14, mmax=64, task1=False)

         print('KIOPS converged at iteration %d to a solution with local error %e' % (stats[2], stats[4]))

         krylov_size = math.floor(0.7 * stats[5] + 0.3 * krylov_size)

         # Save values for the next timestep
         previous_Q   = Q
         previous_rhs = rhs

         # Update solution
         Q = Q + numpy.reshape(phiv, Q.shape) * param.dt

         time_epi3 = time.time() - tic
         print('Elapsed time for EPI3: %0.3f secs' % time_epi3)

      if step % param.stat_freq == 0:
         blockstats(Q, step)

      # Plot solution
      if step % param.plot_freq == 0:
         print('TODO : plot solution')

      communicator.Disconnect()

if __name__ == '__main__':
   numpy.set_printoptions(suppress=True, linewidth=256)
   main()
