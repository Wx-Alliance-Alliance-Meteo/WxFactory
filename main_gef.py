#!/usr/bin/env python3

import sys
import math
import time


import numpy

from blockstats   import blockstats
from definitions  import idx_h, idx_hu1, idx_hu2
from config       import Configuration
from cubed_sphere import cubed_sphere
from initialize   import initialize
from kiops        import kiops
from matrices     import DFR_operators
from matvec       import matvec_fun
from metric       import Metric
from output       import output_init, output_netcdf, output_finalize
from parallel     import create_ptopo
from rhs_sw       import rhs_sw

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
   geom = cubed_sphere(param.nb_elements, param.nbsolpts, my_cube_face)

   # Build differentiation matrice and boundary correction
   mtrx = DFR_operators(geom)

   # Initialize metric tensor
   metric = Metric(geom)

   # Initialize state variables
   Q, topo, h_analytic = initialize(geom, metric, mtrx, param)

   if param.output_freq > 0:
      output_init(geom, param)
      output_netcdf(Q, geom, topo, step, param)  # store initial conditions

   # Time stepping
   t           = 0.0
   krylov_size = param.krylov_size

   rhs_handle = lambda q: rhs_sw(q, geom, mtrx, metric, topo, comm_dist_graph, param.nbsolpts, param.nb_elements, \
         param.α, param.case_number)

   nb_steps = math.ceil(param.t_end / param.dt)

   if param.stat_freq > 0:
      blockstats(Q, step, param.case_number)

   if (param.time_integrator).lower() == "epirk4s3a" or (param.time_integrator).lower() == 'epi4':
      g21 = 1/2
      g31 = 2/3

      alpha21 = 1/2
      alpha31 = 2/3

      alpha32 = 0

      p211 = 1
      p212 = 0
      p213 = 0

      p311 = 1
      p312 = 0
      p313 = 0


      b2p3 = 32
      b2p4 = -144

      b3p3 = -27/2
      b3p4 = 81

      if g21 > g31:
         gCoeffVec = numpy.array([g31, g21])
         KryIndex  = numpy.array([2, 1])
      else:
         gCoeffVec = numpy.array([g21, g31])
         KryIndex = numpy.array([1, 2])

   while t < param.t_end:

      if t + param.dt > param.t_end:
         param.dt = param.t_end - t
         t = param.t_end
      else:
         t += param.dt

      step += 1
      print('\nStep', step, 'of', nb_steps)

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

      elif (param.time_integrator).lower() == "epi4" and step > 2:

         # Using EPI4 time integration
         tic = time.time()

         matvec_handle = lambda v: matvec_fun(v, param.dt, Q, rhs_handle)

         rhs = rhs_handle(Q)

         J_deltaQ1 = matvec_fun(previous_Q - Q, 1., Q, rhs_handle)
         J_deltaQ2 = matvec_fun(prev_previous_Q - Q, 1., Q, rhs_handle)

         r1 = (previous_rhs - rhs)      - numpy.reshape(J_deltaQ1, Q.shape)
         r2 = (prev_previous_rhs - rhs) - numpy.reshape(J_deltaQ2, Q.shape)

         # We need the second and third phi functions (φ_1, φ_3)
         vec = numpy.column_stack((numpy.zeros(len(rhs.flatten())), rhs.flatten(), numpy.zeros(len(rhs.flatten())), 11.0/2.0 * r1.flatten()  - 7.0/8.0 * r2.flatten()  ))

         phiv, stats = kiops([1], matvec_handle, vec, tol=param.tolerance, m_init=krylov_size, mmin=14, mmax=64, task1=False)

         print('KIOPS converged at iteration %d to a solution with local error %e' % (stats[2], stats[4]))

         krylov_size = math.floor(0.7 * stats[5] + 0.3 * krylov_size)

         # Save values for the next timestep
         prev_previous_Q   = previous_Q
         prev_previous_rhs = previous_rhs
         previous_Q        = Q
         previous_rhs      = rhs

         # Update solution
         Q = Q + numpy.reshape(phiv, Q.shape) * param.dt

         time_epi3 = time.time() - tic
         print('Elapsed time for EPI4: %0.3f secs' % time_epi3)


      elif (param.time_integrator).lower() == "tvdrk3":
         # Using a 3th Order TVD-RK time integration
         tic = time.time()

         Q1 =          Q                +           rhs_handle(Q)  * param.dt
         Q2 = 0.75   * Q + 0.25    * Q1 + 0.25    * rhs_handle(Q1) * param.dt
         Q = 1.0/3.0 * Q + 2.0/3.0 * Q2 + 2.0/3.0 * rhs_handle(Q2) * param.dt

         time_tvdrk3 = time.time() - tic
         print('Elapsed time for TVD-RK3: %0.3f secs' % time_tvdrk3)

      elif (param.time_integrator).lower() == "epirk4s3a" or ( (param.time_integrator).lower() == 'epi4' and step <= 2 ):

         # Using a 4th Order 3-stage EPIRK time integration
         tic = time.time()

         rhs = rhs_handle(Q)

         hF = rhs * param.dt
         ni, nj, ne = Q.shape
         zeroVec = numpy.zeros(ni * nj * ne)

         matvec_handle = lambda v: matvec_fun(v, param.dt, Q, rhs_handle)

         # stage 1
         u_mtrx = numpy.column_stack((zeroVec, hF.flatten()))
         phiv, stats = kiops(gCoeffVec, matvec_handle, u_mtrx, tol=param.tolerance, m_init=krylov_size, mmin=14, mmax=64, task1=True)

         U2 = Q + alpha21 * numpy.reshape(phiv[:,0], Q.shape)

         # Calculate residual r(U2)
         mv = numpy.reshape( matvec_handle(U2 - Q), Q.shape)
         hb1 = param.dt * rhs_handle(U2) - hF - mv

         # stage 2
         U3 = Q + alpha31 * numpy.reshape(phiv[:,1], Q.shape)

         # Calculate residual r(U3)
         mv = numpy.reshape( matvec_handle(U3 - Q), Q.shape)
         hb2 = param.dt * rhs_handle(U3) - hF - mv

         # stage 3
         u_mtrx = numpy.column_stack((zeroVec, hF.flatten(), zeroVec, (b2p3*hb1+b3p3*hb2).flatten(), (b2p4*hb1+b3p4*hb2).flatten() ))
         phiv, stats = kiops([1], matvec_handle, u_mtrx, tol=param.tolerance, m_init=krylov_size, mmin=14, mmax=64, task1=False)
         Q = Q + numpy.reshape(phiv, Q.shape)

         if (param.time_integrator).lower() == 'epi4':
            if step == 1:
               prev_previous_Q   = Q
               prev_previous_rhs = rhs
            if step == 2:
               previous_Q   = Q
               previous_rhs = rhs

         time_epirk4s3 = time.time() - tic
         print('Elapsed time for EPIRK4s3A: %0.3f secs' % time_epirk4s3)

      if param.stat_freq > 0:
         if step % param.stat_freq == 0:
            blockstats(Q, step, param.case_number)

      # Plot solution
      if param.output_freq > 0:
         if step % param.output_freq == 0:
            output_netcdf(Q, geom, topo, step, param)

   if param.output_freq > 0:
      output_finalize()


if __name__ == '__main__':
   numpy.set_printoptions(suppress=True, linewidth=256)
   main()
