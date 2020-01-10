#!/usr/bin/env python3

from cubed_sphere import cubed_sphere
from parameters import get_parameters
from plots import *
from matrices import set_operators
from initialize import initialize
#include("matrices.jl")
#include("plot_cubedsphere.jl")
#include("plot_field.jl")

import sys
import numpy

def main():
   if len(sys.argv) == 1:
      cfg_file = 'config/gef_settings.ini'
   else:
      cfg_file = sys.argv[1]

   # Read configuration file
   param = get_parameters(cfg_file)

   # Create the mesh
   geom = cubed_sphere(param.nb_elements, param.degree)

#   plot_cubedsphere(geom)

   # Build differentiation matrices and DFR boundary correction
   mtrx = set_operators(geom)

   # Initialize state variables
   Q = initialize(geom, param.case_number, param.α)

   plot_field(geom, Q[:,:,:,0])

   # Time stepping
   t           = 0.
   step        = 0
   krylov_size = param.krylov_size

#   rhs_handle(q) = rhs_fun(q, grd, mtrx, param.degree+1, param.nb_elements_x, param.nb_elements_z, param.α)
#
#   nb_steps = ceil(Int, param.t_end / param.dt)
#
#   blockstats(Q, step)
#
#   local previous_Q, previous_rhs
#   while t < param.t_end
#
#      if t + param.dt > param.t_end
#         param.dt = param.t_end - t
#         t = param.t_end
#      else
#         t += param.dt
#      end
#
#      step += 1
#      println("\nStep ",  step, " of ", nb_steps, " (", Dates.format(Dates.now(), "yyyymmdd HH:MM:SS")  ,")")
#
#      if lowercase(param.time_integrator) == "epi2" || ( lowercase(param.time_integrator) == "epi3" && step == 1 )
#
         # Using EPI2 time integration
#         tic = time()
#
#         rhs = rhs_handle(Q)
#
         # We only need the second phi function
#         vec = [ zeros(length(rhs[:])) rhs[:] ]
#
#         phiv, stats = kiops(1, v -> matvec_fun(v, param.dt, Q, rhs_handle), vec, tol=param.tolerance, m_init=krylov_size, mmin=14, mmax=64, task1=false)
#
#         @printf("KIOPS converged at iteration %d to a solution with local error %e \n", stats[3], stats[5])
#
#         krylov_size = floor(Int64, (0.7 * stats[6] + 0.3 * krylov_size))
#
#         if lowercase(param.time_integrator) == "epi3"
            # Save values for next timestep with EPI3
#            previous_Q   = Q
#            previous_rhs = rhs
#         end
#
         # Update solution
#         Q = Q .+ reshape(phiv, size(Q)) .* param.dt;
#
#         time_epi2 = time() - tic
#         @printf("Elapsed time for EPI2: %0.3f secs\n",time_epi2)
#
#      elseif lowercase(param.time_integrator) == "epi3" && step > 1
#
         # Using EPI3 time integration
#         tic = time()
#
#         rhs = rhs_handle(Q)
#
#         J_deltaQ = matvec_fun(previous_Q - Q, 1., Q, rhs_handle)
#
#         residual = (previous_rhs - rhs) - reshape(J_deltaQ, size(Q))
#
         # We need the second and third phi functions (φ_1, φ_2)
#         vec = [ zeros(length(rhs[:])) rhs[:] (2/3 * residual[:]) ]
#
#         phiv, stats = kiops(1, v -> matvec_fun(v, param.dt, Q, rhs_handle), vec, tol=param.tolerance, m_init=krylov_size, mmin=14, mmax=64, task1=false)
#
#         @printf("KIOPS converged at iteration %d to a solution with local error %e \n", stats[3], stats[5])
#
#         krylov_size = floor(Int64, (0.7 * stats[6] + 0.3 * krylov_size))
#
         # Save values for the next timestep
#         previous_Q   = Q
#         previous_rhs = rhs
#
         # Update solution
#         Q = Q .+ reshape(phiv, size(Q)) .* param.dt;
#
#         time_epi3 = time() - tic
#         @printf("Elapsed time for EPI3: %0.3f secs\n",time_epi3)
#
#      end
#
#      if (mod(step, param.stat_freq) == 0)
#         blockstats(Q, step)
#      end
#
      # Plot solution
#      if (mod(step, param.plot_freq) == 0)
#         p = contour(grd.X[1,:], grd.Z[:,1], Q[:,:,RHO_THETA]./Q[:,:,RHO], fill=true)
#         display(plot!(p, aspect_ratio=:equal))
#      end
#
#   end

#end; main(ARGS)

if __name__ == '__main__':
   numpy.set_printoptions(suppress=True, linewidth=256)
   main()
