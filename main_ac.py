"""
 This is the main file for running the Allen-Cahn PDE.
 u_t = alpha(u_xx + u_yy) - u + u^3
 x,y in [-1,1] with homogeneous Neumann boundary conditions.

 Integrating from t in [0, 0.02] with dt = 0.001; only 
 taking 20 steps. 

 Run with p = [100, 400, 625, 1600, 2500, 10k ] for 
 strong scaling results. 

 Runtime will be saved for each integrator and ortho method.  

"""

import numpy as np
import math
import sys

from mpi4py      import MPI
from stiff_pdes  import JTV, initWorld, rhs_jac_pdefuncs, print_stuff
from integrators import Integrator, epi_for_others
from time        import time 

#1. initialize world
comm  = MPI.COMM_WORLD
world = initWorld.InitWorld(comm, "Neumann", [-1.0, 1.0], 5002)

#2. read in command line arguments
epi_order    = int(sys.argv[1])
ortho_method = str(sys.argv[2])

#3. set up initial condition 
Q = np.zeros(world.oneDsize)

for j in range(0,world.numPointsY):      #y
  ycoord = world.startY + (j * world.dx)

  for k in range(0, world.numPointsX):    #x

    xcoord = world.startX + (k * world.dx)
    idx = j*world.numPointsY + k
 
    Q[idx] = 0.1 + 0.1*math.cos(2.0*math.pi*xcoord)*math.cos(2.0*math.pi*ycoord)

#4. set up problem parameters
#NOTE: in order for the lambda Jacobian mat_vec handle to be consistent
#accross all pdes, coefficients need to be passed for diffusion, adv, react
epsilon = 0.1  #coeff for diffusion
alpha   = 0.0  #coeff for advection
gamma   = 0.0  #coeff for reaction

#5. set up integrator
#access to functions that are inputs for EPI constructor
rhs = rhs_jac_pdefuncs.allencahn_rhs
jtv = rhs_jac_pdefuncs.allencahn_jtv

#lamdba function of rhs, that way when called inside the integrator
#the coefficients and world info don't have to be consistently passed
#as an argument
rhs_handle = lambda u: rhs(u, epsilon, world)

stepper = epi_for_others.Epi_others(epi_order, rhs_handle, jtv, ortho_method, world, [epsilon, alpha, gamma], init_substeps=10)

#6. set up time integration
#possitle dt from paper: 0.5, 0.25, 0.1250, 0.0625, 0.03125
#dt = 0.000625 works!, 0.001 works!
t_start     = 0.0
t_end       = 0.02
dt          = 0.001
total_steps = math.ceil(t_end/dt)

#7. time step
#--follow gef implementation--
step = 0
start_time = time()
while(step < total_steps):

   #check to make sure we are not overstepping
   if (t_start + dt > t_end):
      dt = t_end - t_start
   else:
      t_start += dt

   step += 1 #increment time step counter

   if MPI.COMM_WORLD.rank == 0: print(f'\nStep {step} of {total_steps}')

   Q = stepper.step(Q, dt)

total_time = time() - start_time

#if world.IamRoot:
#   print("Total runtime = {}".format(total_time))
   
#print out stats
if world.IamRoot:
   size        = MPI.COMM_WORLD.Get_size()
   method      = str(epi_order)
   methodOrtho = str(ortho_method)
   totaltime_name = "results_tanya/runtime_"+ methodOrtho + "_n" +  str(size) + "_e" + str(method) + "ac.txt"

"""   
#print final solution 
print("Gather solution")
finalSolQ = MPI.COMM_WORLD.gather(Q, root=0)

if world.IamRoot:

   #1. gather solin 1 vec and print output file
   filename = "epi5_ac_ts160.txt"
   totalOutLap = print_stuff.print_sol(finalSolQ, filename, world)
"""
