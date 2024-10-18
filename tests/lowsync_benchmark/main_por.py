"""
 This is the main file for running the Porous Medium Eqn
 u_t = alpha(u_x + u_y) + (u^2)_xx + (u^2)_yy
 x,y in [0,1] with periodic boundary conditions. 

 Integrating from t in [0, tf], with dt = .

 Run with p = [625, 1600, 2500, 6400, 10000] for
 strong scaling results. 

 Runtime will be saves for each integrator and ortho method.

"""

import numpy as np
import sys
import math

from mpi4py      import MPI
from stiff_pdes  import JTV, initWorld, rhs_jac_pdefuncs, print_stuff
from integrators import Integrator, epi_for_others
from time        import time 

def heav(x):
   
   if x <0 or x < 1e-8:
     out = 0.0
   elif x == 0.0:
     out = 0.5
   else:
     out = 1.0

   return out


#1. initialize world
comm  = MPI.COMM_WORLD
world = initWorld.InitWorld(comm, "periodic", [0.0, 1.0], 10001)

#2. read in command line arguments
epi_order    = int(sys.argv[1])
ortho_method = str(sys.argv[2])

#3. set up initial condition 
Q = np.zeros(world.oneDsize)

x0 = 0.0
for j in range(0,world.numPointsY):      #y
  ycoord = world.startY + (j * world.dx)

  for k in range(0, world.numPointsX):    #x

    xcoord = world.startX + (k * world.dx)
    idx = j*world.numPointsY + k

    #Q[idx] = 1.0 + heav(0.25 - xcoord) + heav(xcoord - 0.6) \
    #       + heav(0.25 - ycoord) + heav(ycoord - 0.6)
 
    Q[idx] = 1.0 + np.heaviside(0.25 - xcoord, x0) + np.heaviside(xcoord - 0.6, x0) \
           + np.heaviside(0.25 - ycoord, x0) + np.heaviside(ycoord - 0.6, x0)

#4. set up problem parameters
epsilon = 1.0   #coeff for diffusion
alpha   = 10.0  #coeff for advection
gamma   = 0.0   #coeff for reaction

#5. set up integrator
#access to functions that are inputs for EPI constructor
rhs = rhs_jac_pdefuncs.porous_rhs
jtv = rhs_jac_pdefuncs.porous_jtv

#lamdba function of rhs, that way when called inside the integrator
#the coefficients and world info don't have to be consistently passed
#as an argument
rhs_handle = lambda u: rhs(u, alpha, world)

stepper = epi_for_others.Epi_others(epi_order, rhs_handle, jtv, ortho_method, world, [epsilon, alpha, gamma], init_substeps=10)

#6. set up time integration
t_start     = 0.0
t_end       = 0.01 #og, 0.01
dt          = 0.000005 #0.000025 400 steps, 0.00001 1k steps
#dt = 0.000000001 #can get 20 steps in under 10 mins
total_steps = math.ceil(t_end / dt)

#7. time step
#--follow gef implementation--
start_time = time()
step = 0
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

#print out stats
if world.IamRoot:
   size        = MPI.COMM_WORLD.Get_size()
   method      = str(epi_order)
   methodOrtho = str(ortho_method)
   totaltime_name = "results_tanya/runtime_"+ methodOrtho + "_n" +  str(size) + "_e" + str(method) + "por.txt"

   with open(totaltime_name, 'a') as gg:
     gg.write('{} \n'.format(total_time))

   print("Total runtime = {}".format(total_time))

"""   
#print final solution 
print("Gather solution")
finalSolQ = MPI.COMM_WORLD.gather(Q, root=0)

if world.IamRoot:

   #1. gather solin 1 vec and print output file
   filename = "epi3_por_ts1000.txt"
   totalOutLap = print_stuff.print_sol(finalSolQ, filename, world)
"""


