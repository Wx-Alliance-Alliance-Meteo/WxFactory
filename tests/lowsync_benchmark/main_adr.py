"""
 This is the main file for running the Advection-Diffusion-Reaction
 pde

 u_t = epsilon(u_xx + u_yy) - alpha(u_x + u_y) - gamma*u(1-u)

 with x,y in [0,1] with homogeneous Neumann boundary conditions. 

 Integrating from t = [0, 0.02] with dt = 0.001.

 Run with processors p = [100, 400, 625, 1600, 2500] for strong
 scaling results. 

 Runtime of the total simulation will be saved for each ortho method
 and integrator combination.

 Integrators tested: epi4, epi5, epi6, srerk3, srerk6
 Ortho methods: kiops, pmex_{1s,ne1s,ne}, icwy_{1s,ne1s,ne}, cwy_{1s,ne1s,ne} 
"""

import numpy as np
import sys
import math

from mpi4py import MPI
from stiff_pdes import JTV, initWorld, rhs_jac_pdefuncs, print_stuff
from integrators import Integrator, epi_for_others, srerk_for_others
from time import time

# 1. initialize world
comm = MPI.COMM_WORLD
world = initWorld.InitWorld(comm, "Neumann", [0.0, 1.0], 2002)

# 2. read in command line arguments
method = str(sys.argv[1])  # eg epi4 or srerk3
ortho_method = str(sys.argv[2])  # eg kiops or pmex1s

order = int(method[-1])  # the order of the method will always be last
int_type = str(method[:-1])  # check if srerk or epi method

# 3. set up initial condition
Q = np.zeros(world.oneDsize)

for j in range(0, world.numPointsY):  # y
    ycoord = world.startY + (j * world.dx)

    for k in range(0, world.numPointsX):  # x

        xcoord = world.startX + (k * world.dx)
        idx = j * world.numPointsY + k

        Q[idx] = 0.3 + 256.0 * (xcoord * ycoord * (1 - xcoord) * (1 - ycoord)) ** 2

# 4. set up problem parameters
epsilon = 0.01  # coeff for diffusion
alpha = -6.0  # coeff for advection
gamma = 100.0  # coeff for reaction

# 5. set up integrator
# access to functions that are inputs for EPI constructor
rhs = rhs_jac_pdefuncs.adr_rhs
jtv = rhs_jac_pdefuncs.adr_jtv

# lamdba function of rhs, that way when called inside the integrator
# the coefficients and world info don't have to be consistently passed
# as an argument
rhs_handle = lambda u: rhs(u, epsilon, alpha, gamma, world)

# Now set up the 'stepper' function
# the only two options for this test are EPI and SRERK
if int_type == "srerk":
    stepper = srerk_for_others.Srerk_others(order, rhs_handle, jtv, ortho_method, world, [epsilon, alpha, gamma])
    # print("using srerk of order = {}".format( order))

else:
    stepper = epi_for_others.Epi_others(
        order, rhs_handle, jtv, ortho_method, world, [epsilon, alpha, gamma], init_substeps=10
    )
    # print("using epi of order = {}".format(order))

# 6. set up time integration
# possitle dt from paper: 0.01, 0.005, 0.0025, 0.00125, 6.25e-04
# 0.00001 works, 0.0001 works
t_start = 0.0
t_end = 0.02
dt = 0.001
total_steps = math.ceil(t_end / dt)

# 7. time step
# --follow gef implementation--
step = 0
start_time = time()
while step < total_steps:

    # check to make sure we are not overstepping
    if t_start + dt > t_end:
        dt = t_end - t_start
    else:
        t_start += dt

    step += 1  # increment time step counter

    if MPI.COMM_WORLD.rank == 0:
        print(f"\nStep {step} of {total_steps}")

    Q = stepper.step(Q, dt)

total_time = time() - start_time

if world.IamRoot:
    # print("total_time = {}".format(total_time))
    size = MPI.COMM_WORLD.Get_size()
    totaltime_name = "results_tanya/runtime_" + ortho_method + "_n" + str(size) + "_e" + method + "_adr.txt"
    with open(totaltime_name, "a") as gg:
        gg.write("{} \n".format(total_time))

"""
#print final solution 
print("Gather solution")
finalSolQ = MPI.COMM_WORLD.gather(Q, root=0)

if world.IamRoot:

   #1. gather solin 1 vec and print output file
   filename = "epi5_adr_ts100.txt"
   totalOutLap = print_stuff.print_sol(finalSolQ, filename, world)
"""
