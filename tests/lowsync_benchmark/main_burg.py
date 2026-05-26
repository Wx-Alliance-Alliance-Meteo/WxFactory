"""
 This is the main file for running 2D Burger's equation. 

"""

import numpy as np
import sys
import math

import scipy.linalg
import glob

from types import SimpleNamespace

from mpi4py import MPI
from wx_factory.stiff_pdes import JTV, initWorld, rhs_jac_pdefuncs, print_stuff
from wx_factory.integrators import Epi, Srerk
from time import time

# 1. initialize world
comm = MPI.COMM_WORLD
world = initWorld.InitWorld(comm, "Dirichlet", [0.0, 1.0], 2002)

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

        # init condition with Pranab's example
        # temp1  = 1.0 / (1.0 - (2.0*xcoord - 1.0)**2 ) + 1.0 / ( 1.0 - (2.0*ycoord - 1.0)**2)
        # temp2  = -( (xcoord - 0.9)**2 + (ycoord - 0.9)**2) / (2.0*(0.02**2))
        # Q[idx] = 1.0 + math.exp(1.0 - temp1) + 0.5*math.exp(temp2)

        # with dirichlet conditions
        temp1 = 1.0 - ycoord
        Q[idx] = math.sin(3.0 * math.pi * xcoord) ** 2 * math.sin(3.0 * math.pi * ycoord) ** 2 * temp1

# 4. set up problem parameters
epsilon = 0.0003  # coeff for diffusion
alpha = 1.0  # coeff for advection
gamma = 0.0  # coeff for reaction

# 5. set up integrator
rhs = rhs_jac_pdefuncs.burgers_rhs
jtv = rhs_jac_pdefuncs.burgers_jtv

rhs_handle = lambda u: rhs(u, epsilon, alpha, world)
jac_handle = lambda v, Q, scale: jtv(v, Q, scale, epsilon, alpha, gamma, world)

config = SimpleNamespace(
    tolerance=1e-12,
    verbose_solver=0,
    jacobian_method="complex",
    exponential_solver=ortho_method,
    case_number=0,
    time_integrator=method,
    exode_method="",
    exode_controller="",
)

if int_type == "srerk":
    stepper = Srerk(config, order, rhs_handle, jac=jac_handle)
else:
    stepper = Epi(config, order, rhs_handle, jac=jac_handle, init_substeps=10)

# 6. set up time integration
t_start = 0.0
t_end = 0.04
dt = 0.002  # 0.005 for 50 steps, 0.0004 1250 steps
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

# compare with reference
# print("Gather solution")
# finalSolQ = MPI.COMM_WORLD.gather(Q, root=0)
total_time = time() - start_time

if world.IamRoot:

    size = MPI.COMM_WORLD.Get_size()
    totaltime_name = "results_tanya/runtime_" + ortho_method + "_n" + str(size) + "_e" + method + "_burg.txt"
    with open(totaltime_name, "a") as gg:
        gg.write("{} \n".format(total_time))
