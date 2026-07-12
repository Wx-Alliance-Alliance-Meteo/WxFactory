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

from types import SimpleNamespace

from mpi4py import MPI
from wx_factory.stiff_pdes import JTV, initWorld, rhs_jac_pdefuncs, print_stuff
from wx_factory.integrators import Epi, Srerk
from time import time

# 1. initialize world
comm = MPI.COMM_WORLD
world = initWorld.InitWorld(comm, "Neumann", [-1.0, 1.0], 2002)

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

        Q[idx] = 0.1 + 0.1 * math.cos(2.0 * math.pi * xcoord) * math.cos(2.0 * math.pi * ycoord)

# 4. set up problem parameters
# NOTE: in order for the lambda Jacobian mat_vec handle to be consistent
# accross all pdes, coefficients need to be passed for diffusion, adv, react
epsilon = 0.1  # coeff for diffusion
alpha = 0.0  # coeff for advection
gamma = 0.0  # coeff for reaction

# 5. set up integrator
rhs = rhs_jac_pdefuncs.allencahn_rhs
jtv = rhs_jac_pdefuncs.allencahn_jtv

rhs_handle = lambda u: rhs(u, epsilon, world)
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
# possitle dt from paper: 0.5, 0.25, 0.1250, 0.0625, 0.03125
# dt = 0.000625 works!, 0.001 works!
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

# if world.IamRoot:
#   print("Total runtime = {}".format(total_time))


# print out runtime stats
if world.IamRoot:
    size = MPI.COMM_WORLD.Get_size()
    totaltime_name = "results_tanya/runtime_" + ortho_method + "_n" + str(size) + "_e" + method + "_ac.txt"
    with open(totaltime_name, "a") as gg:
        gg.write("{} \n".format(total_time))


"""
#print final solution 
print("Gather solution")
finalSolQ = MPI.COMM_WORLD.gather(Q, root=0)

if world.IamRoot:

   #1. gather solin 1 vec and print output file
   filename = "srerk3_pmex1s_ac_n200_ts500.txt"
   totalOutLap = print_stuff.print_sol(finalSolQ, filename, world)
"""
