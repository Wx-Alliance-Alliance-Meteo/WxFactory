"""
	main file for running laplacian 

"""

import numpy as np
import math
import scipy.linalg
import pandas as pd

from mpi4py import MPI
from stiff_pdes import JTV, initWorld, print_stuff

# 1. initialize world
# how to call InitWorld
# 	1. comm
# 	2. type of boundary conditions: "periodic", "Neumann", "Diriclet" (assumes homogeneous Neumann and Dirichlet)
# 	3. domain, x,y in [start, end]
# 	4. number of points per axis
# 		NOTE: total number of points will be differnt in periodic or other 2 BC
# 		for example: if N = 11, then there will be 10 points per axis w/ periodic bcs but 9 for the others

dxdx = [0.05, 0.025, 0.01, 0.005, 0.0025, 0.001, 0.0005]
N = [int((1.0 / j) + 1) for j in dxdx]  # N = 11, 21, 41, 101, 201, 401, 1001, 2001
error = []

for p in range(len(N)):

    comm = MPI.COMM_WORLD
    world = initWorld.InitWorld(comm, "periodic", [0.0, 1.0], N[p])

    dx = world.dx
    totalN = world.oneDsize  # 1d size for each processor

    rhs_vec = np.zeros(totalN)
    u = np.zeros(totalN)

    print("setting initial condition")
    for j in range(0, world.numPointsY):  # y
        ycoord = world.startY + (j * world.dx)

        for k in range(0, world.numPointsX):  # x

            xcoord = world.startX + (k * world.dx)
            idx = j * world.numPointsY + k

            # rhs_vec[idx] = math.sin(math.pi * xcoord) * math.sin(math.pi * ycoord) #for homo DIRICHLET
            # rhs_vec[idx] = math.cos(math.pi * xcoord) * math.cos(math.pi * ycoord) #for homo NEUMANN
            rhs_vec[idx] = math.cos(2.0 * math.pi * xcoord) * math.cos(2.0 * math.pi * ycoord)  # for periodic
            # rhs_vec[idx] = xcoord*ycoord #quick test of periodic bc's
            u[idx] = 0.5  # with this u, nonlinear laplacian should just be laplacian of v, 1 for adv, 0.5 for lap

    # 2. call laplacian
    print("applying advection")
    # print("applying laplacian")

    # Dirichlet Test
    # out_lap = advectionDirichlet(rhs_vec, 1.0, world)
    # out_lap = laplacianDirichlet(rhs_vec, 1.0, world)

    # Neumann Test
    # out_lap = advectionNeumann(rhs_vec, 1.0, world)
    # out_lap = laplacianNeumann(rhs_vec, 1.0, world)

    # Peroidic Test
    # out_lap = JTV.advectionPeriodic(rhs_vec, 1.0, world)
    # out_lap = laplacianPeriodic(rhs_vec, 1.0, world)

    # nonlinear operators
    # out_lap = nonLinAdvecPeriodic(rhs_vec, u, 1.0, world)
    out_lap = JTV.nonLinLapPeriodic(rhs_vec, u, 1.0, world)

    # 3. compute true solution
    print("computing ref solution")
    true_solution = np.zeros(totalN)

    for j in range(0, world.numPointsY):
        ycoord = world.startY + (j * world.dx)
        for k in range(0, world.numPointsX):
            idx = j * world.numPointsY + k
            xcoord = world.startX + (k * world.dx)

            # true_solution[idx] = -2.0 * math.pi**2 * rhs_vec[idx] #ref for homo laplacian test
            # true_solution[idx] = math.pi * (math.sin(math.pi *xcoord)*math.cos(math.pi*ycoord) + \
            # 	 math.cos(math.pi*xcoord)*math.sin(math.pi*ycoord)) #for homo advection tests

            true_solution[idx] = -8.0 * math.pi**2 * rhs_vec[idx]  # ref for periodic laplacian test
            # true_solution[idx] = -2.0*math.pi * (math.sin(2.0*math.pi *xcoord)*\
            #   math.cos(2.0*math.pi*ycoord) + \
            # 	 math.cos(2.0*math.pi*xcoord)*math.sin(2.0*math.pi*ycoord)) #for periodic advection tests

    print("Gather Solution")
    finalSol_lap = MPI.COMM_WORLD.gather(out_lap, root=0)
    refGather = MPI.COMM_WORLD.gather(true_solution, root=0)

    if world.IamRoot:

        print("gather sol in 1 vec")
        totalOutLap = print_stuff.print_sol(finalSol_lap, world)
        totalRefSol = print_stuff.print_sol(refGather, world)

        # print(totalOutLap)
        # 4. compute error

        print("computing linf error")
        error.append(scipy.linalg.norm(totalRefSol - totalOutLap, np.inf))
        print(error)
        print("all good chief, coming from processor {}".format(world.rank))

# after done computing all error terms, now compute the order
order = np.zeros(len(N))

if world.IamRoot:
    for kk in range(len(N) - 1):
        print("kk = {}".format(kk))
        order[kk] = math.log(error[kk] / error[kk + 1]) / math.log(dxdx[kk] / dxdx[kk + 1])

    table = np.column_stack((dxdx, error, order))
    table = pd.DataFrame(table, columns=["dx", "inf-Error", "order"])
    print(table)
