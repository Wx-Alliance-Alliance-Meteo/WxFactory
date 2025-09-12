import math
import sys

sys.path.insert(1, "/home/siw001/WxFactory_cleanup_for_git/")
import numpy as np

# from solvers import exode
from integrators import Integrator, Epi
from solvers import exode, kiops, pmex, matvec_fun
from math import pi
from collections import deque
from scipy.integrate import solve_ivp
from epi_for_de import Epi_for_DE

import time
import os
import argparse

parser = argparse.ArgumentParser(description="Process one input parameter.")
parser.add_argument("epi_method", help = "2-6")
parser.add_argument("exode_method", help = "BS32, etc.") 
parser.add_argument("tol", help = "1e-5") 
parser.add_argument("rtol", help= "1e-5")

args = parser.parse_args()

output_dir = "/space/hall5/sitestore/eccc/mrd/rpnatm/siw001/testoutput/ADR_2D/"
write_output = "True"
alpha = -10.0
epsil = 1.0 / 100.0
gamma = 100.0
Nx = 401
x0 = 0
xn = 1
dx = (xn - x0) / (Nx - 1)
x = np.linspace(x0, xn, Nx)
y = np.linspace(x0, xn, Nx)
t0 = 0
tf = 0.1
# N_step = 100
# dt = (tf-t0)/N_step
# exode_method='RK23'
controller = "PI3040"
# epi_method = 6
# tol = 1e-16

# initial condition 256(xy(1 −x)(1 −y))^2+0.3
u0 = np.zeros((Nx, Nx))
for i in range(Nx):
    for j in range(Nx):
        u0[i, j] = 256.0 * ((x[i] * y[j] * (1.0 - x[i]) * (1.0 - y[j])) ** 2) + 0.3
u0 = u0.flatten()


def boundary(u, bc):
    if bc == "N":
        u_top = np.concatenate(([0], u[1, :], [0])).reshape(1, Nx + 2)
        u_bot = np.concatenate(([0], u[-2, :], [0])).reshape(1, Nx + 2)
        u_lef = u[:, 1].reshape(Nx, 1)
        u_rig = u[:, -2].reshape(Nx, 1)
        u = np.concatenate((u_lef, u, u_rig), axis=1)
        u = np.concatenate((u_top, u, u_bot), axis=0)
    else:
        print("boudnary condition not supported")

    return u


def f2D_Dx(u):
    Dx = 1 / (2 * dx) * (u[1:-1, 2:] - u[1:-1, :-2] + u[2:, 1:-1] - u[:-2, 1:-1])
    return Dx


def f2D_D2x(u):
    D2x = (
        1
        / (dx ** 2)
        * (u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, :-2] + u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[:-2, 1:-1])
    )
    return D2x


def rhsfunc(u):
    u = u.reshape(Nx, Nx)
    RU = gamma * u * (u - 0.5) * (1 - u)

    uN = boundary(u, "N")
    DU = -alpha * f2D_Dx(uN)
    LU = epsil * f2D_D2x(uN)

    dudt = LU + DU + RU
    dudt = dudt.flatten()
    u = u.flatten()
    return dudt


def rhsfunca_solveivp(t, u):
    u = u.reshape(Nx, Nx)
    RU = gamma * u * (u - 0.5) * (1 - u)

    uN = boundary(u, "N")
    DU = -alpha * f2D_Dx(uN)
    LU = epsil * f2D_D2x(uN)

    dudt = LU + DU + RU
    dudt = dudt.flatten()
    u = u.flatten()
    return dudt


rhs = rhsfunc(u0)
t = t0
u = u0


def l2norm(array):
    return abs(math.sqrt(sum(np.square(abs(array)))))


# reference solution
with open("./ref_sol/2D_ADR_ref_sol_Nx_" + str(Nx) + "_tf_0d1.csv") as ref_file:
    refsol = np.loadtxt(ref_file, delimiter=",")


timing_repeat = 5
rtol= args.rtol
for tol in [float(args.tol)]:
    edit_first_step = "False"
    first_step_controller = 0.9
    for epi_method in [int(args.epi_method)]:
        output_subdir = output_dir + "epi" + str(epi_method)+"/rerun_full_node_rtol_"+rtol
        os.makedirs(output_subdir, exist_ok=True)

        print(
            "epi_method = epi",
            epi_method,
            " with tolerance = ",
            tol,
            " spatial grid = ",
            Nx,
        )

        # repeat $timing_repeat number of times for timing
        for repeat in range(0,timing_repeat):

            print("Run number ", repeat)
            for exode_method in [
            #    "pmex",
            #    "kiops",
            #    "BS32",
            #    "DP54",
            #    "M43",
            #    "KC32",
            #    "EXLRK32",
            #    "EXLRK43",
            #    "ExLRK4(3)minA5",
            #    "ExLRK4(3)minA5param0",
            #    "ExLRK4(3)minA5param0d5",
            #    "ExLRK4(3)minA5paramN0D25",
            #    "Ralston43",
            args.exode_method
            ]:
                # file to save testoutput
                if edit_first_step == "False":
                    testoutput_file = (
                            output_subdir + "/" + exode_method + "_tol_" + str(tol) + "_rtol_"+rtol+"_Nx_" + str(Nx) + "_nt_50_400_repeat_" + str(repeat)+".txt"
                    )
                else:
                    testoutput_file = (
                        output_subdir
                        + "/"
                        + exode_method
                        + "_tol_"
                        + str(tol)
                        + "_first_step_controller_"
                        + str(first_step_controller)
                        + "_Nx_"
                        + str(Nx)
                        + ".txt"
                    )
                with open(testoutput_file, "w") as outputfile:

                    # for convergence
                    prev_err = 1.0
                    prev_dt = 2.0

                    if exode_method == "kiops":
                        method = Epi_for_DE(
                            epi_method,
                            rhsfunc,
                            tol,
                            float(rtol),
                            "kiops",
                            exode_method,
                            controller,
                        )
                    elif exode_method == "pmex":
                         method =  Epi_for_DE(
                            epi_method,
                            rhsfunc,
                            tol,
                            float(rtol),
                            "pmex",
                            exode_method,
                            controller,
                        )
                    else:
                        method = Epi_for_DE(
                            epi_method,
                            rhsfunc,
                            tol,
                            float(rtol),
                            "exode",
                            exode_method,
                            controller,
                        )
                    print("Method: ", exode_method)
                    print("tolerance, dt, N_step, error, order, timing")
                    for N_step in [50,100,200,400]:
                        dt = (tf - t0) / N_step
                        u = u0
                        t = t0

                        begin = time.perf_counter()
                        for step in range(N_step):
                            if edit_first_step == "False":
                                first_step = 1.0
                            else:
                                if step == 0:
                                    first_step = 1.0
                                else:
                                    first_step = first_step_controller * stats[4]

                            u, stats = method.step(u, dt, first_step)
                            t = t + dt
                        end = time.perf_counter()
                        timing = end - begin
                        # print("Elapsed Time: ", f"{end-begin:0.4f}", "seconds")

                        error = l2norm(u - refsol)
                        print(
                            tol,
                            dt,
                            N_step,
                            error,
                            math.log(prev_err / error, prev_dt / dt),
                            timing,
                        )
                        if write_output == "True":
                            outputfile.write(
                                str(tol)
                                + " "
                                + str(dt)
                                + " "
                                + str(N_step)
                                + " "
                                + str(error)
                                + " "
                                + str(math.log(prev_err / error, prev_dt / dt))
                                + " "
                                + str(timing)
                                + "\n"
                            )

                        prev_err = error
                        prev_dt = dt
                    print("\n")
