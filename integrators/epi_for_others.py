from collections import deque
import math
from typing import Callable

from mpi4py import MPI
import numpy

from .integrator import Integrator, SolverInfo
from solvers import (
    kiops,
    matvec_fun,
    pmex,
    pmex_1s,
    pmex_ne1s,
    cwy_ne,
    cwy_1s,
    cwy_ne1s,
    icwy_1s,
    icwy_ne,
    icwy_ne1s,
    icwy_neiop,
    kiops_nest,
    dcgs2,
    kiops_nest,
)


class Epi_others(Integrator):
    def __init__(
        self,
        order: int,
        rhs: Callable,
        jac: Callable,
        exp_solver,
        world_data,
        coeff=None,
        init_method=None,
        init_substeps: int = 1,
    ):
        # super().__init__(param, preconditioner=None)
        self.rhs = rhs  # function for rhs of pde
        self.jac = jac  # function for the jacobian
        self.worldInfo = world_data  # proc data for local comm of jtv and rhs funcs
        self.coeff = coeff
        self.tol = 1e-12  # hard code the tolerance
        self.krylov_size = 1
        self.exponential_solver = exp_solver  # was in param file, now input
        # ---these were in integrator and were causing issues---
        self.preconditioner = None
        self.output_manager = None
        self.solver_info = None
        self.sim_time = -1.0
        # --------------------
        # self.int = time_int                  #was in param file, now input

        if order == 2:
            self.A = numpy.array([[]])
        elif order == 3:
            self.A = numpy.array([[2 / 3]])
        elif order == 4:
            self.A = numpy.array([[-3 / 10, 3 / 40], [32 / 5, -11 / 10]])
        elif order == 5:
            self.A = numpy.array([[-4 / 5, 2 / 5, -4 / 45], [12, -9 / 2, 8 / 9], [3, 0, -1 / 3]])
        elif order == 6:
            self.A = numpy.array(
                [
                    [-49 / 60, 351 / 560, -359 / 1260, 367 / 6720],
                    [92 / 7, -99 / 14, 176 / 63, -1 / 2],
                    [485 / 21, -151 / 14, 23 / 9, -31 / 168],
                ]
            )
        else:
            raise ValueError("Unsupported order for EPI method")

        k, self.n_prev = self.A.shape
        # Limit max phi to 1 for EPI 2
        if order == 2:
            k -= 1
        self.max_phi = k + 1
        self.previous_Q = deque()
        self.previous_rhs = deque()
        self.dt = 0.0

        if init_method or self.n_prev == 0:
            self.init_method = init_method
        else:
            # removed param
            self.init_method = Epi_others(2, rhs, jac, exp_solver, world_data, coeff)

        self.init_substeps = init_substeps

    def __step__(self, Q: numpy.ndarray, dt: float):

        # the additional parameter 'coeff' is a scalar for the laplacian
        # we test different values for adv-diff and ac problem

        # If dt changes, discard saved value and redo initialization
        mpirank = MPI.COMM_WORLD.Get_rank()
        if self.dt and abs(self.dt - dt) > 1e-10:
            self.previous_Q = deque()
            self.previous_rhs = deque()
        self.dt = dt

        # Initialize saved values using init_step method
        if len(self.previous_Q) < self.n_prev:
            self.previous_Q.appendleft(Q)
            self.previous_rhs.appendleft(self.rhs(Q))

            dt /= self.init_substeps
            for i in range(self.init_substeps):
                Q = self.init_method.step(Q, dt)
            return Q

        # Regular EPI step
        rhs = self.rhs(Q)

        # matvec_fun performs a finite difference approximation to the Jacobian
        # instead, I have a function the outputs the action of the jacobian on
        # a vector, so replace the matvec handle with my jtv function

        matvec_handle = lambda v: self.jac(v, Q, dt, self.coeff[0], self.coeff[1], self.coeff[2], self.worldInfo)

        # print("n_prev = {}".format(self.n_prev))
        # print("max_phi+1 = {}".format(self.max_phi+1))

        # def matvec_handle(v): return matvec_fun(v, dt, Q, rhs, self.rhs, self.jacobian_method)

        vec = numpy.zeros((self.max_phi + 1, rhs.size))
        vec[1, :] = rhs.flatten()
        for i in range(self.n_prev):
            # ---original----
            # J_deltaQ = matvec_fun(self.previous_Q[i] - Q, 1., Q, rhs, self.rhs, self.jacobian_method)

            # ---with the JtvFunctions for new problems---
            # previous_Q[i] - Q is the vector that multiplies the Jacobian
            # for reference, look at Val's SWE paper
            J_deltaQ = self.jac(
                self.previous_Q[i] - Q, Q, 1.0, self.coeff[0], self.coeff[1], self.coeff[2], self.worldInfo
            )

            # R(y_{n-i})
            r = (self.previous_rhs[i] - rhs) - numpy.reshape(J_deltaQ, Q.shape)

            for k, alpha in enumerate(self.A[:, i], start=2):
                # v_k = Sum_{i=1}^{n_prev} A_{k,i} R(y_{n-i})
                vec[k, :] += alpha * r.flatten()

        # ----pmex with norm estimate-----
        if self.exponential_solver == "pmex":
            phiv, stats = pmex(
                [1.0], matvec_handle, vec, tol=self.tol, m_init=self.krylov_size, mmin=16, mmax=64, task1=False
            )
            self.krylov_size = math.floor(0.7 * stats[5] + 0.3 * self.krylov_size)

            if mpirank == 0:
                # print to file stats
                # size      = MPI.COMM_WORLD.Get_size()
                # file_name = "results_tanya/pmexne_stats_" + "n" + str(size) + "_" + str(self.int) + "_c" + str(self.case_number) + ".txt"
                # with open(file_name, 'a') as gg:
                #  gg.write('{} {} {} {} {} {} {} {} {} \n'.format(stats[0], stats[1], stats[2], stats[9], stats[5], stats[6], stats[7], stats[8], stats[10]))

                print(
                    f"PMEX NE converged at iteration {stats[2]} (using {stats[0]} internal substeps and {stats[1]} rejected expm)"
                    f" to a solution with local error {stats[4]:.2e}"
                )

        # ----pmex with 1-sync-----
        elif self.exponential_solver == "pmex_1s":
            phiv, stats = pmex_1s(
                [1.0], matvec_handle, vec, tol=self.tol, m_init=self.krylov_size, mmin=16, mmax=64, task1=False
            )
            self.krylov_size = math.floor(0.7 * stats[5] + 0.3 * self.krylov_size)

            if mpirank == 0:
                """
                #print to file stats
                size      = MPI.COMM_WORLD.Get_size()
                file_name = "results_tanya/pmex1s_stats_" + "n" + str(size) + "_" + str(self.int) + "_c" + str(self.case_number) + ".txt"
                with open(file_name, 'a') as gg:
                  gg.write('{} {} {} {} {} {} {} {} \n'.format(stats[0], stats[1], stats[2], stats[5], stats[6], stats[7], stats[8], stats[9]))
                """
                print(
                    f"PMEX 1s converged at iteration {stats[2]} (using {stats[0]} internal substeps and {stats[1]} rejected expm)"
                    f" to a solution with local error {stats[4]:.2e}"
                )

        # ----pmex with norm estimate+1s-----
        elif self.exponential_solver == "pmex_ne1s":
            phiv, stats = pmex_ne1s(
                [1.0], matvec_handle, vec, tol=self.tol, m_init=self.krylov_size, mmin=16, mmax=64, task1=False
            )
            self.krylov_size = math.floor(0.7 * stats[5] + 0.3 * self.krylov_size)

            if mpirank == 0:
                print(
                    f"PMEX NE+1s converged at iteration {stats[2]} (using {stats[0]} internal substeps and {stats[1]} rejected expm)"
                    f" to a solution with local error {stats[4]:.2e}"
                )

        # ----- icwy norm estimate + 1sync-----
        elif self.exponential_solver == "icwy_ne1s":
            phiv, stats = icwy_ne1s(
                [1.0], matvec_handle, vec, tol=self.tol, m_init=self.krylov_size, mmin=16, mmax=64, task1=False
            )
            self.krylov_size = math.floor(0.7 * stats[5] + 0.3 * self.krylov_size)

            if mpirank == 0:
                print(
                    f"ICWY NE+1S converged at iteration {stats[2]} (using {stats[0]} internal substeps and {stats[1]} rejected expm)"
                    f" to a solution with local error {stats[4]:.2e}"
                )

        # ----- icwy norm estimate -----
        elif self.exponential_solver == "icwy_ne":
            phiv, stats = icwy_ne(
                [1.0], matvec_handle, vec, tol=self.tol, m_init=self.krylov_size, mmin=16, mmax=64, task1=False
            )
            self.krylov_size = math.floor(0.7 * stats[5] + 0.3 * self.krylov_size)

            if mpirank == 0:
                print(
                    f"ICWY NE converged at iteration {stats[2]} (using {stats[0]} internal substeps and {stats[1]} rejected expm)"
                    f" to a solution with local error {stats[4]:.2e}"
                )

        # ----- icwy 1-sync-----
        elif self.exponential_solver == "icwy_1s":
            phiv, stats = icwy_1s(
                [1.0], matvec_handle, vec, tol=self.tol, m_init=self.krylov_size, mmin=16, mmax=64, task1=False
            )
            self.krylov_size = math.floor(0.7 * stats[5] + 0.3 * self.krylov_size)

            if mpirank == 0:
                """
                #print stats to file
                size      = MPI.COMM_WORLD.Get_size()
                file_name = "results_tanya/icwy1s_stats_" + "n" + str(size) + "_" + str(self.int) + "_c" + str(self.case_number)+ "_e" +str(self.elem) + ".txt"
                with open(file_name, 'a') as gg:
                  gg.write('{} {} {} {} {} {} {} {} \n'.format(stats[0], stats[1], stats[2], stats[6], stats[7], stats[8], stats[9], stats[10]))
                """

                print(
                    f"ICWY 1S converged at iteration {stats[2]} (using {stats[0]} internal substeps and {stats[1]} rejected expm)"
                    f" to a solution with local error {stats[4]:.2e}"
                )

        # ----- icwy iop+norm estimate-----
        elif self.exponential_solver == "icwy_neiop":
            phiv, stats = icwy_neiop(
                [1.0], matvec_handle, vec, tol=self.tol, m_init=self.krylov_size, mmin=16, mmax=64, task1=False
            )
            self.krylov_size = math.floor(0.7 * stats[5] + 0.3 * self.krylov_size)

            if mpirank == 0:
                print(
                    f"ICWY NE+IOP converged at iteration {stats[2]} (using {stats[0]} internal substeps and {stats[1]} rejected expm)"
                    f" to a solution with local error {stats[4]:.2e}"
                )

        # ----- cwy norm estimate + 1sync-----
        elif self.exponential_solver == "cwy_ne1s":
            phiv, stats = cwy_ne1s(
                [1.0], matvec_handle, vec, tol=self.tol, m_init=self.krylov_size, mmin=16, mmax=64, task1=False
            )
            self.krylov_size = math.floor(0.7 * stats[5] + 0.3 * self.krylov_size)

            if mpirank == 0:
                print(
                    f"CWY NE+1S converged at iteration {stats[2]} (using {stats[0]} internal substeps and {stats[1]} rejected expm)"
                    f" to a solution with local error {stats[4]:.2e}"
                )

        # ----- cwy norm estimate -----
        elif self.exponential_solver == "cwy_ne":
            phiv, stats = cwy_ne(
                [1.0], matvec_handle, vec, tol=self.tol, m_init=self.krylov_size, mmin=16, mmax=64, task1=False
            )
            self.krylov_size = math.floor(0.7 * stats[5] + 0.3 * self.krylov_size)

            if mpirank == 0:
                print(
                    f"CWY NE converged at iteration {stats[2]} (using {stats[0]} internal substeps and {stats[1]} rejected expm)"
                    f" to a solution with local error {stats[4]:.2e}"
                )

        # ----- cwy 1-sync-----
        elif self.exponential_solver == "cwy_1s":
            phiv, stats = cwy_1s(
                [1.0], matvec_handle, vec, tol=self.tol, m_init=self.krylov_size, mmin=16, mmax=64, task1=False
            )
            self.krylov_size = math.floor(0.7 * stats[5] + 0.3 * self.krylov_size)

            if mpirank == 0:

                """
                #print stats to file
                size      = MPI.COMM_WORLD.Get_size()
                file_name = "results_tanya/cwy1s_stats_" + "n" + str(size) + "_" + str(self.int) + "_c" + str(self.case_number) + "_e" + str(self.elem) + ".txt"
                with open(file_name, 'a') as gg:
                  gg.write('{} {} {} {} {} {} {} {} \n'.format(stats[0], stats[1], stats[2], stats[6], stats[7], stats[8], stats[9], stats[10]))
                """

                print(
                    f"CWY 1S converged at iteration {stats[2]} (using {stats[0]} internal substeps and {stats[1]} rejected expm)"
                    f" to a solution with local error {stats[4]:.2e}"
                )

        # ----- dcgs2 -----
        elif self.exponential_solver == "dcgs2":
            phiv, stats = dcgs2(
                [1.0], matvec_handle, vec, tol=self.tol, m_init=self.krylov_size, mmin=16, mmax=64, task1=False
            )
            self.krylov_size = math.floor(0.7 * stats[5] + 0.3 * self.krylov_size)

            if mpirank == 0:
                print(
                    f"DCGS2 converged at iteration {stats[2]} (using {stats[0]} internal substeps and {stats[1]} rejected expm)"
                    f" to a solution with local error {stats[4]:.2e}"
                )

        # ----- kiops + norm estimate-----
        elif self.exponential_solver == "kiops_ne":
            phiv, stats = kiops_nest(
                [1.0], matvec_handle, vec, tol=self.tol, m_init=self.krylov_size, mmin=16, mmax=64, task1=False
            )
            self.krylov_size = math.floor(0.7 * stats[5] + 0.3 * self.krylov_size)

            if mpirank == 0:
                print(
                    f"KIOPS NE converged at iteration {stats[2]} (using {stats[0]} internal substeps and {stats[1]} rejected expm)"
                    f" to a solution with local error {stats[4]:.2e}"
                )

        # ----------default: kiops-----------
        else:
            phiv, stats = kiops(
                [1], matvec_handle, vec, tol=self.tol, m_init=self.krylov_size, mmin=16, mmax=64, task1=False
            )
            self.krylov_size = math.floor(0.7 * stats[5] + 0.3 * self.krylov_size)

            if mpirank == 0:
                """
                #print to file stats
                size      = MPI.COMM_WORLD.Get_size()
                file_name = "results_tanya/kiops_stats_" + "n" + str(size) + "_" + str(self.int) + "_c" + str(self.case_number) + ".txt"
                with open(file_name, 'a') as gg:
                  gg.write('{} {} {} {} {} {} {} {} \n'.format(stats[0], stats[1], stats[2], stats[6], stats[7], stats[8], stats[9], stats[10]))
                """

                print(
                    f"KIOPS converged at iteration {stats[2]} (using {stats[0]} internal substeps and {stats[1]} rejected expm)"
                    f" to a solution with local error {stats[4]:.2e}"
                )

        self.solver_info = SolverInfo(total_num_it=stats[2])

        # Save values for the next timestep
        if self.n_prev > 0:
            self.previous_Q.pop()
            self.previous_Q.appendleft(Q)
            self.previous_rhs.pop()
            self.previous_rhs.appendleft(rhs)

        # Update solution
        return Q + numpy.reshape(phiv, Q.shape) * dt
