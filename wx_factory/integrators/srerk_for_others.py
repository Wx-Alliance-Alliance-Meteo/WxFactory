import math
from typing import Callable, List, Optional

import numpy
from mpi4py import MPI

from common import Configuration
from .integrator import Integrator, alpha_coeff
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


# Computes nodes for SRERK methods with minimal error terms
def opt_nodes(order: int):
    if order < 3:
        raise ValueError("Order should be at least 3")

    coeff = (
        lambda p, q: (-1) ** (p + q)
        * math.factorial(p + q + 2)
        / (math.factorial(q) * math.factorial(q + 2) * math.factorial(p - q))
    )

    c = []
    # Compute optimal nodes for each stage order starting at order 2
    for o in list(range(2, order - 2, 2)) + [order - 2]:
        p = numpy.polynomial.Polynomial([coeff(o, q) for q in range(0, o + 1)])
        c.append(p.roots())

    c.append(numpy.ones(1))
    return c


class Srerk_others(Integrator):
    """Stiffness resilient exponential Runge-Kutta methods"""

    def __init__(
        self, order: int, rhs: Callable, jac: Callable, exp_solver, world_data, coeff=None, nodes: Optional[List] = None
    ):
        """
        If the nodes are NOT specified, return the SRERK method of the specified order with min error terms
        If the nodes are specified, return the SRERK method with these nodes and ignore the 'order' parameter
        """

        # super().__init__(param, preconditioner=None)
        self.rhs = rhs
        self.jac = jac
        self.worldInfo = world_data
        self.coeff = coeff
        self.tol = 1e-12
        self.krylov_size = 1
        # self.jacobian_method = param.jacobian_method
        self.exponential_solver = exp_solver
        # ---these were in integrator and were causing issues---
        self.preconditioner = None
        self.solver_info = None
        self.sim_time = -1.0

        if nodes:
            self.c = nodes
        else:
            self.c = opt_nodes(order)
        self.n_proj = len(self.c)

        self.alpha = []
        for i in range(self.n_proj - 1):
            self.alpha.append(alpha_coeff(self.c[i]))

    def __step__(self, Q: numpy.ndarray, dt: float):
        rhs = self.rhs(Q)

        # --original---
        # matvec_handle = lambda v: matvec_fun(v, dt, Q, rhs, self.rhs, self.jacobian_method)

        # matvec_fun performs a finite difference approximation to the Jacobian
        # instead, I have a function the outputs the action of the jacobian on
        # a vector, so replace the matvec handle with my jtv function

        matvec_handle = lambda v: self.jac(v, Q, dt, self.coeff[0], self.coeff[1], self.coeff[2], self.worldInfo)

        # Initial projection
        vec = numpy.zeros((2, rhs.size))
        vec[1, :] = rhs.flatten()

        # for printing stats
        mpirank = self.device.comm.rank

        # ---original kiops---
        if self.exponential_solver == "kiops":
            z, stats = kiops(
                self.c[0], matvec_handle, vec, tol=self.tol, m_init=self.krylov_size, mmin=16, mmax=64, task1=False
            )

            if mpirank == 0:
                print(
                    f"KIOPS converged at iteration {stats[2]} (using {stats[0]} internal substeps and {stats[1]} rejected expm)"
                    f" to a solution with local error {stats[4]:.2e}"
                )

            self.krylov_size = math.floor(0.7 * stats[5] + 0.3 * self.krylov_size)

        # ---pmex with norm estimate---
        elif self.exponential_solver == "pmex":

            z, stats = pmex(
                self.c[0], matvec_handle, vec, tol=self.tol, m_init=self.krylov_size, mmin=16, mmax=64, task1=False
            )
            self.krylov_size = math.floor(0.7 * stats[5] + 0.3 * self.krylov_size)

            if mpirank == 0:
                print(
                    f"PMEX converged at iteration {stats[2]} (using {stats[0]} internal substeps and {stats[1]} rejected expm)"
                    f" to a solution with local error {stats[4]:.2e}"
                )

        # ----pmex with 1-sync-----
        elif self.exponential_solver == "pmex_1s":
            z, stats = pmex_1s(
                self.c[0], matvec_handle, vec, tol=self.tol, m_init=self.krylov_size, mmin=16, mmax=64, task1=False
            )
            self.krylov_size = math.floor(0.7 * stats[5] + 0.3 * self.krylov_size)

            if mpirank == 0:
                print(
                    f"PMEX 1s converged at iteration {stats[2]} (using {stats[0]} internal substeps and {stats[1]} rejected expm)"
                    f" to a solution with local error {stats[4]:.2e}"
                )

        # ----pmex with norm estimate+1s-----
        elif self.exponential_solver == "pmex_ne1s":
            z, stats = pmex_ne1s(
                self.c[0], matvec_handle, vec, tol=self.tol, m_init=self.krylov_size, mmin=16, mmax=64, task1=False
            )
            self.krylov_size = math.floor(0.7 * stats[5] + 0.3 * self.krylov_size)

            if mpirank == 0:
                print(
                    f"PMEX NE+1s converged at iteration {stats[2]} (using {stats[0]} internal substeps and {stats[1]} rejected expm)"
                    f" to a solution with local error {stats[4]:.2e}"
                )

        # ----- icwy norm estimate -----
        elif self.exponential_solver == "icwy_ne":
            z, stats = icwy_ne(
                self.c[0], matvec_handle, vec, tol=self.tol, m_init=self.krylov_size, mmin=16, mmax=64, task1=False
            )
            self.krylov_size = math.floor(0.7 * stats[5] + 0.3 * self.krylov_size)

            if mpirank == 0:
                print(
                    f"ICWY NE converged at iteration {stats[2]} (using {stats[0]} internal substeps and {stats[1]} rejected expm)"
                    f" to a solution with local error {stats[4]:.2e}"
                )

        # ----- icwy 1-sync-----
        elif self.exponential_solver == "icwy_1s":
            z, stats = icwy_1s(
                self.c[0], matvec_handle, vec, tol=self.tol, m_init=self.krylov_size, mmin=16, mmax=64, task1=False
            )
            self.krylov_size = math.floor(0.7 * stats[5] + 0.3 * self.krylov_size)

            if mpirank == 0:
                print(
                    f"ICWY 1S converged at iteration {stats[2]} (using {stats[0]} internal substeps and {stats[1]} rejected expm)"
                    f" to a solution with local error {stats[4]:.2e}"
                )

        # ----- icwy norm estimate + 1sync-----
        elif self.exponential_solver == "icwy_ne1s":
            z, stats = icwy_ne1s(
                self.c[0], matvec_handle, vec, tol=self.tol, m_init=self.krylov_size, mmin=16, mmax=64, task1=False
            )
            self.krylov_size = math.floor(0.7 * stats[5] + 0.3 * self.krylov_size)

            if mpirank == 0:
                print(
                    f"ICWY NE+1S converged at iteration {stats[2]} (using {stats[0]} internal substeps and {stats[1]} rejected expm)"
                    f" to a solution with local error {stats[4]:.2e}"
                )

        # ----- cwy norm estimate -----
        elif self.exponential_solver == "cwy_ne":
            z, stats = cwy_ne(
                self.c[0], matvec_handle, vec, tol=self.tol, m_init=self.krylov_size, mmin=16, mmax=64, task1=False
            )
            self.krylov_size = math.floor(0.7 * stats[5] + 0.3 * self.krylov_size)

            if mpirank == 0:
                print(
                    f"CWY NE converged at iteration {stats[2]} (using {stats[0]} internal substeps and {stats[1]} rejected expm)"
                    f" to a solution with local error {stats[4]:.2e}"
                )

        # ----- cwy 1-sync-----
        elif self.exponential_solver == "cwy_1s":
            z, stats = cwy_1s(
                self.c[0], matvec_handle, vec, tol=self.tol, m_init=self.krylov_size, mmin=16, mmax=64, task1=False
            )
            self.krylov_size = math.floor(0.7 * stats[5] + 0.3 * self.krylov_size)

            if mpirank == 0:
                print(
                    f"CWY 1S converged at iteration {stats[2]} (using {stats[0]} internal substeps and {stats[1]} rejected expm)"
                    f" to a solution with local error {stats[4]:.2e}"
                )

        # ----- cwy norm estimate + 1sync-----
        elif self.exponential_solver == "cwy_ne1s":
            z, stats = cwy_ne1s(
                self.c[0], matvec_handle, vec, tol=self.tol, m_init=self.krylov_size, mmin=16, mmax=64, task1=False
            )
            self.krylov_size = math.floor(0.7 * stats[5] + 0.3 * self.krylov_size)

            if mpirank == 0:
                print(
                    f"CWY NE+1S converged at iteration {stats[2]} (using {stats[0]} internal substeps and {stats[1]} rejected expm)"
                    f" to a solution with local error {stats[4]:.2e}"
                )

        # ----- dcgs2 -----
        elif self.exponential_solver == "dcgs2":
            z, stats = dcgs2(
                self.c[0], matvec_handle, vec, tol=self.tol, m_init=self.krylov_size, mmin=16, mmax=64, task1=False
            )
            self.krylov_size = math.floor(0.7 * stats[5] + 0.3 * self.krylov_size)

            if mpirank == 0:
                print(
                    f"DCGS2 converged at iteration {stats[2]} (using {stats[0]} internal substeps and {stats[1]} rejected expm)"
                    f" to a solution with local error {stats[4]:.2e}"
                )

        # ----else, integrator not defined---
        else:
            print("There is nothing to see here, go away!")
            exit(0)

        # Loop over all the other projections
        for i_proj in range(1, self.n_proj):

            for i in range(z.shape[0]):
                z[i, :] = Q.flatten() + dt * z[i, :]

            # Compute r(z_i)
            rz = numpy.empty_like(z)
            for i in range(z.shape[0]):
                tmp_z = numpy.reshape(z[i, :], Q.shape)
                rz[i, :] = (self.rhs(tmp_z) - rhs).flatten() - matvec_handle(tmp_z - Q) / dt

            vec = numpy.zeros((z.shape[0] + 3, rhs.size))
            vec[1, :] = rhs.flatten()
            vec[3:, :] = self.alpha[i_proj - 1] @ rz

            # ---original kiops---
            if self.exponential_solver == "kiops":
                z, stats = kiops(
                    self.c[i_proj],
                    matvec_handle,
                    vec,
                    tol=self.tol,
                    m_init=self.krylov_size,
                    mmin=16,
                    mmax=64,
                    task1=False,
                )

                if mpirank == 0:
                    print(
                        f"KIOPS converged at iteration {stats[2]} (using {stats[0]} internal substeps and {stats[1]} rejected expm)"
                        f" to a solution with local error {stats[4]:.2e}"
                    )

                self.krylov_size = math.floor(0.7 * stats[5] + 0.3 * self.krylov_size)

            # ---pmex with norm estimate---
            elif self.exponential_solver == "pmex":
                z, stats = pmex(
                    self.c[i_proj],
                    matvec_handle,
                    vec,
                    tol=self.tol,
                    m_init=self.krylov_size,
                    mmin=16,
                    mmax=64,
                    task1=False,
                )
                self.krylov_size = math.floor(0.7 * stats[5] + 0.3 * self.krylov_size)

                if mpirank == 0:
                    print(
                        f"PMEX converged at iteration {stats[2]} (using {stats[0]} internal substeps and {stats[1]} rejected expm)"
                        f" to a solution with local error {stats[4]:.2e}"
                    )

            # ----pmex with 1-sync-----
            elif self.exponential_solver == "pmex_1s":
                z, stats = pmex_1s(
                    self.c[i_proj],
                    matvec_handle,
                    vec,
                    tol=self.tol,
                    m_init=self.krylov_size,
                    mmin=16,
                    mmax=64,
                    task1=False,
                )
                self.krylov_size = math.floor(0.7 * stats[5] + 0.3 * self.krylov_size)

                if mpirank == 0:
                    print(
                        f"PMEX 1s converged at iteration {stats[2]} (using {stats[0]} internal substeps and {stats[1]} rejected expm)"
                        f" to a solution with local error {stats[4]:.2e}"
                    )

            # ----pmex with norm estimate+1s-----
            elif self.exponential_solver == "pmex_ne1s":
                z, stats = pmex_ne1s(
                    self.c[i_proj],
                    matvec_handle,
                    vec,
                    tol=self.tol,
                    m_init=self.krylov_size,
                    mmin=16,
                    mmax=64,
                    task1=False,
                )
                self.krylov_size = math.floor(0.7 * stats[5] + 0.3 * self.krylov_size)

                if mpirank == 0:
                    print(
                        f"PMEX NE+1s converged at iteration {stats[2]} (using {stats[0]} internal substeps and {stats[1]} rejected expm)"
                        f" to a solution with local error {stats[4]:.2e}"
                    )

            # ----- icwy norm estimate -----
            elif self.exponential_solver == "icwy_ne":
                z, stats = icwy_ne(
                    self.c[i_proj],
                    matvec_handle,
                    vec,
                    tol=self.tol,
                    m_init=self.krylov_size,
                    mmin=16,
                    mmax=64,
                    task1=False,
                )
                self.krylov_size = math.floor(0.7 * stats[5] + 0.3 * self.krylov_size)

                if mpirank == 0:
                    print(
                        f"ICWY NE converged at iteration {stats[2]} (using {stats[0]} internal substeps and {stats[1]} rejected expm)"
                        f" to a solution with local error {stats[4]:.2e}"
                    )

            # ----- icwy 1-sync-----
            elif self.exponential_solver == "icwy_1s":
                z, stats = icwy_1s(
                    self.c[i_proj],
                    matvec_handle,
                    vec,
                    tol=self.tol,
                    m_init=self.krylov_size,
                    mmin=16,
                    mmax=64,
                    task1=False,
                )
                self.krylov_size = math.floor(0.7 * stats[5] + 0.3 * self.krylov_size)

                if mpirank == 0:
                    print(
                        f"ICWY 1S converged at iteration {stats[2]} (using {stats[0]} internal substeps and {stats[1]} rejected expm)"
                        f" to a solution with local error {stats[4]:.2e}"
                    )

            # ----- icwy norm estimate + 1sync-----
            elif self.exponential_solver == "icwy_ne1s":
                z, stats = icwy_ne1s(
                    self.c[i_proj],
                    matvec_handle,
                    vec,
                    tol=self.tol,
                    m_init=self.krylov_size,
                    mmin=16,
                    mmax=64,
                    task1=False,
                )
                self.krylov_size = math.floor(0.7 * stats[5] + 0.3 * self.krylov_size)

                if mpirank == 0:
                    print(
                        f"ICWY NE+1S converged at iteration {stats[2]} (using {stats[0]} internal substeps and {stats[1]} rejected expm)"
                        f" to a solution with local error {stats[4]:.2e}"
                    )

            # ----- cwy norm estimate -----
            elif self.exponential_solver == "cwy_ne":
                z, stats = cwy_ne(
                    self.c[i_proj],
                    matvec_handle,
                    vec,
                    tol=self.tol,
                    m_init=self.krylov_size,
                    mmin=16,
                    mmax=64,
                    task1=False,
                )
                self.krylov_size = math.floor(0.7 * stats[5] + 0.3 * self.krylov_size)

                if mpirank == 0:
                    print(
                        f"CWY NE converged at iteration {stats[2]} (using {stats[0]} internal substeps and {stats[1]} rejected expm)"
                        f" to a solution with local error {stats[4]:.2e}"
                    )

            # ----- cwy 1-sync-----
            elif self.exponential_solver == "cwy_1s":
                z, stats = cwy_1s(
                    self.c[i_proj],
                    matvec_handle,
                    vec,
                    tol=self.tol,
                    m_init=self.krylov_size,
                    mmin=16,
                    mmax=64,
                    task1=False,
                )
                self.krylov_size = math.floor(0.7 * stats[5] + 0.3 * self.krylov_size)

                if mpirank == 0:
                    print(
                        f"CWY 1S converged at iteration {stats[2]} (using {stats[0]} internal substeps and {stats[1]} rejected expm)"
                        f" to a solution with local error {stats[4]:.2e}"
                    )

            # ----- cwy norm estimate + 1sync-----
            elif self.exponential_solver == "cwy_ne1s":
                z, stats = cwy_ne1s(
                    self.c[i_proj],
                    matvec_handle,
                    vec,
                    tol=self.tol,
                    m_init=self.krylov_size,
                    mmin=16,
                    mmax=64,
                    task1=False,
                )
                self.krylov_size = math.floor(0.7 * stats[5] + 0.3 * self.krylov_size)

                if mpirank == 0:
                    print(
                        f"CWY NE+1S converged at iteration {stats[2]} (using {stats[0]} internal substeps and {stats[1]} rejected expm)"
                        f" to a solution with local error {stats[4]:.2e}"
                    )

            # ----- dcgs2 -----
            elif self.exponential_solver == "dcgs2":
                z, stats = dcgs2(
                    self.c[i_proj],
                    matvec_handle,
                    vec,
                    tol=self.tol,
                    m_init=self.krylov_size,
                    mmin=16,
                    mmax=64,
                    task1=False,
                )
                self.krylov_size = math.floor(0.7 * stats[5] + 0.3 * self.krylov_size)

                if mpirank == 0:
                    print(
                        f"DCGS2 converged at iteration {stats[2]} (using {stats[0]} internal substeps and {stats[1]} rejected expm)"
                        f" to a solution with local error {stats[4]:.2e}"
                    )

            # ---integrator not defined
            else:
                print("There is nothing to see here, go away!")
                exit(0)

        # Update solution
        return Q + dt * numpy.reshape(z, Q.shape)
