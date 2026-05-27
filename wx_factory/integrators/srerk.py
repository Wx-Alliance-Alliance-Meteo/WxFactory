import math
from typing import Callable, List, Optional

from mpi4py import MPI
import numpy

from ..common.configuration import Configuration
from ..solvers import (
    kiops,
    matvec_fun,
    pmex,
)

from .integrator import Integrator, alpha_coeff


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


class Srerk(Integrator):
    """Stiffness resilient exponential Runge-Kutta methods"""

    def __init__(self, param: Configuration, order: int, rhs: Callable, jac: Callable = None, nodes: Optional[List] = None, **kwargs):
        """
        If the nodes are NOT specified, return the SRERK method of the specified order with min error terms
        If the nodes are specified, return the SRERK method with these nodes and ignore the 'order' parameter
        """

        super().__init__(param, **kwargs)
        self.rhs = rhs
        self.jac = jac
        self.tol = param.tolerance
        self.krylov_size = 1
        self.jacobian_method = param.jacobian_method
        self.exponential_solver = param.exponential_solver

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
        if self.jac is not None:
            matvec_handle = lambda v: self.jac(v, Q, dt)
        else:
            matvec_handle = lambda v: matvec_fun(v, dt, Q, rhs, self.rhs, self.jacobian_method)

        # Initial projection
        vec = numpy.zeros((2, rhs.size))
        vec[1, :] = rhs.flatten()

        # for printing stats
        mpirank = self.device.comm.rank

        # ---original kiops---
        if self.exponential_solver == "kiops":
            z, stats = kiops(
                self.c[0],
                matvec_handle,
                vec,
                tol=self.tol,
                m_init=self.krylov_size,
                mmin=16,
                mmax=64,
                task1=False,
                device=self.device,
            )

            if mpirank == 0:
                print(
                    f"KIOPS converged at iteration {stats[2]} (using {stats[0]} internal substeps "
                    f"and {stats[1]} rejected expm)"
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
                    f"PMEX converged at iteration {stats[2]} (using {stats[0]} internal substeps "
                    f"and {stats[1]} rejected expm)"
                    f" to a solution with local error {stats[4]:.2e}"
                )

        # ----else, integrator not defined---
        else:
            raise ValueError("Unrecognized solver {self.exponential_solver}")

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
                        f"KIOPS converged at iteration {stats[2]} (using {stats[0]} internal substeps "
                        f"and {stats[1]} rejected expm)"
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
                        f"PMEX converged at iteration {stats[2]} (using {stats[0]} internal substeps "
                        f"and {stats[1]} rejected expm)"
                        f" to a solution with local error {stats[4]:.2e}"
                    )

            # ---integrator not defined
            else:
                raise ValueError("Unrecognized solver {self.exponential_solver}")

        # Update solution
        return Q + dt * numpy.reshape(z, Q.shape)


def _make_srerk_factory(order):
    return lambda cfg, rhs, prec, dev: Srerk(cfg, order, rhs.full, device=dev)


REGISTRY = {f"srerk{o}": _make_srerk_factory(o) for o in range(3, 10)}
