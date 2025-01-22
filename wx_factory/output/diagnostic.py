from mpi4py import MPI
import numpy
from numpy.typing import NDArray

from common.definitions import gravity
from geometry import DFROperators, Metric2D


def relative_vorticity(u1_contra: NDArray, u2_contra: NDArray, metric: Metric2D, mtrx: DFROperators) -> NDArray:

    u1_dual = metric.H_cov_11 * u1_contra + metric.H_cov_12 * u2_contra
    u2_dual = metric.H_cov_21 * u1_contra + metric.H_cov_22 * u2_contra

    du1dx2 = u2_dual @ mtrx.derivative_y
    du2dx1 = u1_dual @ mtrx.derivative_x

    vort = metric.inv_sqrtG * (du2dx1 - du1dx2)

    return vort


def potential_vorticity(
    h: NDArray, u1_contra: NDArray, u2_contra: NDArray, metric: Metric2D, mtrx: DFROperators
) -> NDArray:

    rv = relative_vorticity(u1_contra, u2_contra, metric, mtrx)

    return (rv + metric.coriolis_f) / h


def absolute_vorticity(u1_contra: NDArray, u2_contra: NDArray, metric: Metric2D, mtrx: DFROperators):

    rv = relative_vorticity(u1_contra, u2_contra, metric, mtrx)

    return rv + metric.coriolis_f


def total_energy(h, u1_contra, u2_contra, topo, metric):
    u1_dual = metric.H_cov_11 * u1_contra + metric.H_cov_12 * u2_contra
    u2_dual = metric.H_cov_21 * u1_contra + metric.H_cov_22 * u2_contra

    # Kinetic energy
    kinetic = 0.5 * h * (u1_dual * u1_contra + u2_dual * u2_contra)

    # Potential energy
    if topo is not None:
        potential = 0.5 * gravity * ((h + topo.hsurf) ** 2 - topo.hsurf**2)
    else:
        potential = 0.5 * gravity * h**2

    # "Total" energy
    return kinetic + potential


def potential_enstrophy(h, u1_contra, u2_contra, geom, metric, mtrx, param):
    rv = relative_vorticity(u1_contra, u2_contra, metric, mtrx)
    return (rv + metric.coriolis_f) ** 2 / (2 * h)


def global_integral_2d(field: NDArray, mtrx: DFROperators, metric, num_solpts: int):

    ny, nx = field.shape[:2]
    local_sum = numpy.sum((field * metric.sqrtG).reshape(ny, nx, num_solpts, num_solpts) * mtrx.quad_weights)

    return MPI.COMM_WORLD.allreduce(local_sum)
