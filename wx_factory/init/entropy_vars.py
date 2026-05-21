import numpy
from numpy.typing import NDArray

from common.definitions import (
    idx_rho,
    idx_rho_u1,
    idx_rho_u2,
    idx_rho_w,
    idx_rho_theta,
    idx_h,
    idx_u1,
    idx_u2,
    idx_hu1,
    idx_hu2,
    idx_2d_rho,
    idx_2d_rho_u,
    idx_2d_rho_w,
    idx_2d_rho_theta,
    gravity,
    cpd,
    cvd,
    Rd,
    p0,
)
from common import Configuration
from common.graphx import plot_array
from geometry import Cartesian2D, CubedSphere3D, CubedSphere2D, DFROperators, Metric2D, Metric3DTopo


def conservative_to_entropy(Q, geom, param):
    xp = geom.device.xp
    V = xp.zeros_like(Q)

    ρ, ρu, ρw, ρE, u, w, p, ρe, e = conservative_to_prim(Q)

    gamma = cpd / cvd
    s = xp.log(p) - gamma * xp.log(ρ)

    beta = ρ / p

    V[idx_2d_rho] = (gamma - s) / (gamma - 1.0) - 0.5 * beta * (u**2 + w**2)
    V[idx_2d_rho_u] = beta * u
    V[idx_2d_rho_w] = beta * w
    V[idx_2d_rho_theta] = -beta

    return V


def conservative_to_prim(Q):
    if len(Q.shape) == 4:
        ρ = Q[idx_2d_rho, :, :, :]
        ρu = Q[idx_2d_rho_u, :, :, :]
        ρw = Q[idx_2d_rho_w, :, :, :]
        ρE = Q[idx_2d_rho_theta, :, :, :]  # fourth slot is rho_E
    elif len(Q.shape) == 3:
        ρ = Q[idx_2d_rho, :, :]
        ρu = Q[idx_2d_rho_u, :, :]
        ρw = Q[idx_2d_rho_w, :, :]
        ρE = Q[idx_2d_rho_theta, :, :]  # fourth slot is rho_E

    u = ρu / ρ
    w = ρw / ρ

    kinetic = 0.5 * ρ * (u**2 + w**2)
    ρe = ρE - kinetic
    e = ρe / ρ

    gamma = cpd / cvd
    p = (gamma - 1.0) * ρe

    return ρ, ρu, ρw, ρE, u, w, p, ρe, e

def entropy_to_conservative(V: NDArray, geom: Cartesian2D, param: Configuration) -> NDArray[numpy.float64]:
    """Compute conservative variable from entropy variables"""
    # TODO: Check this function!!!!!!!!!

    num_equations = 4
    xp = geom.device.xp

    gamma = cpd / cvd

    # V = (gamma - 1) * V

    v1 = V[idx_2d_rho, :, :]
    v2 = V[idx_2d_rho_u, :, :]
    v3 = V[idx_2d_rho_w, :, :]
    v4 = V[idx_2d_rho_theta, :, :]

    # _________________________________________________
    beta = -v4

    uu = v2 / beta
    ww = v3 / beta

    s = gamma - (gamma - 1) * v1 - 0.5 * (gamma - 1) * beta * (uu**2 + ww**2)

    ρ = xp.exp(-s/(gamma-1)) * xp.power(beta,-1/(gamma-1))

    p = ρ/beta
    ρ_uu = ρ * uu
    ρ_ww = ρ * ww
    ρ_E = p/(gamma-1) + 0.5 * ρ * (uu**2 + ww**2)
    # _________________________________________________

    # s = gamma - v1 + (v2**2 + v3**2) / (2 * v4)

    # # Check this formula!!!!
    # ρ_e = ((gamma - 1) / (-v4) ** gamma) ** (1 / (gamma - 1)) * xp.exp(-s / (gamma - 1))

    # ρ = -ρ_e * v4
    # ρ_uu = ρ_e * v2
    # ρ_ww = ρ_e * v3
    # ρ_E = ρ_e * (1 - (v2**2 + v3**2) / (2 * v4))

    Q = xp.zeros_like(V)

    Q[idx_2d_rho, :, :] = ρ
    Q[idx_2d_rho_u, :, :] = ρ_uu
    Q[idx_2d_rho_w, :, :] = ρ_ww
    Q[idx_2d_rho_theta, :, :] = ρ_E

    return Q


def du_dv(Q: NDArray, geom: Cartesian2D, param: Configuration):
    """Computes matrix du_dv.
    Shape : (num_equations, num_equations, param.num_elements_vertical, param.num_elements_horizontal, geom.num_solpts**2)
    """
    # Possible issues: diviosion by zero
    num_equations = 4
    xp = geom.device.xp

    if len(Q.shape) == 4:
        K = xp.zeros(
            (
                num_equations,
                num_equations,
                param.num_elements_vertical,
                param.num_elements_horizontal,
                geom.num_solpts**2,
            )
        )
    elif len(Q.shape) == 3:
        K = xp.zeros((num_equations, num_equations, param.num_elements_vertical, param.num_elements_horizontal))

    ρ, ρu, ρw, ρE, u, w, p, ρe, e = conservative_to_prim(Q)

    gamma = cpd / cvd

    k00 = ρ
    k01 = ρu
    k02 = ρw
    k03 = ρE

    k11 = ρu * u + p
    k12 = ρu * w
    k13 = u * (ρE + p)

    k22 = ρw * w + p
    k23 = w * (ρE + p)

    a2 = gamma * p / ρ
    H = (ρE + p) / ρ
    k33 = ρ * H**2 - a2 * p / (gamma - 1.0)

    if len(Q.shape) == 4:
        K[0, 0, :, :, :] = k00
        K[0, 1, :, :, :] = k01
        K[0, 2, :, :, :] = k02
        K[0, 3, :, :, :] = k03

        K[1, 0, :, :, :] = k01
        K[1, 1, :, :, :] = k11
        K[1, 2, :, :, :] = k12
        K[1, 3, :, :, :] = k13

        K[2, 0, :, :, :] = k02
        K[2, 1, :, :, :] = k12
        K[2, 2, :, :, :] = k22
        K[2, 3, :, :, :] = k23

        K[3, 0, :, :, :] = k03
        K[3, 1, :, :, :] = k13
        K[3, 2, :, :, :] = k23
        K[3, 3, :, :, :] = k33

    elif len(Q.shape) == 3:
        K[0, 0, :, :] = k00
        K[0, 1, :, :] = k01
        K[0, 2, :, :] = k02
        K[0, 3, :, :] = k03

        K[1, 0, :, :] = k01
        K[1, 1, :, :] = k11
        K[1, 2, :, :] = k12
        K[1, 3, :, :] = k13

        K[2, 0, :, :] = k02
        K[2, 1, :, :] = k12
        K[2, 2, :, :] = k22
        K[2, 3, :, :] = k23

        K[3, 0, :, :] = k03
        K[3, 1, :, :] = k13
        K[3, 2, :, :] = k23
        K[3, 3, :, :] = k33

    return K


def jacobian_complex_field(func, Q, geom, param, h=1e-20):
    xp = geom.device.xp

    Q = xp.asarray(Q, dtype=float)
    nvar = Q.shape[0]
    spatial_shape = Q.shape[1:]

    J = xp.zeros((nvar, nvar) + spatial_shape, dtype=float)
    Qc = Q.astype(complex)

    for i in range(nvar):
        Q_step = Qc.copy()
        Q_step[i, ...] += 1j * h
        f_step = func(Q_step, geom, param)
        J[:, i, ...] = xp.imag(f_step) / h

    return J


def jacobian_fd_field(func, Q, geom, param, eps=1e-6):
    xp = geom.device.xp
    nvar = Q.shape[0]
    spatial_shape = Q.shape[1:]

    J = xp.zeros((nvar, nvar) + spatial_shape)

    for i in range(nvar):
        Qp = Q.copy()
        Qm = Q.copy()
        Qp[i, ...] += eps
        Qm[i, ...] -= eps

        fp = func(Qp, geom, param)
        fm = func(Qm, geom, param)

        J[:, i, ...] = (fp - fm) / (2 * eps)

    return J


def entropy_potential(Q: NDArray) -> NDArray[numpy.float64]:
    psi_x1 = Q[idx_2d_rho_u, :, :, :]
    psi_x2 = Q[idx_2d_rho_w, :, :, :]
    return psi_x1, psi_x2


def entropy(Q: NDArray, geom: Cartesian2D) -> NDArray[numpy.float64]:
    """Computes physical entropy s = log(p / rho**gamma) for total-energy variables.

    Q = (rho, rho*u, rho*w, rho*E)
    """
    xp = geom.device.xp

    ρ, ρu, ρw, ρE, u, w, p, ρe, e = conservative_to_prim(Q)

    gamma = cpd / cvd

    # Optional safety if you want to avoid log of non-positive values:
    # p = xp.maximum(p, 1e-30)
    # ρ = xp.maximum(ρ, 1e-30)

    s = xp.log(p) - gamma * xp.log(ρ)
    return s


def entropy_function(Q: NDArray, geom: Cartesian2D) -> NDArray[numpy.float64]:
    "Computes mathematical entropy function  S(u) = -rho*s"

    # ρ, _, _, ρ_θ, _ , _, _ = conservative_to_prim(Q)

    ρ, ρu, ρw, ρE, u, w, p, ρe, e = conservative_to_prim(Q)
    s = entropy(Q, geom)
    return -ρ * s
