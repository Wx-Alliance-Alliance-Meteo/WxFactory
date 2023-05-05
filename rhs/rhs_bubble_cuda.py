from typing import Union, Callable
from numpy.typing import NDArray

import cupy as cp
from cupyx import jit

from geometry.cartesian_2d_mesh import Cartesian2D
from geometry.matrices import DFROperators
from common.definitions import *

# --- Interpolate to the element interface

# TODO: could parallelize on idx instead of j
@jit.rawkernel()
def extrap_kfaces_var(kfaces_var, extrap_down, extrap_up, Q, nbsolpts, nb_elements_z):
    """
    Run with `Q.shape[0]` x `Q.shape[1] // nbsolpts` x `Q.shape[2]` total threads,
    with block size `(1, ?, 1)`
    """
    j = jit.blockIdx.y * jit.blockDim.y + jit.threadIdx.y
    if j < nb_elements_z:
        i = jit.blockIdx.x
        k = jit.blockIdx.z

        r, s = 0., 0.
        start = nbsolpts * j
        for l in jit.range(nbsolpts):
            idx = start + l
            r += extrap_down[l] * Q[i, idx, k]
            s += extrap_up[l] * Q[i, idx, k]
    
        kfaces_var[i, j, 0, k] = r
        kfaces_var[i, j, 1, k] = s

# TODO: could parallelize on idx instead of k
@jit.rawkernel()
def extrap_ifaces_var(ifaces_var, extrap_west, extrap_east, Q, nbsolpts, nb_elements_x):
    """
    Run with `Q.shape[0]` x `Q.shape[1]` x `Q.shape[2] // nbsolpts` total threads,
    with block size `(1, 1, ?)`
    """
    k = jit.blockIdx.z * jit.blockDim.z + jit.threadIdx.z
    if k < nb_elements_x:
        i = jit.blockIdx.x
        j = jit.blockIdx.y

        r, s = 0., 0.
        start = nbsolpts * k
        for l in jit.range(nbsolpts):
            idx = start + l
            r += Q[i, j, idx] * extrap_west[l]
            s += Q[i, j, idx] * extrap_east[l]

        ifaces_var[i, k, j, 0] = r
        ifaces_var[i, k, j, 1] = s

# --- Common AUSM fluxes

@jit.rawkernel()
def set_kfaces_flux(kfaces_flux, kfaces_pres, kfaces_var, nb_equations, max_j):
    """
    Run with `nb_interfaces_z - 2` x `nbsolpts * nb_elements_x` x `1` threads,
    with block size `(1, ?, 1)`
    and `max_j = nbsolpts * nb_elements_x`
    """
    j = jit.blockIdx.y * jit.blockDim.y + jit.threadIdx.y
    if j < max_j:
        i = jit.blockIdx.x
        left = i
        right = i + 1
        hcr, rho, rho_w = heat_capacity_ratio, idx_2d_rho, idx_2d_rho_w

        # Left state
        a_L = cp.sqrt(hcr * kfaces_pres[left, 1, j] / kfaces_var[rho, left, 1, j])
        M_L = kfaces_var[rho_w, left, 1, j] / (kfaces_var[rho, left, 1, j] * a_L)

        # Right state
        a_R = cp.sqrt(hcr * kfaces_pres[right, 0, j] / kfaces_var[rho, right, 0, j])
        M_R = kfaces_var[rho_w, right, 0, j] / (kfaces_var[rho, right, 0, j] * a_R)

        M = 0.25 * ((M_L + 1.) ** 2 - (M_R - 1.) ** 2)
        rho_w_extra = 0.5 * ((1. + M_L) * kfaces_pres[left, 1, j] + (1. - M_R) * kfaces_pres[right, 0, j])

        if M > 0:
            for k in jit.range(nb_equations):
                r = kfaces_var[k, left, 1, j] * M * a_L
                if k == rho_w:
                    r += rho_w_extra
                kfaces_flux[k, right, 0, j] = r
                kfaces_flux[k, left, 1, j] = r
        else:
            for k in jit.range(nb_equations):
                r = kfaces_var[k, right, 0, j] * M * a_R
                if k == rho_w:
                    r += rho_w_extra
                kfaces_flux[k, right, 0, j] = r
                kfaces_flux[k, left, 1, j] = r

@jit.rawkernel()
def set_ifaces_flux(ifaces_flux, ifaces_pres, ifaces_var, nb_equations, max_j, xperiodic):
    """
    Run with `nb_interfaces_x - 2` x `nbsolpts * nb_elements_z` x `1` threads,
    with block size `(1, ?, 1)`
    and `max_j = nbsolpts * nb_elements_z`
    """
    j = jit.blockIdx.y * jit.blockDim.y + jit.threadIdx.y
    if j < max_j:
        i = jit.blockIdx.x
        left = i
        right = i + 1
        hcr, rho, rho_u = heat_capacity_ratio, idx_2d_rho, idx_2d_rho_u

        # Left state
        if xperiodic and left == 0:
            left = -1
        
        a_L = cp.sqrt(hcr * ifaces_pres[left, j, 1] / ifaces_var[rho, left, j, 1])
        M_L = ifaces_var[rho_u, left, j, 1] / (ifaces_var[rho, left, j, 1] * a_L)

        # Right state
        a_R = cp.sqrt(hcr * ifaces_pres[right, j, 0] / ifaces_var[rho, right, j, 0])
        M_R = ifaces_var[rho_u, right, j, 0] / (ifaces_var[rho, right, j, 0] * a_R)

        M = 0.25 * ((M_L + 1.) ** 2 - (M_R - 1.) ** 2)
        rho_u_extra = 0.5 * ((1. + M_L) * ifaces_pres[left, j, 1] + (1. - M_R) * ifaces_pres[right, j, 0])

        if M > 0:
            for k in jit.range(nb_equations):
                r = ifaces_var[k, left, j, 1] * M * a_L
                if k == rho_u:
                    r += rho_u_extra
                ifaces_flux[k, right, j, 0] = r
                ifaces_flux[k, left, j, 1] = r
        else:
            for k in jit.range(nb_equations):
                r = ifaces_var[k, right, j, 0] * M * a_R
                if k == rho_u:
                    r += rho_u_extra
                ifaces_flux[k, right, j, 0] = r
                ifaces_flux[k, left, j, 1] = r

# -- Compute the derivatives

# TODO: could parallelize on idx instead of j
@jit.rawkernel()
def set_df3_dx3(df3_dx3, flux_x3, kfaces_flux, diff_solpt, correction, dx3, nbsolpts, nb_elements_z):
    """
    Run with `Q.shape[0]` x `Q.shape[1] // nbsolpts` x `Q.shape[2]` total threads,
    with block size `(1, ?, 1)`
    """
    j = jit.blockIdx.y * jit.blockDim.y + jit.threadIdx.y
    if j < nb_elements_z:
        i = jit.blockIdx.x
        k = jit.blockIdx.z

        start = nbsolpts * j
        for l in jit.range(nbsolpts):
            r = 0.0

            for m in jit.range(nbsolpts):
                r += diff_solpt[l, m] * flux_x3[i, start + m, k]
            for m in jit.range(2):
                r += correction[l, m] * kfaces_flux[i, j, m, k]
            
            df3_dx3[i, start + l, k] = 2.0 * r / dx3

# TODO: could parallelize on idx instead of k
@jit.rawkernel()
def set_df1_dx1(df1_dx1, flux_x1, ifaces_flux, diff_solpt, correction, dx1, nbsolpts, nb_elements_x):
    """
    Run with `Q.shape[0]` x `Q.shape[1]` x `Q.shape[2] // nbsolpts` total threads,
    with block size `(1, 1, ?)`
    note that diff_solpt and correction are automatically transposed
    """
    k = jit.blockIdx.z * jit.blockDim.z + jit.threadIdx.z
    if k < nb_elements_x:
        i = jit.blockIdx.x
        j = jit.blockIdx.y

        start = nbsolpts * k
        for l in jit.range(nbsolpts):
            r = 0.0

            for m in jit.range(nbsolpts):
                r += flux_x1[i, j, start + m] * diff_solpt[l, m]
            for m in jit.range(2):
                r += ifaces_flux[i, k, j, m] * correction[l, m]
            
            df1_dx1[i, j, start + l] = 2.0 * r / dx1

@profile
def rhs_bubble_cuda(Q: NDArray[cp.floating], geom: Cartesian2D, mtrx: DFROperators,
    nbsolpts: int, nb_elements_x: int, nb_elements_z: int) -> NDArray[cp.floating]:

    # TODO: have Q on GPU
    Q = cp.asarray(Q)
    # TODO: make threads per block a global
    B = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())["maxThreadsPerBlock"]

    datatype = Q.dtype
    nb_equations = Q.shape[0] # Number of constituent Euler equations.  Probably 6

    nb_interfaces_x = nb_elements_x + 1
    nb_interfaces_z = nb_elements_z + 1

    flux_x1 = cp.empty_like(Q, dtype=datatype)
    flux_x3 = cp.empty_like(Q, dtype=datatype)

    df1_dx1 = cp.empty_like(Q, dtype=datatype)
    df3_dx3 = cp.empty_like(Q, dtype=datatype)

    kfaces_flux = cp.empty((nb_equations, nb_elements_z, 2, nbsolpts * nb_elements_x), dtype=datatype)
    kfaces_var = cp.empty((nb_equations, nb_elements_z, 2, nbsolpts * nb_elements_z), dtype=datatype)

    ifaces_flux = cp.empty((nb_equations, nb_elements_x, nbsolpts * nb_elements_z, 2), dtype=datatype)
    ifaces_var = cp.empty((nb_equations, nb_elements_x, nbsolpts * nb_elements_z, 2), dtype=datatype)

    # TODO: have mtrx on GPU
    if type(mtrx.extrap_down) != cp.ndarray:
        mtrx.extrap_down = cp.asarray(mtrx.extrap_down)
        mtrx.extrap_up   = cp.asarray(mtrx.extrap_up)
        mtrx.extrap_west = cp.asarray(mtrx.extrap_west)
        mtrx.extrap_east = cp.asarray(mtrx.extrap_east)
        mtrx.correction  = cp.asarray(mtrx.correction)
        mtrx.diff_solpt  = cp.asarray(mtrx.diff_solpt)

    # --- Unpack physical variables
    rho      = Q[idx_2d_rho, :, :]
    uu       = Q[idx_2d_rho_u, :, :] / rho
    ww       = Q[idx_2d_rho_w, :, :] / rho
    pressure = p0 * (Q[idx_2d_rho_theta, :, :] * Rd / p0) ** (cpd / cvd)

    # --- Compute the fluxes
    flux_x1[idx_2d_rho, :, :]       = Q[idx_2d_rho_u, :, :]
    flux_x1[idx_2d_rho_u, :, :]     = Q[idx_2d_rho_u, :, :] * uu + pressure
    flux_x1[idx_2d_rho_w, :, :]     = Q[idx_2d_rho_u, :, :] * ww
    flux_x1[idx_2d_rho_theta, :, :] = Q[idx_2d_rho_theta, :, :] * uu

    flux_x3[idx_2d_rho, :, :]       = Q[idx_2d_rho_w, :, :]
    flux_x3[idx_2d_rho_u, :, :]     = Q[idx_2d_rho_w, :, :] * uu
    flux_x3[idx_2d_rho_w, :, :]     = Q[idx_2d_rho_w, :, :] * ww + pressure
    flux_x3[idx_2d_rho_theta, :, :] = Q[idx_2d_rho_theta, :, :] * ww

    # --- Interpolate to the element interface
    blocks, extra = divmod(Q.shape[1], B)
    gridspec = (Q.shape[0], blocks + (1 if extra else 0), Q.shape[2])
    blockspec = (1, B, 1)
    extrap_kfaces_var[gridspec, blockspec](kfaces_var, mtrx.extrap_down, mtrx.extrap_up, Q, nbsolpts, nb_elements_z)

    blocks, extra = divmod(Q.shape[2], B)
    gridspec = (Q.shape[0], Q.shape[1], blocks + (1 if extra else 0))
    blockspec = (1, 1, B)
    extrap_ifaces_var[gridspec, blockspec](ifaces_var, mtrx.extrap_west, mtrx.extrap_east, Q, nbsolpts, nb_elements_x)

    # --- Interface pressure
    ifaces_pres = p0 * (ifaces_var[idx_2d_rho_theta, :, :, :] * Rd / p0) ** (cpd / cvd)
    kfaces_pres = p0 * (kfaces_var[idx_2d_rho_theta, :, :, :] * Rd / p0) ** (cpd / cvd)

    # --- Boundary treatment

    # zero flux BCs everywhere...
    kfaces_flux[:, 0, 0, :] = 0.0
    kfaces_flux[:, -1, 1, :] = 0.0

    # Skip periodic faces
    if not geom.xperiodic:
        ifaces_flux[:, 0, :, 0] = 0.0
        ifaces_flux[:, -1, :, 1] = 0.0

    # except for momentum eqs where pressure is extrapolated to BCs.
    kfaces_flux[idx_2d_rho_w, 0, 0, :] = kfaces_pres[0, 0, :]
    kfaces_flux[idx_2d_rho_w, -1, 1, :] = kfaces_pres[-1, 1, :]

    ifaces_flux[idx_2d_rho_u, 0, :, 0] = ifaces_pres[0, :, 0] # TODO: pour les cas théoriques seulement...
    ifaces_flux[idx_2d_rho_u, -1, :, 1] = ifaces_pres[-1, :, 1]

    # --- Common AUSM fluxes
    blocks, extra = divmod(nbsolpts * nb_elements_x, B)
    gridspec = (nb_interfaces_z - 2, blocks + (1 if extra else 0), 1)
    blockspec = (1, B, 1)
    set_kfaces_flux[gridspec, blockspec](kfaces_flux, kfaces_pres, kfaces_var, nb_equations, nbsolpts * nb_elements_x)

    if geom.xperiodic:
        ifaces_var[:, 0, :, :] = ifaces_var[:, -1, :, :]
        ifaces_pres[0, :, :] = ifaces_pres[-1, :, :]
    
    blocks, extra = divmod(nbsolpts * nb_elements_z, B)
    gridspec = (nb_interfaces_x - 2, blocks + (1 if extra else 0), 1)
    blockspec = (1, B, 1)
    set_ifaces_flux[gridspec, blockspec](ifaces_flux, ifaces_pres, ifaces_var, nb_equations, nbsolpts * nb_elements_z, geom.xperiodic)

    if geom.xperiodic:
        ifaces_flux[:, 0, :, :] = ifaces_flux[:, -1, :, :]

    # --- Compute the derivatives
    blocks, extra = divmod(Q.shape[1], B)
    gridspec = (Q.shape[0], blocks + (1 if extra else 0), Q.shape[2])
    blockspec = (1, B, 1)
    set_df3_dx3[gridspec, blockspec](df3_dx3, flux_x3, kfaces_flux, mtrx.diff_solpt, mtrx.correction, geom.Δx3, nbsolpts, nb_elements_z)

    blocks, extra = divmod(Q.shape[2], B)
    gridspec = (Q.shape[0], Q.shape[1], blocks + (1 if extra else 0))
    blockspec = (1, 1, B)
    set_df1_dx1[gridspec, blockspec](df1_dx1, flux_x1, ifaces_flux, mtrx.diff_solpt, mtrx.correction, geom.Δx1, nbsolpts, nb_elements_x)

    # --- Assemble the right-hand sides
    rhs = -(df1_dx1 + df3_dx3)
    rhs[idx_2d_rho_w, :, :] -= Q[idx_2d_rho, :, :] * gravity

    return rhs.get()
