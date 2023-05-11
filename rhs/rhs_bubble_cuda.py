import cupy as cp
from cupyx import jit
from common.cuda_module import CudaModule, DimSpec, Dim, cuda_kernel

from geometry.cartesian_2d_mesh import Cartesian2D
from geometry.matrices import DFROperators
from common.definitions import *

from numpy.typing import NDArray

def _dim_kfaces_var(kfaces_var: NDArray[cp.float64], Q: NDArray[cp.float64], extrap_up: NDArray[cp.float64], extrap_down: NDArray[cp.float64],
                    nbsolpts: int, nb_elements_z: int, nb_elements_x: int) \
                    -> tuple[Dim, Dim]:
    return DimSpec.groupby_first_x(lambda s: Dim(s[3], s[1], s[0])) \
        (kfaces_var, Q, extrap_up, extrap_down, nbsolpts, nb_elements_z, nb_elements_x)

def _dim_ifaces_var(ifaces_var: NDArray[cp.float64], Q: NDArray[cp.float64], extrap_west: NDArray[cp.float64], extrap_east: NDArray[cp.float64],
                    nbsolpts: int, nb_elements_z: int, nb_elements_x: int) \
                    -> tuple[Dim, Dim]:
    return DimSpec.groupby_first_x(lambda s: Dim(s[2], s[1], s[0])) \
        (ifaces_var, Q, extrap_west, extrap_east, nbsolpts, nb_elements_z, nb_elements_x)

def _dim_kfaces_flux(kfaces_flux: NDArray[cp.float64], kfaces_pres: NDArray[cp.float64], kfaces_var: NDArray[cp.float64],
                     nb_equations: int, nbsolpts: int, nb_elements_z: int, nb_elements_x: int) \
                     -> tuple[Dim, Dim]:
    return DimSpec.groupby_first_x(lambda s: Dim(s[3], s[1] - 1, 1)) \
        (kfaces_flux, kfaces_pres, kfaces_var, nb_equations, nbsolpts, nb_elements_z, nb_elements_x)

def _dim_ifaces_flux(ifaces_flux: NDArray[cp.float64], ifaces_pres: NDArray[cp.float64], ifaces_var: NDArray[cp.float64],
                     nb_equations: int, nbsolpts: int, nb_elements_z: int, nb_elements_x: int, xperiodic: bool) \
                     -> tuple[Dim, Dim]:
    return DimSpec.groupby_first_x(lambda s: Dim(s[2], s[1] - 1, 1)) \
        (ifaces_flux, ifaces_pres, ifaces_var, nb_equations, nbsolpts, nb_elements_z, nb_elements_x, xperiodic)

def _dim_df3_dx3(df3_dx3: NDArray[cp.float64], flux_x3: NDArray[cp.float64], kfaces_flux: NDArray[cp.float64],
                 diff_solpt: NDArray[cp.float64], correction: NDArray[cp.float64],
                 dx3: float, nbsolpts: int, nb_elements_z: int, nb_elements_x: int) \
                 -> tuple[Dim, Dim]:
    return DimSpec.groupby_first_x(lambda s: Dim(s[2], s[1], s[0])) \
        (df3_dx3, flux_x3, kfaces_flux, diff_solpt, correction, dx3, nbsolpts, nb_elements_z, nb_elements_x)

def _dim_df1_dx1(df1_dx1: NDArray[cp.float64], flux_x1: NDArray[cp.float64], ifaces_flux: NDArray[cp.float64],
                 diff_solpt: NDArray[cp.float64], correction: NDArray[cp.float64],
                 dx1: float, nbsolpts: int, nb_elements_z: int, nb_elements_x: int) \
                 -> tuple[Dim, Dim]:
    return DimSpec.groupby_first_x(lambda s: Dim(s[2], s[1], s[0])) \
        (df1_dx1, flux_x1, ifaces_flux, diff_solpt, correction, dx1, nbsolpts, nb_elements_z, nb_elements_x)

class RHSBubble(CudaModule,
                path="rhs_bubble.cu",
                defines=(("heat_capacity_ratio", heat_capacity_ratio),
                         ("idx_2d_rho", idx_2d_rho),
                         ("idx_2d_rho_u", idx_2d_rho_u),
                         ("idx_2d_rho_w", idx_2d_rho_w))):

    @cuda_kernel(_dim_kfaces_var)
    def set_kfaces_var(kfaces_var: NDArray[cp.float64], Q: NDArray[cp.float64], extrap_up: NDArray[cp.float64], extrap_down: NDArray[cp.float64],
                       nbsolpts: int, nb_elements_z: int, nb_elements_x: int): ...
    
    @cuda_kernel(_dim_ifaces_var)
    def set_ifaces_var(ifaces_var: NDArray[cp.float64], Q: NDArray[cp.float64], extrap_west: NDArray[cp.float64], extrap_east: NDArray[cp.float64],
                       nbsolpts: int, nb_elements_z: int, nb_elements_x: int): ...

    @cuda_kernel(_dim_kfaces_flux)
    def set_kfaces_flux(kfaces_flux: NDArray[cp.float64], kfaces_pres: NDArray[cp.float64], kfaces_var: NDArray[cp.float64],
                        nb_equations: int, nbsolpts: int, nb_elements_z: int, nb_elements_x: int): ...
    
    @cuda_kernel(_dim_ifaces_flux)
    def set_ifaces_flux(ifaces_flux: NDArray[cp.float64], ifaces_pres: NDArray[cp.float64], ifaces_var: NDArray[cp.float64],
                        nb_equations: int, nbsolpts: int, nb_elements_z: int, nb_elements_x: int, xperiodic: bool): ...

    @cuda_kernel(_dim_df3_dx3)
    def set_df3_dx3(df3_dx3: NDArray[cp.float64], flux_x3: NDArray[cp.float64], kfaces_flux: NDArray[cp.float64],
                    diff_solpt: NDArray[cp.float64], correction: NDArray[cp.float64],
                    dx3: float, nbsolpts: int, nb_elements_z: int, nb_elements_x: int): ...
    
    @cuda_kernel(_dim_df1_dx1)
    def set_df1_dx1(df1_dx1: NDArray[cp.float64], flux_x1: NDArray[cp.float64], ifaces_flux: NDArray[cp.float64],
                    diff_solpt: NDArray[cp.float64], correction: NDArray[cp.float64],
                    dx1: float, nbsolpts: int, nb_elements_z: int, nb_elements_x: int): ...

# @profile
def rhs_bubble_cuda(Q: NDArray[cp.floating], geom: Cartesian2D, mtrx: DFROperators,
    nbsolpts: int, nb_elements_x: int, nb_elements_z: int) -> NDArray[cp.floating]:

    # TODO: have Q on GPU
    Q = cp.asarray(Q)
    # TODO: make threads per block a global
    B = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())["maxThreadsPerBlock"] // 16

    datatype = Q.dtype
    nb_equations = Q.shape[0] # Number of constituent Euler equations.  Probably 6

    flux_x1 = cp.empty_like(Q, dtype=datatype)
    flux_x3 = cp.empty_like(Q, dtype=datatype)

    df1_dx1 = cp.empty_like(Q, dtype=datatype)
    df3_dx3 = cp.empty_like(Q, dtype=datatype)

    kfaces_flux = cp.empty((nb_equations, nb_elements_z, 2, nbsolpts * nb_elements_x), dtype=datatype)
    kfaces_var = cp.empty((nb_equations, nb_elements_z, 2, nbsolpts * nb_elements_x), dtype=datatype)

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

    RHSBubble.set_kfaces_var(kfaces_var, Q, mtrx.extrap_up, mtrx.extrap_down, nbsolpts, nb_elements_z, nb_elements_x)

    RHSBubble.set_ifaces_var(ifaces_var, Q, mtrx.extrap_west, mtrx.extrap_east, nbsolpts, nb_elements_z, nb_elements_x)

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

    RHSBubble.set_kfaces_flux(kfaces_flux, kfaces_pres, kfaces_var, nb_equations, nbsolpts, nb_elements_z, nb_elements_x)

    if geom.xperiodic:
        ifaces_var[:, 0, :, :] = ifaces_var[:, -1, :, :]
        ifaces_pres[0, :, :] = ifaces_pres[-1, :, :]
    
    RHSBubble.set_ifaces_flux(ifaces_flux, ifaces_pres, ifaces_var, nb_equations, nbsolpts, nb_elements_z, nb_elements_x, geom.xperiodic)
    
    if geom.xperiodic:
        ifaces_flux[:, 0, :, :] = ifaces_flux[:, -1, :, :]

    # --- Compute the derivatives
    
    RHSBubble.set_df3_dx3(df3_dx3, flux_x3, kfaces_flux, mtrx.diff_solpt, mtrx.correction,
                          geom.Δx3, nbsolpts, nb_elements_z, nb_elements_x)

    RHSBubble.set_df1_dx1(df1_dx1, flux_x1, ifaces_flux, mtrx.diff_solpt, mtrx.correction,
                          geom.Δx1, nbsolpts, nb_elements_z, nb_elements_x)
    
    # --- Assemble the right-hand sides
    rhs = -(df1_dx1 + df3_dx3)
    rhs[idx_2d_rho_w, :, :] -= Q[idx_2d_rho, :, :] * gravity

    return rhs.get()
