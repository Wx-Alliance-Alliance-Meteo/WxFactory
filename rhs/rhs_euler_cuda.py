import numpy as np
import cupy as cp

from common.definitions import idx_rho_u1, idx_rho_u2, idx_rho_w, idx_rho, idx_rho_theta, gravity, p0, Rd, cpd, cvd, heat_capacity_ratio
from common.cuda_module import CudaModule, DimSpec, Dim, cuda_kernel
from init.dcmip import dcmip_schar_damping

# For type hints
from common.cuda_parallel import CudaDistributedWorld
from geometry import CubedSphere
from geometry.cuda_matrices import CudaDFROperators
from geometry.cuda_metric import CudaMetric3DTopo

from numpy.typing import NDArray


class RHSEuler(CudaModule,
               path="rhs_euler.cu",
               defines=(("heat_capacity_ratio", heat_capacity_ratio),
                        ("idx_rho", idx_rho),
                        ("idx_rho_u1", idx_rho_u1),
                        ("idx_rho_u2", idx_rho_u2),
                        ("idx_rho_w", idx_rho_w))):
    
    @cuda_kernel(DimSpec.groupby_first_x(lambda s: Dim(s[3], s[1], s[2] - 1)))
    def compute_flux_i(flux_x1_itf_i: NDArray[cp.float64], wflux_adv_x1_itf_i: NDArray[cp.float64], wflux_pres_x1_itf_i: NDArray[cp.float64],
                       variables_itf_i: NDArray[cp.float64], pressure_itf_i: NDArray[cp.float64], u1_itf_i: NDArray[cp.float64],
                       sqrtG_itf_i: NDArray[cp.float64], H_contra_1x_itf_i: NDArray[cp.float64],
                       nh: int, nk_nh: int, nk_nv: int, ne: int, advection_only: bool): ...
    
    @cuda_kernel(DimSpec.groupby_first_x(lambda s: Dim(s[4], s[1], s[2] - 1)))
    def compute_flux_j(flux_x2_itf_j: NDArray[cp.float64], wflux_adv_x2_itf_j: NDArray[cp.float64], wflux_pres_x2_itf_j: NDArray[cp.float64],
                       variables_itf_j: NDArray[cp.float64], pressure_itf_j: NDArray[cp.float64], u2_itf_j: NDArray[cp.float64],
                       sqrtG_itf_j: NDArray[cp.float64], H_contra_2x_itf_j: NDArray[cp.float64],
                       nh: int, nk_nh: int, nk_nv: int, ne: int, advection_only: bool): ...

    @cuda_kernel(DimSpec.groupby_first_x(lambda s: Dim(s[4], s[1], s[2] - 1)))
    def compute_flux_k(flux_x3_itf_k: NDArray[cp.float64], wflux_adv_x3_itf_k: NDArray[cp.float64], wflux_pres_x3_itf_k: NDArray[cp.float64],
                       variables_itf_k: NDArray[cp.float64], pressure_itf_k: NDArray[cp.float64], w_itf_k: NDArray[cp.float64],
                       sqrtG_itf_k: NDArray[cp.float64], H_contra_3x_itf_k: NDArray[cp.float64],
                       nv: int, nk_nh: int, ne: int, advection_only: bool): ...


def rhs_euler_cuda(Q: NDArray[cp.float64],
                   geom: CubedSphere,
                   mtrx: CudaDFROperators,
                   metric: CudaMetric3DTopo,
                   ptopo: CudaDistributedWorld,
                   nbsolpts: int,
                   nb_elements_hori: int,
                   nb_elements_vert: int,
                   case_number: int) -> NDArray[cp.float64]:
    
    type_vec = Q.dtype
    nb_equations = Q.shape[0]
    nb_interfaces_hori = nb_elements_hori + 1
    nb_interfaces_vert = nb_elements_vert + 1
    nb_pts_hori = nb_elements_hori * nbsolpts
    nb_vertical_levels = nb_elements_vert * nbsolpts

    if type_vec is not cp.dtype(cp.float64):
        raise TypeError(f"rhs_euler_cuda only implemented for double-precision arithmetic, invalid dtype {type_vec}")

    forcing = cp.zeros_like(Q, dtype=type_vec)

    variables_itf_i = cp.ones( (nb_equations, nb_vertical_levels, nb_elements_hori + 2, 2, nb_pts_hori), dtype=type_vec)
    flux_x1_itf_i   = cp.empty((nb_equations, nb_vertical_levels, nb_elements_hori + 2, nb_pts_hori, 2), dtype=type_vec)

    variables_itf_j = cp.ones( (nb_equations, nb_vertical_levels, nb_elements_hori + 2, 2, nb_pts_hori), dtype=type_vec)
    flux_x2_itf_j   = cp.empty((nb_equations, nb_vertical_levels, nb_elements_hori + 2, 2, nb_pts_hori), dtype=type_vec)

    variables_itf_k = cp.ones( (nb_equations, nb_pts_hori, nb_elements_vert + 2, 2, nb_pts_hori), dtype=type_vec)
    flux_x3_itf_k   = cp.empty((nb_equations, nb_pts_hori, nb_elements_vert + 2, 2, nb_pts_hori), dtype=type_vec)

    wflux_adv_x1_itf_i  = cp.zeros_like(flux_x1_itf_i[0])
    wflux_pres_x1_itf_i = cp.zeros_like(flux_x1_itf_i[0])
    wflux_adv_x2_itf_j  = cp.zeros_like(flux_x2_itf_j[0])
    wflux_pres_x2_itf_j = cp.zeros_like(flux_x2_itf_j[0])
    wflux_adv_x3_itf_k  = cp.zeros_like(flux_x3_itf_k[0])
    wflux_pres_x3_itf_k = cp.zeros_like(flux_x3_itf_k[0])

    advection_only = case_number < 13

    variables_itf_i[:, :, 1:-1, :, :] = mtrx.extrapolate_i(Q, geom).transpose((0, 1, 3, 4, 2))
    variables_itf_j[:, :, 1:-1, :, :] = mtrx.extrapolate_j(Q, geom)

    logrho      = cp.log(Q[idx_rho])
    logrhotheta = cp.log(Q[idx_rho_theta])

    variables_itf_i[idx_rho, :, 1:-1, :, :] = cp.exp(mtrx.extrapolate_i(logrho, geom)).transpose((0, 2, 3, 1))
    variables_itf_j[idx_rho, :, 1:-1, :, :] = cp.exp(mtrx.extrapolate_j(logrho, geom))

    variables_itf_i[idx_rho_theta, :, 1:-1, :, :] = cp.exp(mtrx.extrapolate_i(logrhotheta, geom)).transpose((0, 2, 3, 1))
    variables_itf_j[idx_rho_theta, :, 1:-1, :, :] = cp.exp(mtrx.extrapolate_j(logrhotheta, geom))


    all_request = ptopo.xchange_Euler_interfaces(geom, variables_itf_i, variables_itf_j, blocking=False)


    rho = Q[idx_rho]
    u1  = Q[idx_rho_u1] / rho
    u2  = Q[idx_rho_u2] / rho
    w   = Q[idx_rho_w]  / rho

    flux_x1 = metric.sqrtG * u1 * Q
    flux_x2 = metric.sqrtG * u2 * Q
    flux_x3 = metric.sqrtG * w  * Q

    wflux_adv_x1 = metric.sqrtG * u1 * Q[idx_rho_w]
    wflux_adv_x2 = metric.sqrtG * u2 * Q[idx_rho_w]
    wflux_adv_x3 = metric.sqrtG * w  * Q[idx_rho_w]

    pressure = p0 * cp.exp((cpd / cvd) * cp.log((Rd / p0) * Q[idx_rho_theta]))

    flux_x1[idx_rho_u1] += metric.sqrtG * metric.H_contra_11 * pressure
    flux_x1[idx_rho_u2] += metric.sqrtG * metric.H_contra_12 * pressure
    flux_x1[idx_rho_w]  += metric.sqrtG * metric.H_contra_13 * pressure

    wflux_pres_x1 = metric.sqrtG * metric.H_contra_13

    flux_x2[idx_rho_u1] += metric.sqrtG * metric.H_contra_21 * pressure
    flux_x2[idx_rho_u2] += metric.sqrtG * metric.H_contra_22 * pressure
    flux_x2[idx_rho_w]  += metric.sqrtG * metric.H_contra_23 * pressure

    wflux_pres_x2 = metric.sqrtG * metric.H_contra_23 * pressure

    flux_x3[idx_rho_u1] += metric.sqrtG * metric.H_contra_31 * pressure
    flux_x3[idx_rho_u2] += metric.sqrtG * metric.H_contra_32 * pressure
    flux_x3[idx_rho_w]  += metric.sqrtG * metric.H_contra_33 * pressure

    wflux_pres_x3 = metric.sqrtG * metric.H_contra_33

    variables_itf_k[:, :, 1:-1, :, :] = mtrx.extrapolate_k(Q, geom).transpose((0, 3, 1, 2, 4))

    variables_itf_k[idx_rho, :, 1:-1, :, :] = cp.exp(mtrx.extrapolate_k(logrho, geom).transpose((2, 0, 1, 3)))
    variables_itf_k[idx_rho_theta, :, 1:-1, :, :] = cp.exp(mtrx.extrapolate_k(logrhotheta, geom).transpose((2, 0, 1, 3)))

    variables_itf_k[:, :,  0, 1, :] = variables_itf_k[:, :,  1, 0, :]
    variables_itf_k[:, :,  0, 0, :] = variables_itf_k[:, :,  0, 1, :]
    variables_itf_k[:, :, -1, 0, :] = variables_itf_k[:, :, -2, 1, :]
    variables_itf_k[:, :, -1, 1, :] = variables_itf_k[:, :, -1, 0, :]

    pressure_itf_k = p0 * cp.exp((cpd / cvd) * cp.log(variables_itf_k[idx_rho_theta] * (Rd / p0)))

    w_itf_k = variables_itf_k[idx_rho_w] / variables_itf_k[idx_rho]

    w_itf_k[:, 0, 0, :] = 0.
    w_itf_k[:,  0, 1, :] = -w_itf_k[:,  1, 0, :]
    w_itf_k[:, -1, 1, :] = 0.
    w_itf_k[:, -1, 0, :] = -w_itf_k[:, -2, 1, :]

    RHSEuler.compute_flux_k(flux_x3_itf_k, wflux_adv_x3_itf_k, wflux_pres_x3_itf_k,
                            variables_itf_k, pressure_itf_k, w_itf_k,
                            metric.sqrtG_itf_k, metric.H_contra_itf_k[2],
                            nb_elements_vert, nb_pts_hori, nb_equations, advection_only)
    

    all_request.wait()

    u1_itf_i = variables_itf_i[idx_rho_u1] / variables_itf_i[idx_rho]
    u2_itf_j = variables_itf_j[idx_rho_u2] / variables_itf_j[idx_rho]

    pressure_itf_i = p0 * cp.exp((cpd / cvd) * cp.log(variables_itf_i[idx_rho_theta] * (Rd / p0)))
    pressure_itf_j = p0 * cp.exp((cpd / cvd) * cp.log(variables_itf_j[idx_rho_theta] * (Rd / p0)))

    RHSEuler.compute_flux_i(flux_x1_itf_i, wflux_adv_x1_itf_i, wflux_pres_x1_itf_i,
                            variables_itf_i, pressure_itf_i, u1_itf_i,
                            metric.sqrtG_itf_i, metric.H_contra_itf_i[0],
                            nb_elements_hori, nb_pts_hori, nb_vertical_levels, nb_equations, advection_only)
    
    RHSEuler.compute_flux_j(flux_x2_itf_j, wflux_adv_x2_itf_j, wflux_pres_x2_itf_j,
                            variables_itf_j, pressure_itf_j, u2_itf_j,
                            metric.sqrtG_itf_j, metric.H_contra_itf_j[1],
                            nb_elements_hori, nb_pts_hori, nb_vertical_levels, nb_equations, advection_only)
    

    flux_x1_bdy = flux_x1_itf_i.transpose((0, 1, 3, 2, 4))[:, :, :, 1:-1, :].copy()
    df1_dx1 = mtrx.comma_i(flux_x1, flux_x1_bdy, geom)
    flux_x2_bdy = flux_x2_itf_j[:, :, 1:-1, :, :].copy()
    df2_dx2 = mtrx.comma_j(flux_x2, flux_x2_bdy, geom)
    flux_x3_bdy = flux_x3_itf_k[:, :, 1:-1, :, :].transpose(0, 2, 3, 1, 4).copy()
    df3_dx3 = mtrx.comma_k(flux_x3, flux_x3_bdy, geom)

    logp_int = cp.log(pressure)

    pressure_bdy_i = pressure_itf_i[:, 1:-1, :, :].transpose((0, 3, 1, 2)).copy()
    pressure_bdy_j = pressure_itf_j[:, 1:-1, :, :].copy()
    pressure_bdy_k = pressure_itf_k[:, 1:-1, :, :].transpose((1, 2, 0, 3)).copy()

    logp_bdy_i = cp.log(pressure_bdy_i)
    logp_bdy_j = cp.log(pressure_bdy_j)
    logp_bdy_k = cp.log(pressure_bdy_k)

    wflux_adv_x1_bdy_i  =  wflux_adv_x1_itf_i.transpose((0, 2, 1, 3))[:, :, 1:-1, :].copy()
    wflux_pres_x1_bdy_i = wflux_pres_x1_itf_i.transpose((0, 2, 1, 3))[:, :, 1:-1, :].copy()

    wflux_adv_x2_bdy_j  =  wflux_adv_x2_itf_j[:, 1:-1, :, :].copy()
    wflux_pres_x2_bdy_j = wflux_pres_x2_itf_j[:, 1:-1, :, :].copy()

    wflux_adv_x3_bdy_k  =  wflux_adv_x3_itf_k[:, 1:-1, :, :].transpose((1, 2, 0, 3)).copy()
    wflux_pres_x3_bdy_k = wflux_pres_x3_itf_k[:, 1:-1, :, :].transpose((1, 2, 0, 3)).copy()

    w_df1_dx1_adv   = mtrx.comma_i(wflux_adv_x1,  wflux_adv_x1_bdy_i,  geom)
    w_df1_dx1_presa = mtrx.comma_i(wflux_pres_x1, wflux_pres_x1_bdy_i, geom) * pressure
    w_df1_dx1_presb = mtrx.comma_i(logp_int,      logp_bdy_i,          geom) * pressure * wflux_pres_x1
    w_df1_dx1       = w_df1_dx1_adv + w_df1_dx1_presa + w_df1_dx1_presb

    w_df2_dx2_adv   = mtrx.comma_j(wflux_adv_x2,  wflux_adv_x2_bdy_j,  geom)
    w_df2_dx2_presa = mtrx.comma_j(wflux_pres_x2, wflux_pres_x2_bdy_j, geom) * pressure
    w_df2_dx2_presb = mtrx.comma_j(logp_int,      logp_bdy_j,          geom) * pressure * wflux_pres_x2
    w_df2_dx2       = w_df2_dx2_adv + w_df2_dx2_presa + w_df2_dx2_presb

    w_df3_dx3_adv   = mtrx.comma_k(wflux_adv_x3,  wflux_adv_x3_bdy_k,  geom)
    w_df3_dx3_presa = mtrx.comma_k(wflux_pres_x3, wflux_pres_x3_bdy_k, geom) * pressure
    w_df3_dx3_presb = mtrx.comma_k(logp_int,      logp_bdy_k,          geom) * pressure * wflux_pres_x3
    w_df3_dx3       = w_df3_dx3_adv + w_df3_dx3_presa + w_df3_dx3_presb


    forcing[idx_rho] = 0.

    forcing[idx_rho_u1] = 2. * rho * (metric.christoffel_1_01 * u1 + metric.christoffel_1_02 * u2 + metric.christoffel_1_03 * w) \
                        +      metric.christoffel_1_11 * (rho * u1 * u1 + metric.H_contra_11 * pressure) \
                        + 2. * metric.christoffel_1_12 * (rho * u1 * u2 + metric.H_contra_12 * pressure) \
                        + 2. * metric.christoffel_1_13 * (rho * u1 * w  + metric.H_contra_13 * pressure) \
                        +      metric.christoffel_1_22 * (rho * u2 * u2 + metric.H_contra_22 * pressure) \
                        + 2. * metric.christoffel_1_23 * (rho * u2 * w  + metric.H_contra_23 * pressure) \
                        +      metric.christoffel_1_33 * (rho * w  * w  + metric.H_contra_33 * pressure)
    
    forcing[idx_rho_u2] = 2. * rho * (metric.christoffel_2_01 * u1 + metric.christoffel_2_02 * u2 + metric.christoffel_2_03 * w) \
                        +      metric.christoffel_2_11 * (rho * u1 * u1 + metric.H_contra_11 * pressure) \
                        + 2. * metric.christoffel_2_12 * (rho * u1 * u2 + metric.H_contra_12 * pressure) \
                        + 2. * metric.christoffel_2_13 * (rho * u1 * w  + metric.H_contra_13 * pressure) \
                        +      metric.christoffel_2_22 * (rho * u2 * u2 + metric.H_contra_22 * pressure) \
                        + 2. * metric.christoffel_2_23 * (rho * u2 * w  + metric.H_contra_23 * pressure) \
                        +      metric.christoffel_2_33 * (rho * w  * w  + metric.H_contra_33 * pressure)
    
    forcing[idx_rho_w]  = 2. * rho * (metric.christoffel_3_01 * u1 + metric.christoffel_3_02 * u2 + metric.christoffel_3_03 * w) \
                        +      metric.christoffel_3_11 * (rho * u1 * u1 + metric.H_contra_11 * pressure) \
                        + 2. * metric.christoffel_3_12 * (rho * u1 * u2 + metric.H_contra_12 * pressure) \
                        + 2. * metric.christoffel_3_13 * (rho * u1 * w  + metric.H_contra_13 * pressure) \
                        +      metric.christoffel_3_22 * (rho * u2 * u2 + metric.H_contra_22 * pressure) \
                        + 2. * metric.christoffel_3_23 * (rho * u2 * w  + metric.H_contra_23 * pressure) \
                        +      metric.christoffel_3_33 * (rho * w  * w  + metric.H_contra_33 * pressure) \
                        + metric.inv_dzdeta * gravity * metric.inv_sqrtG * mtrx.filter_k(metric.sqrtG * rho, geom)
    
    forcing[idx_rho_theta] = 0.

    if case_number == 21:
        dcmip_schar_damping(forcing, rho, u1, u2, w, metric, geom, shear=False)
    elif case_number == 22:
        dcmip_schar_damping(forcing, rho, u1, u2, w, metric, geom, shear=True)
    

    rhs            = -metric.inv_sqrtG * (  df1_dx1 +   df2_dx2 +   df3_dx3) - forcing
    rhs[idx_rho_w] = -metric.inv_sqrtG * (w_df1_dx1 + w_df2_dx2 + w_df3_dx3) - forcing[idx_rho_w]

    if advection_only:
        rhs[idx_rho]       = 0.
        rhs[idx_rho_u1]    = 0.
        rhs[idx_rho_u2]    = 0.
        rhs[idx_rho_w]     = 0.
        rhs[idx_rho_theta] = 0.
    
    return rhs