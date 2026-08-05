"""
A bunch of flux functions for our RHS functions
"""

from typing import Callable, Tuple

import numpy
import torch
from numpy.typing import NDArray

from ..common.definitions import (
    cpd,
    cvd,
    heat_capacity_ratio,
    p0,
    Rd,
    idx_rho,
    idx_rho_u1,
    idx_rho_u2,
    idx_rho_u3,
)

from ..geometry import Metric3DTopo

FluxFunction2D = Callable[
    [numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray],
    Tuple[numpy.ndarray, numpy.ndarray],
]


def rusanov_3d_vert_new(
    variables_itf_k: NDArray,
    pressure_itf_k: NDArray,
    w_itf_k: NDArray,
    metric: Metric3DTopo,
    advection_only: bool,
    flux_x3_itf_k: NDArray,
    wflux_adv_x3_itf_k: NDArray,
    wflux_pres_x3_itf_k: NDArray,
    num_solpts: int,
):
    # south/north here are relative to an element, *not* an interface
    south = numpy.s_[..., 1:, :, :, : num_solpts**2]
    north = numpy.s_[..., :-1, :, :, num_solpts**2 :]
    w_d = w_itf_k[north]
    w_u = w_itf_k[south]

    if advection_only:
        eig_d = torch.abs(w_d)
        eig_u = torch.abs(w_u)
    else:
        eig_d = torch.abs(w_d) + torch.sqrt(
            metric.h_contra_itf_k_new[2, 2][north]
            * heat_capacity_ratio
            * pressure_itf_k[north]
            / variables_itf_k[idx_rho][north]
        )
        eig_u = torch.abs(w_u) + torch.sqrt(
            metric.h_contra_itf_k_new[2, 2][south]
            * heat_capacity_ratio
            * pressure_itf_k[south]
            / variables_itf_k[idx_rho][south]
        )

    eig = torch.maximum(eig_d, eig_u)

    # Advective part of the flux ...
    flux_d = metric.sqrtG_itf_k_new[north] * w_d * variables_itf_k[north]
    flux_u = metric.sqrtG_itf_k_new[south] * w_u * variables_itf_k[south]

    # Separate variables for rho-w flux
    wflux_adv_d = flux_d[idx_rho_u3, ...].copy()
    wflux_adv_u = flux_u[idx_rho_u3, ...].copy()

    # ... and add the pressure part
    flux_d[idx_rho_u1] += metric.sqrtG_itf_k_new[north] * metric.h_contra_itf_k_new[2, 0][north] * pressure_itf_k[north]
    flux_d[idx_rho_u2] += metric.sqrtG_itf_k_new[north] * metric.h_contra_itf_k_new[2, 1][north] * pressure_itf_k[north]
    flux_d[idx_rho_u3] += metric.sqrtG_itf_k_new[north] * metric.h_contra_itf_k_new[2, 2][north] * pressure_itf_k[north]

    flux_u[idx_rho_u1] += metric.sqrtG_itf_k_new[south] * metric.h_contra_itf_k_new[2, 0][south] * pressure_itf_k[south]
    flux_u[idx_rho_u2] += metric.sqrtG_itf_k_new[south] * metric.h_contra_itf_k_new[2, 1][south] * pressure_itf_k[south]
    flux_u[idx_rho_u3] += metric.sqrtG_itf_k_new[south] * metric.h_contra_itf_k_new[2, 2][south] * pressure_itf_k[south]

    # For rho_w pressure flux, account for the pressure terms separately from the advection terms
    wflux_pres_d = metric.sqrtG_itf_k_new[north] * metric.h_contra_itf_k_new[2, 2][north] * pressure_itf_k[north]
    wflux_pres_u = metric.sqrtG_itf_k_new[south] * metric.h_contra_itf_k_new[2, 2][south] * pressure_itf_k[south]

    # Riemann solver
    flux_x3_itf_k[north] = 0.5 * (
        flux_d + flux_u - eig * metric.sqrtG_itf_k_new[north] * (variables_itf_k[south] - variables_itf_k[north])
    )
    flux_x3_itf_k[south] = flux_x3_itf_k[north]

    # Riemann solver, separating pressure and advection terms for rho-w
    wflux_adv_x3_itf_k[north] = 0.5 * (
        wflux_adv_d
        + wflux_adv_u
        - eig
        * metric.sqrtG_itf_k_new[north]
        * (variables_itf_k[idx_rho_u3][south] - variables_itf_k[idx_rho_u3][north])
    )
    wflux_adv_x3_itf_k[south] = wflux_adv_x3_itf_k[north]
    wflux_pres_x3_itf_k[north] = 0.5 * (wflux_pres_d + wflux_pres_u) / pressure_itf_k[north]
    wflux_pres_x3_itf_k[south] = 0.5 * (wflux_pres_d + wflux_pres_u) / pressure_itf_k[south]


def rusanov_3d_hori_i_new(
    u1_itf_i,
    variables_itf_i,
    pressure_itf_i,
    metric: Metric3DTopo,
    num_interfaces_hori,
    advection_only,
    flux_x1_itf_i,
    wflux_adv_x1_itf_i,
    wflux_pres_x1_itf_i,
    num_solpts,
):
    west = numpy.s_[..., 1:, : num_solpts**2]
    east = numpy.s_[..., :-1, num_solpts**2 :]

    u1_r = u1_itf_i[west]
    u1_l = u1_itf_i[east]

    if advection_only:
        # Advection only, eigenvalues are the velocities alone
        eig_l = torch.abs(u1_l)
        eig_r = torch.abs(u1_r)
    else:
        # Otherwise, maximum eigenvalue is |u| + c_sound
        eig_l = torch.abs(u1_l) + torch.sqrt(
            metric.h_contra_itf_i_new[0, 0][east]
            * heat_capacity_ratio
            * pressure_itf_i[east]
            / variables_itf_i[idx_rho][east]
        )
        eig_r = torch.abs(u1_r) + torch.sqrt(
            metric.h_contra_itf_i_new[0, 0][west]
            * heat_capacity_ratio
            * pressure_itf_i[west]
            / variables_itf_i[idx_rho][west]
        )

    eig = torch.maximum(eig_l, eig_r)

    # Advective part of the flux ...
    flux_l = metric.sqrtG_itf_i_new[east] * u1_l * variables_itf_i[east]
    flux_r = metric.sqrtG_itf_i_new[west] * u1_r * variables_itf_i[west]

    # rho-w specific advective flux
    wflux_adv_L = flux_l[idx_rho_u3, ...].copy()
    wflux_adv_R = flux_r[idx_rho_u3, ...].copy()

    # ... and now add the pressure contribution
    flux_l[idx_rho_u1] += metric.sqrtG_itf_i_new[east] * metric.h_contra_itf_i_new[0, 0][east] * pressure_itf_i[east]
    flux_l[idx_rho_u2] += metric.sqrtG_itf_i_new[east] * metric.h_contra_itf_i_new[0, 1][east] * pressure_itf_i[east]
    flux_l[idx_rho_u3] += metric.sqrtG_itf_i_new[east] * metric.h_contra_itf_i_new[0, 2][east] * pressure_itf_i[east]

    flux_r[idx_rho_u1] += metric.sqrtG_itf_i_new[west] * metric.h_contra_itf_i_new[0, 0][west] * pressure_itf_i[west]
    flux_r[idx_rho_u2] += metric.sqrtG_itf_i_new[west] * metric.h_contra_itf_i_new[0, 1][west] * pressure_itf_i[west]
    flux_r[idx_rho_u3] += metric.sqrtG_itf_i_new[west] * metric.h_contra_itf_i_new[0, 2][west] * pressure_itf_i[west]

    # Pressure contribution specifically for rho-w
    wflux_pres_l = metric.sqrtG_itf_i_new[east] * metric.h_contra_itf_i_new[0, 2][east] * pressure_itf_i[east]
    wflux_pres_r = metric.sqrtG_itf_i_new[west] * metric.h_contra_itf_i_new[0, 2][west] * pressure_itf_i[west]

    # --- Common Rusanov fluxes

    flux_x1_itf_i[east] = 0.5 * (
        flux_l + flux_r - eig * metric.sqrtG_itf_i_new[east] * (variables_itf_i[west] - variables_itf_i[east])
    )
    flux_x1_itf_i[west] = flux_x1_itf_i[east]

    # Separating advective and pressure fluxes for rho-w
    wflux_adv_x1_itf_i[east] = 0.5 * (
        wflux_adv_L
        + wflux_adv_R
        - eig * metric.sqrtG_itf_i_new[east] * (variables_itf_i[idx_rho_u3][west] - variables_itf_i[idx_rho_u3][east])
    )
    wflux_adv_x1_itf_i[west] = wflux_adv_x1_itf_i[east]
    wflux_pres_x1_itf_i[east] = 0.5 * (wflux_pres_l + wflux_pres_r) / pressure_itf_i[east]
    wflux_pres_x1_itf_i[west] = 0.5 * (wflux_pres_l + wflux_pres_r) / pressure_itf_i[west]


def rusanov_3d_hori_j_new(
    u2_itf_j,
    variables_itf_j,
    pressure_itf_j,
    metric: Metric3DTopo,
    num_interfaces_hori,
    advection_only,
    flux_x2_itf_j,
    wflux_adv_x2_itf_j,
    wflux_pres_x2_itf_j,
    num_solpts,
):
    # South and north are relative to an element (*not* an interface)
    south = numpy.s_[..., 1:, :, : num_solpts**2]
    north = numpy.s_[..., :-1, :, num_solpts**2 :]

    u2_l = u2_itf_j[north]
    u2_r = u2_itf_j[south]

    if advection_only:
        eig_l = torch.abs(u2_l)
        eig_r = torch.abs(u2_r)
    else:
        eig_l = torch.abs(u2_l) + torch.sqrt(
            metric.h_contra_itf_j_new[1, 1][north]
            * heat_capacity_ratio
            * pressure_itf_j[north]
            / variables_itf_j[idx_rho][north]
        )
        eig_r = torch.abs(u2_r) + torch.sqrt(
            metric.h_contra_itf_j_new[1, 1][south]
            * heat_capacity_ratio
            * pressure_itf_j[south]
            / variables_itf_j[idx_rho][south]
        )

    eig = torch.maximum(eig_l, eig_r)

    # Advective part of the flux
    flux_l = metric.sqrtG_itf_j_new[north] * u2_l * variables_itf_j[north]
    flux_r = metric.sqrtG_itf_j_new[south] * u2_r * variables_itf_j[south]

    # rho-w specific advective flux
    wflux_adv_l = flux_l[idx_rho_u3, ...].copy()
    wflux_adv_r = flux_r[idx_rho_u3, ...].copy()

    # ... and now add the pressure contribution
    flux_l[idx_rho_u1] += metric.sqrtG_itf_j_new[north] * metric.h_contra_itf_j_new[1, 0][north] * pressure_itf_j[north]
    flux_l[idx_rho_u2] += metric.sqrtG_itf_j_new[north] * metric.h_contra_itf_j_new[1, 1][north] * pressure_itf_j[north]
    flux_l[idx_rho_u3] += metric.sqrtG_itf_j_new[north] * metric.h_contra_itf_j_new[1, 2][north] * pressure_itf_j[north]

    flux_r[idx_rho_u1] += metric.sqrtG_itf_j_new[south] * metric.h_contra_itf_j_new[1, 0][south] * pressure_itf_j[south]
    flux_r[idx_rho_u2] += metric.sqrtG_itf_j_new[south] * metric.h_contra_itf_j_new[1, 1][south] * pressure_itf_j[south]
    flux_r[idx_rho_u3] += metric.sqrtG_itf_j_new[south] * metric.h_contra_itf_j_new[1, 2][south] * pressure_itf_j[south]

    wflux_pres_l = metric.sqrtG_itf_j_new[north] * metric.h_contra_itf_j_new[1, 2][north] * pressure_itf_j[north]
    wflux_pres_r = metric.sqrtG_itf_j_new[south] * metric.h_contra_itf_j_new[1, 2][south] * pressure_itf_j[south]

    # --- Common Rusanov fluxes

    flux_x2_itf_j[north] = 0.5 * (
        flux_l + flux_r - eig * metric.sqrtG_itf_j_new[north] * (variables_itf_j[south] - variables_itf_j[north])
    )
    flux_x2_itf_j[south] = flux_x2_itf_j[north]

    # Separation of advective and pressure flux for rho-w
    wflux_adv_x2_itf_j[north] = 0.5 * (
        wflux_adv_l
        + wflux_adv_r
        - eig
        * metric.sqrtG_itf_j_new[north]
        * (variables_itf_j[idx_rho_u3][south] - variables_itf_j[idx_rho_u3][north])
    )
    wflux_adv_x2_itf_j[south] = wflux_adv_x2_itf_j[north]

    wflux_pres_x2_itf_j[north] = 0.5 * (wflux_pres_l + wflux_pres_r) / pressure_itf_j[north]
    wflux_pres_x2_itf_j[south] = 0.5 * (wflux_pres_l + wflux_pres_r) / pressure_itf_j[south]
