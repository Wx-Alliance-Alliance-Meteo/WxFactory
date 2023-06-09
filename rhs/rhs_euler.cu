#ifndef heat_capacity_ratio
#error "Define heat_capacity_ratio"
#endif
#ifndef idx_rho
#error "Define idx_rho"
#endif
#ifndef idx_rho_u1
#error "Define idx_rho_u1"
#endif
#ifndef idx_rho_u2
#error "Define idx_rho_u2"
#endif
#ifndef idx_rho_w
#error "Define idx_rho_w"
#endif

/**
 * compute_flux_k
 *
 * double       flux_x3_itf_k[ne, nk_nh, nv + 2, 2, nk_nh]
 * double  wflux_adv_x3_itf_k[    nk_nh, nv + 2, 2, nk_nh]
 * double wflux_pres_x3_itf_k[    nk_nh, nv + 2, 2, nk_nh]
 *
 * const double variables_itf_k[ne, nk_nh, nv + 2, 2, nk_nh]
 * const double  pressure_itf_k[    nk_nh, nv + 2, 2, nk_nh]
 * const double         w_itf_k[    nk_nh, nv + 2, 2, nk_nh]
 *
 * const double       sqrtG_itf_k[   nv + 1, nk_nh, nk_nh]
 * const double H_contra_3x_itf_k[3, nv + 1, nk_nh, nk_nh]
 *
 * unsigned int nv
 * unsigned int nk_nh
 * unsigned int ne
 * bool         advection_only
 *
 * run with (nk*nh, nk*nh, nv) threads
 */
extern "C" __global__
void compute_flux_k(
    double* const __restrict__ flux_x3_itf_k,
    double* const __restrict__ wflux_adv_x3_itf_k,
    double* const __restrict__ wflux_pres_x3_itf_k,
    const double* const __restrict__ variables_itf_k,
    const double* const __restrict__ pressure_itf_k,
    const double* const __restrict__ w_itf_k,
    const double* const __restrict__ sqrtG_itf_k,
    const double* const __restrict__ H_contra_3x_itf_k,
    const unsigned int nv,
    const unsigned int nk_nh,
    const unsigned int ne,
    const bool advection_only)
{
    /* stride lengths for flux_x3_itf_k (and most other arrays) */
    const unsigned int F_4 = nk_nh;
    const unsigned int F_3 = 2 * F_4;
    const unsigned int F_2 = (nv + 2) * F_3;
    const unsigned int F_1 = nk_nh * F_2;

    /* stride lengths for metric tensors */
    const unsigned int M_3 = nk_nh;
    const unsigned int M_2 = nk_nh * M_3;
    const unsigned int M_1 = (nv + 1) * M_2;

    /* this thread's location */
    const unsigned int idx_i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int idx_j = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int itf = blockIdx.z * blockDim.z + threadIdx.z;
    const unsigned int elem_D = itf;
    const unsigned int elem_U = itf + 1;

    // this thread's idx for D- or U-elements
    const unsigned int D_idx = idx_j * F_2 + elem_D * F_3 + F_4 + idx_i;
    const unsigned int U_idx = idx_j * F_2 + elem_U * F_3       + idx_i;
    // this thread's idx for metric elements
    const unsigned int M_idx = itf * M_2 + idx_j * M_3 + idx_i;

    const double w_D = w_itf_k[D_idx];
    const double w_U = w_itf_k[U_idx];

    const double pres_D = pressure_itf_k[D_idx];
    const double pres_U = pressure_itf_k[U_idx];

    const double rho_D = variables_itf_k[idx_rho * F_1 + D_idx];
    const double rho_U = variables_itf_k[idx_rho * F_1 + U_idx];

    const double H_33 = H_contra_3x_itf_k[2 * M_1 + M_idx];

    double eig;
    if (advection_only)
    {
        eig = max(abs(w_D), abs(w_U));
    }
    else
    {
        const double eig_D = abs(w_D) + sqrt(H_33 * heat_capacity_ratio * pres_D / rho_D);
        const double eig_U = abs(w_U) + sqrt(H_33 * heat_capacity_ratio * pres_U / rho_U);
        eig = max(eig_D, eig_U);
    }

    const double sqrtG = sqrtG_itf_k[M_idx];

    const double H_31 = H_contra_3x_itf_k[      M_idx];
    const double H_32 = H_contra_3x_itf_k[M_1 + M_idx];

    double flux_D, wflux_adv_D;
    double flux_U, wflux_adv_U;
    for (unsigned int l = 0; l < ne; ++l)
    {
        const double var_D = variables_itf_k[l * F_1 + D_idx];
        const double var_U = variables_itf_k[l * F_1 + U_idx];

        flux_D = sqrtG * w_D * var_D;
        flux_U = sqrtG * w_U * var_U;

        switch (l)
        {
        case idx_rho_u1:
            flux_D += sqrtG * H_31 * pres_D;
            flux_U += sqrtG * H_31 * pres_D;
            break;
        case idx_rho_u2:
            flux_D += sqrtG * H_32 * pres_D;
            flux_U += sqrtG * H_32 * pres_U;
            break;
        case idx_rho_w:
            wflux_adv_D = flux_D;
            wflux_adv_U = flux_U;
            flux_D += sqrtG * H_33 * pres_D;
            flux_U += sqrtG * H_33 * pres_U;
            break;
        }

        flux_x3_itf_k[l * F_1 + D_idx] = flux_x3_itf_k[l * F_1 + U_idx]
            = 0.5 * (flux_D + flux_U - eig * sqrtG * (var_U - var_D));
    }

    const double wflux_pres_D = sqrtG * H_33 * pres_D;
    const double wflux_pres_U = sqrtG * H_33 * pres_U;

    const double rho_w_D = variables_itf_k[idx_rho_w * F_1 + D_idx];
    const double rho_w_U = variables_itf_k[idx_rho_w * F_1 + U_idx];

    wflux_adv_x3_itf_k[D_idx] = wflux_adv_x3_itf_k[U_idx]
        = 0.5 * (wflux_adv_D + wflux_adv_U - eig * sqrtG * (rho_w_U - rho_w_D));
    wflux_pres_x3_itf_k[D_idx] = 0.5 * (wflux_pres_D + wflux_pres_U) / pres_D;
    wflux_pres_x3_itf_k[U_idx] = 0.5 * (wflux_pres_D + wflux_pres_U) / pres_U;
}
