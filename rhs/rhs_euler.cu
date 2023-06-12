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
 * compute_flux_i
 *
 * double       flux_x1_itf_i[ne, nk_nv, nh + 2, nk_nh, 2]
 * double  wflux_adv_x1_itf_i[    nk_nv, nh + 2, nk_nh, 2]
 * double wflux_pres_x1_itf_i[    nk_nv, nh + 2, nk_nh, 2]
 *
 * const double variables_itf_i[ne, nk_nv, nh + 2, 2, nk_nh]
 * const double  pressure_itf_i[    nk_nv, nh + 2, 2, nk_nh]
 * const double        u1_itf_i[    nk_nv, nh + 2, 2, nk_nh]
 *
 * const double       sqrtG_itf_i[   nk_nv, nk_nh, nh + 1]
 * const double H_contra_1x_itf_i[3, nk_nv, nk_nh, nh + 1]
 *
 * usnigned int nh
 * unsigned int nk_nh
 * unsigned int nk_nv
 * unsigned int ne
 * bool         advection_only
 *
 * run with (nk_nh, nk_nv, nh + 1) threads
 */
extern "C" __global__
void compute_flux_i(
    double* const __restrict__ flux_x1_itf_i,
    double* const __restrict__ wflux_adv_x1_itf_i,
    double* const __restrict__ wflux_pres_x1_itf_i,
    const double* const __restrict__ variables_itf_i,
    const double* const __restrict__ pressure_itf_i,
    const double* const __restrict__ u1_itf_i,
    const double* const __restrict__ sqrtG_itf_i,
    const double* const __restrict__ H_contra_1x_itf_i,
    const unsigned int nh,
    const unsigned int nk_nh,
    const unsigned int nk_nv,
    const unsigned int ne,
    const bool advection_only)
{
    /* stride lengths for flux_x1_itf_i (and other outputs) */
    const unsigned int F_4 = 2;
    const unsigned int F_3 = nk_nh * F_4;
    const unsigned int F_2 = (nh + 2) * F_3;
    const unsigned int F_1 = nk_nv * F_2;

    /* stride lengths for variables_itf_i (and other inputs) */
    const unsigned int V_4 = nk_nh;
    const unsigned int V_3 = 2 * V_4;
    const unsigned int V_2 = (nh + 2) * V_3;
    const unsigned int V_1 = nk_nv * V_2;

    /* stride lengths for metric tensors */
    const unsigned int M_3 = nh + 1;
    const unsigned int M_2 = nk_nh * M_3;
    const unsigned int M_1 = nk_nv * M_2;

    /* this thread's location */
    const unsigned int idx_j = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int idx_k = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int itf = blockIdx.z * blockDim.z + threadIdx.z;
    const unsigned int elem_L = itf;
    const unsigned int elem_R = itf + 1;

    /* this thread's index for L- or R-outputs */
    const unsigned int FL_idx = idx_k * F_2 + elem_L * F_3 + idx_j * F_4 + 1;
    const unsigned int FR_idx = idx_k * F_2 + elem_R * F_3 + idx_j * F_4;
    /* this thread's index for L- or R-inputs */
    const unsigned int VL_idx = idx_k * F_2 + elem_L * F_3 + F_4 + idx_j;
    const unsigned int VR_idx = idx_k * F_2 + elem_R * F_3       + idx_j;
    /* this thread's index for metric elementss */
    const unsigned int M_idx = idx_k * M_2 + idx_j * M_3 + itf;

    const double u1_L = u1_itf_i[VL_idx];
    const double u1_R = u1_itf_i[VR_idx];

    const double pres_L = pressure_itf_i[VL_idx];
    const double pres_R = pressure_itf_i[VR_idx];

    const double H_11 = H_contra_1x_itf_i[M_idx];

    double eig;
    if (advection_only)
    {
        eig = max(abs(u1_L), abs(u1_R));
    }
    else
    {
        const double rho_L = variables_itf_i[idx_rho * V_1 + VL_idx];
        const double rho_R = variables_itf_i[idx_rho * V_1 + VR_idx];

        const double eig_L = abs(u1_L) + sqrt(H_11 * heat_capacity_ratio * pres_L / rho_L);
        const double eig_R = abs(u1_R) + sqrt(H_11 * heat_capacity_ratio * pres_R / rho_R);
        eig = max(eig_L, eig_R);
    }

    const double sqrtG = sqrtG_itf_i[M_idx];

    const double H_12 = H_contra_1x_itf_i[    M_1 + M_idx];
    const double H_13 = H_contra_1x_itf_i[2 * M_1 + M_idx];

    double flux_L, wflux_adv_L;
    double flux_R, wflux_adv_R;
    for (unsigned int l = 0; l < ne; ++l)
    {
        const double var_L = variables_itf_i[l * V_1 + VL_idx];
        const double var_R = variables_itf_i[l * V_1 + VR_idx];

        flux_L = sqrtG * u1_L * var_L;
        flux_R = sqrtG * u1_R * var_R;

        switch (l)
        {
        case idx_rho_u1:
            flux_L += sqrtG * H_11 * pres_L;
            flux_R += sqrtG * H_11 * pres_R;
            break;
        case idx_rho_u2:
            flux_L += sqrtG * H_12 * pres_L;
            flux_R += sqrtG * H_12 * pres_R;
            break;
        case idx_rho_w:
            wflux_adv_L = flux_L;
            wflux_adv_R = flux_R;
            flux_L += sqrtG * H_13 * pres_L;
            flux_R += sqrtG * H_13 * pres_R;
            break;
        }

        flux_x1_itf_i[l * F_1 + FL_idx] = flux_x1_itf_i[l * F_1 + FR_idx]
            = 0.5 * (flux_L + flux_R - eig * sqrtG * (var_R - var_L));
    }

    const double rho_w_L = variables_itf_i[idx_rho_w * V_1 + VL_idx];
    const double rho_w_R = variables_itf_i[idx_rho_w * V_1 + VR_idx];

    wflux_adv_x1_itf_i[FL_idx] = wflux_adv_x1_itf_i[FR_idx]
        = 0.5 * (wflux_adv_L + wflux_adv_R - eig * sqrtG * (rho_w_R - rho_w_L));
    
    const double wflux_pres_L = sqrtG * H_13 * pres_L;
    const double wflux_pres_R = sqrtG * H_13 * pres_R;

    wflux_pres_x1_itf_i[FL_idx] = 0.5 * (wflux_pres_L + wflux_pres_R) / pres_L;
    wflux_pres_x1_itf_i[FR_idx] = 0.5 * (wflux_pres_L + wflux_pres_R) / pres_R;
}

/**
 * compute_flux_j
 *
 * double       flux_x2_itf_j[ne, nk_nv, nh + 2, 2, nk_nh]
 * double  wflux_adv_x2_itf_j[    nk_nv, nh + 2, 2, nk_nh]
 * double wflux_pres_x2_itf_j[    nk_nv, nh + 2, 2, nk_nh]
 *
 * const double variables_itf_j[ne, nk_nv, nh + 2, 2, nk_nh]
 * const double  pressure_itf_j[    nk_nv, nh + 2, 2, nk_nh]
 * const double        u2_itf_j[    nk_nv, nh + 2, 2, nk_nh]
 *
 * const double       sqrtG_itf_j[   nk_nv, nh + 1, nk_nh]
 * const double H_contra_2x_itf_j[3, nk_nv, nh + 1, nk_nh]
 *
 * usnigned int nh
 * unsigned int nk_nh
 * unsigned int nk_nv
 * unsigned int ne
 * bool         advection_only
 *
 * run with (nk_nh, nk_nv, nh + 1) threads
 */
extern "C" __global__
void compute_flux_j(
    double* const __restrict__ flux_x2_itf_j,
    double* const __restrict__ wflux_adv_x2_itf_j,
    double* const __restrict__ wflux_pres_x2_itf_j,
    const double* const __restrict__ variables_itf_j,
    const double* const __restrict__ pressure_itf_j,
    const double* const __restrict__ u2_itf_j,
    const double* const __restrict__ sqrtG_itf_j,
    const double* const __restrict__ H_contra_2x_itf_j,
    const unsigned int nh,
    const unsigned int nk_nh,
    const unsigned int nk_nv,
    const unsigned int ne,
    const bool advection_only)
{
    /* stride lengths for flux_x2_itf_j (and most other arrays) */
    const unsigned int F_4 = nk_nh;
    const unsigned int F_3 = 2 * F_4;
    const unsigned int F_2 = (nh + 2) * F_3;
    const unsigned int F_1 = nk_nv * F_2;

    /* stride lengths for metric tensors */
    const unsigned int M_3 = nk_nh;
    const unsigned int M_2 = (nh + 1) * M_3;
    const unsigned int M_1 = nk_nv * M_2;

    /* this thread's location */
    const unsigned int idx_i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int idx_k = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int itf = blockIdx.z * blockDim.z + threadIdx.z;
    const unsigned int elem_L = itf;
    const unsigned int elem_R = itf + 1;

    /* this thread's index for L- or R-elements */
    const unsigned int L_idx = idx_k * F_2 + elem_L * F_3 + F_4 + idx_i;
    const unsigned int R_idx = idx_k * F_2 + elem_R * F_3       + idx_i;
    /* this thread's index for metric elements */
    const unsigned int M_idx = idx_k * M_2 + itf * M_3 + idx_i;

    const double u2_L = u2_itf_j[L_idx];
    const double u2_R = u2_itf_j[R_idx];

    const double pres_L = pressure_itf_j[L_idx];
    const double pres_R = pressure_itf_j[R_idx];

    const double H_22 = H_contra_2x_itf_j[H_1 + M_idx];

    double eig;
    if (advection_only)
    {
        eig = max(abs(u2_L), abs(u2_R));
    }
    else
    {
        const double rho_L = variables_itf_j[idx_rho * F_1 + L_idx];
        const double rho_R = variables_itf_j[idx_rho * F_1 + R_idx];

        const double eig_L = abs(u2_L) + sqrt(H_22 * heat_capacity_ratio * pres_L / rho_L);
        const double eig_R = abs(u2_R) + sqrt(H_22 * heat_capacity_ratio * pres_R / rho_R);
        eig = max(eig_L, eig_R);
    }

    const double sqrtG = sqrtG_itf_j[M_idx];

    const double H_21 = H_contra_2x_itf_j[          M_idx];
    const double H_23 = H_contra_2x_itf_j[2 * M_1 + M_idx];

    double flux_L, wflux_adv_L;
    double flux_R, wflux_adv_R;
    for (unsigned int l = 0; l < ne; ++l)
    {
        const double var_L = variables_itf_j[l * F_1 + L_idx];
        const double var_R = variables_itf_j[l * F_1 + R_idx];

        flux_L = sqrtG * u2_L * var_L;
        flux_R = sqrtG * u2_R * var_R;

        switch (l)
        {
        case idx_rho_u1:
            flux_L += sqrtG * H_21 * pres_L;
            flux_R += sqrtG * H_21 * pres_R;
            break;
        case idx_rho_u2:
            flux_L += sqrtG * H_22 * pres_L;
            flux_R += sqrtG * H_22 * pres_R;
            break;
        case idx_rho_w:
            wflux_adv_L = flux_L;
            wflux_adv_R = flux_R;
            flux_L += sqrtG * H_23 * pres_L;
            flux_R += sqrtG * H_23 * pres_R;
            break;
        }

        flux_x2_itf_j[l * F_1 + L_idx] = flux_x2_itf_j[l * F_1 + R_idx]
            = 0.5 * (flux_L + flux_R - eig * sqrtG * (var_R - var_L));
    }

    const double rho_w_L = variables_itf_j[idx_rho_w * F_1 + L_idx];
    const double rho_w_R = variables_itf_j[idx_rho_w * F_1 + R_idx];

    wflux_adv_x2_itf_j[L_idx] = wflux_adv_x2_itf_j[R_idx]
        = 0.5 * (wflux_adv_L + wflux_adv_R - eig * sqrtG * (rho_w_R - rho_w_L));
    
    const double wflux_pres_L = sqrtG * H_23 * pres_L;
    const double wflux_pres_R = sqrtG * H_23 * pres_R;

    wflux_pres_x2_itf_j[L_idx] = 0.5 * (wflux_pres_L + wflux_pres_R) / pres_L;
    wflux_pres_x2_itf_j[R_idx] = 0.5 * (wflux_pres_L + wflux_pres_R) / pres_R;
}

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
 * run with (nk_nh, nk_nh, nv + 1) threads
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

    const double H_33 = H_contra_3x_itf_k[2 * M_1 + M_idx];

    double eig;
    if (advection_only)
    {
        eig = max(abs(w_D), abs(w_U));
    }
    else
    {
        const double rho_D = variables_itf_k[idx_rho * F_1 + D_idx];
        const double rho_U = variables_itf_k[idx_rho * F_1 + U_idx];

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

    const double rho_w_D = variables_itf_k[idx_rho_w * F_1 + D_idx];
    const double rho_w_U = variables_itf_k[idx_rho_w * F_1 + U_idx];

    wflux_adv_x3_itf_k[D_idx] = wflux_adv_x3_itf_k[U_idx]
        = 0.5 * (wflux_adv_D + wflux_adv_U - eig * sqrtG * (rho_w_U - rho_w_D));

    const double wflux_pres_D = sqrtG * H_33 * pres_D;
    const double wflux_pres_U = sqrtG * H_33 * pres_U;

    wflux_pres_x3_itf_k[D_idx] = 0.5 * (wflux_pres_D + wflux_pres_U) / pres_D;
    wflux_pres_x3_itf_k[U_idx] = 0.5 * (wflux_pres_D + wflux_pres_U) / pres_U;
}
