
/* if s = kfaces_flux.shape
 * then run with (s[3], s[1], s[0]) threads
 */
extern "C" __global__ void set_kfaces_var(
    double* kfaces_var, const double* Q, const double* extrap_up, const double* extrap_down,
    unsigned int num_solpts, unsigned int nb_elements_z, unsigned int nb_elements_x)
{
    unsigned int i, j, k, start, idx, l;
    unsigned int Q_1, Q_2, F_1, F_2, F_3;
    double r, s;

    k = blockIdx.x * blockDim.x + threadIdx.x;
    Q_2 = num_solpts * nb_elements_x;
    if (k < Q_2)
    {
        j = blockIdx.y;
        i = blockIdx.z;

        Q_1 = num_solpts * nb_elements_z * Q_2;
        F_3 = num_solpts * nb_elements_x;
        F_2 = 2 * F_3;
        F_1 = nb_elements_z * F_2;

        r = s = 0.0;
        start = num_solpts * j;
        for (l = 0; l < num_solpts; ++l)
        {
            idx = i * Q_1 + (start + l) * Q_2 + k;
            r += extrap_down[l] * Q[idx];
            s += extrap_up[l] * Q[idx];
        }

        kfaces_var[i * F_1 + j * F_2 + k] = r;
        kfaces_var[i * F_1 + j * F_2 + F_3 + k] = s;
    }
}

/* if s = ifaces_flux.shape
 * then run with (s[2], s[1], s[0]) threads
 */
extern "C" __global__ void set_ifaces_var(
    double* ifaces_var, const double* Q, const double* extrap_west, const double* extrap_east,
    unsigned int num_solpts, unsigned int nb_elements_z, unsigned int nb_elements_x)
{
    unsigned int i, j, k, start, idx, l;
    unsigned int Q_1, Q_2, F_1, F_2, F_3;
    double r, s;

    k = blockIdx.x * blockDim.x + threadIdx.x;
    Q_1 = num_solpts * nb_elements_z;
    if (k < Q_1)
    {
        j = blockIdx.y;
        i = blockIdx.z;

        Q_2 = num_solpts * nb_elements_x;
        Q_1 *= Q_2;
        F_3 = 2;
        F_2 = num_solpts * nb_elements_z * F_3;
        F_1 = nb_elements_x * F_2;

        r = s = 0.0;
        start = num_solpts * j;
        for (l = 0; l < num_solpts; ++l)
        {
            idx = i * Q_1 + k * Q_2 + start + l;
            r += Q[idx] * extrap_west[l];
            s += Q[idx] * extrap_east[l];
        }

        ifaces_var[i * F_1 + j * F_2 + k * F_3] = r;
        ifaces_var[i * F_1 + j * F_2 + k * F_3 + 1] = s;
    }
}

/* if s = kfaces_flux.shape
 * then run with (s[3], s[1], 1) threads
 */
extern "C" __global__ void set_kfaces_flux(
    double* kfaces_flux, const double* kfaces_pres, const double* kfaces_var,
    unsigned int nb_equations, unsigned int num_solpts, unsigned int nb_elements_z, unsigned int nb_elements_x)
{
    unsigned int i, left, right, k;
    unsigned int F_1, F_2, F_3;
    double a_L, M_L, a_R, M_R, M;
    double r, rho_w_extra;

    k = blockIdx.x * blockDim.x + threadIdx.x;
    F_3 = num_solpts * nb_elements_x;
    if (k < F_3)
    {
        left = blockIdx.y;
        right = left + 1;

        F_2 = 2 * F_3;
        F_1 = nb_elements_z * F_2;

        a_L = sqrt(heat_capacity_ratio * kfaces_pres[left * F_2 + F_3 + k] / kfaces_var[idx_2d_rho * F_1 + left * F_2 + F_3 + k]);
        M_L = kfaces_var[idx_2d_rho_w * F_1 + left * F_2 + F_3 + k] / (kfaces_var[idx_2d_rho * F_1 + left * F_2 + F_3 + k] * a_L);

        a_R = sqrt(heat_capacity_ratio * kfaces_pres[right * F_2 + k] / kfaces_var[idx_2d_rho * F_1 + right * F_2 + k]);
        M_R = kfaces_var[idx_2d_rho_w * F_1 + right * F_2 + k] / (kfaces_var[idx_2d_rho * F_1 + right * F_2 + k] * a_R);

        M = 0.25 * ((M_L + 1.) * (M_L + 1.) - (M_R - 1.) * (M_R - 1.));
        rho_w_extra = 0.5 * ((1. + M_L) * kfaces_pres[left * F_2 + F_3 + k] + (1. - M_R) * kfaces_pres[right * F_2 + k]);

        if (M > 0.)
        {
            for (i = 0; i < nb_equations; ++i)
            {
                r = kfaces_var[i * F_1 + left * F_2 + F_3 + k] * M * a_L;
                if (i == idx_2d_rho_w) r += rho_w_extra;
                kfaces_flux[i * F_1 + right * F_2 + k] = kfaces_flux[i * F_1 + left * F_2 + F_3 + k] = r;
            }
        }
        else
        {
            for (i = 0; i < nb_equations; ++i)
            {
                r = kfaces_var[i * F_1 + right * F_2 + k] * M * a_R;
                if (i == idx_2d_rho_w) r += rho_w_extra;
                kfaces_flux[i * F_1 + right * F_2 + k] = kfaces_flux[i * F_1 + left * F_2 + F_3 + k] = r;
            }
        }
    }
}

/* if s = ifaces_flux.shape
 * then run with (s[2], s[1], 1) threads
 */
extern "C" __global__ void set_ifaces_flux(
    double* ifaces_flux, const double* ifaces_pres, const double* ifaces_var,
    unsigned int nb_equations, unsigned int num_solpts, unsigned int nb_elements_z, unsigned int nb_elements_x,
    bool xperiodic)
{
    unsigned int i, left, right, k;
    unsigned int F_1, F_2, F_3;
    double a_L, M_L, a_R, M_R, M;
    double r, rho_u_extra;

    k = blockIdx.x * blockDim.x + threadIdx.x;
    F_2 = num_solpts * nb_elements_z;
    if (k < F_2)
    {
        left = blockIdx.y;
        right = left + 1;

        F_3 = 2;
        F_2 *= F_3;
        F_1 = nb_elements_x * F_2;

        if (xperiodic && left == 0) left = -1;

        a_L = sqrt(heat_capacity_ratio * ifaces_pres[left * F_2 + k * F_3 + 1] / ifaces_var[idx_2d_rho * F_1 + left * F_2 + k * F_3 + 1]);
        M_L = ifaces_var[idx_2d_rho_u * F_1 + left * F_2 + k * F_3 + 1] / (ifaces_var[idx_2d_rho * F_1 + left * F_2 + k * F_3 + 1] * a_L);

        a_R = sqrt(heat_capacity_ratio * ifaces_pres[right * F_2 + k * F_3] / ifaces_var[idx_2d_rho * F_1 + right * F_2 + k * F_3]);
        M_R = ifaces_var[idx_2d_rho_u * F_1 + right * F_2 + k * F_3] / (ifaces_var[idx_2d_rho * F_1 + right * F_2 + k * F_3] * a_R);

        M = 0.25 * ((M_L + 1.) * (M_L + 1.) - (M_R - 1.) * (M_R - 1.));
        rho_u_extra = 0.5 * ((1. + M_L) * ifaces_pres[left * F_2 + k * F_3 + 1] + (1. - M_R) * ifaces_pres[right * F_2 + k * F_3]);

        if (M > 0.)
        {
            for (i = 0; i < nb_equations; ++i)
            {
                r = ifaces_var[i * F_1 + left * F_2 + k * F_3 + 1] * M * a_L;
                if (i == idx_2d_rho_u) r += rho_u_extra;
                ifaces_flux[i * F_1 + right * F_2 + k * F_3] = ifaces_flux[i * F_1 + left * F_2 + k * F_3 + 1] = r;
            }
        }
        else
        {
            for (i = 0; i < nb_equations; ++i)
            {
                r = ifaces_var[i * F_1 + right * F_2 + k * F_3] * M * a_R;
                if (i == idx_2d_rho_u) r += rho_u_extra;
                ifaces_flux[i * F_1 + right * F_2 + k * F_3] = ifaces_flux[i * F_1 + left * F_2 + k * F_3 + 1] = r;
            }
        }
    }
}

/* if s = df3_dx3.shape
 * then run with (s[2], s[1], s[0]) threads
 */
extern "C" __global__ void set_df3_dx3(
    double* df3_dx3, const double* flux_x3, const double* kfaces_flux,
    const double* diff_solpt, const double* correction,
    double dx3, unsigned int num_solpts, unsigned int nb_elements_z, unsigned int nb_elements_x)
{
    unsigned int i, j, k, l, elm, rem, start;
    unsigned int Q_1, Q_2, F_1, F_2, F_3, D_1, C_1;
    double r;
    
    k = blockIdx.x * blockDim.x + threadIdx.x;
    Q_2 = num_solpts * nb_elements_x;
    if (k < Q_2)
    {
        j = blockIdx.y;
        i = blockIdx.z;
        elm = j / num_solpts;
        rem = j % num_solpts;
        start = elm * num_solpts;

        Q_1 = num_solpts * nb_elements_z * Q_2;
        F_3 = num_solpts * nb_elements_x;
        F_2 = 2 * F_3;
        F_1 = nb_elements_z * F_2;
        D_1 = num_solpts;
        C_1 = 2;

        r = 0.0;
        for (l = 0; l < num_solpts; ++l)
        {
            r += diff_solpt[rem * D_1 + l] * flux_x3[i * Q_1 + (start + l) * Q_2 + k];
        }
        for (l = 0; l < 2; ++l)
        {
            r += correction[rem * C_1 + l] * kfaces_flux[i * F_1 + elm * F_2 + l * F_3 + k];
        }
        df3_dx3[i * Q_1 + j * Q_2 + k] = 2.0 * r / dx3;
    }
}

/* if s = df1_dx1.shape
 * then run with (s[2], s[1], s[0]) threads
 */
extern "C" __global__ void set_df1_dx1(
    double* df1_dx1, const double* flux_x1, const double* ifaces_flux,
    const double* diff_solpt, const double* correction,
    double dx1, unsigned int num_solpts, unsigned int nb_elements_z, unsigned int nb_elements_x)
{
    unsigned int i, j, k, l, elm, rem, start;
    unsigned int Q_1, Q_2, F_1, F_2, F_3, D_1, C_1;
    double r;

    k = blockIdx.x * blockDim.x + threadIdx.x;
    Q_2 = num_solpts * nb_elements_x;
    if (k < Q_2)
    {
        j = blockIdx.y;
        i = blockIdx.z;
        elm = k / num_solpts;
        rem = k % num_solpts;
        start = elm * num_solpts;

        Q_1 = num_solpts * nb_elements_z * Q_2;
        F_3 = 2;
        F_2 = nb_elements_z * num_solpts * F_3;
        F_1 = nb_elements_x * F_2;
        D_1 = num_solpts;
        C_1 = 2;

        r = 0.0;
        for (l = 0; l < num_solpts; ++l)
        {
            r += flux_x1[i * Q_1 + j * Q_2 + start + l] * diff_solpt[rem * D_1 + l];
        }
        for (l = 0; l < 2; ++l)
        {
            r += ifaces_flux[i * F_1 + elm * F_2 + j * F_3 + l] * correction[rem * C_1 + l];
        }
        df1_dx1[i * Q_1 + j * Q_2 + k] = 2.0 * r / dx1;
    }
}
