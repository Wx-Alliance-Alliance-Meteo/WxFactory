
extern "C" __global__ void new_df1_dx1(
    double* df1_dx1, const double* flux_x1, const double* ifaces_flux,
    const double* diff_solpt, const double* correction,
    double dx1, unsigned int nbsolpts, unsigned int nb_elements_z, unsigned int nb_elements_x)
{
    unsigned int i, j, k, start;
    unsigned int l, m;
    unsigned int Q_1, Q_2, F_1, F_2, F_3, D_1, C_1;
    double r;

    k = blockIdx.z * blockDim.z + threadIdx.z;
    if (k < nb_elements_x)
    {
        i = blockIdx.x;
        j = blockIdx.y;

        Q_2 = nb_elements_x * nbsolpts;
        Q_1 = nb_elements_z * nbsolpts * Q_2;
        F_3 = 2;
        F_2 = nb_elements_z * nbsolpts * F_3;
        F_1 = nb_elements_x * F_2;
        D_1 = nbsolpts;
        C_1 = 2;

        start = nbsolpts * k;
        for (l = 0; l < nbsolpts; ++l)
        {
            r = 0.0;
            for (m = 0; m < nbsolpts; ++m)
            {
                r += flux_x1[i * Q_1 + j * Q_2 + start + m] * diff_solpt[l * D_1 + m];
            }
            for (m = 0; m < 2; ++m)
            {
                r += ifaces_flux[i * F_1 + k * F_2 + j * F_3 + m] * correction[l * C_1 + m];
            }
            df1_dx1[i * Q_1 + j * Q_2 + start + l] = 2.0 * r / dx1;
        }
    }
}
