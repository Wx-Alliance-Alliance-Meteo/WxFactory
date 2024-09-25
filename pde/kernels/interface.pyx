cdef extern from "euler_cartesian.h":
    void pointwise_euler_flux(double *Q, double *flux_x1, double *flux_x3, const int stride)
    void ausm_solver(double *Ql, double *Qr, double *fl, double *fr, const int nvars, const int direction, const int stride)
    void boundary_flux(double *Q, double *flux, const int direction, const int stride)

cpdef pointwise_fluxes(double[:, :, :] Q,
                       double[:, :, :] flux_x1,
                       double[:, :, :] flux_x3,
                       int nb_elements,
                       int nbsolpts):
    cdef:
        int i
        int stride

    stride = nb_elements * nbsolpts * nbsolpts

    for i in range(nb_elements):
        for j in range(nbsolpts*nbsolpts):
            pointwise_euler_flux(&Q[0, i, j], &flux_x1[0, i, j], &flux_x3[0, i, j], stride)


cpdef riemann_solver(double[:, :, :] Q_itf_x,
                     double[:, :, :] Q_itf_z,
                     double[:, :, :] common_flux_x,
                     double[:, :, :] common_flux_z,
                     const int nb_elements_x,
                     const int nb_elements_z,
                     const int nbsolpts):

    cdef:
        int nvars = Q_itf_x.shape[0]
        int stride = Q_itf_x[0].size

    interface_euler_fluxes_cartesian(Q_itf_x, Q_itf_z,
                                     common_flux_x, common_flux_z,
                                     nb_elements_x, nb_elements_z,
                                     nbsolpts, nvars, stride)

    set_boundary_conditions(Q_itf_x, Q_itf_z,
                            common_flux_x, common_flux_z,
                            nb_elements_x, nb_elements_z,
                            nbsolpts, nvars, stride)


cdef void interface_euler_fluxes_cartesian(double[:, :, :] Q_itf_x,
                                           double[:, :, :] Q_itf_z,
                                           double[:, :, :] common_flux_x,
                                           double[:, :, :] common_flux_z,
                                           const int nb_elements_x,
                                           const int nb_elements_z,
                                           const int nbsolpts,
                                           const int nvars,
                                           const int stride):

    cdef:
        int i, j, k, nb_total
        int ix, iz, ixr, izr, ind

    nb_total = nb_elements_x * nb_elements_z

    # Compute the riemann solves
    ix = 0
    iz = 0
    for i in range(nb_elements_z):
        for j in range(nb_elements_x):

            ixr = ix + 1
            izr = iz + nb_elements_x
            # For every dimension, compute the Riemann flux

            # At the interface, the left element uses the right wall
            # the right element provides its left face

            for k in range(nbsolpts):
                ind = k + nbsolpts
                if j + 1 < nb_elements_x:
                    ausm_solver(&Q_itf_x[0, ix, ind],
                                &Q_itf_x[0, ixr, k],
                                &common_flux_x[0, ix, ind],
                                &common_flux_x[0, ixr, k],
                                nvars,
                                0,
                                stride)

                if izr < nb_total:
                    ausm_solver(&Q_itf_z[0, iz, ind],
                                &Q_itf_z[0, izr, k],
                                &common_flux_z[0, iz, ind],
                                &common_flux_z[0, izr, k],
                                nvars,
                                1,
                                stride)

            # Increase the counters for each direction
            ix += 1
            iz += 1


cdef void set_boundary_conditions(double[:, :, :] Q_itf_x,
                                  double[:, :, :] Q_itf_z,
                                  double[:, :, :] common_flux_x,
                                  double[:, :, :] common_flux_z,
                                  const int nb_elements_x,
                                  const int nb_elements_z,
                                  const int nbsolpts,
                                  const int nvars,
                                  const int stride):

    # Normal fluxes are set to zero, except for the normal momentum
    # which is set to an extrapolated value of the neighbouring pressure

    cdef:
        int j, k
        int count, countr
        int nb_elements_total = nb_elements_x * nb_elements_z
        double rho_theta
        double * bound_flux
        int idx_rho, idx_rhou, idx_rhow, idx_rhot

    # Compute the strides to access state variables
    idx_rho = 0
    idx_rhou = stride
    idx_rhow = 2*stride
    idx_rhot = 3*stride

    # Set the boundary flux states
    count = nb_elements_x*(nb_elements_z-1)

    for j in range(nb_elements_x):
        for k in range(nbsolpts):

            # Bottom boundary
            boundary_flux(&Q_itf_z[0, j, k],
                           &common_flux_z[0, j, k], 1, stride)

            # Top boundary
            boundary_flux(&Q_itf_z[0, j+count, nbsolpts+k],
                           &common_flux_z[0, j+count, nbsolpts+k], 1, stride)

    count = 0
    for j in range(nb_elements_z):
        countr = count + nb_elements_x - 1
        for k in range(nbsolpts):

            # Left boundary
            boundary_flux(&Q_itf_x[0, count, k],
                           &common_flux_x[0, count, k], 0, stride)

            # Right boundary
            boundary_flux(&Q_itf_x[0, countr, k+nbsolpts],
                           &common_flux_x[0, countr, k+nbsolpts], 0, stride)

        count += nb_elements_x
