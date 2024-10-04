cdef extern from "euler.h":
    # Euler equations
    void pointwise_flux_euler(double *q, double *flux_x, double *flux_y, double *flux_z, const double sqrt_g, const double *metrics, const int stride, const int num_dim)
    void riemann_ausm_euler(double *ql, double *qr, double *fl, double *fr, const int nvars, const int direction, const int stride)
    void boundary_flux_euler(double *q, double *flux, const int direction, const int stride)

cdef extern from "shallow_water.h":
    # Shallow water equations
    void pointwise_flux_sw(double *q, double *flux_x, double *flux_y, double *flux_z, const double sqrt_g, const double *metrics, const int stride, const int num_dim)

# Define a function pointer type that matches the function signatures
ctypedef void (*flux_func_type)(double*, double*, double*, double*, const int)


cpdef pointwise_fluxes(double[:, :, :] q,
                       double[:, :, :] flux_x1,
                       double[:, :, :] flux_x2,
                       double[:, :, :] flux_x3,
                       double[:, :, :] metrics,
                       double[:, :] sqrt_g,
                       int nb_elements,
                       int nbsolpts,
                       int num_dim):
    cdef:
        int i
        int stride
        int nbsolpts_total = nbsolpts * nbsolpts

    if num_dim == 3:
      nbsolpts_total *= nbsolpts
  
    stride = nb_elements * nbsolpts_total

    if num_dim == 2:
      for i in range(nb_elements):
          for j in range(nbsolpts_total):
              pointwise_flux_sw(&q[0, i, j], &flux_x1[0, i, j], NULL, &flux_x3[0, i, j], sqrt_g[i, j], &metrics[0, i, j], stride, num_dim)

    elif num_dim == 3:
      for i in range(nb_elements):
          for j in range(nbsolpts_total):
              pointwise_flux_sw(&q[0, i, j], &flux_x1[0, i, j], &flux_x2[0, i, j], &flux_x3[0, i, j], sqrt_g[i, j], &metrics[0, i, j], stride, num_dim)    


cpdef riemann_solver(double[:, :, :] q_itf_x,
                     double[:, :, :] q_itf_z,
                     double[:, :, :] common_flux_x,
                     double[:, :, :] common_flux_z,
                     const int nb_elements_x,
                     const int nb_elements_z,
                     const int nbsolpts):

    cdef:
        int nvars = q_itf_x.shape[0]
        int stride = q_itf_x[0].size

    interface_euler_fluxes_cartesian(q_itf_x, q_itf_z,
                                     common_flux_x, common_flux_z,
                                     nb_elements_x, nb_elements_z,
                                     nbsolpts, nvars, stride)

    set_boundary_conditions(q_itf_x, q_itf_z,
                            common_flux_x, common_flux_z,
                            nb_elements_x, nb_elements_z,
                            nbsolpts, nvars, stride)


cdef void interface_euler_fluxes_cartesian(double[:, :, :] q_itf_x,
                                           double[:, :, :] q_itf_z,
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
                    riemann_ausm_euler(&q_itf_x[0, ix, ind],
                                       &q_itf_x[0, ixr, k],
                                       &common_flux_x[0, ix, ind],
                                       &common_flux_x[0, ixr, k],
                                       nvars,
                                       0,
                                       stride)

                if izr < nb_total:
                    riemann_ausm_euler(&q_itf_z[0, iz, ind],
                                       &q_itf_z[0, izr, k],
                                       &common_flux_z[0, iz, ind],
                                       &common_flux_z[0, izr, k],
                                       nvars,
                                       1,
                                       stride)

            # Increase the counters for each direction
            ix += 1
            iz += 1


cdef void set_boundary_conditions(double[:, :, :] q_itf_x,
                                  double[:, :, :] q_itf_z,
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
            boundary_flux_euler(&q_itf_z[0, j, k],
                           &common_flux_z[0, j, k], 1, stride)

            # Top boundary
            boundary_flux_euler(&q_itf_z[0, j+count, nbsolpts+k],
                           &common_flux_z[0, j+count, nbsolpts+k], 1, stride)

    count = 0
    for j in range(nb_elements_z):
        countr = count + nb_elements_x - 1
        for k in range(nbsolpts):

            # Left boundary
            boundary_flux_euler(&q_itf_x[0, count, k],
                           &common_flux_x[0, count, k], 0, stride)

            # Right boundary
            boundary_flux_euler(&q_itf_x[0, countr, k+nbsolpts],
                           &common_flux_x[0, countr, k+nbsolpts], 0, stride)

        count += nb_elements_x
