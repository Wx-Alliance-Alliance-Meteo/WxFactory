cimport cython

ctypedef fused pynum_t:
    float
    double

cdef extern from "kernels/pointwise_flux.h":
    void pointwise_eulercartesian_2d[num_t](const num_t *q, num_t *flux_x1, num_t *flux_x2, const int stride)
    void pointwise_swcubedsphere_2d[num_t](const num_t *q, num_t *flux_x, num_t *flux_y, num_t *flux_z, const double sqrt_g, const double *metrics, const int stride)

cdef extern from "kernels/riemann_flux.h":
    void riemann_eulercartesian_ausm_2d[num_t](const num_t *ql, const num_t *qr, num_t *fl, num_t* fr, const int nvar, const int direction, const int stride)

cdef extern from "kernels/boundary_flux.h":
    void boundary_eulercartesian_2d[num_t](const num_t* q, num_t*flux, const int direction, const int stride)

# -------------------------------------
# Pointwise flux wrappers
# -------------------------------------
  
cpdef pointwise_eulercartesian_2d_wrapper(pynum_t[:, :, :, :] q, 
                                          pynum_t[:, :, :, :] flux_x1,
                                          pynum_t[:, :, :, :] flux_x2,
                                          int nb_elements_x1,
                                          int nb_elements_x2,
                                          int nb_solpts_total):
   
    """Pointwise flux for the Euler 2D cartesian grid"""
    cdef:
        int i, j, k
        int stride
        int nb_elements_total   
    
    nb_elements_total = nb_elements_x1 * nb_elements_x2
    stride = nb_elements_total * nb_solpts_total 

    for i in range(nb_elements_x2):
        for j in range(nb_elements_x1):
            for k in range(nb_solpts_total):
                pointwise_eulercartesian_2d(&q[0, i, j, k], 
                                            &flux_x1[0, i, j, k], 
                                            &flux_x2[0, i, j, k], 
                                            stride)


# -------------------------------------
# Riemann flux wrappers
# -------------------------------------

cpdef riemann_eulercartesian_ausm_2d_wrapper(pynum_t[:, :, :, :] q_itf_x1,
                                             pynum_t[:, :, :, :] q_itf_x2,
                                             pynum_t[:, :, :, :] common_flux_x1,
                                             pynum_t[:, :, :, :] common_flux_x2,
                                             const int nb_elements_x1,
                                             const int nb_elements_x2,
                                             const int nbsolpts,
                                             const int nvars,
                                             const int stride):
    
    cdef int i, j, k, ind

    # Compute the riemann solves
    ix = 0
    iz = 0
    for i in range(nb_elements_x2):
        for j in range(nb_elements_x1):
            # At the interface, the left element uses the right wall
            # the right element provides its left face

            # Riemann solve along the x1 direction
            if j + 1 < nb_elements_x1:
                for k in range(nbsolpts):
                    ind = k + nbsolpts
                    riemann_eulercartesian_ausm_2d(&q_itf_x1[0, i, j, ind],
                                                   &q_itf_x1[0, i, j + 1, k],
                                                   &common_flux_x1[0, i, j, ind],
                                                   &common_flux_x1[0, i, j + 1, k],
                                                   nvars,
                                                   0,
                                                   stride)

            # Riemann solve along the x2 direction
            if i + 1 < nb_elements_x2:
                for k in range(nbsolpts):
                    ind = k + nbsolpts
                    riemann_eulercartesian_ausm_2d(&q_itf_x2[0, i, j, ind],
                                                   &q_itf_x2[0, i + 1, j, k],
                                                   &common_flux_x2[0, i, j, ind],
                                                   &common_flux_x2[0, i + 1, j, k],
                                                   nvars,
                                                   1,
                                                   stride)

# -------------------------------------
# Boundary flux wrappers
# -------------------------------------

cpdef boundary_eulercartesian_2d_wrapper(pynum_t[:, :, :, :] q_itf_x1,
                                         pynum_t[:, :, :, :] q_itf_x2,
                                         pynum_t[:, :, :, :] common_flux_x1,
                                         pynum_t[:, :, :, :] common_flux_x2,
                                         const int nb_elements_x1,
                                         const int nb_elements_x2,
                                         const int nbsolpts,
                                         const int nvars,
                                         const int stride):

    # Normal fluxes are set to zero, except for the normal momentum
    # which is set to an extrapolated value of the neighbouring pressure

    cdef int j, k

    # Set the boundary flux states
    for j in range(nb_elements_x1):
        for k in range(nbsolpts):

            # Bottom boundary
            boundary_eulercartesian_2d(&q_itf_x2[0, 0, j, k],
                                       &common_flux_x2[0, 0, j, k], 1, stride)

            # Top boundary
            boundary_eulercartesian_2d(&q_itf_x2[0, -1, j, nbsolpts+k],
                                       &common_flux_x2[0, -1, j, nbsolpts+k], 1, stride)


    for j in range(nb_elements_x2):
        for k in range(nbsolpts):

            # Left boundary
            boundary_eulercartesian_2d(&q_itf_x1[0, j, 0, k],
                                       &common_flux_x1[0, j, 0, k], 0, stride)

            # Right boundary
            boundary_eulercartesian_2d(&q_itf_x1[0, j, -1, k+nbsolpts],
                                       &common_flux_x1[0, j, -1, k+nbsolpts], 0, stride)
