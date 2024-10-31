import numpy

from common.definitions import idx_h, idx_hu1, idx_hu2, gravity


def rhs_advection2d(Q: numpy.ndarray, geom, mtrx, metric, ptopo, nbsolpts: int, nb_elements_hori: int):

    type_vec = Q.dtype
    nb_equations = Q.shape[0]
    nb_interfaces_hori = nb_elements_hori + 1

    idx_u1 = 1
    idx_u2 = 2

    df1_dx1, df2_dx2, flux_x1, flux_x2 = [numpy.zeros_like(Q, dtype=type_vec) for _ in range(4)]

    flux_x1_itf_i = numpy.zeros((nb_equations, nb_elements_hori + 2, nbsolpts * nb_elements_hori, 2), dtype=type_vec)
    flux_x2_itf_j, var_itf_i, var_itf_j = [
        numpy.zeros((nb_equations, nb_elements_hori + 2, 2, nbsolpts * nb_elements_hori), dtype=type_vec)
        for _ in range(3)
    ]

    # Offset due to the halo
    offset = 1

    # Interpolate to the element interface
    for elem in range(nb_elements_hori):
        epais = elem * nbsolpts + numpy.arange(nbsolpts)
        pos = elem + offset

        # --- Direction x1
        var_itf_i[:, pos, 0, :] = Q[:, :, epais] @ mtrx.extrap_west
        var_itf_i[:, pos, 1, :] = Q[:, :, epais] @ mtrx.extrap_east

        # --- Direction x2
        var_itf_j[:, pos, 0, :] = mtrx.extrap_south @ Q[:, epais, :]
        var_itf_j[:, pos, 1, :] = mtrx.extrap_north @ Q[:, epais, :]

    # Initiate transfers
    request_u = ptopo.start_exchange_vectors(
        (var_itf_j[idx_hu1, 1, 0], var_itf_j[idx_hu2, 1, 0]),  # South boundary
        (var_itf_j[idx_hu1, -2, 1], var_itf_j[idx_hu2, -2, 1]),  # North boundary
        (var_itf_i[idx_hu1, 1, 0], var_itf_i[idx_hu2, 1, 0]),  # West boundary
        (var_itf_i[idx_hu1, -2, 1], var_itf_i[idx_hu2, -2, 1]),  # East boundary
        geom.X[0, :],
        geom.Y[:, 0],
    )  # Coordinates at the boundary
    request_h = ptopo.start_exchange_scalars(
        var_itf_j[idx_h, 1, 0], var_itf_j[idx_h, -2, 1], var_itf_i[idx_h, 1, 0], var_itf_i[idx_h, -2, 1]
    )

    # Compute the fluxes
    flux_x1[idx_h] = metric.sqrtG * Q[idx_h] * Q[idx_u1]
    flux_x2[idx_h] = metric.sqrtG * Q[idx_h] * Q[idx_u2]

    # Interior contribution to the derivatives, corrections for the boundaries will be added later
    for elem in range(nb_elements_hori):
        epais = elem * nbsolpts + numpy.arange(nbsolpts)

        # --- Direction x1
        df1_dx1[:, :, epais] = flux_x1[:, :, epais] @ mtrx.diff_solpt_tr

        # --- Direction x2
        df2_dx2[:, epais, :] = mtrx.diff_solpt @ flux_x2[:, epais, :]

    # Finish transfers
    (
        (var_itf_j[idx_hu1, 0, 1], var_itf_j[idx_hu2, 0, 1]),
        (var_itf_j[idx_hu1, -1, 0], var_itf_j[idx_hu2, -1, 0]),
        (var_itf_i[idx_hu1, 0, 1], var_itf_i[idx_hu2, 0, 1]),
        (var_itf_i[idx_hu1, -1, 0], var_itf_i[idx_hu2, -1, 0]),
    ) = request_u.wait()
    var_itf_j[idx_h, 0, 1], var_itf_j[idx_h, -1, 0], var_itf_i[idx_h, 0, 1], var_itf_i[idx_h, -1, 0] = request_h.wait()

    # Common AUSM fluxes
    for itf in range(nb_interfaces_hori):

        elem_L = itf
        elem_R = itf + 1

        ################
        # Direction x1 #
        ################

        eig_L = numpy.abs(var_itf_i[idx_u1, elem_L, 1, :])
        eig_R = numpy.abs(var_itf_i[idx_u1, elem_R, 0, :])

        eig = numpy.maximum(eig_L, eig_R)

        flux_L = metric.sqrtG_itf_i[itf, :] * var_itf_i[idx_h, elem_L, 1, :] * var_itf_i[idx_u1, elem_L, 1, :]
        flux_R = metric.sqrtG_itf_i[itf, :] * var_itf_i[idx_h, elem_R, 0, :] * var_itf_i[idx_u1, elem_R, 0, :]

        flux_x1_itf_i[idx_h, elem_L, :, 1] = 0.5 * (
            flux_L
            + flux_R
            - eig * metric.sqrtG_itf_i[itf, :] * (var_itf_i[idx_h, elem_R, 0, :] - var_itf_i[idx_h, elem_L, 1, :])
        )
        flux_x1_itf_i[idx_h, elem_R, :, 0] = flux_x1_itf_i[idx_h, elem_L, :, 1]

        ################
        # Direction x2 #
        ################

        eig_L = numpy.abs(var_itf_j[idx_u2, elem_L, 1, :])
        eig_R = numpy.abs(var_itf_j[idx_u2, elem_R, 0, :])

        eig = numpy.maximum(eig_L, eig_R)

        flux_L = metric.sqrtG_itf_j[itf, :] * var_itf_j[idx_h, elem_L, 1, :] * var_itf_j[idx_u2, elem_L, 1, :]
        flux_R = metric.sqrtG_itf_j[itf, :] * var_itf_j[idx_h, elem_R, 0, :] * var_itf_j[idx_u2, elem_R, 0, :]

        flux_x2_itf_j[idx_h, elem_L, 1, :] = 0.5 * (
            flux_L
            + flux_R
            - eig * metric.sqrtG_itf_j[itf, :] * (var_itf_j[idx_h, elem_R, 0, :] - var_itf_j[idx_h, elem_L, 1, :])
        )
        flux_x2_itf_j[idx_h, elem_R, 0, :] = flux_x2_itf_j[idx_h, elem_L, 1, :]

    # Compute the derivatives
    for elem in range(nb_elements_hori):
        epais = elem * nbsolpts + numpy.arange(nbsolpts)

        # --- Direction x1

        df1_dx1[:, :, epais] += flux_x1_itf_i[:, elem + offset, :, :] @ mtrx.correction_tr

        # --- Direction x2

        df2_dx2[:, epais, :] += mtrx.correction @ flux_x2_itf_j[:, elem + offset, :, :]

    # Assemble the right-hand sides
    rhs = metric.inv_sqrtG * -(df1_dx1 + df2_dx2)

    rhs[idx_u1, :, :] = 0.0
    rhs[idx_u2, :, :] = 0.0

    return rhs
