import numpy

from common.definitions import idx_h, idx_hu1, idx_hu2, gravity


def rhs_sw_nonstiff(Q: numpy.ndarray, geom, mtrx, metric, topo, ptopo, num_solpts: int, num_elements_hori: int):

    type_vec = Q.dtype
    num_equations = Q.shape[0]
    num_interfaces_hori = num_elements_hori + 1

    df1_dx1, df2_dx2, flux_x1, flux_x2 = [numpy.zeros_like(Q, dtype=type_vec) for _ in range(4)]

    flux_x1_itf_i = numpy.zeros(
        (num_equations, num_elements_hori + 2, num_solpts * num_elements_hori, 2), dtype=type_vec
    )
    flux_x2_itf_j, var_itf_i, var_itf_j = [
        numpy.zeros((num_equations, num_elements_hori + 2, 2, num_solpts * num_elements_hori), dtype=type_vec)
        for _ in range(3)
    ]

    forcing = numpy.zeros_like(Q, dtype=type_vec)

    # Offset due to the halo
    offset = 1

    # Unpack dynamical variables
    HH = Q[idx_h] if topo is None else Q[idx_h] + topo.hsurf
    u1 = Q[idx_hu1] / Q[idx_h]
    u2 = Q[idx_hu2] / Q[idx_h]

    # Interpolate to the element interface
    for elem in range(num_elements_hori):
        epais = elem * num_solpts + numpy.arange(num_solpts)
        pos = elem + offset

        # --- Direction x1

        var_itf_i[idx_h, pos, 0, :] = HH[:, epais] @ mtrx.extrap_west
        var_itf_i[idx_h, pos, 1, :] = HH[:, epais] @ mtrx.extrap_east

        var_itf_i[1:, pos, 0, :] = Q[1:, :, epais] @ mtrx.extrap_west
        var_itf_i[1:, pos, 1, :] = Q[1:, :, epais] @ mtrx.extrap_east

        # --- Direction x2
        var_itf_j[idx_h, pos, 0, :] = mtrx.extrap_south @ HH[epais, :]
        var_itf_j[idx_h, pos, 1, :] = mtrx.extrap_north @ HH[epais, :]

        var_itf_j[1:, pos, 0, :] = mtrx.extrap_south @ Q[1:, epais, :]
        var_itf_j[1:, pos, 1, :] = mtrx.extrap_north @ Q[1:, epais, :]

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
    flux_x1[idx_h] = 0.0
    flux_x2[idx_h] = 0.0

    flux_x1[idx_hu1] = metric.sqrtG * (Q[idx_hu1] * u1)
    flux_x2[idx_hu1] = metric.sqrtG * (Q[idx_hu1] * u2)

    flux_x1[idx_hu2] = metric.sqrtG * (Q[idx_hu2] * u1)
    flux_x2[idx_hu2] = metric.sqrtG * (Q[idx_hu2] * u2)

    # Interior contribution to the derivatives, corrections for the boundaries will be added later
    for elem in range(num_elements_hori):
        epais = elem * num_solpts + numpy.arange(num_solpts)

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

    # Substract topo after extrapolation
    if topo is not None:
        var_itf_i[idx_h] -= topo.hsurf_itf_i
        var_itf_j[idx_h] -= topo.hsurf_itf_j

    # Common AUSM fluxes
    for itf in range(num_interfaces_hori):

        elem_L = itf
        elem_R = itf + 1

        ################
        # Direction x1 #
        ################

        # Left state
        aL = numpy.sqrt(gravity * var_itf_i[idx_h, elem_L, 1, :] * metric.H_contra_11_itf_i[itf, :])
        mL = var_itf_i[idx_hu1, elem_L, 1, :] / (var_itf_i[idx_h, elem_L, 1, :] * aL)

        # Right state
        aR = numpy.sqrt(gravity * var_itf_i[idx_h, elem_R, 0, :] * metric.H_contra_11_itf_i[itf, :])
        mR = var_itf_i[idx_hu1, elem_R, 0, :] / (var_itf_i[idx_h, elem_R, 0, :] * aR)

        M = 0.25 * ((mL + 1.0) ** 2 - (mR - 1.0) ** 2)

        # --- Advection part

        flux_x1_itf_i[1:, elem_L, :, 1] = metric.sqrtG_itf_i[itf, :] * (
            numpy.maximum(0.0, M) * aL * var_itf_i[1:, elem_L, 1, :]
            + numpy.minimum(0.0, M) * aR * var_itf_i[1:, elem_R, 0, :]
        )

        flux_x1_itf_i[:, elem_R, :, 0] = flux_x1_itf_i[:, elem_L, :, 1]

        ################
        # Direction x2 #
        ################

        # Left state
        aL = numpy.sqrt(gravity * var_itf_j[idx_h, elem_L, 1, :] * metric.H_contra_22_itf_j[itf, :])
        mL = var_itf_j[idx_hu2, elem_L, 1, :] / (var_itf_j[idx_h, elem_L, 1, :] * aL)

        # Right state
        aR = numpy.sqrt(gravity * var_itf_j[idx_h, elem_R, 0, :] * metric.H_contra_22_itf_j[itf, :])
        mR = var_itf_j[idx_hu2, elem_R, 0, :] / (var_itf_j[idx_h, elem_R, 0, :] * aR)

        M = 0.25 * ((mL + 1.0) ** 2 - (mR - 1.0) ** 2)

        # --- Advection part

        flux_x2_itf_j[1:, elem_L, 1, :] = metric.sqrtG_itf_j[itf, :] * (
            numpy.maximum(0.0, M) * aL * var_itf_j[1:, elem_L, 1, :]
            + numpy.minimum(0.0, M) * aR * var_itf_j[1:, elem_R, 0, :]
        )

        # --- Pressure part

        flux_x2_itf_j[:, elem_R, 0, :] = flux_x2_itf_j[:, elem_L, 1, :]

    # Compute the derivatives
    for elem in range(num_elements_hori):
        epais = elem * num_solpts + numpy.arange(num_solpts)

        # --- Direction x1

        df1_dx1[1:, :, epais] += flux_x1_itf_i[1:, elem + offset, :, :] @ mtrx.correction_tr

        # --- Direction x2

        df2_dx2[1:, epais, :] += mtrx.correction @ flux_x2_itf_j[1:, elem + offset, :, :]

    if topo is None:
        topo_dzdx1 = numpy.zeros_like(metric.H_contra_11)
        topo_dzdx2 = numpy.zeros_like(metric.H_contra_11)
    else:
        topo_dzdx1 = topo.dzdx1
        topo_dzdx2 = topo.dzdx2

    # Add coriolis, metric and terms due to varying bottom topography
    # Note: christoffel_1_22 and metric.christoffel_2_11 are zero
    forcing[idx_hu1, :, :] = 2.0 * (
        metric.christoffel_1_01 * Q[idx_hu1] + metric.christoffel_1_02 * Q[idx_hu2]
    ) + gravity * Q[idx_h] * (metric.H_contra_11 * topo_dzdx1 + metric.H_contra_12 * topo_dzdx2)

    forcing[idx_hu2, :, :] = 2.0 * (
        metric.christoffel_2_01 * Q[idx_hu1] + metric.christoffel_2_02 * Q[idx_hu2]
    ) + gravity * Q[idx_h] * (metric.H_contra_21 * topo_dzdx1 + metric.H_contra_22 * topo_dzdx2)

    # Assemble the right-hand sides
    rhs = metric.inv_sqrtG * -(df1_dx1 + df2_dx2) - forcing

    return rhs
