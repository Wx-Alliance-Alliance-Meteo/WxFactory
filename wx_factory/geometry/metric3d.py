import math
import sys

from mpi4py import MPI
import numpy
from numpy.typing import NDArray

from .cubed_sphere_3d import CubedSphere3D
from .operators import DFROperators


class Metric3DTopo:
    def __init__(self, geom: CubedSphere3D, matrix: DFROperators):
        """Token initialization: store geometry and matrix objects.  Defer construction of the metric itself,
        so that initialization can take place after topography is defined inside the 'geom' object"""

        self.geom = geom
        self.matrix = matrix
        self.deep = geom.deep

    def build_metric(self):
        """Construct the metric terms, with the assurance that topography is now defined.  This defines full, 3D arrays
        for the metric and Christoffel symbols."""

        # Retrieve objects for easier access
        geom = self.geom
        matrix = self.matrix
        xp = geom.device.xp
        dtype = geom.gnomonic.dtype

        # Whether computing deep or shallow metric
        deep = self.deep

        # Shorthand variable for the globe radius, used when computing a shallow-atmosphere metric
        A = geom.earth_radius

        # Gnomonic coordinates in element interiors
        X_int = geom.coordVec_gnom[0, :, :, :]
        Y_int = geom.coordVec_gnom[1, :, :, :]
        R_int = geom.coordVec_gnom[2, :, :, :] + geom.earth_radius
        self.R_int = R_int

        X_int_new = geom.gnomonic[0, ...]
        Y_int_new = geom.gnomonic[1, ...]
        R_int_new = geom.gnomonic[2, ...] + geom.earth_radius

        # Gnomonic coordinates at i-interface
        X_itf_i = geom.coordVec_gnom_itf_i[0, :, :, :]
        Y_itf_i = geom.coordVec_gnom_itf_i[1, :, :, :]
        R_itf_i = geom.coordVec_gnom_itf_i[2, :, :, :] + geom.earth_radius

        X_itf_i_new = geom.gnomonic_itf_i[0, ...]
        Y_itf_i_new = geom.gnomonic_itf_i[1, ...]
        R_itf_i_new = geom.gnomonic_itf_i[2, ...] + geom.earth_radius

        # At j-interface
        X_itf_j = geom.coordVec_gnom_itf_j[0, :, :, :]
        Y_itf_j = geom.coordVec_gnom_itf_j[1, :, :, :]
        R_itf_j = geom.coordVec_gnom_itf_j[2, :, :, :] + geom.earth_radius

        X_itf_j_new = geom.gnomonic_itf_j[0, ...]
        Y_itf_j_new = geom.gnomonic_itf_j[1, ...]
        R_itf_j_new = geom.gnomonic_itf_j[2, ...] + geom.earth_radius

        # Gnomonic coordinates at k-interface
        X_itf_k = geom.coordVec_gnom_itf_k[0, :, :, :]
        Y_itf_k = geom.coordVec_gnom_itf_k[1, :, :, :]
        R_itf_k = geom.coordVec_gnom_itf_k[2, :, :, :] + geom.earth_radius

        X_itf_k_new = geom.gnomonic_itf_k[0, ...]
        Y_itf_k_new = geom.gnomonic_itf_k[1, ...]
        R_itf_k_new = geom.gnomonic_itf_k[2, ...] + geom.earth_radius

        # Grid scaling factors, necessary to define the metric and Christoffel symbols
        # with respect to the standard element
        delta_x = geom.delta_x1
        delta_y = geom.delta_x2
        delta_eta = geom.delta_eta

        # Grid rotation terms
        alpha = geom.angle_p  # Clockwise rotation about the Z axis
        phi = geom.lat_p  # Counterclockwise rotation about the X axis, sending [0,0,1] to a particular latitude
        lam = geom.lon_p  # Counterclockwise rotation about the Z axis, sending [0,-1,0] to a particular longitude
        salp = math.sin(alpha)
        calp = math.cos(alpha)
        sphi = math.sin(phi)
        cphi = math.cos(phi)
        slam = math.sin(lam)
        clam = math.cos(lam)

        ## Compute partial derivatives of R

        # First, we won't use R.  R = H + (radius), and we can improve numerical conditioning by removing that
        # DC offset term.  dR/d(stuff) = dH/d(stuff)
        height_int = geom.coordVec_gnom[2, :, :, :]
        height_itf_i = geom.coordVec_gnom_itf_i[2, :, :, :]
        height_itf_j = geom.coordVec_gnom_itf_j[2, :, :, :]
        height_itf_k = geom.coordVec_gnom_itf_k[2, :, :, :]

        height_int_new = geom.gnomonic[2, ...]
        height_itf_i_new = geom.gnomonic_itf_i[2, ...]
        height_itf_j_new = geom.gnomonic_itf_j[2, ...]
        height_itf_k_new = geom.gnomonic_itf_k[2, ...]

        # Build the boundary-extensions of h, based on the interface boundaries
        # ext_i shape: (nk, nj, num_elements_x1, 2) - west/east boundaries
        height_ext_i = xp.stack((height_itf_i[:, :, :-1], height_itf_i[:, :, 1:]), axis=-1)
        # ext_j shape: (nk, num_elements_x2, 2, ni) - south/north boundaries
        height_ext_j = xp.stack((height_itf_j[:, :-1, :], height_itf_j[:, 1:, :]), axis=-2)
        # ext_k shape: (num_elements_x3, 2, nj, ni) - bottom/top boundaries
        height_ext_k = xp.stack((height_itf_k[:-1, :, :], height_itf_k[1:, :, :]), axis=-3)

        dRdx1_int = matrix.comma_i(height_int, height_ext_i, geom) * 2 / delta_x
        dRdx2_int = matrix.comma_j(height_int, height_ext_j, geom) * 2 / delta_y
        dRdeta_int = matrix.comma_k(height_int, height_ext_k, geom) * 2 / delta_eta

        dRdx1_int_new = (
            height_int_new @ matrix.derivative_x + height_itf_i_new[..., 1:-1, :] @ matrix.correction_WE
        ) * (2 / delta_x)
        dRdx2_int_new = (
            height_int_new @ matrix.derivative_y + height_itf_j_new[..., 1:-1, :, :] @ matrix.correction_SN
        ) * (2 / delta_y)
        dRdeta_int_new = (
            height_int_new @ matrix.derivative_z + height_itf_k_new[..., 1:-1, :, :, :] @ matrix.correction_DU
        ) * (2 / delta_eta)

        # def to_new_itf_j(a):
        #     src_shape = (
        #         geom.num_elements_x3 * geom.num_solpts,
        #         geom.num_elements_x2,
        #         2,
        #         geom.num_elements_x1 * geom.num_solpts,
        #     )
        #     if a.shape[-4:] != src_shape:
        #         raise ValueError(f"Wrong shape {a.shape}, expected (...,) + {src_shape}")

        #     tmp_shape1 = a.shape[:-4] + (
        #         geom.num_elements_x3,
        #         geom.num_solpts,
        #         geom.num_elements_x2,
        #         2,
        #         geom.num_elements_x1,
        #         geom.num_solpts,
        #     )
        #     tmp_shape2 = a.shape[:-4] + (
        #         geom.num_elements_x3,
        #         geom.num_elements_x2,
        #         geom.num_elements_x1,
        #         geom.num_solpts**2 * 2,
        #     )

        #     tmp_1 = a.reshape(tmp_shape1)
        #     tmp_2 = xp.moveaxis(tmp_1, (-5, -2), (-2, -4))
        #     tmp_array = tmp_2.reshape(tmp_shape2)

        #     return tmp_array.copy()

        # sys.stdout.flush()
        # MPI.COMM_WORLD.barrier()

        # diffh = height_int_new - geom._to_new(height_int)
        # diffhn = xp.linalg.norm(diffh)
        # if diffhn > 1e-15:
        #     raise ValueError

        # diffhj = height_itf_j_new[..., 1:-1, :, :] - to_new_itf_j(height_ext_j)
        # diffhjn = xp.linalg.norm(diffhj)
        # if diffhjn > 1e-15:
        #     print(f"diffhjn = {diffhjn:.2e}")
        #     raise ValueError

        # diff1 = dRdx1_int_new - geom._to_new(dRdx1_int)
        # diff1n = xp.linalg.norm(diff1) / xp.linalg.norm(dRdx1_int)
        # if diff1n > 1e-16:
        #     print(f"rank {MPI.COMM_WORLD.rank} diff i {diff1n:.2e}")
        #     raise ValueError

        # diff2 = dRdx2_int_new - geom._to_new(dRdx2_int)
        # diff2n = xp.linalg.norm(diff2) / xp.linalg.norm(dRdx2_int)
        # if diff2n > 1e-16:
        #     print(f"rank {MPI.COMM_WORLD.rank} diff j {diff2n:.2e}")
        #     # if MPI.COMM_WORLD.rank == 2:
        #     #     print(f"new deriv \n{height_int_new @ matrix.derivative_y}")
        #     #     print(f"new corr \n{height_itf_j_new[..., 1:-1, :, :] @ matrix.correction_SN}")
        #     #     print(
        #     #         f"{MPI.COMM_WORLD.rank} diff {diff2n:.2e}\n"
        #     #         f"old = \n{dRdx2_int}\n"
        #     #         f"old w/ new shape = \n{geom._to_new(dRdx2_int)}\n"
        #     #         f"new = \n{dRdx2_int_new}\n"
        #     #         f"diff = \n{diff2}"
        #     #     )

        #     raise ValueError

        # diff3 = dRdeta_int_new - geom._to_new(dRdeta_int)
        # diff3n = xp.linalg.norm(diff3) / xp.linalg.norm(dRdeta_int)
        # if diff3n > 1e-16:
        #     print(f"rank {MPI.COMM_WORLD.rank} diff k {diff3n:.2e}")
        #     raise ValueError

        # The i/j interface values now need to be "fixed up" with a boundary exchange.  However, the existing vector
        # exchange code demands contravariant components, and dRd(...) is covariant.  We can perform the conversion
        # by constructing a (temporary) 2D metric in terms of X and Y only at the interfaces:

        metric_2d_contra_itf_i = xp.zeros((2, 2) + geom.itf_i_shape_3d)
        metric_2d_contra_itf_j = xp.zeros((2, 2) + geom.itf_j_shape_3d)
        metric_2d_cov_itf_i = xp.zeros((2, 2) + geom.itf_i_shape_3d)
        metric_2d_cov_itf_j = xp.zeros((2, 2) + geom.itf_j_shape_3d)

        for metric_contra, metric_cov, X, Y in zip(
            (metric_2d_contra_itf_i, metric_2d_contra_itf_j),
            (metric_2d_cov_itf_i, metric_2d_cov_itf_j),
            (X_itf_i, X_itf_j),
            (Y_itf_i, Y_itf_j),
        ):
            delta2 = 1 + X**2 + Y**2
            metric_contra[0, 0, :, :, :] = delta2 / (1 + X**2)
            metric_contra[0, 1, :, :, :] = delta2 * X * Y / ((1 + X**2) + (1 + Y**2))
            metric_contra[1, 0, :, :, :] = metric_contra[0, 1, :, :, :]
            metric_contra[1, 1, :, :, :] = delta2 / (1 + Y**2)

            metric_cov[0, 0, :, :, :] = (1 + X**2) ** 2 * (1 + Y**2) / delta2**2
            metric_cov[0, 1, :, :, :] = -X * Y * (1 + X**2) * (1 + Y**2) / delta2**2
            metric_cov[1, 0, :, :, :] = metric_cov[0, 1, :, :, :]
            metric_cov[1, 1, :, :, :] = (1 + X**2) * (1 + Y**2) ** 2 / delta2**2

        # Arrays for boundary info:  arrays for parallel exchange have a different shape than the 'natural'
        # extrapolation,
        # in order for the MPI exchange to occur with contiguous subarrays.

        exch_itf_i = xp.zeros((3, geom.nk, geom.num_elements_x1 + 2, 2, geom.nj))
        exch_itf_j = xp.zeros((3, geom.nk, geom.num_elements_x2 + 2, 2, geom.ni))

        # self.itf_i_shape = (self.num_elements_x3, self.num_elements_x2, self.num_elements_x1 + 2, (num_solpts**2) * 2)
        def to_new_i(a: NDArray):
            exp_shape = (geom.nk, geom.num_elements_x1 + 2, 2, geom.nj)
            # exp_shape2 = (geom.nk, geom.num_elements_x1, 2, geom.nj)
            exp_shape2 = exp_shape
            if a.shape not in [exp_shape, exp_shape2]:
                raise ValueError(
                    f"error, expected shape (..., {exp_shape[0]}, {exp_shape[1]}[+2], {exp_shape[2]}, {exp_shape[3]}), "
                    f"not {a.shape}"
                )

            tmp_shape1 = (
                geom.num_elements_x3,
                geom.num_solpts,
                a.shape[1],
                2,
                geom.num_elements_x2,
                geom.num_solpts,
            )

            tmp1 = a.reshape(tmp_shape1)
            tmp2 = tmp1.transpose(0, 4, 2, 3, 1, 5)

            final_shape = geom.itf_i_shape
            # if a.shape[1] == exp_shape2[1]:
            #     result = numpy.zeros(final_shape, dtype=a.dtype)
            #     tmp_shape2 = (geom.itf_i_shape[0], geom.itf_i_shape[1], geom.itf_i_shape[2] - 2, geom.itf_i_shape[3])
            #     result[..., 1:-1, :] = tmp2.reshape(tmp_shape2)
            # else:
            result = tmp2.reshape(final_shape)

            return result

        # Perform extrapolation.  Extrapolation in i and j will be written to arrays for exchange, but k does not
        # require an exchange; we can average directly and will handle this afterwards
        dRdx1_itf_k = xp.empty_like(R_itf_k)
        dRdx2_itf_k = xp.empty_like(R_itf_k)
        dRdeta_itf_k = xp.empty_like(R_itf_k)

        # Extrapolate the interior values to each edge
        dRdx1_extrap_i = matrix.extrapolate_i(dRdx1_int, geom)  # Output dims: (nk,nj,nel_x,2)
        dRdx1_extrap_j = matrix.extrapolate_j(dRdx1_int, geom)  # Output dims: (nk,nel_y,2,ni)
        dRdx1_extrap_k = matrix.extrapolate_k(dRdx1_int, geom)  # Output dims: (nel_z,2,nj,ni)
        dRdx2_extrap_i = matrix.extrapolate_i(dRdx2_int, geom)
        dRdx2_extrap_j = matrix.extrapolate_j(dRdx2_int, geom)
        dRdx2_extrap_k = matrix.extrapolate_k(dRdx2_int, geom)
        dRdeta_extrap_i = matrix.extrapolate_i(dRdeta_int, geom)
        dRdeta_extrap_j = matrix.extrapolate_j(dRdeta_int, geom)
        dRdeta_extrap_k = matrix.extrapolate_k(dRdeta_int, geom)

        dtype = dRdx1_int_new.dtype
        mid_i = xp.s_[..., 1:-1, :]
        mid_j = xp.s_[..., 1:-1, :, :]
        mid_k = xp.s_[..., 1:-1, :, :, :]
        dRdx1_ex_i = xp.zeros(geom.itf_i_shape, dtype=dtype)
        dRdx1_ex_j = xp.zeros(geom.itf_j_shape, dtype=dtype)
        dRdx1_ex_k = xp.zeros(geom.itf_k_shape, dtype=dtype)
        dRdx2_ex_i = xp.zeros(geom.itf_i_shape, dtype=dtype)
        dRdx2_ex_j = xp.zeros(geom.itf_j_shape, dtype=dtype)
        dRdx2_ex_k = xp.zeros(geom.itf_k_shape, dtype=dtype)
        dRdeta_ex_i = xp.zeros(geom.itf_i_shape, dtype=dtype)
        dRdeta_ex_j = xp.zeros(geom.itf_j_shape, dtype=dtype)
        dRdeta_ex_k = xp.zeros(geom.itf_k_shape, dtype=dtype)

        dRdx1_ex_i[mid_i] = dRdx1_int_new @ matrix.extrap_x
        dRdx1_ex_j[mid_j] = dRdx1_int_new @ matrix.extrap_y
        dRdx1_ex_k[mid_k] = dRdx1_int_new @ matrix.extrap_z
        dRdx2_ex_i[mid_i] = dRdx2_int_new @ matrix.extrap_x
        dRdx2_ex_j[mid_j] = dRdx2_int_new @ matrix.extrap_y
        dRdx2_ex_k[mid_k] = dRdx2_int_new @ matrix.extrap_z
        dRdeta_ex_i[mid_i] = dRdeta_int_new @ matrix.extrap_x
        dRdeta_ex_j[mid_j] = dRdeta_int_new @ matrix.extrap_y
        dRdeta_ex_k[mid_k] = dRdeta_int_new @ matrix.extrap_z

        # _k only needs permutation to assign to the exchange arrays
        exch_itf_i[2, :, 1:-1, :, :] = xp.transpose(dRdeta_extrap_i, (0, 2, 3, 1))
        exch_itf_j[2, :, 1:-1, :, :] = xp.transpose(dRdeta_extrap_j, (0, 1, 2, 3))

        # tmp = xp.zeros_like(exch_itf_i[0])
        # tmp[..., 1:-1, :, :]= dRdx1_extrap_i.transpose(0, 2, 3, 1)
        # ref = tmp
        # diff_1i = dRdx1_ex_i - to_new_i(ref)
        # diff_1in = xp.linalg.norm(diff_1i) / xp.linalg.norm(ref)
        # if diff_1in > 1e-15:
        #     print(f"{MPI.COMM_WORLD.rank} large diff (extrap): {diff_1in:.2e}")
        #     if MPI.COMM_WORLD.rank == 0:
        #         print(
        #             f"{MPI.COMM_WORLD.rank} \n"
        #             f"old = \n{to_new_i(ref)}\n"
        #             f"new = \n{dRdeta_ex_i}\n"
        #             f"diff = \n{diff_1i}"
        #         )
        #     raise ValueError

        # _i and _j additionally need conversion to contravariant coordinates
        for el in range(geom.num_elements_x1):
            # Left boundary of the element
            exch_itf_i[0, :, el + 1, 0, :] = (
                metric_2d_contra_itf_i[0, 0, :, :, el] * dRdx1_extrap_i[:, :, el, 0]
                + metric_2d_contra_itf_i[0, 1, :, :, el] * dRdx2_extrap_i[:, :, el, 0]
            )
            exch_itf_i[1, :, el + 1, 0, :] = (
                metric_2d_contra_itf_i[1, 0, :, :, el] * dRdx1_extrap_i[:, :, el, 0]
                + metric_2d_contra_itf_i[1, 1, :, :, el] * dRdx2_extrap_i[:, :, el, 0]
            )
            # Right boundary of the element
            exch_itf_i[0, :, el + 1, 1, :] = (
                metric_2d_contra_itf_i[0, 0, :, :, el + 1] * dRdx1_extrap_i[:, :, el, 1]
                + metric_2d_contra_itf_i[0, 1, :, :, el + 1] * dRdx2_extrap_i[:, :, el, 1]
            )
            exch_itf_i[1, :, el + 1, 1, :] = (
                metric_2d_contra_itf_i[1, 0, :, :, el + 1] * dRdx1_extrap_i[:, :, el, 1]
                + metric_2d_contra_itf_i[1, 1, :, :, el + 1] * dRdx2_extrap_i[:, :, el, 1]
            )

        for el in range(geom.num_elements_x2):
            # 'south' boundary of the element
            exch_itf_j[0, :, el + 1, 0, :] = (
                metric_2d_contra_itf_j[0, 0, :, el, :] * dRdx1_extrap_j[:, el, 0, :]
                + metric_2d_contra_itf_j[0, 1, :, el, :] * dRdx2_extrap_j[:, el, 0, :]
            )
            exch_itf_j[1, :, el + 1, 0, :] = (
                metric_2d_contra_itf_j[1, 0, :, el, :] * dRdx1_extrap_j[:, el, 0, :]
                + metric_2d_contra_itf_j[1, 1, :, el, :] * dRdx2_extrap_j[:, el, 0, :]
            )
            # 'north' boundary of the element
            exch_itf_j[0, :, el + 1, 1, :] = (
                metric_2d_contra_itf_j[0, 0, :, el + 1, :] * dRdx1_extrap_j[:, el, 1, :]
                + metric_2d_contra_itf_j[0, 1, :, el + 1, :] * dRdx2_extrap_j[:, el, 1, :]
            )
            exch_itf_j[1, :, el + 1, 1, :] = (
                metric_2d_contra_itf_j[1, 0, :, el + 1, :] * dRdx1_extrap_j[:, el, 1, :]
                + metric_2d_contra_itf_j[1, 1, :, el + 1, :] * dRdx2_extrap_j[:, el, 1, :]
            )

        ## Perform the exchange.  This function requires u1/u2/u3 contravariant vectors, and in that
        # formulation u3 exchanges like a scalar (because there is no orientation change at panel
        # boundaries))

        s2_ = xp.s_[..., 1, :, : geom.num_solpts**2]
        n2_ = xp.s_[..., -2, :, geom.num_solpts**2 :]
        w2_ = xp.s_[..., 1, : geom.num_solpts**2]
        e2_ = xp.s_[..., -2, geom.num_solpts**2 :]
        s3_ = xp.s_[..., 0, :, geom.num_solpts**2 :]
        n3_ = xp.s_[..., -1, :, : geom.num_solpts**2]
        w3_ = xp.s_[..., 0, geom.num_solpts**2 :]
        e3_ = xp.s_[..., -1, : geom.num_solpts**2]
        (
            (dRdx1_ex_j[s3_], dRdx2_ex_j[s3_], dRdeta_ex_j[s3_]),
            (dRdx1_ex_j[n3_], dRdx2_ex_j[n3_], dRdeta_ex_j[n3_]),
            (dRdx1_ex_i[w3_], dRdx2_ex_i[w3_], dRdeta_ex_i[w3_]),
            (dRdx1_ex_i[e3_], dRdx2_ex_i[e3_], dRdeta_ex_i[e3_]),
        ) = geom.process_topology.start_exchange_vectors(
            (dRdx1_ex_j[s2_], dRdx2_ex_j[s2_], dRdeta_ex_j[s2_]),
            (dRdx1_ex_j[n2_], dRdx2_ex_j[n2_], dRdeta_ex_j[n2_]),
            (dRdx1_ex_i[w2_], dRdx2_ex_i[w2_], dRdeta_ex_i[w2_]),
            (dRdx1_ex_i[e2_], dRdx2_ex_i[e2_], dRdeta_ex_i[e2_]),
            geom.boundary_sn_new,
            geom.boundary_we_new,
            flip_dim=(-3, -1),
            covariant=True,
        ).wait()

        if geom.process_topology.size > 1:
            # Perform exchanges if this is truly a parallel setup.

            s_in = xp.s_[..., 1, 0, :]
            n_in = xp.s_[..., -2, 1, :]
            w_in = s_in
            e_in = n_in
            s_out = xp.s_[..., 0, 1, :]
            n_out = xp.s_[..., -1, 0, :]
            w_out = s_out
            e_out = n_out

            (
                (exch_itf_j[0][s_out], exch_itf_j[1][s_out], exch_itf_j[2][s_out]),
                (exch_itf_j[0][n_out], exch_itf_j[1][n_out], exch_itf_j[2][n_out]),
                (exch_itf_i[0][w_out], exch_itf_i[1][w_out], exch_itf_i[2][w_out]),
                (exch_itf_i[0][e_out], exch_itf_i[1][e_out], exch_itf_i[2][e_out]),
            ) = geom.process_topology.start_exchange_vectors(
                (exch_itf_j[0][s_in], exch_itf_j[1][s_in], exch_itf_j[2][s_in]),
                (exch_itf_j[0][n_in], exch_itf_j[1][n_in], exch_itf_j[2][n_in]),
                (exch_itf_i[0][w_in], exch_itf_i[1][w_in], exch_itf_i[2][w_in]),
                (exch_itf_i[0][e_in], exch_itf_i[1][e_in], exch_itf_i[2][e_in]),
                geom.boundary_sn,
                geom.boundary_we,
            ).wait()

        else:
            # Debugging cases might use a serial setup, so supply copied boundary values as a fallback
            # The right boundary of the 0 element is the left boundary of the 1 element
            exch_itf_i[:, :, 0, 1, :] = exch_itf_i[:, :, 1, 0, :]
            # The left boundary of the -1 (end) element is the right boundary of the -2 element
            exch_itf_i[:, :, -1, 0, :] = exch_itf_i[:, :, -2, 1, :]
            # The north boundary of the 0 element is the south boundary of the 1 element
            exch_itf_j[:, :, 0, 1, :] = exch_itf_j[:, :, 1, 0, :]
            # The south boundary of the -1 element is the north boundary of the -2 element
            exch_itf_j[:, :, -1, 0, :] = exch_itf_j[:, :, -2, 1, :]

        # converted_exch_itf_i = xp.zeros_like(exch_itf_i)
        # for bdy in range(geom.num_elements_x1 + 1):
        #     # Iterate from leftmost to rightmost boundary
        #     converted_exch_itf_i[0, :, bdy + 1, 0, :] = (
        #         metric_2d_cov_itf_i[0, 0, :, :, bdy] * exch_itf_i[0, :, bdy, 0, :]
        #         + metric_2d_cov_itf_i[0, 1, :, :, bdy] * exch_itf_i[1, :, bdy, 0, :]
        #     )
        #     converted_exch_itf_i[0, :, bdy, 1, :] = (
        #         metric_2d_cov_itf_i[0, 0, :, :, bdy] * exch_itf_i[0, :, bdy, 1, :]
        #         + metric_2d_cov_itf_i[0, 1, :, :, bdy] * exch_itf_i[1, :, bdy, 1, :]
        #     )
        # diff_conv = converted_exch_itf_i - exch_itf_i
        # if MPI.COMM_WORLD.rank == 0:
        #     print(f"diff conv= \n{to_new_i(diff_conv[0])}")

        # Define the averaged interface values
        dRdx1_itf_i = xp.empty_like(R_itf_i)
        dRdx2_itf_i = xp.empty_like(R_itf_i)
        dRdeta_itf_i = xp.empty_like(R_itf_i)
        dRdx1_itf_j = xp.empty_like(R_itf_j)
        dRdx2_itf_j = xp.empty_like(R_itf_j)
        dRdeta_itf_j = xp.empty_like(R_itf_j)

        # ref = converted_exch_itf_i[0]
        # diffi1 = dRdx1_ex_i - to_new_i(ref)
        # diffi1n = xp.linalg.norm(diffi1) / xp.linalg.norm(ref)

        # if diffi1n > 1e-15:
        #     print(f"{MPI.COMM_WORLD.rank} diff is so large! {diffi1n:.2e}")
        #     if MPI.COMM_WORLD.rank == 1:
        #         print(
        #             f"{MPI.COMM_WORLD.rank} diff {diffi1n:.2e}\n"
        #             # f"contra itf i: \n{metric_2d_contra_itf_i}\n"
        #             # f"cov itf i: \n{metric_2d_cov_itf_i}\n"
        #             f"old = \n{to_new_i(ref)}\n"
        #             f"new = \n{dRdx1_ex_i}\n"
        #             f"diff = \n{diffi1}"
        #         )
        #     raise ValueError

        # i-interface values
        for bdy in range(geom.num_elements_x1 + 1):
            # Iterate from leftmost to rightmost boundary
            dRdx1_itf_i[:, :, bdy] = metric_2d_cov_itf_i[0, 0, :, :, bdy] * (
                0.5 * exch_itf_i[0, :, bdy, 1, :] + 0.5 * exch_itf_i[0, :, bdy + 1, 0, :]
            ) + metric_2d_cov_itf_i[0, 1, :, :, bdy] * (
                0.5 * exch_itf_i[1, :, bdy, 1, :] + 0.5 * exch_itf_i[1, :, bdy + 1, 0, :]
            )
            dRdx2_itf_i[:, :, bdy] = metric_2d_cov_itf_i[1, 0, :, :, bdy] * (
                0.5 * exch_itf_i[0, :, bdy, 1, :] + 0.5 * exch_itf_i[0, :, bdy + 1, 0, :]
            ) + metric_2d_cov_itf_i[1, 1, :, :, bdy] * (
                0.5 * exch_itf_i[1, :, bdy, 1, :] + 0.5 * exch_itf_i[1, :, bdy + 1, 0, :]
            )
            dRdeta_itf_i[:, :, bdy] = 0.5 * exch_itf_i[2, :, bdy, 1, :] + 0.5 * exch_itf_i[2, :, bdy + 1, 0, :]

        # j-interface values
        for bdy in range(geom.num_elements_x2 + 1):
            # iterate from 'south'most to 'north'most boundary
            dRdx1_itf_j[:, bdy, :] = metric_2d_cov_itf_j[0, 0, :, bdy, :] * (
                0.5 * exch_itf_j[0, :, bdy, 1, :] + 0.5 * exch_itf_j[0, :, bdy + 1, 0, :]
            ) + metric_2d_cov_itf_j[0, 1, :, bdy, :] * (
                0.5 * exch_itf_j[1, :, bdy, 1, :] + 0.5 * exch_itf_j[1, :, bdy + 1, 0, :]
            )
            dRdx2_itf_j[:, bdy, :] = metric_2d_cov_itf_j[1, 0, :, bdy, :] * (
                0.5 * exch_itf_j[0, :, bdy, 1, :] + 0.5 * exch_itf_j[0, :, bdy + 1, 0, :]
            ) + metric_2d_cov_itf_j[1, 1, :, bdy, :] * (
                0.5 * exch_itf_j[1, :, bdy, 1, :] + 0.5 * exch_itf_j[1, :, bdy + 1, 0, :]
            )
            dRdeta_itf_j[:, bdy, :] = 0.5 * exch_itf_j[2, :, bdy, 1, :] + 0.5 * exch_itf_j[2, :, bdy + 1, 0, :]

        # k-interface values require either copying (top/bottom boundary) or simple averaging; no exchange required
        for d_itf_k, d_extrap_k in zip(
            (dRdx1_itf_k, dRdx2_itf_k, dRdeta_itf_k), (dRdx1_extrap_k, dRdx2_extrap_k, dRdeta_extrap_k)
        ):

            # Assign absolute minimum/maximum interface values based on the one-sided extrapolation
            d_itf_k[0, :, :] = d_extrap_k[0, 0, :, :]
            d_itf_k[-1, :, :] = d_extrap_k[-1, 1, :, :]

            # Assign interior values based on the average of the bordering extrapolations
            d_itf_k[1:-1, :, :] = 0.5 * (d_extrap_k[1:, 0, :, :] + d_extrap_k[:-1, 1, :, :])

        # Initialize metric arrays

        # Covariant space-only metric
        def compute_metric(X, Y, R, dRdx1, dRdx2, dRdeta):
            delsq = 1 + X**2 + Y**2  # δ², per Charron May 2022
            del4 = delsq**2

            Hcov = xp.empty((3, 3) + X.shape)
            Hcontra = xp.empty((3, 3) + X.shape)
            rootG = xp.empty_like(X)

            if deep:
                Hcov[0, 0, :] = (delta_x**2 / 4) * (R**2 / del4 * (1 + X**2) ** 2 * (1 + Y**2) + dRdx1**2)  # g_11

                Hcov[0, 1, :] = (delta_x * delta_y / 4) * (
                    -(R**2) / del4 * X * Y * (1 + X**2) * (1 + Y**2) + dRdx1 * dRdx2
                )  # g_12
                Hcov[1, 0, :] = Hcov[0, 1, :]  # g_21 (by symmetry)

                Hcov[0, 2, :] = delta_eta * delta_x / 4 * dRdx1 * dRdeta  # g_13
                Hcov[2, 0, :] = Hcov[0, 2, :]  # g_31 by symmetry

                Hcov[1, 1, :] = delta_y**2 / 4 * (R**2 / del4 * (1 + X**2) * (1 + Y**2) ** 2 + dRdx2**2)  # g_22

                Hcov[1, 2, :] = delta_eta * delta_y / 4 * dRdx2 * dRdeta  # g_23
                Hcov[2, 1, :] = Hcov[1, 2, :]  # g_32 by symmetry

                Hcov[2, 2, :] = (delta_eta**2 / 4) * dRdeta**2  # g_33

                Hcontra[0, 0, :] = (4 / delta_x**2) * (delsq / (R**2 * (1 + X**2)))  # h^11

                Hcontra[0, 1, :] = (4 / delta_x / delta_y) * (X * Y * delsq / (R**2 * (1 + X**2) * (1 + Y**2)))  # h^12
                Hcontra[1, 0, :] = Hcontra[0, 1, :]  # h^21 by symmetry

                Hcontra[0, 2, :] = (4 / delta_x / delta_eta) * (
                    -(dRdx1 * delsq / (R**2 * (1 + X**2)) + dRdx2 * delsq * X * Y / (R**2 * (1 + X**2) * (1 + Y**2)))
                    / (dRdeta)
                )  # h^13
                Hcontra[2, 0, :] = Hcontra[0, 2, :]  # h^31 by symmetry

                Hcontra[1, 1, :] = (4 / delta_y**2) * (delsq / (R**2 * (1 + Y**2)))  # h^22

                Hcontra[1, 2, :] = (4 / delta_y / delta_eta) * (
                    -(dRdx1 * X * Y * delsq / (R**2 * (1 + X**2) * (1 + Y**2)) + dRdx2 * delsq / (R**2 * (1 + Y**2)))
                    / dRdeta
                )  # h^23
                Hcontra[2, 1, :] = Hcontra[1, 2, :]  # h^32 by symmetry

                Hcontra[2, 2, :] = (
                    (4 / delta_eta**2)
                    * (
                        1
                        + dRdx1**2 * delsq / (R**2 * (1 + X**2))
                        + 2 * dRdx1 * dRdx2 * X * Y * delsq / (R**2 * (1 + X**2) * (1 + Y**2))
                        + dRdx2**2 * delsq / (R**2 * (1 + Y**2))
                    )
                    / dRdeta**2
                )

                rootG[:] = (
                    (delta_x / 2)
                    * (delta_y / 2)
                    * (delta_eta / 2)
                    * R**2
                    * (1 + X**2)
                    * (1 + Y**2)
                    * xp.abs(dRdeta)
                    / delsq ** (1.5)
                )
            else:  # Shallow, so all bare R terms become A terms
                Hcov[0, 0, :] = (delta_x**2 / 4) * (A**2 / del4 * (1 + X**2) ** 2 * (1 + Y**2) + dRdx1**2)  # g_11

                Hcov[0, 1, :] = (delta_x * delta_y / 4) * (
                    -(A**2) / del4 * X * Y * (1 + X**2) * (1 + Y**2) + dRdx1 * dRdx2
                )  # g_12
                Hcov[1, 0, :] = Hcov[0, 1, :]  # g_21 (by symmetry)

                Hcov[0, 2, :] = delta_eta * delta_x / 4 * dRdx1 * dRdeta  # g_13
                Hcov[2, 0, :] = Hcov[0, 2, :]  # g_31 by symmetry

                Hcov[1, 1, :] = delta_y**2 / 4 * (A**2 / del4 * (1 + X**2) * (1 + Y**2) ** 2 + dRdx2**2)  # g_22

                Hcov[1, 2, :] = delta_eta * delta_y / 4 * dRdx2 * dRdeta  # g_23
                Hcov[2, 1, :] = Hcov[1, 2, :]  # g_32 by symmetry

                Hcov[2, 2, :] = (delta_eta**2 / 4) * dRdeta**2  # g_33

                Hcontra[0, 0, :] = (4 / delta_x**2) * (delsq / (A**2 * (1 + X**2)))  # h^11

                Hcontra[0, 1, :] = (4 / delta_x / delta_y) * (X * Y * delsq / (A**2 * (1 + X**2) * (1 + Y**2)))  # h^12
                Hcontra[1, 0, :] = Hcontra[0, 1, :]  # h^21 by symmetry

                Hcontra[0, 2, :] = (4 / delta_x / delta_eta) * (
                    -(dRdx1 * delsq / (A**2 * (1 + X**2)) + dRdx2 * delsq * X * Y / (A**2 * (1 + X**2) * (1 + Y**2)))
                    / (dRdeta)
                )  # h^13
                Hcontra[2, 0, :] = Hcontra[0, 2, :]  # h^31 by symmetry

                Hcontra[1, 1, :] = (4 / delta_y**2) * (delsq / (A**2 * (1 + Y**2)))  # h^22

                Hcontra[1, 2, :] = (4 / delta_y / delta_eta) * (
                    -(dRdx1 * X * Y * delsq / (A**2 * (1 + X**2) * (1 + Y**2)) + dRdx2 * delsq / (A**2 * (1 + Y**2)))
                    / dRdeta
                )  # h^23
                Hcontra[2, 1, :] = Hcontra[1, 2, :]  # h^32 by symmetry

                Hcontra[2, 2, :] = (
                    (4 / delta_eta**2)
                    * (
                        1
                        + dRdx1**2 * delsq / (A**2 * (1 + X**2))
                        + 2 * dRdx1 * dRdx2 * X * Y * delsq / (A**2 * (1 + X**2) * (1 + Y**2))
                        + dRdx2**2 * delsq / (A**2 * (1 + Y**2))
                    )
                    / dRdeta**2
                )

                rootG[:] = (
                    (delta_x / 2)
                    * (delta_y / 2)
                    * (delta_eta / 2)
                    * A**2
                    * (1 + X**2)
                    * (1 + Y**2)
                    * xp.abs(dRdeta)
                    / delsq ** (1.5)
                )

            return Hcov, Hcontra, rootG

        H_cov, H_contra, sqrtG = compute_metric(X_int, Y_int, R_int, dRdx1_int, dRdx2_int, dRdeta_int)
        H_cov_itf_i, H_contra_itf_i, sqrtG_itf_i = compute_metric(
            X_itf_i, Y_itf_i, R_itf_i, dRdx1_itf_i, dRdx2_itf_i, dRdeta_itf_i
        )
        H_cov_itf_j, H_contra_itf_j, sqrtG_itf_j = compute_metric(
            X_itf_j, Y_itf_j, R_itf_j, dRdx1_itf_j, dRdx2_itf_j, dRdeta_itf_j
        )
        H_cov_itf_k, H_contra_itf_k, sqrtG_itf_k = compute_metric(
            X_itf_k, Y_itf_k, R_itf_k, dRdx1_itf_k, dRdx2_itf_k, dRdeta_itf_k
        )

        ## Computation of the Christoffel symbols
        ## Part 1: Analytic definition
        ##
        ## This section uses the analytic definition of the Christoffel symbols, which computes the mixed derivatives
        ## with respect to x and y analytically when possible.  For the topographic terms, the derivative is computed
        ## numerically and substituted into the appropriate analytic expression (deriving from the covariant derivative
        ## of the covariant metric tensor being zero).

        ## Even when using the christoffel symbols computed numerically via \sqrt{g}(h^ab)_{;c}, this analytic form is used
        ## for the rotation terms (zero indices)

        # Rotation terms that appear throughout the symbol definitions:
        rot1 = sphi - X_int * cphi * salp + Y_int * cphi * calp
        rot2 = (1 + X_int**2) * cphi * calp - Y_int * sphi + X_int * Y_int * cphi * salp
        rot3 = (1 + Y_int**2) * cphi * salp + X_int * sphi + X_int * Y_int * cphi * calp

        deltasq = 1 + X_int**2 + Y_int**2
        Omega = geom.rotation_speed

        # Γ^1_ab, a≤b
        if deep:
            Christoffel_1_01 = (
                Omega * X_int * Y_int / deltasq * rot1 + dRdx1_int * Omega / (R_int * (1 + X_int**2)) * rot2
            )
            Christoffel_1_02 = (
                -Omega * (-(1 + Y_int**2) / deltasq) * rot1 + dRdx2_int * Omega / (R_int * (1 + X_int**2)) * rot2
            )
            Christoffel_1_03 = dRdeta_int * Omega / (R_int * (1 + X_int**2)) * rot2

            Christoffel_1_11 = 2 * X_int * Y_int**2 / deltasq + dRdx1_int * 2 / R_int
            Christoffel_1_12 = -Y_int * (1 + Y_int**2) / deltasq + dRdx2_int / R_int
            Christoffel_1_13 = dRdeta_int / R_int
        else:  # Shallow atmosphere, R->A
            Christoffel_1_01 = Omega * X_int * Y_int / deltasq * rot1 + dRdx1_int * Omega / (A * (1 + X_int**2)) * rot2
            Christoffel_1_02 = (
                -Omega * (-(1 + Y_int**2) / deltasq) * rot1 + dRdx2_int * Omega / (A * (1 + X_int**2)) * rot2
            )
            Christoffel_1_03 = dRdeta_int * Omega / (A * (1 + X_int**2)) * rot2

            Christoffel_1_11 = 2 * X_int * Y_int**2 / deltasq + dRdx1_int * 2 / A
            Christoffel_1_12 = -Y_int * (1 + Y_int**2) / deltasq + dRdx2_int / A
            Christoffel_1_13 = dRdeta_int / A

        Christoffel_1_22 = 0
        Christoffel_1_23 = 0

        Christoffel_1_33 = 0

        # Γ^2_ab, a≤b
        if deep:
            Christoffel_2_01 = (
                Omega * (1 + X_int**2) / deltasq * rot1 + dRdx1_int * Omega / (R_int * (1 + Y_int**2)) * rot3
            )
            Christoffel_2_02 = (
                -Omega * X_int * Y_int / deltasq * rot2 + dRdx2_int * Omega / (R_int * (1 + Y_int**2)) * rot3
            )
            Christoffel_2_03 = dRdeta_int * Omega / (R_int * (1 + Y_int**2)) * rot3

            Christoffel_2_11 = 0
            Christoffel_2_12 = -X_int * (1 + X_int**2) / deltasq + dRdx1_int / R_int
            Christoffel_2_13 = 0

            Christoffel_2_22 = 2 * X_int**2 * Y_int / deltasq + dRdx2_int * 2 / R_int
            Christoffel_2_23 = dRdeta_int / R_int
        else:  # Shallow
            Christoffel_2_01 = Omega * (1 + X_int**2) / deltasq * rot1 + dRdx1_int * Omega / (A * (1 + Y_int**2)) * rot3
            Christoffel_2_02 = -Omega * X_int * Y_int / deltasq * rot2 + dRdx2_int * Omega / (A * (1 + Y_int**2)) * rot3
            Christoffel_2_03 = dRdeta_int * Omega / (A * (1 + Y_int**2)) * rot3

            Christoffel_2_11 = 0
            Christoffel_2_12 = -X_int * (1 + X_int**2) / deltasq + dRdx1_int / A
            Christoffel_2_13 = 0

            Christoffel_2_22 = 2 * X_int**2 * Y_int / deltasq + dRdx2_int * 2 / A
            Christoffel_2_23 = dRdeta_int / A

        Christoffel_2_33 = 0

        # Γ^3_ab, a≤b
        # For this set of terms, we need the second derivatives of R with respect to x1, x1, and η

        # Build the extensions of R_(i,j,k) to the element boundaries, using the previously-found itf_(i,k,k) values
        # Because we assume the quality of mixed partial derivatives (d^2f/dadb = d^2f/dbda), we need to extend _(i,j,k)
        # for x1, _(j,k) for x2, and only _k for eta.

        dRdx1_ext_i = xp.stack((dRdx1_itf_i[:, :, :-1], dRdx1_itf_i[:, :, 1:]), axis=-1)  # min/max-i boundaries
        dRdx1_ext_j = xp.stack((dRdx1_itf_j[:, :-1, :], dRdx1_itf_j[:, 1:, :]), axis=-2)  # min/max-j boundaries
        dRdx1_ext_k = xp.stack((dRdx1_itf_k[:-1, :, :], dRdx1_itf_k[1:, :, :]), axis=-3)  # min/max-k boundaries

        dRdx2_ext_j = xp.stack((dRdx2_itf_j[:, :-1, :], dRdx2_itf_j[:, 1:, :]), axis=-2)  # min/max-j boundaries
        dRdx2_ext_k = xp.stack((dRdx2_itf_k[:-1, :, :], dRdx2_itf_k[1:, :, :]), axis=-3)  # min/max-k boundaries

        dRdeta_ext_k = xp.stack((dRdeta_itf_k[:-1, :, :], dRdeta_itf_k[1:, :, :]), axis=-3)  # min/max-k boundaries

        # With the extension information, compute the partial derivatives.  We do not need any parallel
        # synchronization here because we only use the Christoffel symbols at element-interior points.
        d2Rdx1x1 = matrix.comma_i(dRdx1_int, dRdx1_ext_i, geom) * 2 / delta_x
        d2Rdx1x2 = matrix.comma_j(dRdx1_int, dRdx1_ext_j, geom) * 2 / delta_y
        d2Rdx1eta = matrix.comma_k(dRdx1_int, dRdx1_ext_k, geom) * 2 / delta_eta

        d2Rdx2x2 = matrix.comma_j(dRdx2_int, dRdx2_ext_j, geom) * 2 / delta_y
        d2Rdx2eta = matrix.comma_k(dRdx2_int, dRdx2_ext_k, geom) * 2 / delta_eta

        d2Rdetaeta = matrix.comma_k(dRdeta_int, dRdeta_ext_k, geom) * 2 / delta_eta

        if deep:
            Christoffel_3_01 = -(dRdeta_int**-1) * (
                dRdx1_int * Christoffel_1_01
                + dRdx2_int * Christoffel_2_01
                + R_int / deltasq * Omega * (1 + X_int**2) * (cphi * calp - Y_int * sphi)
            )
            Christoffel_3_02 = -(dRdeta_int**-1) * (
                dRdx1_int * Christoffel_1_02
                + dRdx2_int * Christoffel_2_02
                + R_int / deltasq * Omega * (1 + Y_int**2) * (cphi * salp + X_int * sphi)
            )
            Christoffel_3_03 = (
                -dRdx1_int * Omega / (R_int * (1 + X_int**2)) * rot2
                - dRdx2_int * Omega / (R_int * (1 + Y_int**2)) * rot3
            )

            Christoffel_3_11 = (dRdeta_int**-1) * (
                d2Rdx1x1 - dRdx1_int * Christoffel_1_11 - R_int / deltasq**2 * (1 + X_int**2) ** 2 * (1 + Y_int**2)
            )
            Christoffel_3_12 = (dRdeta_int**-1) * (
                d2Rdx1x2
                - dRdx1_int * Christoffel_1_12
                - dRdx2_int * Christoffel_2_12
                + R_int / deltasq**2 * X_int * Y_int * (1 + X_int**2) * (1 + Y_int**2)
            )
            Christoffel_3_13 = (dRdeta_int**-1) * d2Rdx1eta - dRdx1_int / R_int

            Christoffel_3_22 = (dRdeta_int**-1) * (
                d2Rdx2x2 - dRdx2_int * Christoffel_2_22 - R_int / deltasq**2 * (1 + X_int**2) * (1 + Y_int**2) ** 2
            )
            Christoffel_3_23 = (dRdeta_int**-1) * d2Rdx2eta - dRdx2_int / R_int

            Christoffel_3_33 = (dRdeta_int**-1) * d2Rdetaeta
        else:  # Shallow atmosphere
            Christoffel_3_01 = -(dRdeta_int**-1) * (
                dRdx1_int * Christoffel_1_01
                + dRdx2_int * Christoffel_2_01
                + A / deltasq * Omega * (1 + X_int**2) * (cphi * calp - Y_int * sphi)
            )
            Christoffel_3_02 = -(dRdeta_int**-1) * (
                dRdx1_int * Christoffel_1_02
                + dRdx2_int * Christoffel_2_02
                + A / deltasq * Omega * (1 + Y_int**2) * (cphi * salp + X_int * sphi)
            )
            Christoffel_3_03 = (
                -dRdx1_int * Omega / (A * (1 + X_int**2)) * rot2 - dRdx2_int * Omega / (A * (1 + Y_int**2)) * rot3
            )

            Christoffel_3_11 = (dRdeta_int**-1) * (
                d2Rdx1x1 - dRdx1_int * Christoffel_1_11 - A / deltasq**2 * (1 + X_int**2) ** 2 * (1 + Y_int**2)
            )
            Christoffel_3_12 = (dRdeta_int**-1) * (
                d2Rdx1x2
                - dRdx1_int * Christoffel_1_12
                - dRdx2_int * Christoffel_2_12
                + A / deltasq**2 * X_int * Y_int * (1 + X_int**2) * (1 + Y_int**2)
            )
            Christoffel_3_13 = (dRdeta_int**-1) * d2Rdx1eta - dRdx1_int / A

            Christoffel_3_22 = (dRdeta_int**-1) * (
                d2Rdx2x2 - dRdx2_int * Christoffel_2_22 - A / deltasq**2 * (1 + X_int**2) * (1 + Y_int**2) ** 2
            )
            Christoffel_3_23 = (dRdeta_int**-1) * d2Rdx2eta - dRdx2_int / A

            Christoffel_3_33 = (dRdeta_int**-1) * d2Rdetaeta

        # Now, normalize the Christoffel symbols by the appropriate grid scaling factor.  To this point, the symbols have
        # been defined in terms of x1, x2, and η, but for compatibility with the numerical differentiation we need to define
        # these symbols in terms of i, j, and k by applying the scaling factors delta_x/2, delta_y/2, and delta_eta/2 respectively.

        # The Christoffel symbol is definitionally Γ^a_bc = (∂e_b/delta_x^c) · ẽ^a, so we apply the scaling factor for the b and c
        # indices and the inverse scaling factor for the a index.

        Christoffel_1_01 *= (2 / delta_x) * (1) * (delta_x / 2)
        self.christoffel_1_01 = Christoffel_1_01
        Christoffel_1_02 *= (2 / delta_x) * (1) * (delta_y / 2)
        self.christoffel_1_02 = Christoffel_1_02
        Christoffel_1_03 *= (2 / delta_x) * (1) * (delta_eta / 2)
        self.christoffel_1_03 = Christoffel_1_03

        Christoffel_1_11 *= (2 / delta_x) * (delta_x / 2) * (delta_x / 2)
        self.christoffel_1_11 = Christoffel_1_11
        Christoffel_1_12 *= (2 / delta_x) * (delta_x / 2) * (delta_y / 2)
        self.christoffel_1_12 = Christoffel_1_12
        Christoffel_1_13 *= (2 / delta_x) * (delta_x / 2) * (delta_eta / 2)
        self.christoffel_1_13 = Christoffel_1_13

        Christoffel_1_22 *= (2 / delta_x) * (delta_y / 2) * (delta_y / 2)
        self.christoffel_1_22 = Christoffel_1_22
        Christoffel_1_23 *= (2 / delta_x) * (delta_y / 2) * (delta_eta / 2)
        self.christoffel_1_23 = Christoffel_1_23

        Christoffel_1_33 *= (2 / delta_x) * (delta_eta / 2) * (delta_eta / 2)
        self.christoffel_1_33 = Christoffel_1_33

        Christoffel_2_01 *= (2 / delta_y) * (1) * (delta_x / 2)
        self.christoffel_2_01 = Christoffel_2_01
        Christoffel_2_02 *= (2 / delta_y) * (1) * (delta_y / 2)
        self.christoffel_2_02 = Christoffel_2_02
        Christoffel_2_03 *= (2 / delta_y) * (1) * (delta_eta / 2)
        self.christoffel_2_03 = Christoffel_2_03

        Christoffel_2_11 *= (2 / delta_y) * (delta_x / 2) * (delta_x / 2)
        self.christoffel_2_11 = Christoffel_2_11
        Christoffel_2_12 *= (2 / delta_y) * (delta_x / 2) * (delta_y / 2)
        self.christoffel_2_12 = Christoffel_2_12
        Christoffel_2_13 *= (2 / delta_y) * (delta_x / 2) * (delta_eta / 2)
        self.christoffel_2_13 = Christoffel_2_13

        Christoffel_2_22 *= (2 / delta_y) * (delta_y / 2) * (delta_y / 2)
        self.christoffel_2_22 = Christoffel_2_22
        Christoffel_2_23 *= (2 / delta_y) * (delta_y / 2) * (delta_eta / 2)
        self.christoffel_2_23 = Christoffel_2_23

        Christoffel_2_33 *= (2 / delta_y) * (delta_eta / 2) * (delta_eta / 2)
        self.christoffel_2_33 = Christoffel_2_33

        Christoffel_3_01 *= (2 / delta_eta) * (1) * (delta_x / 2)
        self.christoffel_3_01 = Christoffel_3_01
        Christoffel_3_02 *= (2 / delta_eta) * (1) * (delta_y / 2)
        self.christoffel_3_02 = Christoffel_3_02
        Christoffel_3_03 *= (2 / delta_eta) * (1) * (delta_eta / 2)
        self.christoffel_3_03 = Christoffel_3_03

        Christoffel_3_11 *= (2 / delta_eta) * (delta_x / 2) * (delta_x / 2)
        self.christoffel_3_11 = Christoffel_3_11
        Christoffel_3_12 *= (2 / delta_eta) * (delta_x / 2) * (delta_y / 2)
        self.christoffel_3_12 = Christoffel_3_12
        Christoffel_3_13 *= (2 / delta_eta) * (delta_x / 2) * (delta_eta / 2)
        self.christoffel_3_13 = Christoffel_3_13

        Christoffel_3_22 *= (2 / delta_eta) * (delta_y / 2) * (delta_y / 2)
        self.christoffel_3_22 = Christoffel_3_22
        Christoffel_3_23 *= (2 / delta_eta) * (delta_y / 2) * (delta_eta / 2)
        self.christoffel_3_23 = Christoffel_3_23

        Christoffel_3_33 *= (2 / delta_eta) * (delta_eta / 2) * (delta_eta / 2)
        self.christoffel_3_33 = Christoffel_3_33

        ## Part 2: Christoffel symbols computed numerically

        ## Here, we have an alternative construction of the spatial Chrstoffel symbols (non-zero indices), based on
        ## the alternative identity √g h^{ij}_{:k} = 0
        ## Inside the flux-form Euler equations, this form effectively enforces that a constant-pressure fluid at rest
        ## remain at rest unless acted on by an external force.

        ## This is more expensive to compute than the analytic form because there is no simple expression for Γ; we must
        ## solve for it pointwise via a linear system.  (The current formulation does not take advantage of symmetry in the
        ## lower indices of Γ).

        # √g (h^ab)_:c = 0 = h^ab * 0  + √g h^ab_:c
        #                  = h^ab (√g,c - √g Γ^d_cd) + √g (h^ab,c + h^db Γ^a_dc + h^ad Γ^b_cd)
        # h^ab √g,c + √g h^ab,c = √g (h^ab Γ^d_cd - h^db Γ^a_dc - h^ad Γ^b_cd )
        # (√g h^ab),c = √g (h^ab Γ^d_cd - h^db Γ^a_dc - h^ad Γ^b_cd )

        # Switch to use the numerical formulation of the Christoffel symbol
        numer_christoffel = True

        verbose = False
        if numer_christoffel:
            if verbose and geom.process_topology.rank == 0:
                print("Computing (√g h^{ab})_{,c}")
            grad_sqrtG_metric_contra = matrix.grad(
                H_contra * sqrtG[numpy.newaxis, numpy.newaxis, :, :, :],
                H_contra_itf_i * sqrtG_itf_i[numpy.newaxis, numpy.newaxis, :, :, :],
                H_contra_itf_j * sqrtG_itf_j[numpy.newaxis, numpy.newaxis, :, :, :],
                H_contra_itf_k * sqrtG_itf_k[numpy.newaxis, numpy.newaxis, :, :, :],
                geom,
            )

            ni = geom.ni
            nj = geom.nj
            nk = geom.nk

            c_rhs = xp.empty((nk, nj, ni, 3, 3, 3))  # h(i,j,k)^(ab)_(,c)
            c_lhs = xp.zeros((nk, nj, ni, 3, 3, 3, 3, 3, 3))  # Γ(i,j,k)^d_{ef} for row (ab,c)

            if verbose and geom.process_topology.rank == 0:
                print("Assembling linear operator for Γ")

            for a in range(3):
                for b in range(3):
                    for c in range(3):
                        c_rhs[:, :, :, a, b, c] = grad_sqrtG_metric_contra[c, a, b, :, :, :]
                        for d in range(3):
                            c_lhs[:, :, :, a, b, c, d, c, d] += sqrtG[:, :, :] * H_contra[a, b, :, :, :]
                            c_lhs[:, :, :, a, b, c, a, d, c] -= sqrtG[:, :, :] * H_contra[d, b, :, :, :]
                            c_lhs[:, :, :, a, b, c, b, c, d] -= sqrtG[:, :, :] * H_contra[a, d, :, :, :]

            if verbose and geom.process_topology.rank == 0:
                print("Solving linear operator for Γ")

            try:
                # This call does not work with numpy 2.x
                # The explicit loop (in the except clause) is fine with numpy, but extremely slow with cupy.
                # That's why we do this call, an only do the explicit loop if it fails.
                # TODO Find a better way to handle this. It's probably doable with numpy 2.x with a single call...
                space_christoffel = numpy.linalg.solve(c_lhs.reshape(nk, nj, ni, 27, 27), c_rhs.reshape(nk, nj, ni, 27))
            except ValueError:
                lhs_tmp = c_lhs.reshape(nk, nj, ni, 27, 27)
                rhs_tmp = c_rhs.reshape(nk, nj, ni, 27)
                space_christoffel = xp.empty_like(rhs_tmp)
                for k in range(nk):
                    for j in range(nj):
                        for i in range(ni):
                            space_christoffel[k, j, i, ...] = xp.linalg.solve(lhs_tmp[k, j, i], rhs_tmp[k, j, i])

            space_christoffel = space_christoffel.reshape((nk, nj, ni, 3, 3, 3))
            space_christoffel = xp.transpose(space_christoffel, (3, 4, 5, 0, 1, 2))

            if verbose and geom.process_topology.rank == 0:
                print("Copying Γ to destination arrays")

            self.num_christoffel = space_christoffel

            self.christoffel_1_11 = space_christoffel[0, 0, 0, :, :, :].copy()
            self.christoffel_1_12 = space_christoffel[0, 0, 1, :, :, :].copy()
            self.christoffel_1_13 = space_christoffel[0, 0, 2, :, :, :].copy()
            self.christoffel_1_22 = space_christoffel[0, 1, 1, :, :, :].copy()
            self.christoffel_1_23 = space_christoffel[0, 1, 2, :, :, :].copy()
            self.christoffel_1_33 = space_christoffel[0, 2, 2, :, :, :].copy()

            self.christoffel_2_11 = space_christoffel[1, 0, 0, :, :, :].copy()
            self.christoffel_2_12 = space_christoffel[1, 0, 1, :, :, :].copy()
            self.christoffel_2_13 = space_christoffel[1, 0, 2, :, :, :].copy()
            self.christoffel_2_22 = space_christoffel[1, 1, 1, :, :, :].copy()
            self.christoffel_2_23 = space_christoffel[1, 1, 2, :, :, :].copy()
            self.christoffel_2_33 = space_christoffel[1, 2, 2, :, :, :].copy()

            self.christoffel_3_11 = space_christoffel[2, 0, 0, :, :, :].copy()
            self.christoffel_3_12 = space_christoffel[2, 0, 1, :, :, :].copy()
            self.christoffel_3_13 = space_christoffel[2, 0, 2, :, :, :].copy()
            self.christoffel_3_22 = space_christoffel[2, 1, 1, :, :, :].copy()
            self.christoffel_3_23 = space_christoffel[2, 1, 2, :, :, :].copy()
            self.christoffel_3_33 = space_christoffel[2, 2, 2, :, :, :].copy()

            if verbose and geom.process_topology.rank == 0:
                print("Done assembling Γ")

        # Assign H_cov and its elements to the object
        self.H_cov = H_cov
        self.H_cov_11 = H_cov[0, 0, :, :, :]
        self.H_cov_12 = H_cov[0, 1, :, :, :]
        self.H_cov_13 = H_cov[0, 2, :, :, :]
        self.H_cov_21 = H_cov[1, 0, :, :, :]
        self.H_cov_22 = H_cov[1, 1, :, :, :]
        self.H_cov_23 = H_cov[1, 2, :, :, :]
        self.H_cov_31 = H_cov[2, 0, :, :, :]
        self.H_cov_32 = H_cov[2, 1, :, :, :]
        self.H_cov_33 = H_cov[2, 2, :, :, :]

        self.H_cov_itf_i = H_cov_itf_i
        self.H_cov_11_itf_i = H_cov_itf_i[0, 0, :, :, :]
        self.H_cov_12_itf_i = H_cov_itf_i[0, 1, :, :, :]
        self.H_cov_13_itf_i = H_cov_itf_i[0, 2, :, :, :]
        self.H_cov_21_itf_i = H_cov_itf_i[1, 0, :, :, :]
        self.H_cov_22_itf_i = H_cov_itf_i[1, 1, :, :, :]
        self.H_cov_23_itf_i = H_cov_itf_i[1, 2, :, :, :]
        self.H_cov_31_itf_i = H_cov_itf_i[2, 0, :, :, :]
        self.H_cov_32_itf_i = H_cov_itf_i[2, 1, :, :, :]
        self.H_cov_33_itf_i = H_cov_itf_i[2, 2, :, :, :]

        self.H_cov_itf_j = H_cov_itf_j
        self.H_cov_11_itf_j = H_cov_itf_j[0, 0, :, :, :]
        self.H_cov_12_itf_j = H_cov_itf_j[0, 1, :, :, :]
        self.H_cov_13_itf_j = H_cov_itf_j[0, 2, :, :, :]
        self.H_cov_21_itf_j = H_cov_itf_j[1, 0, :, :, :]
        self.H_cov_22_itf_j = H_cov_itf_j[1, 1, :, :, :]
        self.H_cov_23_itf_j = H_cov_itf_j[1, 2, :, :, :]
        self.H_cov_31_itf_j = H_cov_itf_j[2, 0, :, :, :]
        self.H_cov_32_itf_j = H_cov_itf_j[2, 1, :, :, :]
        self.H_cov_33_itf_j = H_cov_itf_j[2, 2, :, :, :]

        self.H_cov_itf_k = H_cov_itf_k
        self.H_cov_11_itf_k = H_cov_itf_k[0, 0, :, :, :]
        self.H_cov_12_itf_k = H_cov_itf_k[0, 1, :, :, :]
        self.H_cov_13_itf_k = H_cov_itf_k[0, 2, :, :, :]
        self.H_cov_21_itf_k = H_cov_itf_k[1, 0, :, :, :]
        self.H_cov_22_itf_k = H_cov_itf_k[1, 1, :, :, :]
        self.H_cov_23_itf_k = H_cov_itf_k[1, 2, :, :, :]
        self.H_cov_31_itf_k = H_cov_itf_k[2, 0, :, :, :]
        self.H_cov_32_itf_k = H_cov_itf_k[2, 1, :, :, :]
        self.H_cov_33_itf_k = H_cov_itf_k[2, 2, :, :, :]

        # Assign H_contra and its elements to the object
        self.H_contra = H_contra
        self.H_contra_11 = H_contra[0, 0, :, :, :]
        self.H_contra_12 = H_contra[0, 1, :, :, :]
        self.H_contra_13 = H_contra[0, 2, :, :, :]
        self.H_contra_21 = H_contra[1, 0, :, :, :]
        self.H_contra_22 = H_contra[1, 1, :, :, :]
        self.H_contra_23 = H_contra[1, 2, :, :, :]
        self.H_contra_31 = H_contra[2, 0, :, :, :]
        self.H_contra_32 = H_contra[2, 1, :, :, :]
        self.H_contra_33 = H_contra[2, 2, :, :, :]

        self.H_contra_itf_i = H_contra_itf_i
        self.H_contra_11_itf_i = H_contra_itf_i[0, 0, :, :, :]
        self.H_contra_12_itf_i = H_contra_itf_i[0, 1, :, :, :]
        self.H_contra_13_itf_i = H_contra_itf_i[0, 2, :, :, :]
        self.H_contra_21_itf_i = H_contra_itf_i[1, 0, :, :, :]
        self.H_contra_22_itf_i = H_contra_itf_i[1, 1, :, :, :]
        self.H_contra_23_itf_i = H_contra_itf_i[1, 2, :, :, :]
        self.H_contra_31_itf_i = H_contra_itf_i[2, 0, :, :, :]
        self.H_contra_32_itf_i = H_contra_itf_i[2, 1, :, :, :]
        self.H_contra_33_itf_i = H_contra_itf_i[2, 2, :, :, :]

        self.H_contra_itf_j = H_contra_itf_j
        self.H_contra_11_itf_j = H_contra_itf_j[0, 0, :, :, :]
        self.H_contra_12_itf_j = H_contra_itf_j[0, 1, :, :, :]
        self.H_contra_13_itf_j = H_contra_itf_j[0, 2, :, :, :]
        self.H_contra_21_itf_j = H_contra_itf_j[1, 0, :, :, :]
        self.H_contra_22_itf_j = H_contra_itf_j[1, 1, :, :, :]
        self.H_contra_23_itf_j = H_contra_itf_j[1, 2, :, :, :]
        self.H_contra_31_itf_j = H_contra_itf_j[2, 0, :, :, :]
        self.H_contra_32_itf_j = H_contra_itf_j[2, 1, :, :, :]
        self.H_contra_33_itf_j = H_contra_itf_j[2, 2, :, :, :]

        self.H_contra_itf_k = H_contra_itf_k
        self.H_contra_11_itf_k = H_contra_itf_k[0, 0, :, :, :]
        self.H_contra_12_itf_k = H_contra_itf_k[0, 1, :, :, :]
        self.H_contra_13_itf_k = H_contra_itf_k[0, 2, :, :, :]
        self.H_contra_21_itf_k = H_contra_itf_k[1, 0, :, :, :]
        self.H_contra_22_itf_k = H_contra_itf_k[1, 1, :, :, :]
        self.H_contra_23_itf_k = H_contra_itf_k[1, 2, :, :, :]
        self.H_contra_31_itf_k = H_contra_itf_k[2, 0, :, :, :]
        self.H_contra_32_itf_k = H_contra_itf_k[2, 1, :, :, :]
        self.H_contra_33_itf_k = H_contra_itf_k[2, 2, :, :, :]

        self.sqrtG = sqrtG
        self.sqrtG_itf_i = sqrtG_itf_i
        self.sqrtG_itf_j = sqrtG_itf_j
        self.sqrtG_itf_k = sqrtG_itf_k
        self.inv_sqrtG = 1 / sqrtG

        self.coriolis_f = (
            2
            * geom.rotation_speed
            / geom.delta_block
            * (
                math.sin(geom.lat_p)
                - geom.X_block * math.cos(geom.lat_p) * math.sin(geom.angle_p)
                + geom.Y_block * math.cos(geom.lat_p) * math.cos(geom.angle_p)
            )
        )

        self.inv_dzdeta = 1 / dRdeta_int * 2 / delta_eta
        self.inv_dzdeta_new = geom._to_new(self.inv_dzdeta)

        self.christoffel = xp.zeros((3, 9) + geom.grid_shape_3d_new, dtype=dtype)
        self.christoffel[0, 0] = geom._to_new(self.christoffel_1_01)
        self.christoffel[0, 1] = geom._to_new(self.christoffel_1_02)
        self.christoffel[0, 2] = geom._to_new(self.christoffel_1_03)
        self.christoffel[0, 3] = geom._to_new(self.christoffel_1_11)
        self.christoffel[0, 4] = geom._to_new(self.christoffel_1_12)
        self.christoffel[0, 5] = geom._to_new(self.christoffel_1_13)
        self.christoffel[0, 6] = geom._to_new(self.christoffel_1_22)
        self.christoffel[0, 7] = geom._to_new(self.christoffel_1_23)
        self.christoffel[0, 8] = geom._to_new(self.christoffel_1_33)

        self.christoffel[1, 0] = geom._to_new(self.christoffel_2_01)
        self.christoffel[1, 1] = geom._to_new(self.christoffel_2_02)
        self.christoffel[1, 2] = geom._to_new(self.christoffel_2_03)
        self.christoffel[1, 3] = geom._to_new(self.christoffel_2_11)
        self.christoffel[1, 4] = geom._to_new(self.christoffel_2_12)
        self.christoffel[1, 5] = geom._to_new(self.christoffel_2_13)
        self.christoffel[1, 6] = geom._to_new(self.christoffel_2_22)
        self.christoffel[1, 7] = geom._to_new(self.christoffel_2_23)
        self.christoffel[1, 8] = geom._to_new(self.christoffel_2_33)

        self.christoffel[2, 0] = geom._to_new(self.christoffel_3_01)
        self.christoffel[2, 1] = geom._to_new(self.christoffel_3_02)
        self.christoffel[2, 2] = geom._to_new(self.christoffel_3_03)
        self.christoffel[2, 3] = geom._to_new(self.christoffel_3_11)
        self.christoffel[2, 4] = geom._to_new(self.christoffel_3_12)
        self.christoffel[2, 5] = geom._to_new(self.christoffel_3_13)
        self.christoffel[2, 6] = geom._to_new(self.christoffel_3_22)
        self.christoffel[2, 7] = geom._to_new(self.christoffel_3_23)
        self.christoffel[2, 8] = geom._to_new(self.christoffel_3_33)

        self.h_contra_new = geom._to_new(self.H_contra)
        self.h_contra = self.h_contra_new
        self.h_contra_itf_i_new = geom._to_new_itf_i(self.H_contra_itf_i)
        self.h_contra_itf_j_new = geom._to_new_itf_j(self.H_contra_itf_j)
        self.h_contra_itf_k_new = geom._to_new_itf_k(self.H_contra_itf_k)

        self.h_cov_new = geom._to_new(self.H_cov)
        self.h_cov_itf_i_new = geom._to_new_itf_i(self.H_cov_itf_i)
        self.h_cov_itf_j_new = geom._to_new_itf_j(self.H_cov_itf_j)
        self.h_cov_itf_k_new = geom._to_new_itf_k(self.H_cov_itf_k)

        self.sqrtG_new = geom._to_new(self.sqrtG)
        self.sqrtG_itf_i_new = geom._to_new_itf_i(self.sqrtG_itf_i)
        self.sqrtG_itf_j_new = geom._to_new_itf_j(self.sqrtG_itf_j)
        self.sqrtG_itf_k_new = geom._to_new_itf_k(self.sqrtG_itf_k)
        self.inv_sqrtG_new = 1.0 / self.sqrtG_new
