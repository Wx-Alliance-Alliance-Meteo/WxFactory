from typing import Optional

from mpi4py import MPI
from numpy.typing import NDArray

from init.initialize import Topo
from common.definitions import idx_h, idx_hu1, idx_hu2, gravity
from common.process_topology import ProcessTopology
from geometry import CubedSphere2D, DFROperators, Metric2D, Metric3DTopo
from .rhs import RHS


class RhsShallowWater:
    """
    RHS for the shallow water equation
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        geom: CubedSphere2D,
        operators: DFROperators,
        metric: Metric2D | Metric3DTopo,
        topo: Optional[Topo],
        ptopo: ProcessTopology,
        nbsolpts: int,
        nb_elements_hori: int,
    ):
        self.shape = shape
        self.geom = geom
        self.operators = operators
        self.metric = metric
        self.topo = topo
        self.ptopo = ptopo
        self.nbsolpts = nbsolpts
        self.nb_elements_hori = nb_elements_hori

    def __call__(self, vec: NDArray) -> NDArray:
        """Compute the value of the right-hand side based on the input state.

        :param vec: Vector containing the input state. It can have any shape, as long as its size is the same as the
                    one used to create this RHS object
        :return: Value of the right-hand side, in the same shape as the input
        """
        old_shape = vec.shape
        result = self.__compute_rhs__(
            vec.reshape(self.shape),
            self.geom,
            self.operators,
            self.metric,
            self.topo,
            self.ptopo,
            self.nbsolpts,
            self.nb_elements_hori,
        )
        return result.reshape(old_shape)

    def __compute_rhs__(
        self,
        Q: NDArray,
        geom: CubedSphere2D,
        mtrx: DFROperators,
        metric: Metric2D | Metric3DTopo,
        topo: Optional[Topo],
        ptopo: ProcessTopology,
        nbsolpts: int,
        nb_elements_hori: int,
    ) -> NDArray:
        """
        Compute the RHS
        """

        rank = MPI.COMM_WORLD.rank
        xp = geom.device.xp

        nb_equations = Q.shape[0]

        itf_i_shape = (nb_equations, nb_elements_hori, nb_elements_hori + 2, 2 * nbsolpts)
        itf_j_shape = (nb_equations, nb_elements_hori + 2, nb_elements_hori, 2 * nbsolpts)

        Q_new = Q

        # Unpack dynamical variables
        Q_new_unpacked = Q_new.copy()
        if topo is not None:
            Q_new_unpacked[idx_h] += topo.hsurf

        # Interpolate to the element interface
        var_itf_i_new = xp.zeros(itf_i_shape, dtype=Q.dtype)
        var_itf_i_new[:, :, 1:-1, :] = (Q_new_unpacked @ mtrx.extrap_x).reshape(
            (nb_equations, nb_elements_hori, nb_elements_hori, nbsolpts * 2)
        )

        var_itf_j_new = xp.zeros(itf_j_shape, dtype=Q.dtype)
        var_itf_j_new[:, 1:-1, :, :] = (Q_new_unpacked @ mtrx.extrap_y).reshape(
            (nb_equations, nb_elements_hori, nb_elements_hori, nbsolpts * 2)
        )

        Q_new_unpacked[idx_hu1] /= Q_new[idx_h]
        Q_new_unpacked[idx_hu2] /= Q_new[idx_h]

        u1_new = Q_new_unpacked[idx_hu1]
        u2_new = Q_new_unpacked[idx_hu2]

        # Initiate transfers
        request_u_new = ptopo.start_exchange_vectors(
            (
                xp.ravel(var_itf_j_new[idx_hu1, 1, :, :nbsolpts]),
                xp.ravel(var_itf_j_new[idx_hu2, 1, :, :nbsolpts]),
            ),  # South boundary
            (
                xp.ravel(var_itf_j_new[idx_hu1, -2, :, nbsolpts:]),
                xp.ravel(var_itf_j_new[idx_hu2, -2, :, nbsolpts:]),
            ),  # North boundary
            (
                xp.ravel(var_itf_i_new[idx_hu1, :, 1, :nbsolpts]),
                xp.ravel(var_itf_i_new[idx_hu2, :, 1, :nbsolpts]),
            ),  # West boundary
            (
                xp.ravel(var_itf_i_new[idx_hu1, :, -2, nbsolpts:]),
                xp.ravel(var_itf_i_new[idx_hu2, :, -2, nbsolpts:]),
            ),  # East boundary
            geom.boundary_sn,
            geom.boundary_we,
        )
        request_h_new = ptopo.start_exchange_scalars(
            xp.ravel(var_itf_j_new[idx_h, 1, :, :nbsolpts]),
            xp.ravel(var_itf_j_new[idx_h, -2, :, nbsolpts:]),
            xp.ravel(var_itf_i_new[idx_h, :, 1, :nbsolpts]),
            xp.ravel(var_itf_i_new[idx_h, :, -2, nbsolpts:]),
        )

        # Compute the fluxes
        flux_x1_new = xp.empty_like(Q_new)
        flux_x2_new = xp.empty_like(Q_new)

        flux_x1_new[idx_h] = metric.sqrtG * Q_new[idx_hu1]
        flux_x2_new[idx_h] = metric.sqrtG * Q_new[idx_hu2]

        hsquared = Q_new[idx_h] ** 2
        flux_x1_new[idx_hu1] = metric.sqrtG * (Q_new[idx_hu1] * u1_new + 0.5 * gravity * metric.H_contra_11 * hsquared)
        flux_x2_new[idx_hu1] = metric.sqrtG * (Q_new[idx_hu1] * u2_new + 0.5 * gravity * metric.H_contra_12 * hsquared)

        flux_x1_new[idx_hu2] = metric.sqrtG * (Q_new[idx_hu2] * u1_new + 0.5 * gravity * metric.H_contra_21 * hsquared)
        flux_x2_new[idx_hu2] = metric.sqrtG * (Q_new[idx_hu2] * u2_new + 0.5 * gravity * metric.H_contra_22 * hsquared)

        # Interior contribution to the derivatives, corrections for the boundaries will be added later
        df1_dx1_new = flux_x1_new @ mtrx.derivative_x
        df2_dx2_new = flux_x2_new @ mtrx.derivative_y

        # Finish transfers

        # TODO make this simpler
        (s1, s2), (n1, n2), (w1, w2), (e1, e2) = request_u_new.wait()
        var_itf_j_new[idx_hu1, 0, :, nbsolpts:] = s1.reshape(nb_elements_hori, nbsolpts)
        var_itf_j_new[idx_hu2, 0, :, nbsolpts:] = s2.reshape(nb_elements_hori, nbsolpts)
        var_itf_j_new[idx_hu1, -1, :, :nbsolpts] = n1.reshape(nb_elements_hori, nbsolpts)
        var_itf_j_new[idx_hu2, -1, :, :nbsolpts] = n2.reshape(nb_elements_hori, nbsolpts)
        var_itf_i_new[idx_hu1, :, 0, nbsolpts:] = w1.reshape(nb_elements_hori, nbsolpts)
        var_itf_i_new[idx_hu2, :, 0, nbsolpts:] = w2.reshape(nb_elements_hori, nbsolpts)
        var_itf_i_new[idx_hu1, :, -1, :nbsolpts] = e1.reshape(nb_elements_hori, nbsolpts)
        var_itf_i_new[idx_hu2, :, -1, :nbsolpts] = e2.reshape(nb_elements_hori, nbsolpts)

        s, n, w, e = request_h_new.wait()
        var_itf_j_new[idx_h, 0, :, nbsolpts:] = s.reshape(nb_elements_hori, nbsolpts)
        var_itf_j_new[idx_h, -1, :, :nbsolpts] = n.reshape(nb_elements_hori, nbsolpts)
        var_itf_i_new[idx_h, :, 0, nbsolpts:] = w.reshape(nb_elements_hori, nbsolpts)
        var_itf_i_new[idx_h, :, -1, :nbsolpts] = e.reshape(nb_elements_hori, nbsolpts)

        # Substract topo after extrapolation
        if topo is not None:
            var_itf_i_new[idx_h] -= topo.hsurf_itf_i
            var_itf_j_new[idx_h] -= topo.hsurf_itf_j

        # West and east are defined relative to the elements, *not* to the interface itself.
        # Therefore, a certain interface will be the western interface of its eastern element and vice-versa
        #
        #   western-elem   itf  eastern-elem
        #   ________________|_____________________|
        #                   |
        #   west .  east -->|<-- west  .  east -->
        #                   |
        west = xp.s_[..., 1:, :nbsolpts]
        east = xp.s_[..., :-1, nbsolpts:]
        south = xp.s_[..., 1:, :, :nbsolpts]
        north = xp.s_[..., :-1, :, nbsolpts:]

        a = xp.sqrt(gravity * var_itf_i_new[idx_h] * metric.H_contra_11_itf_i)
        tmp = var_itf_i_new[idx_h] * a
        m = xp.where(tmp != 0.0, var_itf_i_new[idx_hu1] / tmp, 0.0)

        big_M = 0.25 * ((m[east] + 1.0) ** 2 - (m[west] - 1.0) ** 2)

        flux_x1_itf_new = xp.zeros_like(var_itf_i_new)
        # ------ Advection part
        flux_x1_itf_new[east] = metric.sqrtG_itf_i[east] * (
            xp.maximum(0.0, big_M) * a[east] * var_itf_i_new[east]
            + xp.minimum(0.0, big_M) * a[west] * var_itf_i_new[west]
        )
        # ------ Pressure part
        p11 = metric.sqrtG_itf_i * (0.5 * gravity) * metric.H_contra_11_itf_i * var_itf_i_new[idx_h] ** 2
        p21 = metric.sqrtG_itf_i * (0.5 * gravity) * metric.H_contra_21_itf_i * var_itf_i_new[idx_h] ** 2
        flux_x1_itf_new[idx_hu1][east] += 0.5 * ((1.0 + m[east]) * p11[east] + (1.0 - m[west]) * p11[west])
        flux_x1_itf_new[idx_hu2][east] += 0.5 * ((1.0 + m[east]) * p21[east] + (1.0 - m[west]) * p21[west])

        # ------ Copy to west interface of eastern element
        flux_x1_itf_new[west] = flux_x1_itf_new[east]

        # Common AUSM fluxes
        a = xp.sqrt(gravity * var_itf_j_new[idx_h] * metric.H_contra_22_itf_j)
        m = var_itf_j_new[idx_hu2] / (var_itf_j_new[idx_h] * a)
        m[xp.where(xp.isnan(m))] = 0.0
        big_M = 0.25 * ((m[north] + 1.0) ** 2 - (m[south] - 1.0) ** 2)

        flux_x2_itf_new = xp.zeros_like(var_itf_j_new)
        # ------ Advection part
        flux_x2_itf_new[north] = metric.sqrtG_itf_j[north] * (
            xp.maximum(0.0, big_M) * a[north] * var_itf_j_new[north]
            + xp.minimum(0.0, big_M) * a[south] * var_itf_j_new[south]
        )
        # ------ Pressure part
        p12 = metric.sqrtG_itf_j * (0.5 * gravity) * metric.H_contra_12_itf_j * var_itf_j_new[idx_h] ** 2
        p22 = metric.sqrtG_itf_j * (0.5 * gravity) * metric.H_contra_22_itf_j * var_itf_j_new[idx_h] ** 2
        flux_x2_itf_new[idx_hu1][north] += 0.5 * ((1.0 + m[north]) * p12[north] + (1.0 - m[south]) * p12[south])
        flux_x2_itf_new[idx_hu2][north] += 0.5 * ((1.0 + m[north]) * p22[north] + (1.0 - m[south]) * p22[south])
        # ------ Copy to south interface of northern element
        flux_x2_itf_new[south] = flux_x2_itf_new[north]

        # Compute the derivatives
        df1_dx1_new[...] += geom.middle_itf_i(flux_x1_itf_new) @ mtrx.correction_WE
        df2_dx2_new[...] += geom.middle_itf_j(flux_x2_itf_new) @ mtrx.correction_SN

        if topo is None:
            topo_dzdx1_new = xp.zeros_like(metric.H_contra_11)
            topo_dzdx2_new = xp.zeros_like(metric.H_contra_11)

        else:
            topo_dzdx1_new = topo.dzdx1
            topo_dzdx2_new = topo.dzdx2

        # Add coriolis, metric and terms due to varying bottom topography
        # Note: christoffel_1_22 and metric.christoffel_2_11 are zero
        forcing_new = xp.zeros_like(Q_new)
        forcing_new[idx_hu1] = (
            2.0 * (metric.christoffel_1_01 * Q_new[idx_hu1] + metric.christoffel_1_02 * Q_new[idx_hu2])
            + metric.christoffel_1_11 * Q_new[idx_hu1] * u1_new
            + 2.0 * metric.christoffel_1_12 * Q_new[idx_hu1] * u2_new
            + gravity * Q_new[idx_h] * (metric.H_contra_11 * topo_dzdx1_new + metric.H_contra_12 * topo_dzdx2_new)
        )
        forcing_new[idx_hu2] = (
            2.0 * (metric.christoffel_2_01 * Q_new[idx_hu1] + metric.christoffel_2_02 * Q_new[idx_hu2])
            + 2.0 * metric.christoffel_2_12 * Q_new[idx_hu1] * u2_new
            + metric.christoffel_2_22 * Q_new[idx_hu2] * u2_new
            + gravity * Q_new[idx_h] * (metric.H_contra_21 * topo_dzdx1_new + metric.H_contra_22 * topo_dzdx2_new)
        )

        # Assemble the right-hand sides
        rhs_new = metric.inv_sqrtG * (-df1_dx1_new - df2_dx2_new) - forcing_new

        return rhs_new
