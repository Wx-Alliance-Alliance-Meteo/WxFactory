from typing import Optional

from mpi4py import MPI
from numpy.typing import NDArray

from common.definitions import idx_h, idx_hu1, idx_hu2, gravity
from geometry import CubedSphere2D, DFROperators, Metric2D
from init.initialize import Topo
from .rhs import RHS
from wx_mpi import ProcessTopology


class RhsShallowWater:
    """
    RHS for the shallow water equation
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        geom: CubedSphere2D,
        operators: DFROperators,
        metric: Metric2D,
        topo: Optional[Topo],
        ptopo: ProcessTopology,
        num_solpts: int,
        num_elements_hori: int,
    ):
        self.shape = shape
        self.geom = geom
        self.operators = operators
        self.metric = metric
        self.topo = topo
        self.ptopo = ptopo
        self.num_solpts = num_solpts
        self.num_elements_hori = num_elements_hori

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
            self.num_solpts,
            self.num_elements_hori,
        )
        return result.reshape(old_shape)

    def __compute_rhs__(
        self,
        Q: NDArray,
        geom: CubedSphere2D,
        mtrx: DFROperators,
        metric: Metric2D,
        topo: Optional[Topo],
        ptopo: ProcessTopology,
        num_solpts: int,
        num_elements_hori: int,
    ) -> NDArray:
        """
        Compute the RHS
        """

        xp = geom.device.xp

        num_equations = Q.shape[0]

        itf_i_shape = (num_equations,) + geom.itf_i_shape
        itf_j_shape = (num_equations,) + geom.itf_j_shape

        # Prepare array for unpacked dynamical variables
        Q_unpacked = Q.copy()
        if topo is not None:
            Q_unpacked[idx_h] += topo.hsurf

        # Interpolate to the element interface (middle elements only, halo remains 0)
        var_itf_i = xp.zeros(itf_i_shape, dtype=Q.dtype)
        var_itf_i[:, :, 1:-1, :] = Q_unpacked @ mtrx.extrap_x

        var_itf_j = xp.zeros(itf_j_shape, dtype=Q.dtype)
        var_itf_j[:, 1:-1, :, :] = Q_unpacked @ mtrx.extrap_y

        # Unpack dynamical variables
        Q_unpacked[idx_hu1] /= Q[idx_h]
        Q_unpacked[idx_hu2] /= Q[idx_h]

        u1 = Q_unpacked[idx_hu1]
        u2 = Q_unpacked[idx_hu2]

        # Initiate transfers. The first and last row (column) of elements of each array is part of the halo.
        # Each PE must thus send the second and second-to-last row (column) of elements.
        # There is a separate function for sending vector data, since they must potentially be converted to the
        # neighbor PE's coordinate system
        request_u = ptopo.start_exchange_vectors(
            south=((var_itf_j[idx_hu1, 1, :, :num_solpts]), (var_itf_j[idx_hu2, 1, :, :num_solpts])),
            north=((var_itf_j[idx_hu1, -2, :, num_solpts:]), (var_itf_j[idx_hu2, -2, :, num_solpts:])),
            west=((var_itf_i[idx_hu1, :, 1, :num_solpts]), (var_itf_i[idx_hu2, :, 1, :num_solpts])),
            east=((var_itf_i[idx_hu1, :, -2, num_solpts:]), (var_itf_i[idx_hu2, :, -2, num_solpts:])),
            boundary_sn=geom.boundary_sn,
            boundary_we=geom.boundary_we,
        )
        request_h = ptopo.start_exchange_scalars(
            south=var_itf_j[idx_h, 1, :, :num_solpts],
            north=var_itf_j[idx_h, -2, :, num_solpts:],
            west=var_itf_i[idx_h, :, 1, :num_solpts],
            east=var_itf_i[idx_h, :, -2, num_solpts:],
            boundary_shape=(num_elements_hori * num_solpts,),
        )

        # Compute fluxes
        flux_x1 = xp.empty_like(Q)
        flux_x2 = xp.empty_like(Q)

        flux_x1[idx_h] = metric.sqrtG * Q[idx_hu1]
        flux_x2[idx_h] = metric.sqrtG * Q[idx_hu2]

        hsquared = Q[idx_h] ** 2
        flux_x1[idx_hu1] = metric.sqrtG * (Q[idx_hu1] * u1 + 0.5 * gravity * metric.H_contra_11 * hsquared)
        flux_x2[idx_hu1] = metric.sqrtG * (Q[idx_hu1] * u2 + 0.5 * gravity * metric.H_contra_12 * hsquared)

        flux_x1[idx_hu2] = metric.sqrtG * (Q[idx_hu2] * u1 + 0.5 * gravity * metric.H_contra_21 * hsquared)
        flux_x2[idx_hu2] = metric.sqrtG * (Q[idx_hu2] * u2 + 0.5 * gravity * metric.H_contra_22 * hsquared)

        # Interior contribution to the derivatives, corrections for the boundaries will be added later
        df1_dx1 = flux_x1 @ mtrx.derivative_x
        df2_dx2 = flux_x2 @ mtrx.derivative_y

        # Finish transfers. We receive the halo, so it is stored in the first and last row/column of each array
        (
            (var_itf_j[idx_hu1, 0, :, num_solpts:], var_itf_j[idx_hu2, 0, :, num_solpts:]),  # South boundary
            (var_itf_j[idx_hu1, -1, :, :num_solpts], var_itf_j[idx_hu2, -1, :, :num_solpts]),  # North boundary
            (var_itf_i[idx_hu1, :, 0, num_solpts:], var_itf_i[idx_hu2, :, 0, num_solpts:]),  # West boundary
            (var_itf_i[idx_hu1, :, -1, :num_solpts], var_itf_i[idx_hu2, :, -1, :num_solpts]),  # East boundary
        ) = request_u.wait()

        (
            var_itf_j[idx_h, 0, :, num_solpts:],  # South boundary
            var_itf_j[idx_h, -1, :, :num_solpts],  # North boundary
            var_itf_i[idx_h, :, 0, num_solpts:],  # West boundary
            var_itf_i[idx_h, :, -1, :num_solpts],  # East boundary
        ) = request_h.wait()

        # Substract topo after extrapolation
        if topo is not None:
            var_itf_i[idx_h] -= topo.hsurf_itf_i
            var_itf_j[idx_h] -= topo.hsurf_itf_j

        # West and east are defined relative to the elements, *not* to the interface itself.
        # Therefore, a certain interface will be the western interface of its eastern element and vice-versa
        #
        #   western-elem   itf  eastern-elem
        #   ________________|_____________________|
        #                   |
        #   west .  east -->|<-- west  .  east -->
        #                   |
        west = xp.s_[..., 1:, :num_solpts]
        east = xp.s_[..., :-1, num_solpts:]
        south = xp.s_[..., 1:, :, :num_solpts]
        north = xp.s_[..., :-1, :, num_solpts:]

        a = xp.sqrt(gravity * var_itf_i[idx_h] * metric.H_contra_11_itf_i)
        tmp = var_itf_i[idx_h] * a
        m = xp.where(tmp != 0.0, var_itf_i[idx_hu1] / tmp, 0.0)

        big_M = 0.25 * ((m[east] + 1.0) ** 2 - (m[west] - 1.0) ** 2)

        flux_x1_itf = xp.zeros_like(var_itf_i)
        # ------ Advection part
        flux_x1_itf[east] = metric.sqrtG_itf_i[east] * (
            xp.maximum(0.0, big_M) * a[east] * var_itf_i[east] + xp.minimum(0.0, big_M) * a[west] * var_itf_i[west]
        )
        # ------ Pressure part
        p11 = metric.sqrtG_itf_i * (0.5 * gravity) * metric.H_contra_11_itf_i * var_itf_i[idx_h] ** 2
        p21 = metric.sqrtG_itf_i * (0.5 * gravity) * metric.H_contra_21_itf_i * var_itf_i[idx_h] ** 2
        flux_x1_itf[idx_hu1][east] += 0.5 * ((1.0 + m[east]) * p11[east] + (1.0 - m[west]) * p11[west])
        flux_x1_itf[idx_hu2][east] += 0.5 * ((1.0 + m[east]) * p21[east] + (1.0 - m[west]) * p21[west])

        # ------ Copy to west interface of eastern element
        flux_x1_itf[west] = flux_x1_itf[east]

        # Common AUSM fluxes
        a = xp.sqrt(gravity * var_itf_j[idx_h] * metric.H_contra_22_itf_j)
        m = var_itf_j[idx_hu2] / (var_itf_j[idx_h] * a)
        m[xp.where(xp.isnan(m))] = 0.0
        big_M = 0.25 * ((m[north] + 1.0) ** 2 - (m[south] - 1.0) ** 2)

        flux_x2_itf = xp.zeros_like(var_itf_j)
        # ------ Advection part
        flux_x2_itf[north] = metric.sqrtG_itf_j[north] * (
            xp.maximum(0.0, big_M) * a[north] * var_itf_j[north] + xp.minimum(0.0, big_M) * a[south] * var_itf_j[south]
        )
        # ------ Pressure part
        p12 = metric.sqrtG_itf_j * (0.5 * gravity) * metric.H_contra_12_itf_j * var_itf_j[idx_h] ** 2
        p22 = metric.sqrtG_itf_j * (0.5 * gravity) * metric.H_contra_22_itf_j * var_itf_j[idx_h] ** 2
        flux_x2_itf[idx_hu1][north] += 0.5 * ((1.0 + m[north]) * p12[north] + (1.0 - m[south]) * p12[south])
        flux_x2_itf[idx_hu2][north] += 0.5 * ((1.0 + m[north]) * p22[north] + (1.0 - m[south]) * p22[south])
        # ------ Copy to south interface of northern element
        flux_x2_itf[south] = flux_x2_itf[north]

        # Compute the derivatives
        df1_dx1[...] += flux_x1_itf[:, :, 1:-1, :] @ mtrx.correction_WE
        df2_dx2[...] += flux_x2_itf[:, 1:-1, :, :] @ mtrx.correction_SN

        if topo is None:
            topo_dzdx1 = 0.0
            topo_dzdx2 = 0.0

        else:
            topo_dzdx1 = topo.dzdx1
            topo_dzdx2 = topo.dzdx2

        # Add coriolis, metric and terms due to varying bottom topography
        # Note: christoffel_1_22 and metric.christoffel_2_11 are zero
        forcing = xp.zeros_like(Q)
        forcing[idx_hu1] = (
            2.0 * (metric.christoffel_1_01 * Q[idx_hu1] + metric.christoffel_1_02 * Q[idx_hu2])
            + metric.christoffel_1_11 * Q[idx_hu1] * u1
            + 2.0 * metric.christoffel_1_12 * Q[idx_hu1] * u2
            + gravity * Q[idx_h] * (metric.H_contra_11 * topo_dzdx1 + metric.H_contra_12 * topo_dzdx2)
        )
        forcing[idx_hu2] = (
            2.0 * (metric.christoffel_2_01 * Q[idx_hu1] + metric.christoffel_2_02 * Q[idx_hu2])
            + 2.0 * metric.christoffel_2_12 * Q[idx_hu1] * u2
            + metric.christoffel_2_22 * Q[idx_hu2] * u2
            + gravity * Q[idx_h] * (metric.H_contra_21 * topo_dzdx1 + metric.H_contra_22 * topo_dzdx2)
        )

        # Assemble the right-hand sides
        rhs = metric.inv_sqrtG * (-df1_dx1 - df2_dx2) - forcing

        return rhs
