import numpy
import torch
from numpy.typing import NDArray

from ..common.definitions import idx_h, idx_u1, idx_u2
from ..common.matmul import maximum
from ..geometry import CubedSphere2D, DFROperators, Metric2D
from ..process_topology import ProcessTopology


class RhsAdvection2d:
    """
    RHS for 2D advection-only (passive tracer transport with prescribed velocities)
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        geom: CubedSphere2D,
        operators_real: DFROperators,
        operators_complex: DFROperators,
        metric: Metric2D,
        ptopo: ProcessTopology,
        num_solpts: int,
        num_elements_hori: int,
    ):
        self.shape = shape
        self.geom = geom
        self.operators_real = operators_real
        self.operators_complex = operators_complex
        self.metric = metric
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
        operators = self.operators_complex if torch.is_complex(vec) else self.operators_real
        result = self.__compute_rhs__(
            vec.reshape(self.shape),
            self.geom,
            operators,
            self.metric,
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
        ptopo: ProcessTopology,
        num_solpts: int,
        num_elements_hori: int,
    ) -> NDArray:
        """
        Compute the RHS for advection-only cases
        """

        num_equations = Q.shape[0]

        itf_i_shape = (num_equations,) + geom.itf_i_shape
        itf_j_shape = (num_equations,) + geom.itf_j_shape

        # Interpolate to the element interface (middle elements only, halo remains 0)
        var_itf_i = torch.zeros(itf_i_shape, dtype=Q.dtype)
        var_itf_i[:, :, 1:-1, :] = Q @ mtrx.extrap_x

        var_itf_j = torch.zeros(itf_j_shape, dtype=Q.dtype)
        var_itf_j[:, 1:-1, :, :] = Q @ mtrx.extrap_y

        # For advection-only cases, velocities are stored directly as u1 and u2 (not hu1 and hu2)
        u1 = Q[idx_u1]
        u2 = Q[idx_u2]

        # Initiate transfers. The first and last row (column) of elements of each array is part of the halo.
        # Each PE must thus send the second and second-to-last row (column) of elements.
        # There is a separate function for sending vector data, since they must potentially be converted to the
        # neighbor PE's coordinate system
        request_u = ptopo.start_exchange_vectors(
            south=((var_itf_j[idx_u1, 1, :, :num_solpts]), (var_itf_j[idx_u2, 1, :, :num_solpts])),
            north=((var_itf_j[idx_u1, -2, :, num_solpts:]), (var_itf_j[idx_u2, -2, :, num_solpts:])),
            west=((var_itf_i[idx_u1, :, 1, :num_solpts]), (var_itf_i[idx_u2, :, 1, :num_solpts])),
            east=((var_itf_i[idx_u1, :, -2, num_solpts:]), (var_itf_i[idx_u2, :, -2, num_solpts:])),
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
        flux_x1 = torch.empty_like(Q)
        flux_x2 = torch.empty_like(Q)

        flux_x1[idx_h] = metric.sqrtG * Q[idx_h] * u1
        flux_x2[idx_h] = metric.sqrtG * Q[idx_h] * u2

        # Velocity fluxes are zero for advection-only (velocities not evolved)
        flux_x1[idx_u1] = 0.0
        flux_x2[idx_u1] = 0.0
        flux_x1[idx_u2] = 0.0
        flux_x2[idx_u2] = 0.0

        # Interior contribution to the derivatives, corrections for the boundaries will be added later
        df1_dx1 = flux_x1 @ mtrx.derivative_x
        df2_dx2 = flux_x2 @ mtrx.derivative_y

        # Finish transfers.
        # We receive the halo, so it is stored in the first and last row/column of each array
        (
            (var_itf_j[idx_u1, 0, :, num_solpts:], var_itf_j[idx_u2, 0, :, num_solpts:]),  # South boundary
            (var_itf_j[idx_u1, -1, :, :num_solpts], var_itf_j[idx_u2, -1, :, :num_solpts]),  # North boundary
            (var_itf_i[idx_u1, :, 0, num_solpts:], var_itf_i[idx_u2, :, 0, num_solpts:]),  # West boundary
            (var_itf_i[idx_u1, :, -1, :num_solpts], var_itf_i[idx_u2, :, -1, :num_solpts]),  # East boundary
        ) = request_u.wait()

        (
            var_itf_j[idx_h, 0, :, num_solpts:],  # South boundary
            var_itf_j[idx_h, -1, :, :num_solpts],  # North boundary
            var_itf_i[idx_h, :, 0, num_solpts:],  # West boundary
            var_itf_i[idx_h, :, -1, :num_solpts],  # East boundary
        ) = request_h.wait()

        # West and east are defined relative to the elements, *not* to the interface itself.
        # Therefore, a certain interface will be the western interface of its eastern element and vice-versa
        #
        #   western-elem   itf  eastern-elem
        #   ________________|_____________________|
        #                   |
        #   west .  east -->|<-- west  .  east -->
        #                   |
        west = numpy.s_[..., 1:, :num_solpts]
        east = numpy.s_[..., :-1, num_solpts:]
        south = numpy.s_[..., 1:, :, :num_solpts]
        north = numpy.s_[..., :-1, :, num_solpts:]

        # Rusanov flux for advection
        # Direction x1
        eig = maximum(torch.abs(var_itf_i[idx_u1][west]), torch.abs(var_itf_i[idx_u1][east]))

        flux_x1_itf = torch.zeros_like(var_itf_i)
        flux_L = metric.sqrtG_itf_i[east] * var_itf_i[idx_h][east] * var_itf_i[idx_u1][east]
        flux_R = metric.sqrtG_itf_i[east] * var_itf_i[idx_h][west] * var_itf_i[idx_u1][west]

        flux_x1_itf[idx_h][east] = 0.5 * (
            flux_L + flux_R - eig * metric.sqrtG_itf_i[east] * (var_itf_i[idx_h][west] - var_itf_i[idx_h][east])
        )
        flux_x1_itf[idx_h][west] = flux_x1_itf[idx_h][east]

        # Direction x2
        eig = maximum(torch.abs(var_itf_j[idx_u2][south]), torch.abs(var_itf_j[idx_u2][north]))

        flux_x2_itf = torch.zeros_like(var_itf_j)
        flux_L = metric.sqrtG_itf_j[north] * var_itf_j[idx_h][north] * var_itf_j[idx_u2][north]
        flux_R = metric.sqrtG_itf_j[north] * var_itf_j[idx_h][south] * var_itf_j[idx_u2][south]

        flux_x2_itf[idx_h][north] = 0.5 * (
            flux_L + flux_R - eig * metric.sqrtG_itf_j[north] * (var_itf_j[idx_h][south] - var_itf_j[idx_h][north])
        )
        flux_x2_itf[idx_h][south] = flux_x2_itf[idx_h][north]

        # Add boundary flux corrections
        df1_dx1 = df1_dx1 + flux_x1_itf[:, :, 1:-1, :] @ mtrx.correction_WE
        df2_dx2 = df2_dx2 + flux_x2_itf[:, 1:-1, :, :] @ mtrx.correction_SN

        # No forcing terms for pure advection
        forcing = torch.zeros_like(Q)

        # Assemble the right-hand side
        rhs = metric.inv_sqrtG * -(df1_dx1 + df2_dx2) - forcing

        # For advection-only, velocity fields are prescribed (not evolved)
        rhs[idx_u1] = 0.0
        rhs[idx_u2] = 0.0

        return rhs
