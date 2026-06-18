from typing import Optional

from mpi4py import MPI
from numpy.typing import NDArray

from ..common.definitions import idx_h, idx_hu1, idx_hu2, gravity
from ..geometry import CubedSphere2D, DFROperators, Metric2D
from ..init.initialize import Topo
from .rhs import RHS
from ..process_topology import ProcessTopology


class RhsShallowWater(RHS):
    def __init__(
        self,
        shape: tuple[int, ...],
        geom: CubedSphere2D,
        operators_real: DFROperators,
        operators_complex: DFROperators,
        metric: Metric2D,
        topo: Optional[Topo],
        ptopo: ProcessTopology,
    ):
        super().__init__(
            pde=None,
            geometry=geom,
            operators_real=operators_real,
            operators_complex=operators_complex,
            metric=metric,
            topography=topo,
            process_topo=ptopo,
            config=None,
            expected_shape=shape,
        )
        self.num_solpts = geom.num_solpts
        self.num_elements_hori = geom.num_elements_horizontal

        self.var_itf_j = None
        self.var_itf_i = None
        self.u1 = None
        self.u2 = None

    def __call__(self, q):
        if q.ndim == 5:
            results = []
            for k in range(q.shape[0]):
                rhs_k = super().__call__(q[k])
                results.append(rhs_k)
            return self.geom.device.xp.stack(results, axis=0)
        else:
            return super().__call__(q)

    def allocate_arrays(self, q):
        return

    def solution_extrapolation(self, q: NDArray) -> None:
        xp = self.geom.device.xp

        num_equations = q.shape[0]
        itf_i_shape = (num_equations,) + self.geom.itf_i_shape
        itf_j_shape = (num_equations,) + self.geom.itf_j_shape

        # Prepare array for unpacked dynamical variables
        Q_unpacked = q.copy()
        if self.topo is not None:
            Q_unpacked[idx_h] += self.topo.hsurf

        # Interpolate to the element interface (middle elements only, halo remains 0)
        self.var_itf_i = xp.zeros(itf_i_shape, dtype=q.dtype)
        self.var_itf_i[:, :, 1:-1, :] = Q_unpacked @ self.ops_real.extrap_x

        self.var_itf_j = xp.zeros(itf_j_shape, dtype=q.dtype)
        self.var_itf_j[:, 1:-1, :, :] = Q_unpacked @ self.ops_real.extrap_y

        # Unpack dynamical variables
        Q_unpacked[idx_hu1] /= q[idx_h]
        Q_unpacked[idx_hu2] /= q[idx_h]

        self.u1 = Q_unpacked[idx_hu1]
        self.u2 = Q_unpacked[idx_hu2]

    def start_communication(self):

        # Initiate transfers. The first and last row (column) of elements of each array is part of the halo.
        # Each PE must thus send the second and second-to-last row (column) of elements.
        # There is a separate function for sending vector data, since they must potentially be converted to the
        # neighbor PE's coordinate system
        self.request_u = self.ptopo.start_exchange_vectors(
            south=(
                (self.var_itf_j[idx_hu1, 1, :, : self.num_solpts]),
                (self.var_itf_j[idx_hu2, 1, :, : self.num_solpts]),
            ),
            north=(
                (self.var_itf_j[idx_hu1, -2, :, self.num_solpts :]),
                (self.var_itf_j[idx_hu2, -2, :, self.num_solpts :]),
            ),
            west=(
                (self.var_itf_i[idx_hu1, :, 1, : self.num_solpts]),
                (self.var_itf_i[idx_hu2, :, 1, : self.num_solpts]),
            ),
            east=(
                (self.var_itf_i[idx_hu1, :, -2, self.num_solpts :]),
                (self.var_itf_i[idx_hu2, :, -2, self.num_solpts :]),
            ),
            boundary_sn=self.geom.boundary_sn,
            boundary_we=self.geom.boundary_we,
        )
        self.request_h = self.ptopo.start_exchange_scalars(
            south=self.var_itf_j[idx_h, 1, :, : self.num_solpts],
            north=self.var_itf_j[idx_h, -2, :, self.num_solpts :],
            west=self.var_itf_i[idx_h, :, 1, : self.num_solpts],
            east=self.var_itf_i[idx_h, :, -2, self.num_solpts :],
            boundary_shape=(self.num_elements_hori * self.num_solpts,),
        )

    def pointwise_fluxes(self, q):
        xp = self.geom.device.xp
        self.f_x1 = xp.empty_like(q)
        self.f_x2 = xp.empty_like(q)

        self.f_x1[idx_h] = self.metric.sqrtG * q[idx_hu1]
        self.f_x2[idx_h] = self.metric.sqrtG * q[idx_hu2]

        hsquared = q[idx_h] ** 2
        self.f_x1[idx_hu1] = self.metric.sqrtG * (
            q[idx_hu1] * self.u1 + 0.5 * gravity * self.metric.H_contra_11 * hsquared
        )
        self.f_x2[idx_hu1] = self.metric.sqrtG * (
            q[idx_hu1] * self.u2 + 0.5 * gravity * self.metric.H_contra_12 * hsquared
        )

        self.f_x1[idx_hu2] = self.metric.sqrtG * (
            q[idx_hu2] * self.u1 + 0.5 * gravity * self.metric.H_contra_21 * hsquared
        )
        self.f_x2[idx_hu2] = self.metric.sqrtG * (
            q[idx_hu2] * self.u2 + 0.5 * gravity * self.metric.H_contra_22 * hsquared
        )

    def flux_divergence_partial(self):
        # Interior contribution to the derivatives, corrections for the boundaries will be added later
        self.df1_dx1 = self.f_x1 @ self.ops_real.derivative_x
        self.df2_dx2 = self.f_x2 @ self.ops_real.derivative_y

    def end_communication(self):
        # Finish transfers. We receive the halo, so it is stored in the first and last row/column of each array
        (
            (
                self.var_itf_j[idx_hu1, 0, :, self.num_solpts :],
                self.var_itf_j[idx_hu2, 0, :, self.num_solpts :],
            ),  # South boundary
            (
                self.var_itf_j[idx_hu1, -1, :, : self.num_solpts],
                self.var_itf_j[idx_hu2, -1, :, : self.num_solpts],
            ),  # North boundary
            (
                self.var_itf_i[idx_hu1, :, 0, self.num_solpts :],
                self.var_itf_i[idx_hu2, :, 0, self.num_solpts :],
            ),  # West boundary
            (
                self.var_itf_i[idx_hu1, :, -1, : self.num_solpts],
                self.var_itf_i[idx_hu2, :, -1, : self.num_solpts],
            ),  # East boundary
        ) = self.request_u.wait()

        (
            self.var_itf_j[idx_h, 0, :, self.num_solpts :],  # South boundary
            self.var_itf_j[idx_h, -1, :, : self.num_solpts],  # North boundary
            self.var_itf_i[idx_h, :, 0, self.num_solpts :],  # West boundary
            self.var_itf_i[idx_h, :, -1, : self.num_solpts],  # East boundary
        ) = self.request_h.wait()

        # Substract topo after extrapolation
        if self.topo is not None:
            self.var_itf_i[idx_h] -= self.topo.hsurf_itf_i
            self.var_itf_j[idx_h] -= self.topo.hsurf_itf_j

    def riemann_fluxes(self):
        # West and east are defined relative to the elements, *not* to the interface itself.
        # Therefore, a certain interface will be the western interface of its eastern element and vice-versa
        #
        #   western-elem   itf  eastern-elem
        #   ________________|_____________________|
        #                   |
        #   west .  east -->|<-- west  .  east -->
        #                   |
        xp = self.geom.device.xp

        west = xp.s_[..., 1:, : self.num_solpts]
        east = xp.s_[..., :-1, self.num_solpts :]
        south = xp.s_[..., 1:, :, : self.num_solpts]
        north = xp.s_[..., :-1, :, self.num_solpts :]

        a = xp.sqrt(gravity * self.var_itf_i[idx_h] * self.metric.H_contra_11_itf_i)
        m = xp.where(xp.real(a) > 0.0, self.var_itf_i[idx_hu1] / (self.var_itf_i[idx_h] * a), 0.0)

        # Workaround for CuPy bug where n**2 is wrong when n is complex with a negative real value
        mw2 = (m[west] - 1.0) * (m[west] - 1.0)
        big_M = 0.25 * ((m[east] + 1.0) ** 2 - mw2)

        self.flux_x1_itf = xp.zeros_like(self.var_itf_i)
        # ------ Advection part
        self.flux_x1_itf[east] = self.metric.sqrtG_itf_i[east] * xp.where(
            xp.real(big_M) > 0.0, big_M * a[east] * self.var_itf_i[east], big_M * a[west] * self.var_itf_i[west]
        )
        # ------ Pressure part
        p11 = self.metric.sqrtG_itf_i * (0.5 * gravity) * self.metric.H_contra_11_itf_i * self.var_itf_i[idx_h] ** 2
        p21 = self.metric.sqrtG_itf_i * (0.5 * gravity) * self.metric.H_contra_21_itf_i * self.var_itf_i[idx_h] ** 2
        self.flux_x1_itf[idx_hu1][east] += 0.5 * ((1.0 + m[east]) * p11[east] + (1.0 - m[west]) * p11[west])
        self.flux_x1_itf[idx_hu2][east] += 0.5 * ((1.0 + m[east]) * p21[east] + (1.0 - m[west]) * p21[west])

        # ------ Copy to west interface of eastern element
        self.flux_x1_itf[west] = self.flux_x1_itf[east]

        # Common AUSM fluxes
        a = xp.sqrt(gravity * self.var_itf_j[idx_h] * self.metric.H_contra_22_itf_j)
        m = xp.where(xp.real(a) > 0.0, self.var_itf_j[idx_hu2] / (self.var_itf_j[idx_h] * a), 0.0)

        # Workaround for CuPy bug where n**2 is wrong when n is complex with a negative real value
        ms2 = (m[south] - 1.0) * (m[south] - 1.0)
        big_M = 0.25 * ((m[north] + 1.0) ** 2 - ms2)

        self.flux_x2_itf = xp.zeros_like(self.var_itf_j)
        # ------ Advection part
        self.flux_x2_itf[north] = self.metric.sqrtG_itf_j[north] * xp.where(
            xp.real(big_M) > 0.0, big_M * a[north] * self.var_itf_j[north], big_M * a[south] * self.var_itf_j[south]
        )
        # ------ Pressure part
        p12 = self.metric.sqrtG_itf_j * (0.5 * gravity) * self.metric.H_contra_12_itf_j * self.var_itf_j[idx_h] ** 2
        p22 = self.metric.sqrtG_itf_j * (0.5 * gravity) * self.metric.H_contra_22_itf_j * self.var_itf_j[idx_h] ** 2
        self.flux_x2_itf[idx_hu1][north] += 0.5 * ((1.0 + m[north]) * p12[north] + (1.0 - m[south]) * p12[south])
        self.flux_x2_itf[idx_hu2][north] += 0.5 * ((1.0 + m[north]) * p22[north] + (1.0 - m[south]) * p22[south])
        # ------ Copy to south interface of northern element
        self.flux_x2_itf[south] = self.flux_x2_itf[north]

    def flux_divergence(self):
        self.df1_dx1[...] += self.flux_x1_itf[:, :, 1:-1, :] @ self.ops_real.correction_WE
        self.df2_dx2[...] += self.flux_x2_itf[:, 1:-1, :, :] @ self.ops_real.correction_SN

    def forcing_terms(self, q: NDArray):
        xp = self.geom.device.xp

        if self.topo is None:
            topo_dzdx1 = 0.0
            topo_dzdx2 = 0.0

        else:
            topo_dzdx1 = self.topo.dzdx1
            topo_dzdx2 = self.topo.dzdx2

        # Add coriolis, metric and terms due to varying bottom topography
        # Note: christoffel_1_22 and metric.christoffel_2_11 are zero
        forcing = xp.zeros_like(q)
        forcing[idx_hu1] = (
            2.0 * (self.metric.christoffel_1_01 * q[idx_hu1] + self.metric.christoffel_1_02 * q[idx_hu2])
            + self.metric.christoffel_1_11 * q[idx_hu1] * self.u1
            + 2.0 * self.metric.christoffel_1_12 * q[idx_hu1] * self.u2
            + gravity * q[idx_h] * (self.metric.H_contra_11 * topo_dzdx1 + self.metric.H_contra_12 * topo_dzdx2)
        )
        forcing[idx_hu2] = (
            2.0 * (self.metric.christoffel_2_01 * q[idx_hu1] + self.metric.christoffel_2_02 * q[idx_hu2])
            + 2.0 * self.metric.christoffel_2_12 * q[idx_hu1] * self.u2
            + self.metric.christoffel_2_22 * q[idx_hu2] * self.u2
            + gravity * q[idx_h] * (self.metric.H_contra_21 * topo_dzdx1 + self.metric.H_contra_22 * topo_dzdx2)
        )

        # Assemble the right-hand sides
        self.rhs = self.metric.inv_sqrtG * (-self.df1_dx1 - self.df2_dx2) - forcing
