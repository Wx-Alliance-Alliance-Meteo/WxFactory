from mpi4py import MPI
from numpy.typing import NDArray

from common import Configuration
from common.definitions import idx_rho, idx_rho_u1, idx_rho_u2, idx_rho_w, idx_rho_theta, p0, cpd, cvd, Rd, gravity
from device import CudaDevice
from geometry import CubedSphere3D, Metric3DTopo
from init.dcmip import dcmip_schar_damping

from .pde import PDE


def compute_forcing_1(f, r, u1, u2, w, p, c01, c02, c03, c11, c12, c13, c22, c23, c33, h11, h12, h13, h22, h23, h33):
    """Compute forcing for fluid velocity in a single direction based on metric terms and coriolis effect."""

    # fmt: off
    f[:] = (
          2.0 *   r * (c01 * u1 + c02 * u2 + c03 * w)
        +       c11 * (r * u1 * u1 + h11 * p)
        + 2.0 * c12 * (r * u1 * u2 + h12 * p)
        + 2.0 * c13 * (r * u1 * w  + h13 * p)
        +       c22 * (r * u2 * u2 + h22 * p)
        + 2.0 * c23 * (r * u2 * w  + h23 * p)
        +       c33 * (r * w  * w  + h33 * p)
    )
    # fmt: on


def compute_forcings(
    # fmt: off
    # Outputs (u1, u2, and w forcings)
    f2, f3, f4,
    # Field variables (rho, u1, u2, w and pressure)
    r, u1, u2, w, p,
    # Christoffel symbols
    c101, c102, c103,
    c111, c112, c113, c122, c123, c133,
    c201, c202, c203, c211, c212, c213, c222, c223, c233,
    c301, c302, c303, c311, c312, c313, c322, c323, c333,
    # Metric terms
    h11, h12, h13, h22, h23, h33,
    # fmt: on
):
    """Compute forcings for fluid velocity (u1, u2, w) based on metric terms and coriolis effect."""
    compute_forcing_1(
        f2, r, u1, u2, w, p, c101, c102, c103, c111, c112, c113, c122, c123, c133, h11, h12, h13, h22, h23, h33
    )

    compute_forcing_1(
        f3, r, u1, u2, w, p, c201, c202, c203, c211, c212, c213, c222, c223, c233, h11, h12, h13, h22, h23, h33
    )
    compute_forcing_1(
        f4, r, u1, u2, w, p, c301, c302, c303, c311, c312, c313, c322, c323, c333, h11, h12, h13, h22, h23, h33
    )


class PDEEulerCubesphere(PDE):
    def __init__(self, geometry: CubedSphere3D, config: Configuration, metric: Metric3DTopo):
        pde = geometry.device.pde
        super().__init__(
            geometry,
            config,
            metric,
            num_dim=3,
            num_var=5,
            num_elem=geometry.num_elements_horizontal**2 * geometry.num_elements_vertical,
            pointwise_func=pde.pointwise_euler_cubedsphere_3d,
            riemann_func=pde.riemann_euler_cubedsphere_rusanov_3d,
        )

        self.num_solpts = geometry.num_solpts

        self.case_number = config.case_number
        self.advection_only = config.case_number < 13

        self.compute_forcings = compute_forcings
        if isinstance(self.device, CudaDevice):
            self.compute_forcings = self.device.cupy.fuse(compute_forcings)

        if self.config.desired_device in ["numpy", "cupy"]:
            self.compute_forcings_inner = self.compute_forcings_py
        else:
            self.compute_forcings_inner = self.compute_forcings_code

    def pointwise_fluxes(
        self,
        q: NDArray,
        flux_x1: NDArray,
        flux_x2: NDArray,
        flux_x3: NDArray,
        pressure: NDArray,
        wflux_adv_x1: NDArray,
        wflux_adv_x2: NDArray,
        wflux_adv_x3: NDArray,
        wflux_pres_x1: NDArray,
        wflux_pres_x2: NDArray,
        wflux_pres_x3: NDArray,
        logp: NDArray,
    ):

        # Call appropriate backend kernel
        self.pointwise_func(
            q,
            self.metric.sqrtG_new,
            self.metric.h_contra_new,
            flux_x1,
            flux_x2,
            flux_x3,
            pressure,
            wflux_adv_x1,
            wflux_adv_x2,
            wflux_adv_x3,
            wflux_pres_x1,
            wflux_pres_x2,
            wflux_pres_x3,
            logp,
            self.geometry.num_elements_x1,
            self.geometry.num_elements_x2,
            self.geometry.num_elements_x3,
            self.num_solpts**3,
            False,
        )

    def riemann_fluxes(
        self,
        q_itf_x1: NDArray,
        q_itf_x2: NDArray,
        q_itf_x3: NDArray,
        flux_itf_x1: NDArray,
        flux_itf_x2: NDArray,
        flux_itf_x3: NDArray,
        pressure_itf_x1: NDArray,
        pressure_itf_x2: NDArray,
        pressure_itf_x3: NDArray,
        wflux_adv_itf_x1: NDArray,
        wflux_pres_itf_x1: NDArray,
        wflux_adv_itf_x2: NDArray,
        wflux_pres_itf_x2: NDArray,
        wflux_adv_itf_x3: NDArray,
        wflux_pres_itf_x3: NDArray,
        metric: Metric3DTopo,
    ):

        # Call Riemann kernel in appropriate backend
        self.riemann_func(
            q_itf_x1,
            q_itf_x2,
            q_itf_x3,
            metric.sqrtG_itf_i_new,
            metric.sqrtG_itf_j_new,
            metric.sqrtG_itf_k_new,
            metric.h_contra_itf_i_new,
            metric.h_contra_itf_j_new,
            metric.h_contra_itf_k_new,
            self.geometry.num_elements_x1,
            self.geometry.num_elements_x2,
            self.geometry.num_elements_x3,
            self.num_solpts,
            flux_itf_x1,
            flux_itf_x2,
            flux_itf_x3,
            pressure_itf_x1,
            pressure_itf_x2,
            pressure_itf_x3,
            wflux_adv_itf_x1,
            wflux_pres_itf_x1,
            wflux_adv_itf_x2,
            wflux_pres_itf_x2,
            wflux_adv_itf_x3,
            wflux_pres_itf_x3,
        )

        return pressure_itf_x1, pressure_itf_x2

    def compute_forcings_py(
        self,
        q: NDArray,
        rho: NDArray,
        u1: NDArray,
        u2: NDArray,
        w: NDArray,
        pressure: NDArray,
        metric: Metric3DTopo,
        forcing: NDArray,
    ):
        self.compute_forcings(
            forcing[idx_rho_u1],
            forcing[idx_rho_u2],
            forcing[idx_rho_w],
            rho,
            u1,
            u2,
            w,
            pressure,
            metric.christoffel[0, 0],
            metric.christoffel[0, 1],
            metric.christoffel[0, 2],
            metric.christoffel[0, 3],
            metric.christoffel[0, 4],
            metric.christoffel[0, 5],
            metric.christoffel[0, 6],
            metric.christoffel[0, 7],
            metric.christoffel[0, 8],
            metric.christoffel[1, 0],
            metric.christoffel[1, 1],
            metric.christoffel[1, 2],
            metric.christoffel[1, 3],
            metric.christoffel[1, 4],
            metric.christoffel[1, 5],
            metric.christoffel[1, 6],
            metric.christoffel[1, 7],
            metric.christoffel[1, 8],
            metric.christoffel[2, 0],
            metric.christoffel[2, 1],
            metric.christoffel[2, 2],
            metric.christoffel[2, 3],
            metric.christoffel[2, 4],
            metric.christoffel[2, 5],
            metric.christoffel[2, 6],
            metric.christoffel[2, 7],
            metric.christoffel[2, 8],
            metric.h_contra_new[0, 0],
            metric.h_contra_new[0, 1],
            metric.h_contra_new[0, 2],
            metric.h_contra_new[1, 1],
            metric.h_contra_new[1, 2],
            metric.h_contra_new[2, 2],
        )

    def compute_forcings_code(
        self,
        q: NDArray,
        rho: NDArray,
        u1: NDArray,
        u2: NDArray,
        w: NDArray,
        pressure: NDArray,
        metric: Metric3DTopo,
        forcing: NDArray,
    ):
        num_x1 = self.geometry.num_elements_horizontal
        num_x2 = num_x1
        num_x3 = self.geometry.num_elements_vertical
        num_solpts = self.geometry.num_solpts
        self.device.pde.forcing_euler_cubesphere_3d(
            q,
            pressure,
            metric.sqrtG_new,
            metric.h_contra_new,
            metric.christoffel,
            forcing,
            num_x1,
            num_x2,
            num_x3,
            num_solpts**3,
            0,  # Verbose flag
        )

    def forcing_terms(self, rhs, q, pressure, metric, ops, forcing):
        # Add coriolis, metric terms and other forcings

        rho = q[idx_rho]
        u1 = q[idx_rho_u1] / rho
        u2 = q[idx_rho_u2] / rho
        w = q[idx_rho_w] / rho

        self.compute_forcings_inner(q, rho, u1, u2, w, pressure, metric, forcing)

        # if MPI.COMM_WORLD.rank == 0:
        #     print(f"filter k = \n{ops.highfilter_k}")
        # raise ValueError

        # Gravity effect, in vertical direction
        forcing[idx_rho_w] += (
            metric.inv_dzdeta_new * gravity * metric.inv_sqrtG_new * ((metric.sqrtG_new * rho) @ ops.highfilter_k)
        )

        # DCMIP cases 2-1 and 2-2 involve rayleigh damping
        # dcmip_schar_damping modifies the 'forcing' variable to apply the requried Rayleigh damping
        if self.case_number == 21:
            dcmip_schar_damping(forcing, rho, u1, u2, w, metric, self.geometry, shear=False, new_layout=True)
        elif self.case_number == 22:
            dcmip_schar_damping(forcing, rho, u1, u2, w, metric, self.geometry, shear=True, new_layout=True)

        rhs -= forcing
