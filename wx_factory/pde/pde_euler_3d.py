import numpy
import torch
from numpy.typing import NDArray

from ..common import Configuration
from ..common.definitions import Rd, cpd, cvd, idx_rho, idx_rho_theta, idx_rho_u1, idx_rho_u2, idx_rho_u3, p0
from ..geometry import CubedSphere3D, Metric3DTopo
from ..init.dcmip import dcmip_schar_damping
from .fluxes import outward_faces, rusanov_3d
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
    # Velocity-forcing outputs
    f2,
    f3,
    f4,
    # Field variables (rho, u1, u2, w and pressure)
    r,
    u1,
    u2,
    w,
    p,
    # Christoffel symbols
    c101,
    c102,
    c103,
    c111,
    c112,
    c113,
    c122,
    c123,
    c133,
    c201,
    c202,
    c203,
    c211,
    c212,
    c213,
    c222,
    c223,
    c233,
    c301,
    c302,
    c303,
    c311,
    c312,
    c313,
    c322,
    c323,
    c333,
    # Metric terms
    h11,
    h12,
    h13,
    h22,
    h23,
    h33,
):
    """Compute velocity forcing from metric and Coriolis terms."""
    compute_forcing_1(
        f2, r, u1, u2, w, p, c101, c102, c103, c111, c112, c113, c122, c123, c133, h11, h12, h13, h22, h23, h33
    )

    compute_forcing_1(
        f3, r, u1, u2, w, p, c201, c202, c203, c211, c212, c213, c222, c223, c233, h11, h12, h13, h22, h23, h33
    )
    compute_forcing_1(
        f4, r, u1, u2, w, p, c301, c302, c303, c311, c312, c313, c322, c323, c333, h11, h12, h13, h22, h23, h33
    )


class PDEEuler3D(PDE):
    def __init__(self, geometry: CubedSphere3D, config: Configuration, metric: Metric3DTopo, num_var: int = 5):
        # Passive tracers follow the five Euler variables.
        super().__init__(
            geometry,
            config,
            metric,
            num_dim=3,
            num_var=num_var,
            num_elem=geometry.num_elements_horizontal**2 * geometry.num_elements_vertical,
        )

        self.num_solpts = geometry.num_solpts

        self.case_number = config.case_number
        # DCMIP 1 transport cases prescribe the wind and freeze the Euler state.
        # ``auto`` uses the case number; Cartesian bubble cases override it.
        mode = getattr(config, "advection_only", "auto")
        self.advection_only = {"on": True, "off": False}.get(mode, config.case_number <= 13)

    def pointwise_fluxes(
        self,
        q: NDArray,
        flux_x1: NDArray,
        flux_x2: NDArray,
        flux_x3: NDArray,
        pressure: NDArray,
    ):
        rho = q[idx_rho]
        u1 = q[idx_rho_u1] / rho
        u2 = q[idx_rho_u2] / rho
        w = q[idx_rho_u3] / rho

        # Advective fluxes.
        flux_x1[...] = self.metric.sqrtG_new * u1 * q
        flux_x2[...] = self.metric.sqrtG_new * u2 * q
        flux_x3[...] = self.metric.sqrtG_new * w * q

        # Pressure contribution.
        pressure[...] = p0 * torch.exp((cpd / cvd) * torch.log((Rd / p0) * q[idx_rho_theta]))

        # Reuse sqrt(G) p for all momentum fluxes.
        sqrtG_pressure = self.metric.sqrtG_new * pressure
        h_contra = self.metric.h_contra_new

        flux_x1[idx_rho_u1] += sqrtG_pressure * h_contra[0, 0]
        flux_x1[idx_rho_u2] += sqrtG_pressure * h_contra[0, 1]
        flux_x1[idx_rho_u3] += sqrtG_pressure * h_contra[0, 2]

        flux_x2[idx_rho_u1] += sqrtG_pressure * h_contra[1, 0]
        flux_x2[idx_rho_u2] += sqrtG_pressure * h_contra[1, 1]
        flux_x2[idx_rho_u3] += sqrtG_pressure * h_contra[1, 2]

        flux_x3[idx_rho_u1] += sqrtG_pressure * h_contra[2, 0]
        flux_x3[idx_rho_u2] += sqrtG_pressure * h_contra[2, 1]
        flux_x3[idx_rho_u3] += sqrtG_pressure * h_contra[2, 2]

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
        metric: Metric3DTopo,
    ):
        velocity_itf = (
            q_itf_x1[idx_rho_u1] / q_itf_x1[idx_rho],
            q_itf_x2[idx_rho_u2] / q_itf_x2[idx_rho],
            q_itf_x3[idx_rho_u3] / q_itf_x3[idx_rho],
        )
        q_itf = (q_itf_x1, q_itf_x2, q_itf_x3)
        pressure_itf = (pressure_itf_x1, pressure_itf_x2, pressure_itf_x3)
        flux_itf = (flux_itf_x1, flux_itf_x2, flux_itf_x3)

        # Reflect vertical velocity across the rigid top and bottom walls.
        num_solpts_2d = self.num_solpts**2
        wall_bottom = numpy.s_[..., 0, :, :, num_solpts_2d:]  # Bottom ghost trace.
        wall_top = numpy.s_[..., -1, :, :, :num_solpts_2d]  # Top ghost trace.
        first_element = numpy.s_[..., 1, :, :, :num_solpts_2d]
        last_element = numpy.s_[..., -2, :, :, num_solpts_2d:]

        w_itf_x3 = velocity_itf[2]
        w_itf_x3[wall_bottom] = -w_itf_x3[first_element]
        w_itf_x3[wall_top] = -w_itf_x3[last_element]

        for pressure, q in zip(pressure_itf, q_itf):
            pressure[...] = p0 * torch.exp((cpd / cvd) * torch.log(q[idx_rho_theta] * (Rd / p0)))

        # Clear unused outer halo faces.
        for direction in range(3):
            for face in outward_faces(direction, self.num_solpts):
                pressure_itf[direction][face] = 0.0
                if direction == 2:
                    w_itf_x3[face] = 0.0

        for direction in range(3):
            rusanov_3d(
                direction,
                velocity_itf[direction],
                q_itf[direction],
                pressure_itf[direction],
                metric,
                self.advection_only,
                flux_itf[direction],
                self.num_solpts,
            )

    def metric_forcings(
        self,
        rho: NDArray,
        u1: NDArray,
        u2: NDArray,
        w: NDArray,
        pressure: NDArray,
        metric: Metric3DTopo,
        forcing: NDArray,
    ):
        """Write the Christoffel and Coriolis forcing of each momentum row into ``forcing``."""
        compute_forcings(
            forcing[idx_rho_u1],
            forcing[idx_rho_u2],
            forcing[idx_rho_u3],
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

    def vertical_pressure_forcing(self, pressure: NDArray, metric: Metric3DTopo) -> NDArray:
        """Return the vertical-momentum pressure metric source ``Gamma^3_jk h^jk p``."""
        return metric.gamma3_h_contra_new * pressure

    def forcing_terms(self, rhs, q, pressure, metric, ops, forcing):
        # Add coriolis, metric terms and other forcings

        rho = q[idx_rho]
        u1 = q[idx_rho_u1] / rho
        u2 = q[idx_rho_u2] / rho
        w = q[idx_rho_u3] / rho

        self.metric_forcings(rho, u1, u2, w, pressure, metric, forcing)

        # if MPI.COMM_WORLD.rank == 0:

        # Gravity effect, in vertical direction
        forcing[idx_rho_u3] += (
            metric.inv_dzdeta_new
            * metric.gravity_new
            * metric.inv_sqrtG_new
            * ((metric.sqrtG_new * rho) @ ops.highfilter_k)
        )

        # DCMIP cases 2-1 and 2-2 involve rayleigh damping
        # dcmip_schar_damping modifies the 'forcing' variable to apply the requried Rayleigh damping
        if self.case_number == 21:
            dcmip_schar_damping(forcing, rho, u1, u2, w, metric, self.geometry, shear=False)
        elif self.case_number == 22:
            dcmip_schar_damping(forcing, rho, u1, u2, w, metric, self.geometry, shear=True)

        rhs -= forcing
