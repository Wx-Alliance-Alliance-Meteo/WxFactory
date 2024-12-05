from numpy.typing import NDArray
from mpi4py import MPI

from common.definitions import (
    idx_rho_u1,
    idx_rho_u2,
    idx_rho_w,
    idx_rho,
    idx_rho_theta,
    gravity,
    p0,
    Rd,
    cpd,
    cvd,
    heat_capacity_ratio,
)
from common.device import CpuDevice, CudaDevice, Device, default_device
from common.process_topology import ProcessTopology
from .fluxes import (
    rusanov_3d_vert,
    rusanov_3d_hori_i,
    rusanov_3d_hori_j,
    rusanov_3d_hori_i_new,
    rusanov_3d_hori_j_new,
    rusanov_3d_vert_new,
)
from geometry import CubedSphere3D, DFROperators, Metric3DTopo
from init.dcmip import dcmip_schar_damping
from rhs.rhs import RHS

rhs_euler_kernels = None


def compute_forcing_1(f, r, u1, u2, w, p, c01, c02, c03, c11, c12, c13, c22, c23, c33, h11, h12, h13, h22, h23, h33):
    """Compute forcing for fluid velocity in a single direction based on metric terms and coriolis effect."""
    f[:] = (
        2.0 * r * (c01 * u1 + c02 * u2 + c03 * w)
        + c11 * (r * u1 * u1 + h11 * p)
        + 2.0 * c12 * (r * u1 * u2 + h12 * p)
        + 2.0 * c13 * (r * u1 * w + h13 * p)
        + c22 * (r * u2 * u2 + h22 * p)
        + 2.0 * c23 * (r * u2 * w + h23 * p)
        + c33 * (r * w * w + h33 * p)
    )


def compute_forcings(
    # Outputs (u1, u2, and w forcings)
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


class RhsEuler:
    def __init__(
        self,
        shape: tuple[int, ...],
        geom: CubedSphere3D,
        operators: DFROperators,
        metric: Metric3DTopo,
        ptopo: ProcessTopology,
        nbsolpts: int,
        nb_elements_hori: int,
        nb_elements_vert: int,
        case_number: int,
        device: Device = default_device,
    ) -> None:
        # super().__init__(shape, geom, operators, metric, ptopo, nbsolpts, nb_elements_hori, nb_elements_vert,
        #                  case_number, device)
        self.shape = shape
        self.geom = geom
        self.operators = operators
        self.metric = metric
        self.ptopo = ptopo
        self.nbsolpts = nbsolpts
        self.nb_elements_hori = nb_elements_hori
        self.nb_elements_vert = nb_elements_vert
        self.case_number = case_number
        self.device = device

        self.compute_forcings = compute_forcings
        if isinstance(device, CudaDevice):
            self.compute_forcings = device.cupy.fuse(self.compute_forcings)

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
            self.ptopo,
            self.nbsolpts,
            self.nb_elements_hori,
            self.nb_elements_vert,
            self.case_number,
            self.device,
        )
        return result.reshape(old_shape)

    def __compute_rhs__(
        self,
        Q: NDArray,
        geom: CubedSphere3D,
        operators: DFROperators,
        metric: Metric3DTopo,
        ptopo: ProcessTopology,
        nbsolpts: int,
        nb_elements_hori: int,
        nb_elements_vert: int,
        case_number: int,
        device: Device,
    ) -> NDArray:
        """Evaluate the right-hand side of the three-dimensional Euler equations.

        This function evaluates RHS of the Euler equations using the four-demsional tensor
        formulation (see Charron 2014), returning an array consisting of the time-derivative of the conserved
        variables (ρ,ρu,ρv,ρw,ρθ).  A "curried" version of this function, with non-Q parameters predefined,
        should be passed to the time-stepping routine to use as a RHS black-box. Since some of the
        time-stepping routines perform a Jacobian evaluation via complex derivative, this function should also
        be safe with respect to complex-valued inputs inside Q.

        Note that this function includes MPI communication for inter-process boundary interactions, so it must be
        called collectively.

        :param Q: numpy.ndarray
           Input array of the current model state, indexed as (var,k,j,i)
        :param geom: CubedSphere
           Geometry definition, containing parameters relating to the spherical coordinate system
        :param operators: DFR_operators
           Contains matrix operators for the DFR discretization, notably boundary extrapolation and
           local (partial) derivatives
        :param metric: Metric
           Contains the various metric terms associated with the tensor formulation, notably including the
           scalar √g, the spatial metric h, and the Christoffel symbols
        :param ptopo: :py:class:`~process_topology.ProcessTopology`
           Wraps the information and communication functions necessary for MPI distribution
        :param nbsolpts: int
           Number of interior nodal points per element.  A 3D element will contain nbsolpts**3 internal points.
        :param nb_elements_hori: int
           Number of elements in x/y on each panel of the cubed sphere
        :param nb_elements_vert: int
           Number of elements in the vertical
        :param case_number: int
           DCMIP case number, used to selectively enable or disable parts of the Euler equations to accomplish
           specialized tests like advection-only

        :return: numpy.ndarray
           Output of right-hand-side terms of Euler equations
        """
        # Shortcuts
        op = operators
        xp = device.xp
        mid_i = xp.s_[..., 1:-1, :]
        mid_j = xp.s_[..., 1:-1, :, :]
        mid_k = xp.s_[..., 1:-1, :, :, :]

        nb_equations = Q.shape[0]  # Number of constituent Euler equations.  Probably 6.

        # Flag for advection-only processing, with DCMIP test cases 11 and 12
        advection_only = case_number < 13

        itf_i_shape = (nb_equations,) + geom.itf_i_shape
        itf_j_shape = (nb_equations,) + geom.itf_j_shape
        itf_k_shape = (nb_equations,) + geom.itf_k_shape

        # Interpolate to the element interface (middle elements only, halo remains 0)
        var_itf_i = xp.ones(itf_i_shape, dtype=Q.dtype)
        var_itf_i[mid_i] = Q @ operators.extrap_x

        var_itf_j = xp.ones(itf_j_shape, dtype=Q.dtype)
        var_itf_j[mid_j] = Q @ operators.extrap_y

        # Scaled variables for separate reconstruction
        logrho = xp.log(Q[idx_rho])
        logrhotheta = xp.log(Q[idx_rho_theta])

        var_itf_i[idx_rho][mid_i] = xp.exp(logrho @ operators.extrap_x)
        var_itf_i[idx_rho_theta][mid_i] = xp.exp(logrhotheta @ operators.extrap_x)
        var_itf_j[idx_rho][mid_j] = xp.exp(logrho @ operators.extrap_y)
        var_itf_j[idx_rho_theta][mid_j] = xp.exp(logrhotheta @ operators.extrap_y)

        # Initiate transfers
        s2_ = xp.s_[..., 1, :, : nbsolpts**2]
        n2_ = xp.s_[..., -2, :, nbsolpts**2 :]
        w2_ = xp.s_[..., 1, : nbsolpts**2]
        e2_ = xp.s_[..., -2, nbsolpts**2 :]
        req_r = ptopo.start_exchange_scalars(
            var_itf_j[idx_rho][s2_],
            var_itf_j[idx_rho][n2_],
            var_itf_i[idx_rho][w2_],
            var_itf_i[idx_rho][e2_],
            boundary_shape=(nb_elements_hori, nbsolpts, nbsolpts),
            flip_dim=(-3, -1),
        )
        req_u = ptopo.start_exchange_vectors(
            (var_itf_j[idx_rho_u1][s2_], var_itf_j[idx_rho_u2][s2_], var_itf_j[idx_rho_w][s2_]),
            (var_itf_j[idx_rho_u1][n2_], var_itf_j[idx_rho_u2][n2_], var_itf_j[idx_rho_w][n2_]),
            (var_itf_i[idx_rho_u1][w2_], var_itf_i[idx_rho_u2][w2_], var_itf_i[idx_rho_w][w2_]),
            (var_itf_i[idx_rho_u1][e2_], var_itf_i[idx_rho_u2][e2_], var_itf_i[idx_rho_w][e2_]),
            geom.boundary_sn_new,
            geom.boundary_we_new,
            flip_dim=(-3, -1),
        )
        req_t = ptopo.start_exchange_scalars(
            var_itf_j[idx_rho_theta][s2_],
            var_itf_j[idx_rho_theta][n2_],
            var_itf_i[idx_rho_theta][w2_],
            var_itf_i[idx_rho_theta][e2_],
            boundary_shape=(nb_elements_hori, nbsolpts, nbsolpts),
            flip_dim=(-3, -1),
        )

        # Unpack dynamical variables, each to arrays of size [nk,nj,ni]
        rho = Q[idx_rho]
        u1 = Q[idx_rho_u1] / rho
        u2 = Q[idx_rho_u2] / rho
        w = Q[idx_rho_w] / rho

        # Compute the advective fluxes ...
        flux_x1 = metric.sqrtG_new * u1 * Q
        flux_x2 = metric.sqrtG_new * u2 * Q
        flux_x3 = metric.sqrtG_new * w * Q

        wflux_adv_x1 = metric.sqrtG_new * u1 * Q[idx_rho_w]
        wflux_adv_x2 = metric.sqrtG_new * u2 * Q[idx_rho_w]
        wflux_adv_x3 = metric.sqrtG_new * w * Q[idx_rho_w]

        # ... and add the pressure component
        # Performance note: exp(log) is measurably faster than ** (pow)
        pressure = p0 * xp.exp((cpd / cvd) * xp.log((Rd / p0) * Q[idx_rho_theta]))

        flux_x1[idx_rho_u1] += metric.sqrtG_new * metric.h_contra_new[0, 0] * pressure
        flux_x1[idx_rho_u2] += metric.sqrtG_new * metric.h_contra_new[0, 1] * pressure
        flux_x1[idx_rho_w] += metric.sqrtG_new * metric.h_contra_new[0, 2] * pressure

        wflux_pres_x1 = (metric.sqrtG_new * metric.h_contra_new[0, 2]).astype(Q.dtype)

        flux_x2[idx_rho_u1] += metric.sqrtG_new * metric.h_contra_new[1, 0] * pressure
        flux_x2[idx_rho_u2] += metric.sqrtG_new * metric.h_contra_new[1, 1] * pressure
        flux_x2[idx_rho_w] += metric.sqrtG_new * metric.h_contra_new[1, 2] * pressure

        wflux_pres_x2 = (metric.sqrtG_new * metric.h_contra_new[1, 2]).astype(Q.dtype)

        flux_x3[idx_rho_u1] += metric.sqrtG_new * metric.h_contra_new[2, 0] * pressure
        flux_x3[idx_rho_u2] += metric.sqrtG_new * metric.h_contra_new[2, 1] * pressure
        flux_x3[idx_rho_w] += metric.sqrtG_new * metric.h_contra_new[2, 2] * pressure

        wflux_pres_x3 = (metric.sqrtG_new * metric.h_contra_new[2, 2]).astype(Q.dtype)

        var_itf_k = xp.empty(itf_k_shape, dtype=Q.dtype)
        var_itf_k[...] = 0.0
        var_itf_k[mid_k] = Q @ operators.extrap_z
        var_itf_k[idx_rho][mid_k] = xp.exp(logrho @ operators.extrap_z)
        var_itf_k[idx_rho_theta][mid_k] = xp.exp(logrhotheta @ operators.extrap_z)

        south = xp.s_[: nbsolpts**2]
        north = xp.s_[nbsolpts**2 :]

        # For consistency at the surface and top boundaries, treat the extrapolation as continuous.  That is,
        # the "top" of the ground is equal to the "bottom" of the atmosphere, and the "bottom" of the model top
        # is equal to the "top" of the atmosphere.
        var_itf_k[..., 0, :, :, south] = var_itf_k[..., 1, :, :, south]
        var_itf_k[..., 0, :, :, north] = var_itf_k[..., 0, :, :, south]
        var_itf_k[..., -1, :, :, north] = var_itf_k[..., -2, :, :, north]
        var_itf_k[..., -1, :, :, south] = var_itf_k[..., -1, :, :, north]

        # Evaluate pressure at the vertical element interfaces based on ρθ.
        pressure_itf_k = p0 * xp.exp((cpd / cvd) * xp.log(var_itf_k[idx_rho_theta] * (Rd / p0)))

        # Take w ← (wρ)/ ρ at the vertical interfaces
        w_itf_k = var_itf_k[idx_rho_w] / var_itf_k[idx_rho]

        # Surface and top boundary treatement, imposing no flow (w=0) through top and bottom
        # csubich -- apply odd symmetry to w at boundary so there is no advective _flux_ through boundary
        w_itf_k[0, :, :, south] = 0.0  # Bottom of bottom element (unused)
        w_itf_k[0, :, :, north] = -w_itf_k[1, :, :, south]  # Top of bottom element (negative symmetry)
        w_itf_k[-1, :, :, north] = 0.0  # Top of top element (unused)
        w_itf_k[-1, :, :, south] = -w_itf_k[-2, :, :, north]  # Bottom of top element (negative symmetry)

        flux_x3_itf_k = xp.zeros_like(var_itf_k)
        wflux_adv_x3_itf_k = xp.zeros_like(flux_x3_itf_k[0])
        wflux_pres_x3_itf_k = xp.zeros_like(flux_x3_itf_k[0])
        rusanov_3d_vert_new(
            var_itf_k,
            pressure_itf_k,
            w_itf_k,
            metric,
            advection_only,
            flux_x3_itf_k,
            wflux_adv_x3_itf_k,
            wflux_pres_x3_itf_k,
            xp,
            nbsolpts,
        )

        # Finish transfers
        s3_ = xp.s_[..., 0, :, nbsolpts**2 :]
        n3_ = xp.s_[..., -1, :, : nbsolpts**2]
        w3_ = xp.s_[..., 0, nbsolpts**2 :]
        e3_ = xp.s_[..., -1, : nbsolpts**2]
        (
            var_itf_j[idx_rho][s3_],
            var_itf_j[idx_rho][n3_],
            var_itf_i[idx_rho][w3_],
            var_itf_i[idx_rho][e3_],
        ) = req_r.wait()
        (
            (var_itf_j[idx_rho_u1][s3_], var_itf_j[idx_rho_u2][s3_], var_itf_j[idx_rho_w][s3_]),
            (var_itf_j[idx_rho_u1][n3_], var_itf_j[idx_rho_u2][n3_], var_itf_j[idx_rho_w][n3_]),
            (var_itf_i[idx_rho_u1][w3_], var_itf_i[idx_rho_u2][w3_], var_itf_i[idx_rho_w][w3_]),
            (var_itf_i[idx_rho_u1][e3_], var_itf_i[idx_rho_u2][e3_], var_itf_i[idx_rho_w][e3_]),
        ) = req_u.wait()
        (
            var_itf_j[idx_rho_theta][s3_],
            var_itf_j[idx_rho_theta][n3_],
            var_itf_i[idx_rho_theta][w3_],
            var_itf_i[idx_rho_theta][e3_],
        ) = req_t.wait()

        # Define u, v at the interface by dividing momentum and density
        u1_itf_i = var_itf_i[idx_rho_u1] / var_itf_i[idx_rho]
        u2_itf_j = var_itf_j[idx_rho_u2] / var_itf_j[idx_rho]

        # Evaluate pressure at the lateral interfaces
        pressure_itf_i = p0 * xp.exp((cpd / cvd) * xp.log(var_itf_i[idx_rho_theta] * (Rd / p0)))
        pressure_itf_j = p0 * xp.exp((cpd / cvd) * xp.log(var_itf_j[idx_rho_theta] * (Rd / p0)))

        # Riemann solver
        flux_x1_itf_i = xp.zeros_like(var_itf_i)
        wflux_adv_x1_itf_i = xp.zeros_like(flux_x1_itf_i[0])
        wflux_pres_x1_itf_i = xp.zeros_like(flux_x1_itf_i[0])
        flux_x2_itf_j = xp.zeros_like(var_itf_j)
        wflux_adv_x2_itf_j = xp.zeros_like(flux_x2_itf_j[0])
        wflux_pres_x2_itf_j = xp.zeros_like(flux_x2_itf_j[0])

        rusanov_3d_hori_i_new(
            u1_itf_i,
            var_itf_i,
            pressure_itf_i,
            metric,
            0,
            advection_only,
            flux_x1_itf_i,
            wflux_adv_x1_itf_i,
            wflux_pres_x1_itf_i,
            nbsolpts,
            xp,
        )
        rusanov_3d_hori_j_new(
            u2_itf_j,
            var_itf_j,
            pressure_itf_j,
            metric,
            0,
            advection_only,
            flux_x2_itf_j,
            wflux_adv_x2_itf_j,
            wflux_pres_x2_itf_j,
            nbsolpts,
            xp,
        )

        # Perform flux derivatives
        df1_dx1 = flux_x1 @ operators.derivative_x + flux_x1_itf_i[mid_i] @ operators.correction_WE
        df2_dx2 = flux_x2 @ operators.derivative_y + flux_x2_itf_j[mid_j] @ operators.correction_SN
        df3_dx3 = flux_x3 @ operators.derivative_z + flux_x3_itf_k[mid_k] @ operators.correction_DU

        logp_int = xp.log(pressure)

        logp_bdy_i = xp.log(pressure_itf_i[mid_i])
        logp_bdy_j = xp.log(pressure_itf_j[mid_j])
        logp_bdy_k = xp.log(pressure_itf_k[mid_k])

        w_df1_dx1_adv = wflux_adv_x1 @ op.derivative_x + wflux_adv_x1_itf_i[mid_i] @ op.correction_WE
        w_df1_dx1_presa = (wflux_pres_x1 @ op.derivative_x + wflux_pres_x1_itf_i[mid_i] @ op.correction_WE) * pressure
        w_df1_dx1_presb = (logp_int @ op.derivative_x + logp_bdy_i @ op.correction_WE) * pressure * wflux_pres_x1
        w_df1_dx1 = w_df1_dx1_adv + w_df1_dx1_presa + w_df1_dx1_presb

        w_df2_dx2_adv = wflux_adv_x2 @ op.derivative_y + wflux_adv_x2_itf_j[mid_j] @ op.correction_SN
        w_df2_dx2_presa = (wflux_pres_x2 @ op.derivative_y + wflux_pres_x2_itf_j[mid_j] @ op.correction_SN) * pressure
        w_df2_dx2_presb = (logp_int @ op.derivative_y + logp_bdy_j @ op.correction_SN) * pressure * wflux_pres_x2
        w_df2_dx2 = w_df2_dx2_adv + w_df2_dx2_presa + w_df2_dx2_presb

        w_df3_dx3_adv = wflux_adv_x3 @ op.derivative_z + wflux_adv_x3_itf_k[mid_k] @ op.correction_DU
        w_df3_dx3_presa = (wflux_pres_x3 @ op.derivative_z + wflux_pres_x3_itf_k[mid_k] @ op.correction_DU) * pressure
        w_df3_dx3_presb = (logp_int @ op.derivative_z + logp_bdy_k @ op.correction_DU) * pressure * wflux_pres_x3
        w_df3_dx3 = w_df3_dx3_adv + w_df3_dx3_presa + w_df3_dx3_presb

        # Add coriolis, metric terms and other forcings
        forcing = xp.zeros_like(Q)
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

        # Gravity effect, in vertical direction
        forcing[idx_rho_w] += (
            metric.inv_dzdeta_new * gravity * metric.inv_sqrtG_new * ((metric.sqrtG_new * rho) @ op.highfilter_k)
        )

        # DCMIP cases 2-1 and 2-2 involve rayleigh damping
        # dcmip_schar_damping modifies the 'forcing' variable to apply the requried Rayleigh damping
        if case_number == 21:
            dcmip_schar_damping(forcing, rho, u1, u2, w, metric, geom, shear=False, new_layout=True)
        elif case_number == 22:
            dcmip_schar_damping(forcing, rho, u1, u2, w, metric, geom, shear=True, new_layout=True)

        # Assemble the right-hand side
        rhs = -metric.inv_sqrtG_new * (df1_dx1 + df2_dx2 + df3_dx3) - forcing
        rhs[idx_rho_w] = -metric.inv_sqrtG_new * (w_df1_dx1 + w_df2_dx2 + w_df3_dx3) - forcing[idx_rho_w]

        # For pure advection problems, we do not update the dynamical variables
        if advection_only:
            rhs[idx_rho] = 0.0
            rhs[idx_rho_u1] = 0.0
            rhs[idx_rho_u2] = 0.0
            rhs[idx_rho_w] = 0.0
            rhs[idx_rho_theta] = 0.0

        return rhs
