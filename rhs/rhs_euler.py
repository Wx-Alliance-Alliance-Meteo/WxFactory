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

num_it = 0


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

        This function evaluates RHS of the Euler equations using the four-demsional tensor formulation (see Charron 2014), returning
        an array consisting of the time-derivative of the conserved variables (ρ,ρu,ρv,ρw,ρθ).  A "curried" version of this function,
        with non-Q parameters predefined, should be passed to the time-stepping routine to use as a RHS black-box.  Since some of the
        time-stepping routines perform a Jacobian evaluation via complex derivative, this function should also be safe with respect to
        complex-valued inputs inside Q.

        Note that this function includes MPI communication for inter-process boundary interactions, so it must be called collectively.

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

        global num_it

        # Load CUDA kernels if needed (and not done already)
        global rhs_euler_kernels
        if isinstance(device, CudaDevice) and rhs_euler_kernels is None:
            from wx_cupy import Rusanov

            rhs_euler_kernels = Rusanov

        op = operators
        xp = device.xp
        rank = MPI.COMM_WORLD.rank

        Q_new = geom._to_new(Q)

        type_vec = Q.dtype  #  Output/processing type -- may be complex
        nb_equations = Q.shape[0]  # Number of constituent Euler equations.  Probably 6.
        nb_pts_hori = nb_elements_hori * nbsolpts  # Total number of solution points per horizontal dimension
        nb_vertical_levels = nb_elements_vert * nbsolpts  # Total number of solution points in the vertical dimension

        def to_new_itf_i(a):
            src_shape = (nb_vertical_levels, nb_elements_hori + 2, 2, nb_pts_hori)
            if a.shape[-4:] != src_shape:
                raise ValueError(f"Wrong shape {a.shape}, expected (...,) + {src_shape}")

            tmp_shape1 = a.shape[:-4] + (
                geom.nb_elements_x3,
                geom.nbsolpts,
                geom.nb_elements_x1 + 2,
                2,
                geom.nb_elements_x2,
                geom.nbsolpts,
            )
            offset = len(a.shape[:-4])
            transp = ()
            for i in range(offset):
                transp += (i,)
            transp += (offset, offset + 4, offset + 2, offset + 3, offset + 1, offset + 5)
            # print(f"transp = {transp}")

            tmp_shape2 = a.shape[:-4] + geom.itf_i_shape
            tmp_1 = a.reshape(tmp_shape1)
            tmp_2 = tmp_1.transpose(transp)
            tmp_array = tmp_2.reshape(tmp_shape2)

            return tmp_array.copy()

        def to_new_itf_j(a):
            src_shape = (nb_vertical_levels, nb_elements_hori + 2, 2, nb_pts_hori)
            if a.shape[-4:] != src_shape:
                raise ValueError(f"Wrong shape {a.shape}, expected (...,) + {src_shape}")

            tmp_shape1 = a.shape[:-4] + (
                geom.nb_elements_x3,
                geom.nbsolpts,
                geom.nb_elements_x2 + 2,
                2,
                geom.nb_elements_x1,
                geom.nbsolpts,
            )
            tmp_shape2 = a.shape[:-4] + geom.itf_j_shape

            tmp_1 = a.reshape(tmp_shape1)
            tmp_2 = xp.moveaxis(tmp_1, (-5, -2), (-2, -4))
            tmp_array = tmp_2.reshape(tmp_shape2)

            return tmp_array.copy()

        def to_new_itf_k(a):
            src_shape = (nb_pts_hori, nb_elements_vert + 2, 2, nb_pts_hori)
            if a.shape[-4:] != src_shape:
                raise ValueError(f"Wrong shape {a.shape}, expected (...,) + {src_shape}")

            tmp_shape1 = a.shape[:-4] + (
                geom.nb_elements_x2,
                geom.nbsolpts,
                geom.nb_elements_x3 + 2,
                2,
                geom.nb_elements_x1,
                geom.nbsolpts,
            )
            tmp_shape2 = a.shape[:-4] + geom.itf_k_shape

            offset = len(a.shape[:-4])
            transp = ()
            for i in range(offset):
                transp += (i,)
            transp += (offset + 2, offset, offset + 4, offset + 3, offset + 1, offset + 5)

            tmp_1 = a.reshape(tmp_shape1)
            tmp_2 = tmp_1.transpose(transp)
            tmp_array = tmp_2.reshape(tmp_shape2)

            return tmp_array.copy()

        # Array for forcing: Coriolis terms, metric corrections from the curvilinear coordinate, and gravity
        forcing = xp.zeros_like(Q, dtype=type_vec)

        # Array to extrapolate variables and fluxes to the boundaries along x (i)
        variables_itf_i = xp.ones(
            (nb_equations, nb_vertical_levels, nb_elements_hori + 2, 2, nb_pts_hori), dtype=type_vec
        )  # Initialized to one in the halo to avoid division by zero later
        # Note that flux_x1_itf_i has a different shape than variables_itf_i
        flux_x1_itf_i = xp.empty(
            (nb_equations, nb_vertical_levels, nb_elements_hori + 2, nb_pts_hori, 2), dtype=type_vec
        )
        flux_x1_itf_i[...] = 0.0

        # Extrapolation arrays along y (j)
        variables_itf_j = xp.ones(
            (nb_equations, nb_vertical_levels, nb_elements_hori + 2, 2, nb_pts_hori), dtype=type_vec
        )  # Initialized to one in the halo to avoid division by zero later
        flux_x2_itf_j = xp.empty(
            (nb_equations, nb_vertical_levels, nb_elements_hori + 2, 2, nb_pts_hori), dtype=type_vec
        )
        flux_x2_itf_j[...] = 0.0

        # Extrapolation arrays along z (k), note dimensions of (6, nj, nk+2, 2, ni)
        variables_itf_k = xp.empty((nb_equations, nb_pts_hori, nb_elements_vert + 2, 2, nb_pts_hori), dtype=type_vec)
        flux_x3_itf_k = xp.empty((nb_equations, nb_pts_hori, nb_elements_vert + 2, 2, nb_pts_hori), dtype=type_vec)
        flux_x3_itf_k[...] = 0.0

        # Special arrays for calculation of (ρw) flux
        wflux_adv_x1_itf_i = xp.zeros_like(flux_x1_itf_i[0])
        wflux_pres_x1_itf_i = xp.zeros_like(flux_x1_itf_i[0])
        wflux_adv_x2_itf_j = xp.zeros_like(flux_x2_itf_j[0])
        wflux_pres_x2_itf_j = xp.zeros_like(flux_x2_itf_j[0])
        wflux_adv_x3_itf_k = xp.zeros_like(flux_x3_itf_k[0])
        wflux_pres_x3_itf_k = xp.zeros_like(flux_x3_itf_k[0])

        # Flag for advection-only processing, with DCMIP test cases 11 and 12
        advection_only = case_number < 13

        variables_itf_i[:, :, 1:-1, :, :] = operators.extrapolate_i(Q, geom).transpose((0, 1, 3, 4, 2))
        variables_itf_j[:, :, 1:-1, :, :] = operators.extrapolate_j(Q, geom)

        itf_i_shape = (nb_equations,) + geom.itf_i_shape
        itf_j_shape = (nb_equations,) + geom.itf_j_shape
        itf_k_shape = (nb_equations,) + geom.itf_k_shape

        # Interpolate to the element interface (middle elements only, halo remains 0)
        var_itf_i_new = xp.ones(itf_i_shape, dtype=Q.dtype)
        var_itf_i_new[..., 1:-1, :] = Q_new @ operators.extrap_x

        var_itf_j_new = xp.ones(itf_j_shape, dtype=Q.dtype)
        var_itf_j_new[..., 1:-1, :, :] = Q_new @ operators.extrap_y

        # Scaled variables for separate reconstruction
        logrho = xp.log(Q[idx_rho])
        logrhotheta = xp.log(Q[idx_rho_theta])

        logrho_new = xp.log(Q_new[idx_rho])
        logrhotheta_new = xp.log(Q_new[idx_rho_theta])

        variables_itf_i[idx_rho, :, 1:-1, :, :] = xp.exp(operators.extrapolate_i(logrho, geom)).transpose((0, 2, 3, 1))
        variables_itf_j[idx_rho, :, 1:-1, :, :] = xp.exp(operators.extrapolate_j(logrho, geom))

        variables_itf_i[idx_rho_theta, :, 1:-1, :, :] = xp.exp(operators.extrapolate_i(logrhotheta, geom)).transpose(
            (0, 2, 3, 1)
        )
        variables_itf_j[idx_rho_theta, :, 1:-1, :, :] = xp.exp(operators.extrapolate_j(logrhotheta, geom))

        var_itf_i_new[idx_rho, ..., 1:-1, :] = xp.exp(logrho_new @ operators.extrap_x)
        var_itf_i_new[idx_rho_theta, ..., 1:-1, :] = xp.exp(logrhotheta_new @ operators.extrap_x)
        var_itf_j_new[idx_rho, ..., 1:-1, :, :] = xp.exp(logrho_new @ operators.extrap_y)
        var_itf_j_new[idx_rho_theta, ..., 1:-1, :, :] = xp.exp(logrhotheta_new @ operators.extrap_y)

        # Initiate transfers
        s_ = xp.s_[..., 1, 0, :]
        n_ = xp.s_[..., -2, 1, :]
        w_ = s_
        e_ = n_
        req_r = ptopo.start_exchange_scalars(
            variables_itf_j[idx_rho][s_],
            variables_itf_j[idx_rho][n_],
            variables_itf_i[idx_rho][w_],
            variables_itf_i[idx_rho][e_],
            (nb_pts_hori,),
        )
        req_u = ptopo.start_exchange_vectors(
            (variables_itf_j[idx_rho_u1][s_], variables_itf_j[idx_rho_u2][s_], variables_itf_j[idx_rho_w][s_]),
            (variables_itf_j[idx_rho_u1][n_], variables_itf_j[idx_rho_u2][n_], variables_itf_j[idx_rho_w][n_]),
            (variables_itf_i[idx_rho_u1][w_], variables_itf_i[idx_rho_u2][w_], variables_itf_i[idx_rho_w][w_]),
            (variables_itf_i[idx_rho_u1][e_], variables_itf_i[idx_rho_u2][e_], variables_itf_i[idx_rho_w][e_]),
            geom.boundary_sn,
            geom.boundary_we,
        )
        req_t = ptopo.start_exchange_scalars(
            variables_itf_j[idx_rho_theta][s_],
            variables_itf_j[idx_rho_theta][n_],
            variables_itf_i[idx_rho_theta][w_],
            variables_itf_i[idx_rho_theta][e_],
            (nb_pts_hori,),
        )

        s2_ = xp.s_[..., 1, :, : nbsolpts**2]
        n2_ = xp.s_[..., -2, :, nbsolpts**2 :]
        w2_ = xp.s_[..., 1, : nbsolpts**2]
        e2_ = xp.s_[..., -2, nbsolpts**2 :]
        req_r_new = ptopo.start_exchange_scalars(
            var_itf_j_new[idx_rho][s2_],
            var_itf_j_new[idx_rho][n2_],
            var_itf_i_new[idx_rho][w2_],
            var_itf_i_new[idx_rho][e2_],
            boundary_shape=(nb_elements_hori, nbsolpts, nbsolpts),
            flip_dim=(-3, -1),
        )
        req_u_new = ptopo.start_exchange_vectors(
            (var_itf_j_new[idx_rho_u1][s2_], var_itf_j_new[idx_rho_u2][s2_], var_itf_j_new[idx_rho_w][s2_]),
            (var_itf_j_new[idx_rho_u1][n2_], var_itf_j_new[idx_rho_u2][n2_], var_itf_j_new[idx_rho_w][n2_]),
            (var_itf_i_new[idx_rho_u1][w2_], var_itf_i_new[idx_rho_u2][w2_], var_itf_i_new[idx_rho_w][w2_]),
            (var_itf_i_new[idx_rho_u1][e2_], var_itf_i_new[idx_rho_u2][e2_], var_itf_i_new[idx_rho_w][e2_]),
            geom.boundary_sn_new,
            geom.boundary_we_new,
            flip_dim=(-3, -1),
        )
        req_t_new = ptopo.start_exchange_scalars(
            var_itf_j_new[idx_rho_theta][s2_],
            var_itf_j_new[idx_rho_theta][n2_],
            var_itf_i_new[idx_rho_theta][w2_],
            var_itf_i_new[idx_rho_theta][e2_],
            boundary_shape=(nb_elements_hori, nbsolpts, nbsolpts),
            flip_dim=(-3, -1),
        )

        # Unpack dynamical variables, each to arrays of size [nk,nj,ni]
        rho = Q[idx_rho]
        u1 = Q[idx_rho_u1] / rho
        u2 = Q[idx_rho_u2] / rho
        w = Q[idx_rho_w] / rho

        rho_new = Q_new[idx_rho]
        u1_new = Q_new[idx_rho_u1] / rho_new
        u2_new = Q_new[idx_rho_u2] / rho_new
        w_new = Q_new[idx_rho_w] / rho_new

        # Compute the advective fluxes ...
        flux_x1 = metric.sqrtG * u1 * Q
        flux_x2 = metric.sqrtG * u2 * Q
        flux_x3 = metric.sqrtG * w * Q

        flux_x1_new = metric.sqrtG_new * u1_new * Q_new
        flux_x2_new = metric.sqrtG_new * u2_new * Q_new
        flux_x3_new = metric.sqrtG_new * w_new * Q_new

        # flux_x1_new = metric.sqrtG_new * u1_new * Q_new
        # flux_x2_new = metric.sqrtG_new * u2_new * Q_new
        # flux_x3_new = metric.sqrtG_new * w_new * Q_new

        wflux_adv_x1 = metric.sqrtG * u1 * Q[idx_rho_w]
        wflux_adv_x2 = metric.sqrtG * u2 * Q[idx_rho_w]
        wflux_adv_x3 = metric.sqrtG * w * Q[idx_rho_w]

        wflux_adv_x1_new = metric.sqrtG_new * u1_new * Q_new[idx_rho_w]
        wflux_adv_x2_new = metric.sqrtG_new * u2_new * Q_new[idx_rho_w]
        wflux_adv_x3_new = metric.sqrtG_new * w_new * Q_new[idx_rho_w]

        # ... and add the pressure component
        # Performance note: exp(log) is measurably faster than ** (pow)
        pressure = p0 * xp.exp((cpd / cvd) * xp.log((Rd / p0) * Q[idx_rho_theta]))
        pressure_new = p0 * xp.exp((cpd / cvd) * xp.log((Rd / p0) * Q_new[idx_rho_theta]))

        flux_x1[idx_rho_u1] += metric.sqrtG * metric.H_contra_11 * pressure
        flux_x1[idx_rho_u2] += metric.sqrtG * metric.H_contra_12 * pressure
        flux_x1[idx_rho_w] += metric.sqrtG * metric.H_contra_13 * pressure

        flux_x1_new[idx_rho_u1] += metric.sqrtG_new * metric.h_contra_new[0, 0] * pressure_new
        flux_x1_new[idx_rho_u2] += metric.sqrtG_new * metric.h_contra_new[0, 1] * pressure_new
        flux_x1_new[idx_rho_w] += metric.sqrtG_new * metric.h_contra_new[0, 2] * pressure_new

        wflux_pres_x1 = (metric.sqrtG * metric.H_contra_13).astype(type_vec)
        wflux_pres_x1_new = (metric.sqrtG_new * metric.h_contra_new[0, 2]).astype(type_vec)

        diff_wpr_1 = wflux_pres_x1_new - geom._to_new(wflux_pres_x1)
        diff_wpr_1n = xp.linalg.norm(diff_wpr_1) / xp.linalg.norm(wflux_pres_x1)
        if diff_wpr_1n > 1e-13:
            print(f"{MPI.COMM_WORLD.rank} Large wflux pres x1 diff: {diff_wpr_1n:.2e}")
            raise ValueError

        flux_x2[idx_rho_u1] += metric.sqrtG * metric.H_contra_21 * pressure
        flux_x2[idx_rho_u2] += metric.sqrtG * metric.H_contra_22 * pressure
        flux_x2[idx_rho_w] += metric.sqrtG * metric.H_contra_23 * pressure

        flux_x2_new[idx_rho_u1] += metric.sqrtG_new * metric.h_contra_new[1, 0] * pressure_new
        flux_x2_new[idx_rho_u2] += metric.sqrtG_new * metric.h_contra_new[1, 1] * pressure_new
        flux_x2_new[idx_rho_w] += metric.sqrtG_new * metric.h_contra_new[1, 2] * pressure_new

        wflux_pres_x2 = (metric.sqrtG * metric.H_contra_23).astype(type_vec)
        wflux_pres_x2_new = (metric.sqrtG_new * metric.h_contra_new[1, 2]).astype(type_vec)

        flux_x3[idx_rho_u1] += metric.sqrtG * metric.H_contra_31 * pressure
        flux_x3[idx_rho_u2] += metric.sqrtG * metric.H_contra_32 * pressure
        flux_x3[idx_rho_w] += metric.sqrtG * metric.H_contra_33 * pressure

        flux_x3_new[idx_rho_u1] += metric.sqrtG_new * metric.h_contra_new[2, 0] * pressure_new
        flux_x3_new[idx_rho_u2] += metric.sqrtG_new * metric.h_contra_new[2, 1] * pressure_new
        flux_x3_new[idx_rho_w] += metric.sqrtG_new * metric.h_contra_new[2, 2] * pressure_new

        wflux_pres_x3 = (metric.sqrtG * metric.H_contra_33).astype(type_vec)
        wflux_pres_x3_new = (metric.sqrtG_new * metric.h_contra_new[2, 2]).astype(type_vec)

        variables_itf_k[...] = 0.0
        variables_itf_k[:, :, 1:-1, :, :] = operators.extrapolate_k(Q, geom).transpose((0, 3, 1, 2, 4))

        var_itf_k_new = xp.empty(itf_k_shape, dtype=type_vec)
        var_itf_k_new[...] = 0.0
        # var_itf_k_new[1][..., 1:-1, :, :, :] = Q_new[1] @ operators.extrap_z
        # var_itf_k_new[2][..., 1:-1, :, :, :] = Q_new[2] @ operators.extrap_z
        # var_itf_k_new[3][..., 1:-1, :, :, :] = Q_new[3] @ operators.extrap_z
        var_itf_k_new[..., 1:-1, :, :, :] = Q_new @ operators.extrap_z

        variables_itf_k[idx_rho, :, 1:-1, :, :] = xp.exp(operators.extrapolate_k(logrho, geom).transpose((2, 0, 1, 3)))
        variables_itf_k[idx_rho_theta, :, 1:-1, :, :] = xp.exp(
            operators.extrapolate_k(logrhotheta, geom).transpose((2, 0, 1, 3))
        )

        var_itf_k_new[idx_rho][..., 1:-1, :, :, :] = xp.exp(logrho_new @ operators.extrap_z)
        var_itf_k_new[idx_rho_theta][..., 1:-1, :, :, :] = xp.exp(logrhotheta_new @ operators.extrap_z)

        south = xp.s_[: nbsolpts**2]
        north = xp.s_[nbsolpts**2 :]

        # For consistency at the surface and top boundaries, treat the extrapolation as continuous.  That is,
        # the "top" of the ground is equal to the "bottom" of the atmosphere, and the "bottom" of the model top
        # is equal to the "top" of the atmosphere.
        variables_itf_k[:, :, 0, 1, :] = variables_itf_k[:, :, 1, 0, :]
        variables_itf_k[:, :, 0, 0, :] = variables_itf_k[:, :, 0, 1, :]
        variables_itf_k[:, :, -1, 0, :] = variables_itf_k[:, :, -2, 1, :]
        variables_itf_k[:, :, -1, 1, :] = variables_itf_k[:, :, -1, 0, :]

        var_itf_k_new[..., 0, :, :, south] = var_itf_k_new[..., 1, :, :, south]
        var_itf_k_new[..., 0, :, :, north] = var_itf_k_new[..., 0, :, :, south]
        var_itf_k_new[..., -1, :, :, north] = var_itf_k_new[..., -2, :, :, north]
        var_itf_k_new[..., -1, :, :, south] = var_itf_k_new[..., -1, :, :, north]

        diffz = var_itf_k_new - to_new_itf_k(variables_itf_k)
        denom = xp.linalg.norm(variables_itf_k)
        diffzn = xp.linalg.norm(diffz) / denom
        if diffzn > 1e-15:
            print(f"{MPI.COMM_WORLD.rank} large diff: {diffzn:.2e}")
            raise ValueError

        # Evaluate pressure at the vertical element interfaces based on ρθ.
        pressure_itf_k = p0 * xp.exp((cpd / cvd) * xp.log(variables_itf_k[idx_rho_theta] * (Rd / p0)))
        pressure_itf_k_new = p0 * xp.exp((cpd / cvd) * xp.log(var_itf_k_new[idx_rho_theta] * (Rd / p0)))

        # Take w ← (wρ)/ ρ at the vertical interfaces
        w_itf_k = variables_itf_k[idx_rho_w] / variables_itf_k[idx_rho]
        w_itf_k_new = var_itf_k_new[idx_rho_w] / var_itf_k_new[idx_rho]

        # Surface and top boundary treatement, imposing no flow (w=0) through top and bottom
        # csubich -- apply odd symmetry to w at boundary so there is no advective _flux_ through boundary
        w_itf_k[:, 0, 0, :] = 0.0  # Bottom of bottom element (unused)
        w_itf_k[:, 0, 1, :] = -w_itf_k[:, 1, 0, :]  # Top of bottom element (negative symmetry)
        w_itf_k[:, -1, 1, :] = 0.0  # Top of top element (unused)
        w_itf_k[:, -1, 0, :] = -w_itf_k[:, -2, 1, :]  # Bottom of top boundary element (negative symmetry)

        w_itf_k_new[0, :, :, south] = 0.0
        w_itf_k_new[0, :, :, north] = -w_itf_k_new[1, :, :, south]
        w_itf_k_new[-1, :, :, north] = 0.0
        w_itf_k_new[-1, :, :, south] = -w_itf_k_new[-2, :, :, north]

        diffw = w_itf_k_new - to_new_itf_k(w_itf_k)
        denom = xp.linalg.norm(w_itf_k)
        diffwn = xp.linalg.norm(diffw)
        if denom > 0.0:
            diffwn /= denom
        if diffwn > 1e-15:
            print(f"{MPI.COMM_WORLD.rank} diff for w itf: {diffwn:.2e}")
            raise ValueError

        diff_vk = var_itf_k_new - to_new_itf_k(variables_itf_k)
        diff_vkn = xp.linalg.norm(diff_vk)
        denom = xp.linalg.norm(variables_itf_k)
        if denom > 0.0:
            diff_vkn /= denom
        if diff_vkn > 1e-15:
            print(f"{MPI.COMM_WORLD.rank} diff for var k itf: {diff_vkn:.2e}")
            raise ValueError

        flux_x3_itf_k_new = xp.zeros_like(var_itf_k_new)
        wflux_adv_x3_itf_k_new = xp.zeros_like(flux_x3_itf_k_new[0])
        wflux_pres_x3_itf_k_new = xp.zeros_like(flux_x3_itf_k_new[0])
        rusanov_3d_vert_new(
            var_itf_k_new,
            pressure_itf_k_new,
            w_itf_k_new,
            metric,
            advection_only,
            flux_x3_itf_k_new,
            wflux_adv_x3_itf_k_new,
            wflux_pres_x3_itf_k_new,
            xp,
            nbsolpts,
        )

        if isinstance(device, CpuDevice):
            rusanov_3d_vert(
                variables_itf_k,
                pressure_itf_k,
                w_itf_k,
                metric,
                nb_elements_vert + 1,
                advection_only,
                flux_x3_itf_k,
                wflux_adv_x3_itf_k,
                wflux_pres_x3_itf_k,
            )
        elif isinstance(device, CudaDevice):
            rhs_euler_kernels.compute_flux_k(
                flux_x3_itf_k,
                wflux_adv_x3_itf_k,
                wflux_pres_x3_itf_k,
                variables_itf_k,
                pressure_itf_k,
                w_itf_k,
                metric.sqrtG_itf_k,
                metric.H_contra_itf_k[2],
                nb_elements_vert,
                nb_pts_hori,
                nb_equations,
                advection_only,
            )
        else:
            raise ValueError(f"Device is not of a recognized type: {device}")

        diff_fk = flux_x3_itf_k_new - to_new_itf_k(flux_x3_itf_k)
        denom = xp.linalg.norm(flux_x3_itf_k)
        diff_fkn = xp.linalg.norm(diff_fk)
        if denom > 0.0:
            diff_fkn /= denom
        if diff_fkn > 1e-11:
            print(
                f"{MPI.COMM_WORLD.rank} large error with flux_x3_k: {diff_fkn:.2e} (shape {diff_fk.shape})\n"
                # f"{diff_fk}"
            )
            raise ValueError

        # Finish transfers
        s_ = xp.s_[..., 0, 1, :]
        n_ = xp.s_[..., -1, 0, :]
        w_ = s_
        e_ = n_
        (
            variables_itf_j[idx_rho][s_],
            variables_itf_j[idx_rho][n_],
            variables_itf_i[idx_rho][w_],
            variables_itf_i[idx_rho][e_],
        ) = req_r.wait()
        (
            (variables_itf_j[idx_rho_u1][s_], variables_itf_j[idx_rho_u2][s_], variables_itf_j[idx_rho_w][s_]),
            (variables_itf_j[idx_rho_u1][n_], variables_itf_j[idx_rho_u2][n_], variables_itf_j[idx_rho_w][n_]),
            (variables_itf_i[idx_rho_u1][w_], variables_itf_i[idx_rho_u2][w_], variables_itf_i[idx_rho_w][w_]),
            (variables_itf_i[idx_rho_u1][e_], variables_itf_i[idx_rho_u2][e_], variables_itf_i[idx_rho_w][e_]),
        ) = req_u.wait()
        (
            variables_itf_j[idx_rho_theta][s_],
            variables_itf_j[idx_rho_theta][n_],
            variables_itf_i[idx_rho_theta][w_],
            variables_itf_i[idx_rho_theta][e_],
        ) = req_t.wait()

        s3_ = xp.s_[..., 0, :, nbsolpts**2 :]
        n3_ = xp.s_[..., -1, :, : nbsolpts**2]
        w3_ = xp.s_[..., 0, nbsolpts**2 :]
        e3_ = xp.s_[..., -1, : nbsolpts**2]
        (
            var_itf_j_new[idx_rho][s3_],
            var_itf_j_new[idx_rho][n3_],
            var_itf_i_new[idx_rho][w3_],
            var_itf_i_new[idx_rho][e3_],
        ) = req_r_new.wait()
        (
            (var_itf_j_new[idx_rho_u1][s3_], var_itf_j_new[idx_rho_u2][s3_], var_itf_j_new[idx_rho_w][s3_]),
            (var_itf_j_new[idx_rho_u1][n3_], var_itf_j_new[idx_rho_u2][n3_], var_itf_j_new[idx_rho_w][n3_]),
            (var_itf_i_new[idx_rho_u1][w3_], var_itf_i_new[idx_rho_u2][w3_], var_itf_i_new[idx_rho_w][w3_]),
            (var_itf_i_new[idx_rho_u1][e3_], var_itf_i_new[idx_rho_u2][e3_], var_itf_i_new[idx_rho_w][e3_]),
        ) = req_u_new.wait()
        (
            var_itf_j_new[idx_rho_theta][s3_],
            var_itf_j_new[idx_rho_theta][n3_],
            var_itf_i_new[idx_rho_theta][w3_],
            var_itf_i_new[idx_rho_theta][e3_],
        ) = req_t_new.wait()

        diffi = var_itf_i_new - to_new_itf_i(variables_itf_i)
        diffin = xp.linalg.norm(diffi) / xp.linalg.norm(variables_itf_i)
        if diffin > 1e-15:
            if rank != 0:
                print(
                    f"rank {rank}, i\n"
                    f"Q[0] = \n{Q[0]}\n"
                    f"Q_new[0] = \n{Q_new[0]}\n"
                    f"old: \n{variables_itf_i[0]}\n"
                    f"old w/ new shape: \n{to_new_itf_i(variables_itf_i)[0]}\n"
                    f"new: \n{var_itf_i_new[0]}\n"
                    f"diff!!!! \n{diffi[0]}"
                )
            raise ValueError

        MPI.COMM_WORLD.barrier()

        diffj = var_itf_j_new - to_new_itf_j(variables_itf_j)
        diffjn = xp.linalg.norm(diffj) / xp.linalg.norm(variables_itf_j)
        if diffjn > 1e-15:
            if rank == 0:
                print(
                    f"rank {rank}, j\n"
                    f"Q[0] = \n{Q[0]}\n"
                    f"Q_new[0] = \n{Q_new[0]}\n"
                    f"old: \n{variables_itf_j[0]}\n"
                    f"old w/ new shape: \n{to_new_itf_j(variables_itf_j)[0]}\n"
                    f"new: \n{var_itf_j_new[0]}\n"
                    f"diff!!!! \n{diffj[0]}"
                )
            raise ValueError

        # Define u, v at the interface by dividing momentum and density
        u1_itf_i = variables_itf_i[idx_rho_u1] / variables_itf_i[idx_rho]
        u2_itf_j = variables_itf_j[idx_rho_u2] / variables_itf_j[idx_rho]

        u1_itf_i_new = var_itf_i_new[idx_rho_u1] / var_itf_i_new[idx_rho]
        u2_itf_j_new = var_itf_j_new[idx_rho_u2] / var_itf_j_new[idx_rho]

        # Evaluate pressure at the lateral interfaces
        pressure_itf_i = p0 * xp.exp((cpd / cvd) * xp.log(variables_itf_i[idx_rho_theta] * (Rd / p0)))
        pressure_itf_j = p0 * xp.exp((cpd / cvd) * xp.log(variables_itf_j[idx_rho_theta] * (Rd / p0)))

        pressure_itf_i_new = p0 * xp.exp((cpd / cvd) * xp.log(var_itf_i_new[idx_rho_theta] * (Rd / p0)))
        pressure_itf_j_new = p0 * xp.exp((cpd / cvd) * xp.log(var_itf_j_new[idx_rho_theta] * (Rd / p0)))

        # Riemann solver
        flux_x1_itf_i_new = xp.zeros_like(var_itf_i_new)
        wflux_adv_x1_itf_i_new = xp.zeros_like(flux_x1_itf_i_new[0])
        wflux_pres_x1_itf_i_new = xp.zeros_like(flux_x1_itf_i_new[0])
        flux_x2_itf_j_new = xp.zeros_like(var_itf_j_new)
        wflux_adv_x2_itf_j_new = xp.zeros_like(flux_x2_itf_j_new[0])
        wflux_pres_x2_itf_j_new = xp.zeros_like(flux_x2_itf_j_new[0])

        rusanov_3d_hori_i_new(
            u1_itf_i_new,
            var_itf_i_new,
            pressure_itf_i_new,
            metric,
            0,
            advection_only,
            flux_x1_itf_i_new,
            wflux_adv_x1_itf_i_new,
            wflux_pres_x1_itf_i_new,
            nbsolpts,
            xp,
        )
        rusanov_3d_hori_j_new(
            u2_itf_j_new,
            var_itf_j_new,
            pressure_itf_j_new,
            metric,
            0,
            advection_only,
            flux_x2_itf_j_new,
            wflux_adv_x2_itf_j_new,
            wflux_pres_x2_itf_j_new,
            nbsolpts,
            xp,
        )

        if isinstance(device, CpuDevice):
            rusanov_3d_hori_i(
                u1_itf_i,
                variables_itf_i,
                pressure_itf_i,
                metric,
                nb_elements_hori + 1,
                advection_only,
                flux_x1_itf_i,
                wflux_adv_x1_itf_i,
                wflux_pres_x1_itf_i,
            )

            rusanov_3d_hori_j(
                u2_itf_j,
                variables_itf_j,
                pressure_itf_j,
                metric,
                nb_elements_hori + 1,
                advection_only,
                flux_x2_itf_j,
                wflux_adv_x2_itf_j,
                wflux_pres_x2_itf_j,
            )
        elif isinstance(device, CudaDevice):
            rhs_euler_kernels.compute_flux_i(
                flux_x1_itf_i,
                wflux_adv_x1_itf_i,
                wflux_pres_x1_itf_i,
                variables_itf_i,
                pressure_itf_i,
                u1_itf_i,
                metric.sqrtG_itf_i,
                metric.H_contra_itf_i[0],
                nb_elements_hori,
                nb_pts_hori,
                nb_vertical_levels,
                nb_equations,
                advection_only,
            )

            rhs_euler_kernels.compute_flux_j(
                flux_x2_itf_j,
                wflux_adv_x2_itf_j,
                wflux_pres_x2_itf_j,
                variables_itf_j,
                pressure_itf_j,
                u2_itf_j,
                metric.sqrtG_itf_j,
                metric.H_contra_itf_j[1],
                nb_elements_hori,
                nb_pts_hori,
                nb_vertical_levels,
                nb_equations,
                advection_only,
            )
        else:
            raise ValueError(f"Device is not of a recognized type: {device}")

        diff_fi = flux_x1_itf_i_new - to_new_itf_i(flux_x1_itf_i.transpose(0, 1, 2, 4, 3))
        diff_fin = xp.linalg.norm(diff_fi) / xp.linalg.norm(flux_x1_itf_i)

        if diff_fin > 1e-13:
            print(f"Large diff at iteration {num_it}! {diff_fin:.2e}")
            raise ValueError

        diff_fwai = wflux_adv_x1_itf_i_new - to_new_itf_i(wflux_adv_x1_itf_i.transpose(0, 1, 3, 2))
        denom = xp.linalg.norm(wflux_adv_x1_itf_i)
        diff_fwain = xp.linalg.norm(diff_fwai)
        if denom > 0.0:
            diff_fwain /= denom
        if diff_fwain > 1e-14:
            print(
                f"{MPI.COMM_WORLD.rank} Large diff wflux adv i at iteration {num_it}! {diff_fwain:.2e}\n" f"{diff_fwai}"
            )
            raise ValueError

        diff_fwpi = wflux_pres_x1_itf_i_new - to_new_itf_i(wflux_pres_x1_itf_i.transpose(0, 1, 3, 2))
        diff_fwpin = xp.linalg.norm(diff_fwpi) / xp.linalg.norm(wflux_pres_x1_itf_i)
        if diff_fwpin > 1e-13:
            print(
                f"{MPI.COMM_WORLD.rank} Large diff wflux pres i at iteration {num_it}! {diff_fwpin:.2e}\n"
                f"old: \n{to_new_itf_i(wflux_pres_x1_itf_i.transpose(0, 1, 3, 2))}\n"
                f"new: \n{wflux_pres_x1_itf_i_new}\n"
                f"diff: \n{diff_fwpi}"
            )
            raise ValueError

        diff_fj = flux_x2_itf_j_new - to_new_itf_j(flux_x2_itf_j)
        diff_fjn = xp.linalg.norm(diff_fj) / xp.linalg.norm(flux_x2_itf_j)

        if diff_fjn > 1e-13:
            print(f"Large diff flux j at iteration {num_it}! {diff_fjn:.2e}")
            raise ValueError

        # Perform flux derivatives

        flux_x1_bdy = flux_x1_itf_i.transpose((0, 1, 3, 2, 4))[:, :, :, 1:-1, :].copy()
        df1_dx1 = operators.comma_i(flux_x1, flux_x1_bdy, geom)
        flux_x2_bdy = flux_x2_itf_j[:, :, 1:-1, :, :].copy()
        df2_dx2 = operators.comma_j(flux_x2, flux_x2_bdy, geom)
        flux_x3_bdy = flux_x3_itf_k[:, :, 1:-1, :, :].transpose(0, 2, 3, 1, 4).copy()
        df3_dx3 = operators.comma_k(flux_x3, flux_x3_bdy, geom)

        mid_i = xp.s_[..., 1:-1, :]
        mid_j = xp.s_[..., 1:-1, :, :]
        mid_k = xp.s_[..., 1:-1, :, :, :]

        df1_dx1_new = flux_x1_new @ operators.derivative_x + flux_x1_itf_i_new[mid_i] @ operators.correction_WE
        df2_dx2_new = flux_x2_new @ operators.derivative_y + flux_x2_itf_j_new[mid_j] @ operators.correction_SN
        df3_dx3_new = flux_x3_new @ operators.derivative_z + flux_x3_itf_k_new[mid_k] @ operators.correction_DU

        diff_dx1 = df1_dx1_new - geom._to_new(df1_dx1)
        diff_dx2 = df2_dx2_new - geom._to_new(df2_dx2)
        diff_dx3 = df3_dx3_new - geom._to_new(df3_dx3)

        diff_dx1n = xp.linalg.norm(diff_dx1) / xp.linalg.norm(df1_dx1)
        diff_dx2n = xp.linalg.norm(diff_dx2) / xp.linalg.norm(df2_dx2)
        diff_dx3n = xp.linalg.norm(diff_dx3) / xp.linalg.norm(df3_dx3)

        if diff_dx1n > 1e-12 or diff_dx2n > 1e-12 or diff_dx3n > 1e-11:
            print(
                f"{MPI.COMM_WORLD.rank} "
                f"Large diff dfx at iteration {num_it}! {diff_dx1n:.2e} {diff_dx2n:.2e} {diff_dx3n:.2e}"
            )
            raise ValueError

        logp_int = xp.log(pressure)
        logp_int_new = xp.log(pressure_new)

        pressure_bdy_i = pressure_itf_i[:, 1:-1, :, :].transpose((0, 3, 1, 2)).copy()
        pressure_bdy_j = pressure_itf_j[:, 1:-1, :, :].copy()
        pressure_bdy_k = pressure_itf_k[:, 1:-1, :, :].transpose((1, 2, 0, 3)).copy()

        logp_bdy_i = xp.log(pressure_bdy_i)
        logp_bdy_j = xp.log(pressure_bdy_j)
        logp_bdy_k = xp.log(pressure_bdy_k)

        logp_bdy_i_new = xp.log(pressure_itf_i_new[mid_i])
        logp_bdy_j_new = xp.log(pressure_itf_j_new[mid_j])
        logp_bdy_k_new = xp.log(pressure_itf_k_new[mid_k])

        wflux_adv_x1_bdy_i = wflux_adv_x1_itf_i.transpose((0, 2, 1, 3))[:, :, 1:-1, :].copy()
        wflux_pres_x1_bdy_i = wflux_pres_x1_itf_i.transpose((0, 2, 1, 3))[:, :, 1:-1, :].copy()

        wflux_adv_x2_bdy_j = wflux_adv_x2_itf_j[:, 1:-1, :, :].copy()
        wflux_pres_x2_bdy_j = wflux_pres_x2_itf_j[:, 1:-1, :, :].copy()

        wflux_adv_x3_bdy_k = wflux_adv_x3_itf_k[:, 1:-1, :, :].transpose((1, 2, 0, 3)).copy()
        wflux_pres_x3_bdy_k = wflux_pres_x3_itf_k[:, 1:-1, :, :].transpose((1, 2, 0, 3)).copy()

        w_df1_dx1_adv = operators.comma_i(wflux_adv_x1, wflux_adv_x1_bdy_i, geom)
        w_df1_dx1_presa = operators.comma_i(wflux_pres_x1, wflux_pres_x1_bdy_i, geom) * pressure
        w_df1_dx1_presb = operators.comma_i(logp_int, logp_bdy_i, geom) * pressure * wflux_pres_x1
        w_df1_dx1 = w_df1_dx1_adv + w_df1_dx1_presa + w_df1_dx1_presb

        w_df1_dx1_adv_new = wflux_adv_x1_new @ op.derivative_x + wflux_adv_x1_itf_i_new[mid_i] @ op.correction_WE
        w_df1_dx1_presa_new = (
            wflux_pres_x1_new @ op.derivative_x + wflux_pres_x1_itf_i_new[mid_i] @ op.correction_WE
        ) * pressure_new
        w_df1_dx1_presb_new = (
            (logp_int_new @ op.derivative_x + logp_bdy_i_new @ op.correction_WE) * pressure_new * wflux_pres_x1_new
        )
        w_df1_dx1_new = w_df1_dx1_adv_new + w_df1_dx1_presa_new + w_df1_dx1_presb_new

        diff_wa1 = w_df1_dx1_adv_new - geom._to_new(w_df1_dx1_adv)
        denom = xp.linalg.norm(w_df1_dx1_adv)
        diff_wa1n = xp.linalg.norm(diff_wa1)
        if denom > 0.0:
            diff_wa1n /= denom
        diff_wpa1 = w_df1_dx1_presa_new - geom._to_new(w_df1_dx1_presa)
        denom = xp.linalg.norm(w_df1_dx1_presa)
        diff_wpa1n = xp.linalg.norm(diff_wpa1)
        if denom > 0.0:
            diff_wpa1n /= denom
        diff_wpb1 = w_df1_dx1_presb_new - geom._to_new(w_df1_dx1_presb)
        denom = xp.linalg.norm(w_df1_dx1_presb)
        diff_wpb1n = xp.linalg.norm(diff_wpb1)
        if denom > 0.0:
            diff_wpb1n /= denom

        if diff_wa1n > 1e-10 or diff_wpa1n > 1e-10 or diff_wpb1n > 1e-10:
            print(f"{MPI.COMM_WORLD.rank} Large diff w d1: {diff_wa1n:.2e} {diff_wpa1n:.2e} {diff_wpb1n:.2e}")
            raise ValueError

        diff_wd1 = w_df1_dx1_new - geom._to_new(w_df1_dx1)
        diff_wd1n = xp.linalg.norm(diff_wd1) / xp.linalg.norm(w_df1_dx1)
        if diff_wd1n > 1e-13:
            print(f"{MPI.COMM_WORLD.rank} Large diff w dx1: {diff_wd1n:.2e}\n" f"{diff_wd1}")
            raise ValueError

        w_df2_dx2_adv = operators.comma_j(wflux_adv_x2, wflux_adv_x2_bdy_j, geom)
        w_df2_dx2_presa = operators.comma_j(wflux_pres_x2, wflux_pres_x2_bdy_j, geom) * pressure
        w_df2_dx2_presb = operators.comma_j(logp_int, logp_bdy_j, geom) * pressure * wflux_pres_x2
        w_df2_dx2 = w_df2_dx2_adv + w_df2_dx2_presa + w_df2_dx2_presb

        w_df2_dx2_adv_new = wflux_adv_x2_new @ op.derivative_y + wflux_adv_x2_itf_j_new[mid_j] @ op.correction_SN
        w_df2_dx2_presa_new = (
            wflux_pres_x2_new @ op.derivative_y + wflux_pres_x2_itf_j_new[mid_j] @ op.correction_SN
        ) * pressure_new
        w_df2_dx2_presb_new = (
            (logp_int_new @ op.derivative_y + logp_bdy_j_new @ op.correction_SN) * pressure_new * wflux_pres_x2_new
        )
        w_df2_dx2_new = w_df2_dx2_adv_new + w_df2_dx2_presa_new + w_df2_dx2_presb_new

        diff_wdf2 = w_df2_dx2_new - geom._to_new(w_df2_dx2)
        diff_wdf2n = xp.linalg.norm(diff_wdf2) / xp.linalg.norm(w_df2_dx2)
        if diff_wdf2n > 1e-13:
            print(f"{MPI.COMM_WORLD.rank} Large diff w dx2 at it {num_it}: {diff_wdf2n:.2e}\n" f"{diff_wdf2}")
            raise ValueError

        w_df3_dx3_adv = operators.comma_k(wflux_adv_x3, wflux_adv_x3_bdy_k, geom)
        w_df3_dx3_presa = operators.comma_k(wflux_pres_x3, wflux_pres_x3_bdy_k, geom) * pressure
        w_df3_dx3_presb = operators.comma_k(logp_int, logp_bdy_k, geom) * pressure * wflux_pres_x3
        w_df3_dx3 = w_df3_dx3_adv + w_df3_dx3_presa + w_df3_dx3_presb

        w_df3_dx3_adv_new = wflux_adv_x3_new @ op.derivative_z + wflux_adv_x3_itf_k_new[mid_k] @ op.correction_DU
        w_df3_dx3_presa_new = (
            wflux_pres_x3_new @ op.derivative_z + wflux_pres_x3_itf_k_new[mid_k] @ op.correction_DU
        ) * pressure_new
        w_df3_dx3_presb_new = (
            (logp_int_new @ op.derivative_z + logp_bdy_k_new @ op.correction_DU) * pressure_new * wflux_pres_x3_new
        )
        w_df3_dx3_new = w_df3_dx3_adv_new + w_df3_dx3_presa_new + w_df3_dx3_presb_new

        diff_wdf3 = w_df3_dx3_new - geom._to_new(w_df3_dx3)
        diff_wdf3n = xp.linalg.norm(diff_wdf3) / xp.linalg.norm(w_df3_dx3)
        if diff_wdf3n > 1e-13:
            print(f"{MPI.COMM_WORLD.rank} Large diff w dx3 at it {num_it}: {diff_wdf3n:.2e}\n" f"{diff_wdf3}")
            raise ValueError

        # Add coriolis, metric terms and other forcings

        forcing[idx_rho] = 0.0
        forcing[idx_rho_theta] = 0.0

        self.compute_forcings(
            forcing[idx_rho_u1],
            forcing[idx_rho_u2],
            forcing[idx_rho_w],
            rho,
            u1,
            u2,
            w,
            pressure,
            metric.christoffel_1_01,
            metric.christoffel_1_02,
            metric.christoffel_1_03,
            metric.christoffel_1_11,
            metric.christoffel_1_12,
            metric.christoffel_1_13,
            metric.christoffel_1_22,
            metric.christoffel_1_23,
            metric.christoffel_1_33,
            metric.christoffel_2_01,
            metric.christoffel_2_02,
            metric.christoffel_2_03,
            metric.christoffel_2_11,
            metric.christoffel_2_12,
            metric.christoffel_2_13,
            metric.christoffel_2_22,
            metric.christoffel_2_23,
            metric.christoffel_2_33,
            metric.christoffel_3_01,
            metric.christoffel_3_02,
            metric.christoffel_3_03,
            metric.christoffel_3_11,
            metric.christoffel_3_12,
            metric.christoffel_3_13,
            metric.christoffel_3_22,
            metric.christoffel_3_23,
            metric.christoffel_3_33,
            metric.H_contra_11,
            metric.H_contra_12,
            metric.H_contra_13,
            metric.H_contra_22,
            metric.H_contra_23,
            metric.H_contra_33,
        )

        forcing_new = xp.empty_like(Q_new)
        forcing_new[idx_rho] = 0.0
        forcing_new[idx_rho_theta] = 0.0
        self.compute_forcings(
            forcing_new[idx_rho_u1],
            forcing_new[idx_rho_u2],
            forcing_new[idx_rho_w],
            rho_new,
            u1_new,
            u2_new,
            w_new,
            pressure_new,
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
            metric.inv_dzdeta * gravity * metric.inv_sqrtG * operators.filter_k(metric.sqrtG * rho, geom)
        )

        # if MPI.COMM_WORLD.rank == 0:
        #     print(f"shape left = {(metric.sqrtG_new * rho_new).shape}, shape right  = {op.highfilter_k.shape}")

        forcing_new[idx_rho_w] += (
            metric.inv_dzdeta_new * gravity * metric.inv_sqrtG_new * ((metric.sqrtG_new * rho_new) @ op.highfilter_k)
        )

        # DCMIP cases 2-1 and 2-2 involve rayleigh damping
        # dcmip_schar_damping modifies the 'forcing' variable to apply the requried Rayleigh damping
        if case_number == 21:
            dcmip_schar_damping(forcing, rho, u1, u2, w, metric, geom, shear=False, new_layout=False)
            dcmip_schar_damping(forcing_new, rho_new, u1_new, u2_new, w_new, metric, geom, shear=False, new_layout=True)
        elif case_number == 22:
            dcmip_schar_damping(forcing, rho, u1, u2, w, metric, geom, shear=True, new_layout=False)
            dcmip_schar_damping(forcing_new, rho_new, u1_new, u2_new, w_new, metric, geom, shear=True, new_layout=True)

        diff_f = forcing_new - geom._to_new(forcing)
        diff_fn = xp.linalg.norm(diff_f) / xp.linalg.norm(forcing)
        if diff_fn > 1e-14:
            print(f"{MPI.COMM_WORLD.rank} Diff forcing at it {num_it}: {diff_fn:.2e}")
            raise ValueError

        # Assemble the right-hand side
        rhs = -metric.inv_sqrtG * (df1_dx1 + df2_dx2 + df3_dx3) - forcing
        rhs[idx_rho_w] = -metric.inv_sqrtG * (w_df1_dx1 + w_df2_dx2 + w_df3_dx3) - forcing[idx_rho_w]

        rhs_new = -metric.inv_sqrtG_new * (df1_dx1_new + df2_dx2_new + df3_dx3_new) - forcing_new
        rhs_new[idx_rho_w] = (
            -metric.inv_sqrtG_new * (w_df1_dx1_new + w_df2_dx2_new + w_df3_dx3_new) - forcing_new[idx_rho_w]
        )
        diff_rhs = rhs_new - geom._to_new(rhs)
        diff_rhsn = xp.linalg.norm(diff_rhs) / xp.linalg.norm(rhs)

        if diff_rhsn > 1e-9:
            print(f"{MPI.COMM_WORLD.rank} Large RHS diff at it {num_it}: {diff_rhsn:.2e}")
            raise ValueError

        # For pure advection problems, we do not update the dynamical variables
        if advection_only:
            rhs[idx_rho] = 0.0
            rhs[idx_rho_u1] = 0.0
            rhs[idx_rho_u2] = 0.0
            rhs[idx_rho_w] = 0.0
            rhs[idx_rho_theta] = 0.0

        num_it += 1

        return rhs
