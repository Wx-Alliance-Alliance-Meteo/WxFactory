import numpy
import sys

from common.definitions import idx_rho_u1, idx_rho_u2, idx_rho_w, idx_rho, idx_rho_theta, gravity, p0, Rd, cpd, cvd, heat_capacity_ratio
from .fluxes import rusanov_3d_vert, rusanov_3d_hori_i, rusanov_3d_hori_j

# For type hints
from common.parallel import DistributedWorld
from geometry        import CubedSphere, DFROperators, Metric3DTopo
from init.dcmip      import dcmip_schar_damping

#@profile
def rhs_euler (Q: numpy.ndarray, geom: CubedSphere, mtrx: DFROperators, metric: Metric3DTopo, ptopo: DistributedWorld,
               nbsolpts: int, nb_elements_hori: int, nb_elements_vert: int, case_number: int):
   '''Evaluate the right-hand side of the three-dimensional Euler equations.

   This function evaluates RHS of the Euler equations using the four-demsional tensor formulation (see Charron 2014), returning
   an array consisting of the time-derivative of the conserved variables (ρ,ρu,ρv,ρw,ρθ).  A "curried" version of this function,
   with non-Q parameters predefined, should be passed to the time-stepping routine to use as a RHS black-box.  Since some of the
   time-stepping routines perform a Jacobian evaluation via complex derivative, this function should also be safe with respect to
   complex-valued inputs inside Q.

   Note that this function includes MPI communication for inter-process boundary interactions, so it must be called collectively.

   Parameters
   ----------
   Q : numpy.ndarray
      Input array of the current model state, indexed as (var,k,j,i)
   geom : CubedSphere
      Geometry definition, containing parameters relating to the spherical coordinate system
   mtrx : DFR_operators
      Contains matrix operators for the DFR discretization, notably boundary extrapolation and
      local (partial) derivatives
   metric : Metric
      Contains the various metric terms associated with the tensor formulation, notably including the
      scalar √g, the spatial metric h, and the Christoffel symbols
   ptopo : Distributed_World
      Wraps the information and communication functions necessary for MPI distribution
   nbsolpts : int
      Number of interior nodal points per element.  A 3D element will contain nbsolpts**3 internal points.
   nb_elements_hori : int
      Number of elements in x/y on each panel of the cubed sphere
   nb_elements_vert : int
      Number of elements in the vertical
   case_number : int
      DCMIP case number, used to selectively enable or disable parts of the Euler equations to accomplish
      specialized tests like advection-only

   Returns:
   --------
   rhs : numpy.ndarray
      Output of right-hand-side terms of Euler equations
   '''

   type_vec = Q.dtype #  Output/processing type -- may be complex
   nb_equations = Q.shape[0] # Number of constituent Euler equations.  Probably 6.
   nb_interfaces_hori = nb_elements_hori + 1 # Number of element interfaces per horizontal dimension
   nb_interfaces_vert = nb_elements_vert + 1 # Number of element interfaces in the vertical dimension
   nb_pts_hori = nb_elements_hori * nbsolpts # Total number of solution points per horizontal dimension
   nb_vertical_levels = nb_elements_vert * nbsolpts # Total number of solution points in the vertical dimension

   # Array for forcing: Coriolis terms, metric corrections from the curvilinear coordinate, and gravity
   forcing = numpy.zeros_like(Q, dtype=type_vec)

   # Array to extrapolate variables and fluxes to the boundaries along x (i)
   variables_itf_i = numpy.ones((nb_equations, nb_vertical_levels, nb_elements_hori + 2, 2, nb_pts_hori), dtype=type_vec) # Initialized to one in the halo to avoid division by zero later
   # Note that flux_x1_itf_i has a different shape than variables_itf_i
   flux_x1_itf_i   = numpy.empty((nb_equations, nb_vertical_levels, nb_elements_hori + 2, nb_pts_hori, 2), dtype=type_vec)

   # Extrapolation arrays along y (j)
   variables_itf_j = numpy.ones((nb_equations, nb_vertical_levels, nb_elements_hori + 2, 2, nb_pts_hori), dtype=type_vec) # Initialized to one in the halo to avoid division by zero later
   flux_x2_itf_j   = numpy.empty((nb_equations, nb_vertical_levels, nb_elements_hori + 2, 2, nb_pts_hori), dtype=type_vec)

   # Extrapolation arrays along z (k), note dimensions of (6, nj, nk+2, 2, ni)
   variables_itf_k = numpy.empty((nb_equations, nb_pts_hori, nb_elements_vert + 2, 2, nb_pts_hori), dtype=type_vec)
   flux_x3_itf_k   = numpy.empty((nb_equations, nb_pts_hori, nb_elements_vert + 2, 2, nb_pts_hori), dtype=type_vec)

   # Special arrays for calculation of (ρw) flux
   wflux_adv_x1_itf_i  = numpy.zeros_like(flux_x1_itf_i[0])
   wflux_pres_x1_itf_i = numpy.zeros_like(flux_x1_itf_i[0])
   wflux_adv_x2_itf_j  = numpy.zeros_like(flux_x2_itf_j[0])
   wflux_pres_x2_itf_j = numpy.zeros_like(flux_x2_itf_j[0])
   wflux_adv_x3_itf_k  = numpy.zeros_like(flux_x3_itf_k[0])
   wflux_pres_x3_itf_k = numpy.zeros_like(flux_x3_itf_k[0])

   # Flag for advection-only processing, with DCMIP test cases 11 and 12
   advection_only = case_number < 13

   variables_itf_i[:, :, 1:-1, :, :] = mtrx.extrapolate_i(Q, geom).transpose((0, 1, 3, 4, 2))
   variables_itf_j[:, :, 1:-1, :, :] = mtrx.extrapolate_j(Q, geom)

   # Scaled variables for separate reconstruction
   logrho      = numpy.log(Q[idx_rho])
   logrhotheta = numpy.log(Q[idx_rho_theta])

   variables_itf_i[idx_rho, :, 1:-1, :, :] = numpy.exp(mtrx.extrapolate_i(logrho, geom)).transpose((0, 2, 3, 1))
   variables_itf_j[idx_rho, :, 1:-1, :, :] = numpy.exp(mtrx.extrapolate_j(logrho, geom))

   variables_itf_i[idx_rho_theta, :, 1:-1, :, :] = numpy.exp(mtrx.extrapolate_i(logrhotheta, geom)).transpose((0, 2, 3, 1))
   variables_itf_j[idx_rho_theta, :, 1:-1, :, :] = numpy.exp(mtrx.extrapolate_j(logrhotheta, geom))

   # Initiate transfers
   all_request = ptopo.xchange_Euler_interfaces(geom, variables_itf_i, variables_itf_j, blocking=False)

   # Unpack dynamical variables, each to arrays of size [nk,nj,ni]
   rho = Q[idx_rho]
   u1  = Q[idx_rho_u1] / rho
   u2  = Q[idx_rho_u2] / rho
   w   = Q[idx_rho_w]  / rho

   # Compute the advective fluxes ...
   flux_x1 = metric.sqrtG * u1 * Q
   flux_x2 = metric.sqrtG * u2 * Q
   flux_x3 = metric.sqrtG * w  * Q

   wflux_adv_x1 = metric.sqrtG * u1 * Q[idx_rho_w]
   wflux_adv_x2 = metric.sqrtG * u2 * Q[idx_rho_w]
   wflux_adv_x3 = metric.sqrtG * w  * Q[idx_rho_w]

   # ... and add the pressure component
   # Performance note: exp(log) is measurably faster than ** (pow)
   pressure = p0 * numpy.exp((cpd/cvd) * numpy.log((Rd/p0)*Q[idx_rho_theta]))

   flux_x1[idx_rho_u1] += metric.sqrtG * metric.H_contra_11 * pressure
   flux_x1[idx_rho_u2] += metric.sqrtG * metric.H_contra_12 * pressure
   flux_x1[idx_rho_w]  += metric.sqrtG * metric.H_contra_13 * pressure

   wflux_pres_x1 = (metric.sqrtG * metric.H_contra_13).astype(type_vec)

   flux_x2[idx_rho_u1] += metric.sqrtG * metric.H_contra_21 * pressure
   flux_x2[idx_rho_u2] += metric.sqrtG * metric.H_contra_22 * pressure
   flux_x2[idx_rho_w]  += metric.sqrtG * metric.H_contra_23 * pressure

   wflux_pres_x2 = (metric.sqrtG * metric.H_contra_23).astype(type_vec)

   flux_x3[idx_rho_u1] += metric.sqrtG * metric.H_contra_31 * pressure
   flux_x3[idx_rho_u2] += metric.sqrtG * metric.H_contra_32 * pressure
   flux_x3[idx_rho_w]  += metric.sqrtG * metric.H_contra_33 * pressure

   wflux_pres_x3 = (metric.sqrtG * metric.H_contra_33).astype(type_vec)

   variables_itf_k[:, :, 1:-1, :, :] = mtrx.extrapolate_k(Q, geom).transpose((0, 3, 1, 2, 4))

   variables_itf_k[idx_rho, :, 1:-1, :, :] = numpy.exp(mtrx.extrapolate_k(logrho,geom).transpose((2, 0, 1, 3)))
   variables_itf_k[idx_rho_theta, :, 1:-1, :, :] = numpy.exp(mtrx.extrapolate_k(logrhotheta,geom).transpose((2, 0, 1, 3)))

   # For consistency at the surface and top boundaries, treat the extrapolation as continuous.  That is,
   # the "top" of the ground is equal to the "bottom" of the atmosphere, and the "bottom" of the model top
   # is equal to the "top" of the atmosphere.
   variables_itf_k[:, :,  0, 1, :] = variables_itf_k[:, :,  1, 0, :]
   variables_itf_k[:, :,  0, 0, :] = variables_itf_k[:, :,  0, 1, :]
   variables_itf_k[:, :, -1, 0, :] = variables_itf_k[:, :, -2, 1, :]
   variables_itf_k[:, :, -1, 1, :] = variables_itf_k[:, :, -1, 0, :]

   # Evaluate pressure at the vertical element interfaces based on ρθ.
   pressure_itf_k = p0 * numpy.exp((cpd / cvd) * numpy.log(variables_itf_k[idx_rho_theta] * (Rd / p0)))

   # Take w ← (wρ)/ ρ at the vertical interfaces
   w_itf_k = variables_itf_k[idx_rho_w] / variables_itf_k[idx_rho]

   # Surface and top boundary treatement, imposing no flow (w=0) through top and bottom
   # csubich -- apply odd symmetry to w at boundary so there is no advective _flux_ through boundary
   w_itf_k[:,  0, 0, :] = 0. # Bottom of bottom element (unused)
   w_itf_k[:,  0, 1, :] = -w_itf_k[:, 1, 0, :] # Top of bottom element (negative symmetry)
   w_itf_k[:, -1, 1, :] = 0. # Top of top element (unused)
   w_itf_k[:, -1, 0, :] = -w_itf_k[:, -2, 1, :] # Bottom of top boundary element (negative symmetry)

   rusanov_3d_vert(variables_itf_k, pressure_itf_k, w_itf_k, metric, nb_interfaces_vert, advection_only,
                   flux_x3_itf_k, wflux_adv_x3_itf_k, wflux_pres_x3_itf_k)

   # Finish transfers
   all_request.wait()

   # Define u, v at the interface by dividing momentum and density
   u1_itf_i = variables_itf_i[idx_rho_u1] / variables_itf_i[idx_rho]
   u2_itf_j = variables_itf_j[idx_rho_u2] / variables_itf_j[idx_rho]

   # Evaluate pressure at the lateral interfaces
   pressure_itf_i = p0 * numpy.exp((cpd/cvd) * numpy.log(variables_itf_i[idx_rho_theta] * (Rd / p0)))
   pressure_itf_j = p0 * numpy.exp((cpd/cvd) * numpy.log(variables_itf_j[idx_rho_theta] * (Rd / p0)))

   # Riemann solver
   rusanov_3d_hori_i(u1_itf_i, variables_itf_i, pressure_itf_i, metric, nb_interfaces_hori, advection_only,
                     flux_x1_itf_i, wflux_adv_x1_itf_i, wflux_pres_x1_itf_i)
   rusanov_3d_hori_j(u2_itf_j, variables_itf_j, pressure_itf_j, metric, nb_interfaces_hori, advection_only,
                     flux_x2_itf_j, wflux_adv_x2_itf_j, wflux_pres_x2_itf_j)

   # Perform flux derivatives

   flux_x1_bdy = flux_x1_itf_i.transpose((0, 1, 3, 2, 4))[:, :, :, 1:-1, :].copy()
   df1_dx1 = mtrx.comma_i(flux_x1, flux_x1_bdy, geom)
   flux_x2_bdy = flux_x2_itf_j[:, :, 1:-1, :, :].copy()
   df2_dx2 = mtrx.comma_j(flux_x2, flux_x2_bdy, geom)
   flux_x3_bdy = flux_x3_itf_k[:, :, 1:-1, :, :].transpose(0, 2, 3, 1, 4).copy()
   df3_dx3 = mtrx.comma_k(flux_x3, flux_x3_bdy, geom)

   logp_int = numpy.log(pressure)

   pressure_bdy_i = pressure_itf_i[:, 1:-1, :, :].transpose((0, 3, 1, 2)).copy()
   pressure_bdy_j = pressure_itf_j[:, 1:-1, :, :].copy()
   pressure_bdy_k = pressure_itf_k[:, 1:-1, :, :].transpose((1, 2, 0, 3)).copy()

   logp_bdy_i = numpy.log(pressure_bdy_i)
   logp_bdy_j = numpy.log(pressure_bdy_j)
   logp_bdy_k = numpy.log(pressure_bdy_k)

   wflux_adv_x1_bdy_i  =  wflux_adv_x1_itf_i.transpose((0, 2, 1, 3))[:, :, 1:-1, :].copy()
   wflux_pres_x1_bdy_i = wflux_pres_x1_itf_i.transpose((0, 2, 1, 3))[:, :, 1:-1, :].copy()

   wflux_adv_x2_bdy_j  =  wflux_adv_x2_itf_j[:, 1:-1, :, :].copy()
   wflux_pres_x2_bdy_j = wflux_pres_x2_itf_j[:, 1:-1, :, :].copy()

   wflux_adv_x3_bdy_k  =  wflux_adv_x3_itf_k[:, 1:-1, :, :].transpose((1, 2, 0, 3)).copy()
   wflux_pres_x3_bdy_k = wflux_pres_x3_itf_k[:, 1:-1, :, :].transpose((1, 2, 0, 3)).copy()

   w_df1_dx1_adv   = mtrx.comma_i(wflux_adv_x1,  wflux_adv_x1_bdy_i,  geom)
   w_df1_dx1_presa = mtrx.comma_i(wflux_pres_x1, wflux_pres_x1_bdy_i, geom) * pressure
   w_df1_dx1_presb = mtrx.comma_i(logp_int,      logp_bdy_i,          geom) * pressure * wflux_pres_x1
   w_df1_dx1       = w_df1_dx1_adv + w_df1_dx1_presa + w_df1_dx1_presb

   w_df2_dx2_adv   = mtrx.comma_j(wflux_adv_x2,  wflux_adv_x2_bdy_j,  geom)
   w_df2_dx2_presa = mtrx.comma_j(wflux_pres_x2, wflux_pres_x2_bdy_j, geom) * pressure
   w_df2_dx2_presb = mtrx.comma_j(logp_int,      logp_bdy_j,          geom) * pressure * wflux_pres_x2
   w_df2_dx2       = w_df2_dx2_adv + w_df2_dx2_presa + w_df2_dx2_presb

   w_df3_dx3_adv   = mtrx.comma_k(wflux_adv_x3,  wflux_adv_x3_bdy_k,  geom) 
   w_df3_dx3_presa = mtrx.comma_k(wflux_pres_x3, wflux_pres_x3_bdy_k, geom) * pressure
   w_df3_dx3_presb = mtrx.comma_k(logp_int,      logp_bdy_k,          geom) * pressure * wflux_pres_x3
   w_df3_dx3       = w_df3_dx3_adv + w_df3_dx3_presa + w_df3_dx3_presb

   # Add coriolis, metric terms and other forcings
   forcing[idx_rho] = 0.0

   forcing[idx_rho_u1] = \
           2.0 * rho * (metric.christoffel_1_01 * u1 + metric.christoffel_1_02 * u2 + metric.christoffel_1_03 * w) \
         +       metric.christoffel_1_11 * (rho * u1 * u1 + metric.H_contra_11 * pressure) \
         + 2.0 * metric.christoffel_1_12 * (rho * u1 * u2 + metric.H_contra_12 * pressure) \
         + 2.0 * metric.christoffel_1_13 * (rho * u1 * w  + metric.H_contra_13 * pressure) \
         +       metric.christoffel_1_22 * (rho * u2 * u2 + metric.H_contra_22 * pressure) \
         + 2.0 * metric.christoffel_1_23 * (rho * u2 * w  + metric.H_contra_23 * pressure) \
         +       metric.christoffel_1_33 * (rho * w  * w  + metric.H_contra_33 * pressure)

   forcing[idx_rho_u2] = \
           2.0 * rho * (metric.christoffel_2_01 * u1 + metric.christoffel_2_02 * u2 + metric.christoffel_2_03 * w) \
         +       metric.christoffel_2_11 * (rho * u1 * u1 + metric.H_contra_11 * pressure) \
         + 2.0 * metric.christoffel_2_12 * (rho * u1 * u2 + metric.H_contra_12 * pressure) \
         + 2.0 * metric.christoffel_2_13 * (rho * u1 * w  + metric.H_contra_13 * pressure) \
         +       metric.christoffel_2_22 * (rho * u2 * u2 + metric.H_contra_22 * pressure) \
         + 2.0 * metric.christoffel_2_23 * (rho * u2 * w  + metric.H_contra_23 * pressure) \
         +       metric.christoffel_2_33 * (rho * w  * w  + metric.H_contra_33 * pressure)

   forcing[idx_rho_w] = \
           2.0 * rho * (metric.christoffel_3_01 * u1 + metric.christoffel_3_02 * u2 + metric.christoffel_3_03 * w) \
         +       metric.christoffel_3_11 * (rho * u1 * u1 + metric.H_contra_11 * pressure) \
         + 2.0 * metric.christoffel_3_12 * (rho * u1 * u2 + metric.H_contra_12 * pressure) \
         + 2.0 * metric.christoffel_3_13 * (rho * u1 * w  + metric.H_contra_13 * pressure) \
         +       metric.christoffel_3_22 * (rho * u2 * u2 + metric.H_contra_22 * pressure) \
         + 2.0 * metric.christoffel_3_23 * (rho * u2 * w  + metric.H_contra_23 * pressure) \
         +       metric.christoffel_3_33 * (rho * w  * w  + metric.H_contra_33 * pressure) \
         + metric.inv_dzdeta * gravity * metric.inv_sqrtG * mtrx.filter_k(metric.sqrtG * rho, geom)

   forcing[idx_rho_theta] = 0.0

   # DCMIP cases 2-1 and 2-2 involve rayleigh damping
   # dcmip_schar_damping modifies the 'forcing' variable to apply the requried Rayleigh damping
   if case_number == 21:
      dcmip_schar_damping(forcing, rho, u1, u2, w, metric, geom, shear=False)
   elif case_number == 22:
      dcmip_schar_damping(forcing, rho, u1, u2, w, metric, geom, shear=True)

   # Assemble the right-hand side
   rhs            = -metric.inv_sqrtG * (  df1_dx1 +   df2_dx2 +   df3_dx3) - forcing
   rhs[idx_rho_w] = -metric.inv_sqrtG * (w_df1_dx1 + w_df2_dx2 + w_df3_dx3) - forcing[idx_rho_w]

   # For pure advection problems, we do not update the dynamical variables
   if advection_only:
      rhs[idx_rho]       = 0.0
      rhs[idx_rho_u1]    = 0.0
      rhs[idx_rho_u2]    = 0.0
      rhs[idx_rho_w]     = 0.0
      rhs[idx_rho_theta] = 0.0

   return rhs
