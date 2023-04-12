import numpy
import sys

from common.definitions import idx_rho_u1, idx_rho_u2, idx_rho_w, idx_rho, idx_rho_theta, gravity, p0, Rd, cpd, cvd, heat_capacity_ratio

# For type hints
from common.parallel import DistributedWorld
from geometry        import CubedSphere, DFROperators, Metric3DTopo
from init.dcmip      import dcmip_schar_damping

#@profile
def rhs_euler_convective (Q: numpy.ndarray, geom: CubedSphere, mtrx: DFROperators, metric: Metric3DTopo, ptopo: DistributedWorld,
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

   # Create new arrays for each component of T^μν_:ν, plus one more for the final right hand side
   df1_dx1, df2_dx2, df3_dx3, rhs = [numpy.empty_like(Q, dtype=type_vec) for _ in range(4)]

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

   # Flag for advection-only processing, with DCMIP test cases 11 and 12
   advection_only = case_number < 13

   # Offset due to the halo
   offset = 1

   # Interpolate to the element interface
   for elem in range(nb_elements_hori):
      # This loop performs extrapolation to element boundaries through the mtrix.extrap_* operator (matrix multiplication).
      # Thanks to numpy's broadcasting, each iteration of this loop extrapolates an entire row/column of elements at once,
      # operating on all variables simultaneously

      # Index of the 'live' interior elements inside the Q array, to be extrapolated
      epais = elem * nbsolpts + numpy.arange(nbsolpts)
      # Position in the output interface array for writing.  'pos' 1 corresponds to the west/southmost element, with
      # 'pos' 0 (and nb_elements_hori+1) reserved for exchanges from neighbouring panels
      pos   = elem + offset

      # --- Direction x1
      # The implied matrix multiplication here sees a [numvar, nk] array of matrices, each 
      # of size [nj, nbsolpoints], and the extrapolation is performed via right multiplication.
      # (Note C-ordering of indices; in fortran or matlab the indices would be reversed)
      variables_itf_i[:, :, pos, 0, :] = Q[:, :, :, epais] @ mtrx.extrap_west
      variables_itf_i[:, :, pos, 1, :] = Q[:, :, :, epais] @ mtrx.extrap_east

      # --- Direction x2
      # The matrix multiplication here sees a [numvar, nk] array of matrices, each of size
      # [nbsolpoints, ni], and the extrapolation is performed by left multiplication
      variables_itf_j[:, :, pos, 0, :] = mtrx.extrap_south @ Q[:, :, epais, :]
      variables_itf_j[:, :, pos, 1, :] = mtrx.extrap_north @ Q[:, :, epais, :]

   # Transfer boundary values to neighbouring proessors/panels, including conversion of vector quantities
   # to the recipient's local coordinate system

   # Initiate transfers
   all_request = ptopo.xchange_Euler_interfaces(geom, variables_itf_i, variables_itf_j, blocking=False)

   # Unpack dynamical variables, each to arrays of size [nk,nj,ni]
   rho = Q[idx_rho]
   u1  = Q[idx_rho_u1] / rho
   u2  = Q[idx_rho_u2] / rho
   w   = Q[idx_rho_w]  / rho # TODO : u3

   # Compute the fluxes (equation 3 of Charron & Gaudreault 2021, LHS)

   # Compute the advective fluxes ...
   flux_x1 = metric.sqrtG * u1 * Q
   flux_x2 = metric.sqrtG * u2 * Q
   flux_x3 = metric.sqrtG * w  * Q

   # ... and add the pressure component
   # Performance note: exp(log) is measuably faster than ** (pow)
   pressure = p0 * numpy.exp((cpd/cvd) * numpy.log((Rd/p0)*Q[idx_rho_theta]))

   flux_x1[idx_rho_u1] += metric.sqrtG * metric.H_contra_11 * pressure
   flux_x1[idx_rho_u2] += metric.sqrtG * metric.H_contra_12 * pressure
   flux_x1[idx_rho_w]  += metric.sqrtG * metric.H_contra_13 * pressure

   flux_x2[idx_rho_u1] += metric.sqrtG * metric.H_contra_21 * pressure
   flux_x2[idx_rho_u2] += metric.sqrtG * metric.H_contra_22 * pressure
   flux_x2[idx_rho_w]  += metric.sqrtG * metric.H_contra_23 * pressure

   flux_x3[idx_rho_u1] += metric.sqrtG * metric.H_contra_31 * pressure
   flux_x3[idx_rho_u2] += metric.sqrtG * metric.H_contra_32 * pressure
   flux_x3[idx_rho_w]  += metric.sqrtG * metric.H_contra_33 * pressure

   # if (ptopo.rank == 0): print('√g: %e, H^33: %e' % (metric.sqrtG[0,0],metric.H_contra_33[0,0]))

   # Interior contribution to the derivatives, corrections for the boundaries will be added later
   # The "interior contribution" here is evaluated as if the fluxes at the element boundaries are
   # zero.
   for elem in range(nb_elements_hori):
      epais = elem * nbsolpts + numpy.arange(nbsolpts)

      # --- Direction x1
      df1_dx1[:, :, :, epais] = flux_x1[:, :, :, epais] @ mtrx.diff_solpt_tr

      # --- Direction x2
      df2_dx2[:, :, epais, :] = mtrx.diff_solpt @ flux_x2[:, :, epais, :]

   # --- Direction x3

    # Important notice : all the vertical stuff should be done before the synchronization of the horizontal communications.
    # Since there is no communication step in the vertical, we can compute the boundary correction first

   # Extrapolate to top/bottom for each element
   for slab in range(nb_pts_hori):
      for elem in range(nb_elements_vert):
         epais = elem * nbsolpts + numpy.arange(nbsolpts)
         pos = elem + offset

         # Extrapolate by left multiplication.  Note that this assignment also permutes the indices, going from
         # (var,nk,nj,ni) to (var,nj,nk,top/bot,ni)
         variables_itf_k[:, slab, pos, 0, :] = mtrx.extrap_down @ Q[:, epais, slab, :]
         variables_itf_k[:, slab, pos, 1, :] = mtrx.extrap_up   @ Q[:, epais, slab, :]

   # For consistency at the surface and top boundaries, treat the extrapolation as continuous.  That is,
   # the "top" of the ground is equal to the "bottom" of the atmosphere, and the "bottom" of the model top
   # is equal to the "top" of the atmosphere.
   variables_itf_k[:, :, 0, 1, :] = variables_itf_k[:, :, 1, 0, :]
   variables_itf_k[:, :, 0, 0, :] = variables_itf_k[:, :, 0, 1, :] # Unused?
   variables_itf_k[:, :, -1, 0, :] = variables_itf_k[:, :, -2, 1, :]
   variables_itf_k[:, :, -1, 1, :] = variables_itf_k[:, :, -1, 0, :] # Unused?

   # Evaluate pressure at the vertical element interfaces based on ρθ.
   pressure_itf_k = p0 * numpy.exp((cpd/cvd)*numpy.log(variables_itf_k[idx_rho_theta] * (Rd / p0)))

   # Take w ← (wρ)/ ρ at the vertical interfaces
   w_itf_k = variables_itf_k[idx_rho_w] / variables_itf_k[idx_rho]

   # Surface and top boundary treatement, imposing no flow (w=0) through top and bottom
   w_itf_k[:, 0, :, :] = 0.
   w_itf_k[:, 1, 0, :] = 0.
   w_itf_k[:, -1, :, :] = 0.
   w_itf_k[:, -2, 1, :] = 0.

   # Common Rusanov vertical fluxes
   # flux_D_itf_k = numpy.empty((nb_equations, nb_interfaces_vert, geom.nj, geom.ni))
   # flux_U_itf_k = numpy.empty((nb_equations, nb_interfaces_vert, geom.nj, geom.ni))
   # eig_D_itf_k = numpy.empty((nb_interfaces_vert, geom.nj, geom.ni))
   # eig_U_itf_k = numpy.empty((nb_interfaces_vert, geom.nj, geom.ni))
   for itf in range(nb_interfaces_vert):

      elem_D = itf
      elem_U = itf + 1

      # Direction x3

      w_D = w_itf_k[:, elem_D, 1, :] # w at the top of the lower element
      w_U = w_itf_k[:, elem_U, 0, :] # w at the bottom of the upper element

      # Advective part of the flux ...
      flux_D = metric.sqrtG_itf_k[itf,:,:] * w_D * variables_itf_k[:, :, elem_D, 1, :]
      flux_U = metric.sqrtG_itf_k[itf,:,:] * w_U * variables_itf_k[:, :, elem_U, 0, :]

      # eig_D_itf_k[itf,:,:] = eig_D
      # eig_U_itf_k[itf,:,:] = eig_U
      # flux_D_itf_k[:,itf,:,:] = flux_D
      # flux_U_itf_k[:,itf,:,:] = flux_U

      # ... and add the pressure part
      flux_D[idx_rho_u1] += metric.sqrtG_itf_k[itf,:,:] * metric.H_contra_31_itf_k[itf,:,:] * pressure_itf_k[:, elem_D, 1, :]
      flux_D[idx_rho_u2] += metric.sqrtG_itf_k[itf,:,:] * metric.H_contra_32_itf_k[itf,:,:] * pressure_itf_k[:, elem_D, 1, :]
      flux_D[idx_rho_w]  += metric.sqrtG_itf_k[itf,:,:] * metric.H_contra_33_itf_k[itf,:,:] * pressure_itf_k[:, elem_D, 1, :]

      flux_U[idx_rho_u1] += metric.sqrtG_itf_k[itf,:,:] * metric.H_contra_31_itf_k[itf,:,:] * pressure_itf_k[:, elem_U, 0, :]
      flux_U[idx_rho_u2] += metric.sqrtG_itf_k[itf,:,:] * metric.H_contra_32_itf_k[itf,:,:] * pressure_itf_k[:, elem_U, 0, :]
      flux_U[idx_rho_w]  += metric.sqrtG_itf_k[itf,:,:] * metric.H_contra_33_itf_k[itf,:,:] * pressure_itf_k[:, elem_U, 0, :]

      # Riemann solver
      flux_x3_itf_k[:, :, elem_D, 1, :] = 0.5 * ( flux_D + flux_U )
      flux_x3_itf_k[:, :, elem_U, 0, :] = flux_x3_itf_k[:, :, elem_D, 1, :]

   for slab in range(nb_pts_hori):
      for elem in range(nb_elements_vert):
         epais = elem * nbsolpts + numpy.arange(nbsolpts)
         # TODO : inclure la transformation vers l'élément de référence dans la vitesse w.
         df3_dx3[:, epais, slab, :] = ( mtrx.diff_solpt @ flux_x3[:, epais, slab, :] + mtrx.correction @ flux_x3_itf_k[:, slab, elem+offset, :, :] ) #* 2.0 / geom.Δx3

   # Finish transfers
   all_request.wait()

   # sys.exit(1)

   # Define u, v at the interface by dividing momentum and density
   u1_itf_i = variables_itf_i[idx_rho_u1] / variables_itf_i[idx_rho]
   u2_itf_j = variables_itf_j[idx_rho_u2] / variables_itf_j[idx_rho]

   # Evaluate pressure at the lateral interfaces
   pressure_itf_i = p0 * numpy.exp((cpd/cvd) * numpy.log(variables_itf_i[idx_rho_theta] * (Rd / p0)))
   pressure_itf_j = p0 * numpy.exp((cpd/cvd) * numpy.log(variables_itf_j[idx_rho_theta] * (Rd / p0)))

   # Riemann solver
   for itf in range(nb_interfaces_hori):

      elem_L = itf
      elem_R = itf + 1

      # Direction x1
      u1_L = u1_itf_i[:, elem_L, 1, :] # u at the right interface of the left element
      u1_R = u1_itf_i[:, elem_R, 0, :] # u at the left interface of the right element

      # Advective part of the flux ...
      flux_L = metric.sqrtG_itf_i[:, :, itf] * u1_L * variables_itf_i[:, :, elem_L, 1, :]
      flux_R = metric.sqrtG_itf_i[:, :, itf] * u1_R * variables_itf_i[:, :, elem_R, 0, :]

      # ... and now add the pressure contribution
      flux_L[idx_rho_u1] += metric.sqrtG_itf_i[:, :, itf] * metric.H_contra_11_itf_i[:, :, itf] * pressure_itf_i[:, elem_L, 1, :]
      flux_L[idx_rho_u2] += metric.sqrtG_itf_i[:, :, itf] * metric.H_contra_12_itf_i[:, :, itf] * pressure_itf_i[:, elem_L, 1, :]
      flux_L[idx_rho_w]  += metric.sqrtG_itf_i[:, :, itf] * metric.H_contra_13_itf_i[:, :, itf] * pressure_itf_i[:, elem_L, 1, :]
                                                                                        
      flux_R[idx_rho_u1] += metric.sqrtG_itf_i[:, :, itf] * metric.H_contra_11_itf_i[:, :, itf] * pressure_itf_i[:, elem_R, 0, :]
      flux_R[idx_rho_u2] += metric.sqrtG_itf_i[:, :, itf] * metric.H_contra_12_itf_i[:, :, itf] * pressure_itf_i[:, elem_R, 0, :]
      flux_R[idx_rho_w]  += metric.sqrtG_itf_i[:, :, itf] * metric.H_contra_13_itf_i[:, :, itf] * pressure_itf_i[:, elem_R, 0, :]

      # --- Common Rusanov fluxes

      flux_x1_itf_i[:, :, elem_L, :, 1] = 0.5 * ( flux_L  + flux_R )
      flux_x1_itf_i[:, :, elem_R, :, 0] = flux_x1_itf_i[:, :, elem_L, :, 1]

      # Direction x2

      u2_L = u2_itf_j[:, elem_L, 1, :] # v at the north interface of the south element
      u2_R = u2_itf_j[:, elem_R, 0, :] # v at the south interface of the north element

      # Advective part of the flux
      flux_L = metric.sqrtG_itf_j[:, itf, :] * u2_L * variables_itf_j[:, :, elem_L, 1, :]
      flux_R = metric.sqrtG_itf_j[:, itf, :] * u2_R * variables_itf_j[:, :, elem_R, 0, :]

      # ... and now add the pressure contribution
      flux_L[idx_rho_u1] += metric.sqrtG_itf_j[:, itf, :]  * metric.H_contra_21_itf_j[:, itf, :]  * pressure_itf_j[:, elem_L, 1, :]
      flux_L[idx_rho_u2] += metric.sqrtG_itf_j[:, itf, :]  * metric.H_contra_22_itf_j[:, itf, :]  * pressure_itf_j[:, elem_L, 1, :]
      flux_L[idx_rho_w] += metric.sqrtG_itf_j[:, itf, :]  * metric.H_contra_23_itf_j[:, itf, :]  * pressure_itf_j[:, elem_L, 1, :]

      flux_R[idx_rho_u1] += metric.sqrtG_itf_j[:, itf, :]  * metric.H_contra_21_itf_j[:, itf, :]  * pressure_itf_j[:, elem_R, 0, :]
      flux_R[idx_rho_u2] += metric.sqrtG_itf_j[:, itf, :]  * metric.H_contra_22_itf_j[:, itf, :]  * pressure_itf_j[:, elem_R, 0, :]
      flux_R[idx_rho_w] += metric.sqrtG_itf_j[:, itf, :]  * metric.H_contra_23_itf_j[:, itf, :]  * pressure_itf_j[:, elem_R, 0, :]

      # --- Common Rusanov fluxes

      flux_x2_itf_j[:, :, elem_L, 1, :] = 0.5 * ( flux_L + flux_R )
      flux_x2_itf_j[:, :, elem_R, 0, :] = flux_x2_itf_j[:, :, elem_L, 1, :]

   # Add corrections to the derivatives
   for elem in range(nb_elements_hori):
      epais = elem * nbsolpts + numpy.arange(nbsolpts)

      # --- Direction x1

      df1_dx1[:, :, :, epais] += flux_x1_itf_i[:, :, elem+offset, :, :] @ mtrx.correction_tr

      # --- Direction x2

      df2_dx2[:, :, epais, :] += mtrx.correction @ flux_x2_itf_j[:, :, elem+offset, :, :]

   # Add coriolis, metric terms and other forcings
   forcing[idx_rho,:,:,:] = 0.0



   # TODO: could be simplified
   #pressure = 0
   forcing[idx_rho_u1] = 2.0 * ( metric.christoffel_1_01 * rho * u1 + metric.christoffel_1_02 * rho * u2 + metric.christoffel_1_03 * rho * w) \
         +       metric.christoffel_1_11 * (rho * u1 * u1 + metric.H_contra_11*pressure) \
         + 2.0 * metric.christoffel_1_12 * (rho * u1 * u2 + metric.H_contra_12*pressure) \
         + 2.0 * metric.christoffel_1_13 * (rho * u1 * w  + metric.H_contra_13*pressure) \
         +       metric.christoffel_1_22 * (rho * u2 * u2 + metric.H_contra_22*pressure) \
         + 2.0 * metric.christoffel_1_23 * (rho * u2 * w  + metric.H_contra_23*pressure) \
         +       metric.christoffel_1_33 * (rho * w * w   + metric.H_contra_33*pressure) 

   forcing[idx_rho_u2] = 2.0 * (metric.christoffel_2_01 * rho * u1 + metric.christoffel_2_02 * rho * u2 + metric.christoffel_2_03 * rho * w) \
         +       metric.christoffel_2_11 * (rho * u1 * u1 + metric.H_contra_11*pressure) \
         + 2.0 * metric.christoffel_2_12 * (rho * u1 * u2 + metric.H_contra_12*pressure) \
         + 2.0 * metric.christoffel_2_13 * (rho * u1 * w  + metric.H_contra_13*pressure) \
         +       metric.christoffel_2_22 * (rho * u2 * u2 + metric.H_contra_22*pressure) \
         + 2.0 * metric.christoffel_2_23 * (rho * u2 * w  + metric.H_contra_23*pressure) \
         +       metric.christoffel_2_33 * (rho * w * w   + metric.H_contra_33*pressure) 

   forcing[idx_rho_w] = 2.0 * (metric.christoffel_3_01 * rho * u1 + metric.christoffel_3_02 * rho * u2 + metric.christoffel_3_03 * rho * w) \
         +       metric.christoffel_3_11 * (rho * u1 * u1 + metric.H_contra_11*pressure) \
         + 2.0 * metric.christoffel_3_12 * (rho * u1 * u2 + metric.H_contra_12*pressure) \
         + 2.0 * metric.christoffel_3_13 * (rho * u1 * w  + metric.H_contra_13*pressure) \
         +       metric.christoffel_3_22 * (rho * u2 * u2 + metric.H_contra_22*pressure) \
         + 2.0 * metric.christoffel_3_23 * (rho * u2 * w  + metric.H_contra_23*pressure) \
         +       metric.christoffel_3_33 * (rho * w * w   + metric.H_contra_33*pressure) \
         + metric.inv_dzdeta * rho * gravity

   forcing[idx_rho_theta] = 0.0


   # DCMIP cases 2-1 and 2-2 involve rayleigh damping
   if (case_number == 21):
      # dcmip_schar_damping modifies the 'forcing' variable to apply the requried Rayleigh damping
      dcmip_schar_damping(forcing, rho, u1, u2, w, metric, geom, shear=False)
   elif (case_number == 22):
      dcmip_schar_damping(forcing, rho, u1, u2, w, metric, geom, shear=True)


   # Assemble the right-hand sides
   rhs = - metric.inv_sqrtG * ( df1_dx1 + df2_dx2 + df3_dx3 ) - forcing

   # For pure advection problems, we do not update the dynamical variables
   if advection_only:
      rhs[idx_rho]       = 0.0
      rhs[idx_rho_u1]    = 0.0
      rhs[idx_rho_u2]    = 0.0
      rhs[idx_rho_w]     = 0.0
      rhs[idx_rho_theta] = 0.0
   return rhs
