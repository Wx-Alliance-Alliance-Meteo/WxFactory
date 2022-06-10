import numpy

from definitions import idx_rho_u1, idx_rho_u2, idx_rho_w, idx_rho, idx_rho_theta, gravity, p0, Rd, cpd, cvd, heat_capacity_ratio

def rhs_euler (Q: numpy.ndarray, geom, mtrx, metric, topo, ptopo, nbsolpts: int, nb_elements_hori: int, nb_elements_vert: int, case_number: int):

   type_vec = Q.dtype
   nb_equations = Q.shape[0]
   nb_interfaces_hori = nb_elements_hori + 1
   nb_interfaces_vert = nb_elements_vert + 1
   nb_pts_hori = nb_elements_hori * nbsolpts
   nb_vertical_levels = nb_elements_vert * nbsolpts

   df1_dx1, df2_dx2, df3_dx3, rhs = [numpy.empty_like(Q, dtype=type_vec) for _ in range(4)]

   forcing = numpy.zeros_like(Q, dtype=type_vec)

   variables_itf_i = numpy.ones((nb_equations, nb_vertical_levels, nb_elements_hori + 2, 2, nb_pts_hori), dtype=type_vec) # Initialized to one in the halo to avoid division by zero later
   flux_x1_itf_i   = numpy.empty((nb_equations, nb_vertical_levels, nb_elements_hori + 2, nb_pts_hori, 2), dtype=type_vec)

   variables_itf_j = numpy.ones((nb_equations, nb_vertical_levels, nb_elements_hori + 2, 2, nb_pts_hori), dtype=type_vec) # Initialized to one in the halo to avoid division by zero later
   flux_x2_itf_j   = numpy.empty((nb_equations, nb_vertical_levels, nb_elements_hori + 2, 2, nb_pts_hori), dtype=type_vec)

   variables_itf_k = numpy.empty((nb_equations, nb_pts_hori, nb_elements_vert + 2, 2, nb_pts_hori), dtype=type_vec)
   flux_x3_itf_k   = numpy.empty((nb_equations, nb_pts_hori, nb_elements_vert + 2, 2, nb_pts_hori), dtype=type_vec)

   advection_only = case_number < 13

   # Offset due to the halo
   offset = 1

   # Interpolate to the element interface
   for elem in range(nb_elements_hori):
      epais = elem * nbsolpts + numpy.arange(nbsolpts)
      pos   = elem + offset

      # --- Direction x1
      variables_itf_i[:, :, pos, 0, :] = Q[:, :, :, epais] @ mtrx.extrap_west
      variables_itf_i[:, :, pos, 1, :] = Q[:, :, :, epais] @ mtrx.extrap_east

      # --- Direction x2
      variables_itf_j[:, :, pos, 0, :] = mtrx.extrap_south @ Q[:, :, epais, :]
      variables_itf_j[:, :, pos, 1, :] = mtrx.extrap_north @ Q[:, :, epais, :]

   # Initiate transfers
   all_request = ptopo.xchange_Euler_interfaces(geom, variables_itf_i, variables_itf_j, blocking=False)

   # Unpack dynamical variables
   rho = Q[idx_rho]
   u1  = Q[idx_rho_u1] / rho
   u2  = Q[idx_rho_u2] / rho
   w   = Q[idx_rho_w]  / rho # TODO : u3

   # Compute the advective fluxes ...
   flux_x1 = metric.sqrtG * u1 * Q
   flux_x2 = metric.sqrtG * u2 * Q
   flux_x3 = metric.sqrtG * w  * Q

   # ... and add the pressure component
   pressure = p0 * (Q[idx_rho_theta] * Rd / p0)**(cpd / cvd)

   flux_x1[idx_rho_u1] += metric.sqrtG * metric.H_contra_11 * pressure
   flux_x1[idx_rho_u2] += metric.sqrtG * metric.H_contra_12 * pressure
   flux_x1[idx_rho_w]  += metric.sqrtG * metric.H_contra_13 * pressure

   flux_x2[idx_rho_u1] += metric.sqrtG * metric.H_contra_21 * pressure
   flux_x2[idx_rho_u2] += metric.sqrtG * metric.H_contra_22 * pressure
   flux_x2[idx_rho_w]  += metric.sqrtG * metric.H_contra_23 * pressure

   flux_x3[idx_rho_u1] += metric.sqrtG * metric.H_contra_31 * pressure
   flux_x3[idx_rho_u2] += metric.sqrtG * metric.H_contra_32 * pressure
   flux_x3[idx_rho_w]  += metric.sqrtG * metric.H_contra_33 * pressure

   # Interior contribution to the derivatives, corrections for the boundaries will be added later
   for elem in range(nb_elements_hori):
      epais = elem * nbsolpts + numpy.arange(nbsolpts)

      # --- Direction x1
      df1_dx1[:, :, :, epais] = flux_x1[:, :, :, epais] @ mtrx.diff_solpt_tr

      # --- Direction x2
      df2_dx2[:, :, epais, :] = mtrx.diff_solpt @ flux_x2[:, :, epais, :]

   # --- Direction x3

   # Important notice : all the vertical stuff should be done before the synchronization of the horizontal communications.

   for slab in range(nb_pts_hori):
      for elem in range(nb_elements_vert):
         epais = elem * nbsolpts + numpy.arange(nbsolpts)
         pos = elem + offset

         variables_itf_k[:, slab, pos, 0, :] = mtrx.extrap_down @ Q[:, epais, slab, :]
         variables_itf_k[:, slab, pos, 1, :] = mtrx.extrap_up   @ Q[:, epais, slab, :]

   # For consistency at the surface and top boundaries
   variables_itf_k[:, :, 0, 1, :] = variables_itf_k[:, :, 1, 0, :]
   variables_itf_k[:, :, 0, 0, :] = variables_itf_k[:, :, 0, 1, :]
   variables_itf_k[:, :, -1, 0, :] = variables_itf_k[:, :, -2, 1, :]
   variables_itf_k[:, :, -1, 1, :] = variables_itf_k[:, :, -1, 0, :]

   pressure_itf_k = p0 * (variables_itf_k[idx_rho_theta] * Rd / p0)**(cpd / cvd)

   w_itf_k = variables_itf_k[idx_rho_w] / variables_itf_k[idx_rho]

   # Surface and top boundary treatement
   w_itf_k[:, 0, :, :] = 0.
   w_itf_k[:, 1, 0, :] = 0.
   w_itf_k[:, -1, :, :] = 0.
   w_itf_k[:, -2, 1, :] = 0.

   # Common Rusanov vertical fluxes
   for itf in range(nb_interfaces_vert):

      elem_D = itf
      elem_U = itf + 1

      # Direction x3

      w_D = w_itf_k[:, elem_D, 1, :]
      w_U = w_itf_k[:, elem_U, 0, :]

      if advection_only:
         eig_D = numpy.abs(w_D)
         eig_U = numpy.abs(w_U)
      else:
         eig_D = numpy.abs(w_D) + numpy.sqrt(metric.H_contra_33 * heat_capacity_ratio * pressure_itf_k[:, elem_D, 1, :] / variables_itf_k[idx_rho, :, elem_D, 1, :])
         eig_U = numpy.abs(w_U) + numpy.sqrt(metric.H_contra_33 * heat_capacity_ratio * pressure_itf_k[:, elem_U, 0, :] / variables_itf_k[idx_rho, :, elem_U, 0, :])

      eig = numpy.maximum(eig_D, eig_U)

      # Advective part of the flux ...
      flux_D = metric.sqrtG * w_D * variables_itf_k[:, :, elem_D, 1, :]
      flux_U = metric.sqrtG * w_U * variables_itf_k[:, :, elem_U, 0, :]

      # ... and add the pressure part
      flux_D[idx_rho_u1] += metric.sqrtG * metric.H_contra_31 * pressure_itf_k[:, elem_D, 1, :]
      flux_D[idx_rho_u2] += metric.sqrtG * metric.H_contra_32 * pressure_itf_k[:, elem_D, 1, :]
      flux_D[idx_rho_w] += metric.sqrtG * metric.H_contra_33 * pressure_itf_k[:, elem_D, 1, :]

      flux_U[idx_rho_u1] += metric.sqrtG * metric.H_contra_31 * pressure_itf_k[:, elem_U, 0, :]
      flux_U[idx_rho_u2] += metric.sqrtG * metric.H_contra_32 * pressure_itf_k[:, elem_U, 0, :]
      flux_U[idx_rho_w] += metric.sqrtG * metric.H_contra_33 * pressure_itf_k[:, elem_U, 0, :]

      # Riemann solver
      flux_x3_itf_k[:, :, elem_D, 1, :] = 0.5 * ( flux_D + flux_U - eig * metric.sqrtG * ( variables_itf_k[:, :, elem_U, 0, :] - variables_itf_k[:, :, elem_D, 1, :] ) )
      flux_x3_itf_k[:, :, elem_U, 0, :] = flux_x3_itf_k[:, :, elem_D, 1, :]

   for slab in range(nb_pts_hori):
      for elem in range(nb_elements_vert):
         epais = elem * nbsolpts + numpy.arange(nbsolpts)
         # TODO : inclure la transformation vers l'élément de référence dans la vitesse w.
         df3_dx3[:, epais, slab, :] = ( mtrx.diff_solpt @ flux_x3[:, epais, slab, :] + mtrx.correction @ flux_x3_itf_k[:, slab, elem+offset, :, :] ) * 2.0 / geom.Δx3

   # Finish transfers
   all_request.wait()

   u1_itf_i = variables_itf_i[idx_rho_u1] / variables_itf_i[idx_rho]
   u2_itf_j = variables_itf_j[idx_rho_u2] / variables_itf_j[idx_rho]

   pressure_itf_i = p0 * (variables_itf_i[idx_rho_theta] * Rd / p0)**(cpd / cvd)
   pressure_itf_j = p0 * (variables_itf_j[idx_rho_theta] * Rd / p0)**(cpd / cvd)

   # Riemann solver
   for itf in range(nb_interfaces_hori):

      elem_L = itf
      elem_R = itf + 1

      # Direction x1
      u1_L = u1_itf_i[:, elem_L, 1, :]
      u1_R = u1_itf_i[:, elem_R, 0, :]

      if advection_only:
         eig_L = numpy.abs( u1_L )
         eig_R = numpy.abs( u1_R )
      else:
         eig_L = numpy.abs( u1_L ) + numpy.sqrt(metric.H_contra_11_itf_i[itf, :] * heat_capacity_ratio * pressure_itf_i[:, elem_L, 1, :] / variables_itf_i[idx_rho, :, elem_L, 1, :])
         eig_R = numpy.abs( u1_R ) + numpy.sqrt(metric.H_contra_11_itf_i[itf, :] * heat_capacity_ratio * pressure_itf_i[:, elem_R, 0, :] / variables_itf_i[idx_rho, :, elem_R, 0, :])

      eig = numpy.maximum(eig_L, eig_R)

      # Advective part of the flux ...
      flux_L = metric.sqrtG_itf_i[itf, :] * u1_L * variables_itf_i[:, :, elem_L, 1, :]
      flux_R = metric.sqrtG_itf_i[itf, :] * u1_R * variables_itf_i[:, :, elem_R, 0, :]

      # ... and now add the pressure contribution
      flux_L[idx_rho_u1] += metric.sqrtG_itf_i[itf, :] * metric.H_contra_11_itf_i[itf, :] * pressure_itf_i[:, elem_L, 1, :]
      flux_L[idx_rho_u2] += metric.sqrtG_itf_i[itf, :] * metric.H_contra_12_itf_i[itf, :] * pressure_itf_i[:, elem_L, 1, :]
      flux_L[idx_rho_w]  += metric.sqrtG_itf_i[itf, :] * metric.H_contra_13_itf_i[itf, :] * pressure_itf_i[:, elem_L, 1, :]
                                                                                        
      flux_R[idx_rho_u1] += metric.sqrtG_itf_i[itf, :] * metric.H_contra_11_itf_i[itf, :] * pressure_itf_i[:, elem_R, 0, :]
      flux_R[idx_rho_u2] += metric.sqrtG_itf_i[itf, :] * metric.H_contra_12_itf_i[itf, :] * pressure_itf_i[:, elem_R, 0, :]
      flux_R[idx_rho_w]  += metric.sqrtG_itf_i[itf, :] * metric.H_contra_13_itf_i[itf, :] * pressure_itf_i[:, elem_R, 0, :]

      # --- Common Rusanov fluxes

      flux_x1_itf_i[:, :, elem_L, :, 1] = 0.5 * ( flux_L  + flux_R - eig * metric.sqrtG_itf_i[itf, :] * ( variables_itf_i[:, :, elem_R, 0, :] - variables_itf_i[:, :, elem_L, 1, :] ) )
      flux_x1_itf_i[:, :, elem_R, :, 0] = flux_x1_itf_i[:, :, elem_L, :, 1]

      # Direction x2

      u2_L = u2_itf_j[:, elem_L, 1, :]
      u2_R = u2_itf_j[:, elem_R, 0, :]

      if advection_only:
         eig_L = numpy.abs( u2_L )
         eig_R = numpy.abs( u2_R )
      else:
         eig_L = numpy.abs( u2_L ) + numpy.sqrt(metric.H_contra_22_itf_j[itf, :] * heat_capacity_ratio * pressure_itf_j[:, elem_L, 1, :] / variables_itf_j[idx_rho, :, elem_L, 1, :])
         eig_R = numpy.abs( u2_R ) + numpy.sqrt(metric.H_contra_22_itf_j[itf, :] * heat_capacity_ratio * pressure_itf_j[:, elem_R, 0, :] / variables_itf_j[idx_rho, :, elem_R, 0, :])

      eig = numpy.maximum(eig_L, eig_R)

      # Advective part of the flux
      flux_L = metric.sqrtG_itf_j[itf, :] * u2_L * variables_itf_j[:, :, elem_L, 1, :]
      flux_R = metric.sqrtG_itf_j[itf, :] * u2_R * variables_itf_j[:, :, elem_R, 0, :]

      # ... and now add the pressure contribution
      flux_L[idx_rho_u1] += metric.sqrtG_itf_j[itf, :] * metric.H_contra_21_itf_j[itf, :] * pressure_itf_j[:, elem_L, 1, :]
      flux_L[idx_rho_u2] += metric.sqrtG_itf_j[itf, :] * metric.H_contra_22_itf_j[itf, :] * pressure_itf_j[:, elem_L, 1, :]
      flux_L[idx_rho_w] += metric.sqrtG_itf_j[itf, :] * metric.H_contra_23_itf_j[itf, :] * pressure_itf_j[:, elem_L, 1, :]

      flux_R[idx_rho_u1] += metric.sqrtG_itf_j[itf, :] * metric.H_contra_21_itf_j[itf, :] * pressure_itf_j[:, elem_R, 0, :]
      flux_R[idx_rho_u2] += metric.sqrtG_itf_j[itf, :] * metric.H_contra_22_itf_j[itf, :] * pressure_itf_j[:, elem_R, 0, :]
      flux_R[idx_rho_w] += metric.sqrtG_itf_j[itf, :] * metric.H_contra_23_itf_j[itf, :] * pressure_itf_j[:, elem_R, 0, :]

      # --- Common Rusanov fluxes

      flux_x2_itf_j[:, :, elem_L, 1, :] = 0.5 * ( flux_L + flux_R - eig * metric.sqrtG_itf_j[itf, :] * ( variables_itf_j[:, :, elem_R, 0, :] - variables_itf_j[:, :, elem_L, 1, :] ) )
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
   forcing[idx_rho_u1] = 2.0 * ( metric.christoffel_1_01 * rho * u1 + metric.christoffel_1_02 * rho * u2 + metric.christoffel_1_03 * rho * w) \
         +       metric.christoffel_1_11 * rho * u1 * u1  \
         + 2.0 * metric.christoffel_1_12 * rho * u1 * u2 \
         + 2.0 * metric.christoffel_1_13 * rho * u1 * w \
         +       metric.christoffel_1_22 * rho * u2 * u2 \
         + 2.0 * metric.christoffel_1_23 * rho * u2 * w \
         +       metric.christoffel_1_33 * rho * w * w

   forcing[idx_rho_u2] = 2.0 * (metric.christoffel_2_01 * rho * u1 + metric.christoffel_2_02 * rho * u2 + metric.christoffel_2_03 * rho * w) \
         +       metric.christoffel_2_11 * rho * u1 * u1  \
         + 2.0 * metric.christoffel_2_12 * rho * u1 * u2 \
         + 2.0 * metric.christoffel_2_13 * rho * u1 * w \
         +       metric.christoffel_2_22 * rho * u2 * u2 \
         + 2.0 * metric.christoffel_2_23 * rho * u2 * w \
         +       metric.christoffel_2_33 * rho * w * w

   forcing[idx_rho_w] = 2.0 * (metric.christoffel_3_01 * rho * u1 + metric.christoffel_3_02 * rho * u2 + metric.christoffel_3_03 * rho * w) \
         +       metric.christoffel_3_11 * rho * u1 * u1 \
         + 2.0 * metric.christoffel_3_12 * rho * u1 * u2 \
         + 2.0 * metric.christoffel_3_13 * rho * u1 * w \
         +       metric.christoffel_3_22 * rho * u2 * u2 \
         + 2.0 * metric.christoffel_3_23 * rho * u2 * w \
         +       metric.christoffel_3_33 * rho * w * w \
         + metric.inv_dzdeta * rho * gravity

   forcing[idx_rho_theta] = 0.0

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
