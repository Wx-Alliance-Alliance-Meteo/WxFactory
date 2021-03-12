import numpy

from definitions import idx_rho_u1, idx_rho_u2, idx_rho_u3, idx_rho, idx_rho_theta, gravity, p0, Rd, cpd, cvd
from dgfilter import apply_filter

def rhs_euler(Q, geom, mtrx, metric, topo, ptopo, nb_sol_pts: int, nb_elements_horiz: int, nb_elements_vert: int, case_number: int, filter_rhs: bool = False):

   datatype = Q.dtype
   nb_equations = 5
   nb_interfaces_horiz = nb_elements_horiz + 1
   nb_total_sol_pt_horiz = nb_elements_horiz * nb_sol_pts
   nb_total_sol_pt_vert = nb_elements_vert * nb_sol_pts

   # Result
   rhs = numpy.zeros_like(Q)

   # Work arrays
   flux_x1 = numpy.zeros_like(Q)
   flux_x2 = numpy.zeros_like(Q)
   flux_x3 = numpy.zeros_like(Q)

   df1_dx1 = numpy.zeros_like(Q)
   df2_dx2 = numpy.zeros_like(Q)
   df3_dx3 = numpy.zeros_like(Q)

   # forcing = numpy.zeros_like(Q, dtype=type_vec)

   itf_variable_i  = numpy.zeros((nb_equations, nb_total_sol_pt_vert, nb_elements_horiz + 2, 2, nb_total_sol_pt_horiz), dtype=datatype)
   itf_flux_i      = numpy.zeros_like(itf_variable_i)
   itf_diffusion_i = numpy.zeros_like(itf_variable_i)

   itf_variable_j  = numpy.zeros((nb_equations, nb_total_sol_pt_vert, nb_elements_horiz + 2, 2, nb_total_sol_pt_horiz), dtype=datatype)
   itf_flux_j      = numpy.zeros_like(itf_variable_j)
   itf_diffusion_j = numpy.zeros_like(itf_variable_j)

   itf_variable_k  = numpy.zeros((nb_equations, nb_elements_vert + 2, 2, nb_total_sol_pt_horiz, nb_total_sol_pt_horiz), dtype=datatype)
   itf_flux_k      = numpy.zeros_like(itf_variable_k)
   itf_diffusion_k = numpy.zeros_like(itf_variable_k)

   # eig_L          = numpy.zeros(nb_sol_pts * nb_elements_horiz, dtype=type_vec)
   # eig_R          = numpy.zeros(nb_sol_pts * nb_elements_horiz, dtype=type_vec)
   # eig            = numpy.zeros(nb_sol_pts * nb_elements_horiz, dtype=type_vec)
   #
   # flux_L         = numpy.zeros(nb_sol_pts * nb_elements_horiz, dtype=type_vec)
   # flux_R         = numpy.zeros(nb_sol_pts * nb_elements_horiz, dtype=type_vec)

   # Unpack dynamical variables
   density = Q[idx_rho,:, :, :]
   rho_u1  = Q[idx_rho_u1, :, :, :]
   rho_u2  = Q[idx_rho_u2, :, :, :]
   rho_u3  = Q[idx_rho_u3, :, :, :]
   rho_potential_temp = Q[idx_rho_theta, :, :, :]

   u1 = rho_u1 / density
   u2 = rho_u2 / density
   u3 = rho_u3 / density
   potential_temp = rho_potential_temp / density
   pressure = p0 * (rho_potential_temp * Rd / p0)**(cpd/cvd)

   #######################
   # Compute the fluxes
   flux_x1[idx_rho_u1,    :, :, :] = metric.sqrtG * (rho_u1 * u1 + metric.H_contra_11 * pressure)
   flux_x1[idx_rho_u2,    :, :, :] = metric.sqrtG * (rho_u1 * u2 + metric.H_contra_21 * pressure)
   flux_x1[idx_rho_u3,    :, :, :] = metric.sqrtG * rho_u1 * u3
   flux_x1[idx_rho,       :, :, :] = metric.sqrtG * rho_u1
   flux_x1[idx_rho_theta, :, :, :] = metric.sqrtG * rho_potential_temp * u1

   flux_x2[idx_rho_u1,    :, :, :] = metric.sqrtG * (rho_u2 * u1 + metric.H_contra_12 * pressure)
   flux_x2[idx_rho_u2,    :, :, :] = metric.sqrtG * (rho_u2 * u2 + metric.H_contra_22 * pressure)
   flux_x2[idx_rho_u3,    :, :, :] = metric.sqrtG * rho_u2 * u3
   flux_x2[idx_rho,       :, :, :] = metric.sqrtG * rho_u2
   flux_x2[idx_rho_theta, :, :, :] = metric.sqrtG * rho_potential_temp * u2

   flux_x3[idx_rho_u1,    :, :, :] = metric.sqrtG * (rho_u3 * u1 + metric.H_contra_13 * pressure)
   flux_x3[idx_rho_u2,    :, :, :] = metric.sqrtG * (rho_u3 * u2 + metric.H_contra_23 * pressure)
   flux_x3[idx_rho_u3,    :, :, :] = metric.sqrtG * rho_u3 * u3
   flux_x3[idx_rho,       :, :, :] = metric.sqrtG * rho_u3
   flux_x3[idx_rho_theta, :, :, :] = metric.sqrtG * rho_potential_temp * u3

   # Offset due to the halo
   offset = 1

   ########################################
   # Interpolate to the element interface
   for elem in range(nb_elements_horiz):
      epais = elem * nb_sol_pts + numpy.arange(nb_sol_pts)
      pos = elem + offset

      # --- Direction x1
      itf_variable_i[:, :, pos, 0, :] = Q[:, :, :, epais] @ mtrx.extrap_west
      itf_variable_i[:, :, pos, 1, :] = Q[:, :, :, epais] @ mtrx.extrap_east

      itf_flux_i[:, :, pos, 0, :] = flux_x1[:, :, :, epais] @ mtrx.extrap_west
      itf_flux_i[:, :, pos, 1, :] = flux_x1[:, :, :, epais] @ mtrx.extrap_east

      # --- Direction x2
      itf_variable_j[:, :, pos, 0, :] = mtrx.extrap_south @ Q[:, :, epais, :]
      itf_variable_j[:, :, pos, 1, :] = mtrx.extrap_north @ Q[:, :, epais, :]

      itf_flux_j[:, :, pos, 0, :] = mtrx.extrap_south @ flux_x2[:, :, epais, :]
      itf_flux_j[:, :, pos, 1, :] = mtrx.extrap_north @ flux_x2[:, :, epais, :]

   # --- Direction x3
   for i, (val_ground, val_sky) in enumerate(zip(mtrx.extrap_ground, mtrx.extrap_sky)):
      slice = [i + nb_sol_pts * x for x in range(nb_elements_vert)]
      itf_variable_k[:, 1:-1, 0, :, :] += val_ground * Q[:, slice, :, :]
      itf_variable_k[:, 1:-1, 1, :, :] += val_sky * Q[:, slice, :, :]

      itf_flux_k[:, 1:-1, 0, :, :] += val_ground * flux_x3[:, slice, :, :]
      itf_flux_k[:, 1:-1, 1, :, :] += val_sky * flux_x3[:, slice, :, :]

   ##########################################
   # Exchange necessary info with neighbors
   # TODO which of these do we need?
   ptopo.xchange_vectors(geom, u1_itf_i, u2_itf_i, u1_itf_j, u2_itf_j, u3_itf_i, u3_itf_j)
   ptopo.xchange_scalars(geom, density_itf_i, density_itf_j)
   ptopo.xchange_scalars(geom, temp_itf_i, temp_itf_j)

   #########################
   # Common Rusanov fluxes
   for itf in range(nb_interfaces_horiz):

      elem_L = itf
      elem_R = itf + 1

      h_itf_i[layer, elem_L, 1, :] -= topo.hsurf_itf_i[elem_L, :, 1]
      h_itf_i[layer, elem_R, 0, :] -= topo.hsurf_itf_i[elem_R, :, 0]

      h_itf_j[layer, elem_L, 1, :] -= topo.hsurf_itf_j[elem_L, 1, :]
      h_itf_j[layer, elem_R, 0, :] -= topo.hsurf_itf_j[elem_R, 0, :]

      # Direction x1

      if shallow_water_equations:
         eig_L[:] = numpy.abs( u1_itf_i[layer, elem_L, 1, :] ) + numpy.sqrt( gravity * h_itf_i[layer, elem_L, 1, :] * metric.H_contra_11_itf_i[:, itf] )
         eig_R[:] = numpy.abs( u1_itf_i[layer, elem_R, 0, :] ) + numpy.sqrt( gravity * h_itf_i[layer, elem_R, 0, :] * metric.H_contra_11_itf_i[:, itf] )
      else:
         eig_L[:] = numpy.abs( u1_itf_i[layer, elem_L, 1, :] )
         eig_R[:] = numpy.abs( u1_itf_i[layer, elem_R, 0, :] )

      eig[:] = numpy.maximum(eig_L, eig_R)

      # --- Continuity equation

      flux_L[:] = metric.sqrtG_itf_i[:, itf] * h_itf_i[layer, elem_L, 1, :] * u1_itf_i[layer, elem_L, 1, :]
      flux_R[:] = metric.sqrtG_itf_i[:, itf] * h_itf_i[layer, elem_R, 0, :] * u1_itf_i[layer, elem_R, 0, :]

      flux_Eq0_itf_i[elem_L, :, 1] = 0.5 * ( flux_L  + flux_R - eig * metric.sqrtG_itf_i[:, itf] *
                                             ( h_itf_i[layer, elem_R, 0, :] - h_itf_i[layer, elem_L, 1, :] ) )
      flux_Eq0_itf_i[elem_R, :, 0] = flux_Eq0_itf_i[elem_L, :, 1]

      # --- u1 equation

      flux_L[:] = metric.sqrtG_itf_i[:, itf] * ( h_itf_i[layer, elem_L, 1, :] * u1_itf_i[layer, elem_L, 1, :]**2 \
            + 0.5 * gravity * metric.H_contra_11_itf_i[:, itf] * h_itf_i[layer, elem_L, 1, :]**2 )
      flux_R[:] = metric.sqrtG_itf_i[:, itf] * ( h_itf_i[layer, elem_R, 0, :] * u1_itf_i[layer, elem_R, 0, :]**2 \
            + 0.5 * gravity * metric.H_contra_11_itf_i[:, itf] * h_itf_i[layer, elem_R, 0, :]**2 )

      flux_Eq1_itf_i[elem_L, :, 1] = 0.5 * ( flux_L  + flux_R - eig * metric.sqrtG_itf_i[:, itf] \
            * ( h_itf_i[layer, elem_R, 0, :] * u1_itf_i[layer, elem_R, 0, :] - h_itf_i[layer, elem_L, 1, :] * u1_itf_i[layer, elem_L, 1, :] ) )
      flux_Eq1_itf_i[elem_R, :, 0] = flux_Eq1_itf_i[elem_L, :, 1]

      # --- u2 equation

      flux_L[:] = metric.sqrtG_itf_i[:, itf] * ( h_itf_i[layer, elem_L, 1, :] * u2_itf_i[layer, elem_L, 1, :] * u1_itf_i[layer, elem_L, 1, :] \
            + 0.5 * gravity * metric.H_contra_21_itf_i[:, itf] * h_itf_i[layer, elem_L, 1, :]**2 )
      flux_R[:] = metric.sqrtG_itf_i[:, itf] * ( h_itf_i[layer, elem_R, 0, :] * u2_itf_i[layer, elem_R, 0, :] * u1_itf_i[layer, elem_R, 0, :] \
            + 0.5 * gravity * metric.H_contra_21_itf_i[:, itf] * h_itf_i[layer, elem_R, 0, :]**2 )

      flux_Eq2_itf_i[elem_L, :, 1] = 0.5 * ( flux_L  + flux_R - eig * metric.sqrtG_itf_i[:, itf] \
            * ( h_itf_i[layer, elem_R, 0, :] * u2_itf_i[layer, elem_R, 0, :] - h_itf_i[layer, elem_L, 1, :] * u2_itf_i[layer, elem_L, 1, :] ) )
      flux_Eq2_itf_i[elem_R, :, 0] = flux_Eq2_itf_i[elem_L, :, 1]

      # Direction x2

      if shallow_water_equations:
         eig_L[:] = numpy.abs( u2_itf_j[layer, elem_L, 1, :] ) + numpy.sqrt( gravity * h_itf_j[layer, elem_L, 1, :] * metric.H_contra_22_itf_j[itf, :] )
         eig_R[:] = numpy.abs( u2_itf_j[layer, elem_R, 0, :] ) + numpy.sqrt( gravity * h_itf_j[layer, elem_R, 0, :] * metric.H_contra_22_itf_j[itf, :] )
      else:
         eig_L[:] = numpy.abs( u2_itf_j[layer, elem_L, 1, :] )
         eig_R[:] = numpy.abs( u2_itf_j[layer, elem_R, 0, :] )

      eig[:] = numpy.maximum(eig_L, eig_R)

      # --- Continuity equation

      flux_L[:] = metric.sqrtG_itf_j[itf, :] * h_itf_j[layer, elem_L, 1, :] * u2_itf_j[layer, elem_L, 1, :]
      flux_R[:] = metric.sqrtG_itf_j[itf, :] * h_itf_j[layer, elem_R, 0, :] * u2_itf_j[layer, elem_R, 0, :]

      flux_Eq0_itf_j[elem_L, 1, :] = 0.5 * ( flux_L + flux_R - eig * metric.sqrtG_itf_j[itf, :] *
                                             ( h_itf_j[layer, elem_R, 0, :] - h_itf_j[layer, elem_L, 1, :] ) )
      flux_Eq0_itf_j[elem_R, 0, :] = flux_Eq0_itf_j[elem_L, 1, :]

      # --- u1 equation

      flux_L[:] = metric.sqrtG_itf_j[itf, :] * ( h_itf_j[layer, elem_L, 1, :] * u1_itf_j[layer, elem_L, 1, :] * u2_itf_j[layer, elem_L, 1, :] \
            + 0.5 * gravity * metric.H_contra_12_itf_j[itf, :] * h_itf_j[layer, elem_L, 1, :]**2 )
      flux_R[:] = metric.sqrtG_itf_j[itf, :] * ( h_itf_j[layer, elem_R, 0, :] * u1_itf_j[layer, elem_R, 0, :] * u2_itf_j[layer, elem_R, 0, :] \
            + 0.5 * gravity * metric.H_contra_12_itf_j[itf, :] * h_itf_j[layer, elem_R, 0, :]**2 )

      flux_Eq1_itf_j[elem_L, 1, :] = 0.5 * ( flux_L + flux_R - eig * metric.sqrtG_itf_j[itf, :] \
            * ( h_itf_j[layer, elem_R, 0, :] * u1_itf_j[layer, elem_R, 0, :] - h_itf_j[layer, elem_L, 1, :] * u1_itf_j[layer, elem_L, 1, :]) )
      flux_Eq1_itf_j[elem_R, 0, :] = flux_Eq1_itf_j[elem_L, 1, :]

      # --- u2 equation

      flux_L[:] = metric.sqrtG_itf_j[itf, :] * ( h_itf_j[layer, elem_L, 1, :] * u2_itf_j[layer, elem_L, 1, :]**2 \
            + 0.5 * gravity * metric.H_contra_22_itf_j[itf, :] * h_itf_j[layer, elem_L, 1, :]**2 )
      flux_R[:] = metric.sqrtG_itf_j[itf, :] * ( h_itf_j[layer, elem_R, 0, :] * u2_itf_j[layer, elem_R, 0, :]**2 \
            + 0.5 * gravity * metric.H_contra_22_itf_j[itf, :] * h_itf_j[layer, elem_R, 0, :]**2 )

      flux_Eq2_itf_j[elem_L, 1, :] = 0.5 * ( flux_L + flux_R - eig * metric.sqrtG_itf_j[itf, :] \
            * ( h_itf_j[layer, elem_R, 0, :] * u2_itf_j[layer, elem_R, 0, :] - h_itf_j[layer, elem_L, 1, :] * u2_itf_j[layer, elem_L, 1, :]) )
      flux_Eq2_itf_j[elem_R, 0, :] = flux_Eq2_itf_j[elem_L, 1, :]

   ###########################
   # Compute the derivatives
   for elem in range(nb_elements_horiz):
      epais = elem * nb_sol_pts + numpy.arange(nb_sol_pts)

      # --- Direction x1

      df1_dx1[idx_h][:,epais]   = flux_Eq0_x1[:,epais] @ mtrx.diff_solpt_tr + flux_Eq0_itf_i[elem+offset,:,:] @ mtrx.correction_tr
      df1_dx1[idx_hu1][:,epais] = flux_Eq1_x1[:,epais] @ mtrx.diff_solpt_tr + flux_Eq1_itf_i[elem+offset,:,:] @ mtrx.correction_tr
      df1_dx1[idx_hu2][:,epais] = flux_Eq2_x1[:,epais] @ mtrx.diff_solpt_tr + flux_Eq2_itf_i[elem+offset,:,:] @ mtrx.correction_tr

      # --- Direction x2

      df2_dx2[idx_h,epais,:]   = mtrx.diff_solpt @ flux_Eq0_x2[epais,:] + mtrx.correction @ flux_Eq0_itf_j[elem+offset,:,:]
      df2_dx2[idx_hu1,epais,:] = mtrx.diff_solpt @ flux_Eq1_x2[epais,:] + mtrx.correction @ flux_Eq1_itf_j[elem+offset,:,:]
      df2_dx2[idx_hu2,epais,:] = mtrx.diff_solpt @ flux_Eq2_x2[epais,:] + mtrx.correction @ flux_Eq2_itf_j[elem+offset,:,:]

   # Add coriolis, metric and terms due to varying bottom topography
   forcing[idx_h,:,:] = 0.0

   # Note: christoffel_1_22 is zero
   forcing[idx_hu1,:,:] = 2.0 * ( metric.christoffel_1_01 * h * u1 + metric.christoffel_1_02 * h * u2) \
         + metric.christoffel_1_11 * h * u1**2 + 2.0 * metric.christoffel_1_12 * h * u1 * u2 \
         + gravity * h * ( metric.H_contra_11 * topo.dzdx1 + metric.H_contra_12 * topo.dzdx2)

   # Note: metric.christoffel_2_11 is zero
   forcing[idx_hu2,:,:] = 2.0 * (metric.christoffel_2_01 * h * u1 + metric.christoffel_2_02 * h * u2) \
         + 2.0 * metric.christoffel_2_12 * h * u1 * u2 + metric.christoffel_2_22 * h * u2**2 \
         + gravity * h * ( metric.H_contra_21 * topo.dzdx1 + metric.H_contra_22 * topo.dzdx2)

   # Assemble the right-hand sides
   for var in range(3):
      rhs[var] = metric.inv_sqrtG * -( df1_dx1[var] + df2_dx2[var] ) - forcing[var]


   if filter_rhs:
      for var in range(3):
         rhs[var,:,:] = apply_filter(rhs[var,:,:], mtrx, nb_elements_horiz, nb_sol_pts)

   return rhs
