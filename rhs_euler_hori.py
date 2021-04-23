import numpy

from definitions import idx_rho_u1, idx_rho_u2, idx_rho_u3, idx_rho, idx_rho_theta, gravity, p0, Rd, cpd, cvd, heat_capacity_ratio
from dgfilter import apply_filter

def rhs_euler_hori(Q, geom, mtrx, metric, topo, ptopo, nbsolpts: int, nb_elements_horiz: int, case_number: int, filter_rhs: bool = False):

   type_vec = Q.dtype

   shallow_water_equations = ( case_number > 1 )

   nb_interfaces_horiz = nb_elements_horiz + 1

   df1_dx1 = numpy.zeros_like(Q, dtype=type_vec)
   df2_dx2 = numpy.zeros_like(Q, dtype=type_vec)
   forcing = numpy.zeros_like(Q, dtype=type_vec)
   rhs = numpy.zeros_like(Q, dtype=type_vec)

   flux_Eq0_itf_j = numpy.zeros((nb_elements_horiz+2, 2, nbsolpts*nb_elements_horiz), dtype=type_vec)
   flux_Eq1_itf_j = numpy.zeros((nb_elements_horiz+2, 2, nbsolpts*nb_elements_horiz), dtype=type_vec)
   flux_Eq2_itf_j = numpy.zeros((nb_elements_horiz+2, 2, nbsolpts*nb_elements_horiz), dtype=type_vec)
   flux_Eq3_itf_j = numpy.zeros((nb_elements_horiz+2, 2, nbsolpts*nb_elements_horiz), dtype=type_vec)
   flux_Eq4_itf_j = numpy.zeros((nb_elements_horiz+2, 2, nbsolpts*nb_elements_horiz), dtype=type_vec)
   rho_itf_j      = numpy.zeros((nb_elements_horiz+2, 2, nbsolpts*nb_elements_horiz), dtype=type_vec)
   u1_itf_j       = numpy.zeros((nb_elements_horiz+2, 2, nbsolpts*nb_elements_horiz), dtype=type_vec)
   u2_itf_j       = numpy.zeros((nb_elements_horiz+2, 2, nbsolpts*nb_elements_horiz), dtype=type_vec)
   u3_itf_j       = numpy.zeros((nb_elements_horiz+2, 2, nbsolpts*nb_elements_horiz), dtype=type_vec)
   theta_itf_j    = numpy.zeros((nb_elements_horiz+2, 2, nbsolpts*nb_elements_horiz), dtype=type_vec)
   pressure_itf_j    = numpy.zeros((nb_elements_horiz+2, 2, nbsolpts*nb_elements_horiz), dtype=type_vec)

   flux_Eq0_itf_i = numpy.zeros((nb_elements_horiz+2, nbsolpts*nb_elements_horiz, 2), dtype=type_vec)
   flux_Eq1_itf_i = numpy.zeros((nb_elements_horiz+2, nbsolpts*nb_elements_horiz, 2), dtype=type_vec)
   flux_Eq2_itf_i = numpy.zeros((nb_elements_horiz+2, nbsolpts*nb_elements_horiz, 2), dtype=type_vec)
   flux_Eq3_itf_i = numpy.zeros((nb_elements_horiz+2, nbsolpts*nb_elements_horiz, 2), dtype=type_vec)
   flux_Eq4_itf_i = numpy.zeros((nb_elements_horiz+2, nbsolpts*nb_elements_horiz, 2), dtype=type_vec)
   rho_itf_i      = numpy.zeros((nb_elements_horiz+2, 2, nbsolpts*nb_elements_horiz), dtype=type_vec)
   u1_itf_i       = numpy.zeros((nb_elements_horiz+2, 2, nbsolpts*nb_elements_horiz), dtype=type_vec)
   u2_itf_i       = numpy.zeros((nb_elements_horiz+2, 2, nbsolpts*nb_elements_horiz), dtype=type_vec)
   u3_itf_i       = numpy.zeros((nb_elements_horiz+2, 2, nbsolpts*nb_elements_horiz), dtype=type_vec)
   theta_itf_i    = numpy.zeros((nb_elements_horiz+2, 2, nbsolpts*nb_elements_horiz), dtype=type_vec)
   pressure_itf_i    = numpy.zeros((nb_elements_horiz+2, 2, nbsolpts*nb_elements_horiz), dtype=type_vec)

   eig_L          = numpy.zeros(nbsolpts*nb_elements_horiz, dtype=type_vec)
   eig_R          = numpy.zeros(nbsolpts*nb_elements_horiz, dtype=type_vec)
   eig            = numpy.zeros(nbsolpts*nb_elements_horiz, dtype=type_vec)

   flux_L         = numpy.zeros(nbsolpts*nb_elements_horiz, dtype=type_vec)
   flux_R         = numpy.zeros(nbsolpts*nb_elements_horiz, dtype=type_vec)

   # Unpack dynamical variables
   rho     = Q[idx_rho, :, :]
   rho_u1  = Q[idx_rho_u1, :, :]
   rho_u2  = Q[idx_rho_u2, :, :]
   rho_u3  = Q[idx_rho_u3, :, :]
   rho_theta = Q[idx_rho_theta, :, :]

   u1 = rho_u1 / rho
   u2 = rho_u2 / rho
   u3 = rho_u3 / rho
   theta = rho_theta / rho
   pressure = p0 * (rho_theta * Rd / p0)**(cpd/cvd)

   # Compute the fluxes
   flux_Eq0_x1 = metric.sqrtG[0, :, :] * rho_u1 # TODO : sqrtG et les autres "metric" devrait être 2D
   flux_Eq0_x2 = metric.sqrtG[0, :, :] * rho_u2

   flux_Eq1_x1 = metric.sqrtG[0, :, :] * ( rho_u1 * u1 + metric.H_contra_11[0, :, :] * pressure )
   flux_Eq1_x2 = metric.sqrtG[0, :, :] * ( rho_u1 * u2 + metric.H_contra_12[0, :, :] * pressure )

   flux_Eq2_x1 = metric.sqrtG[0, :, :] * ( rho_u2 * u1 + metric.H_contra_21[0, :, :] * pressure )
   flux_Eq2_x2 = metric.sqrtG[0, :, :] * ( rho_u2 * u2 + metric.H_contra_22[0, :, :] * pressure )

   flux_Eq3_x1 = metric.sqrtG[0, :, :] * ( rho_u3 * u1 + metric.H_contra_31[0, :, :] * pressure )
   flux_Eq3_x2 = metric.sqrtG[0, :, :] * ( rho_u3 * u2 + metric.H_contra_32[0, :, :] * pressure )

   flux_Eq4_x1 = metric.sqrtG[0, :, :] * rho_theta * u1
   flux_Eq4_x2 = metric.sqrtG[0, :, :] * rho_theta * u2

   # Offset due to the halo
   offset = 1

   # Interpolate to the element interface
   for elem in range(nb_elements_horiz):
      epais = elem * nbsolpts + numpy.arange(nbsolpts)

      pos = elem + offset

      # --- Direction x1

      rho_itf_i[pos, 0, :] = rho[:, epais] @ mtrx.extrap_west
      rho_itf_i[pos, 1, :] = rho[:, epais] @ mtrx.extrap_east

      u1_itf_i[pos, 0, :] = u1[:, epais] @ mtrx.extrap_west
      u1_itf_i[pos, 1, :] = u1[:, epais] @ mtrx.extrap_east

      u2_itf_i[pos, 0, :] = u2[:, epais] @ mtrx.extrap_west
      u2_itf_i[pos, 1, :] = u2[:, epais] @ mtrx.extrap_east

      u3_itf_i[pos, 0, :] = u3[:, epais] @ mtrx.extrap_west
      u3_itf_i[pos, 1, :] = u3[:, epais] @ mtrx.extrap_east

      theta_itf_i[pos, 0, :] = theta[:, epais] @ mtrx.extrap_west
      theta_itf_i[pos, 1, :] = theta[:, epais] @ mtrx.extrap_east

      pressure_itf_i[pos, 0, :] = theta[:, epais] @ mtrx.extrap_west
      pressure_itf_i[pos, 1, :] = theta[:, epais] @ mtrx.extrap_east

      # --- Direction x2

      rho_itf_j[pos, 0, :] = mtrx.extrap_south @ rho[epais, :]
      rho_itf_j[pos, 1, :] = mtrx.extrap_north @ rho[epais, :]

      u1_itf_j[pos, 0, :] = mtrx.extrap_south @ u1[epais, :]
      u1_itf_j[pos, 1, :] = mtrx.extrap_north @ u1[epais, :]

      u2_itf_j[pos, 0, :] = mtrx.extrap_south @ u2[epais, :]
      u2_itf_j[pos, 1, :] = mtrx.extrap_north @ u2[epais, :]

      u3_itf_j[pos, 0, :] = mtrx.extrap_south @ u3[epais, :]
      u3_itf_j[pos, 1, :] = mtrx.extrap_north @ u3[epais, :]

      theta_itf_j[pos, 0, :] = mtrx.extrap_south @ theta[epais, :]
      theta_itf_j[pos, 1, :] = mtrx.extrap_north @ theta[epais, :]

      pressure_itf_j[pos, 0, :] = mtrx.extrap_south @ theta[epais, :]
      pressure_itf_j[pos, 1, :] = mtrx.extrap_north @ theta[epais, :]

   # TODO : faire une seule fois les échanges ... 
   ptopo.xchange_scalars(geom, rho_itf_i, rho_itf_j)
   ptopo.xchange_vectors(geom, u1_itf_i, u2_itf_i, u1_itf_j, u2_itf_j, u3_itf_i, u3_itf_j)
   ptopo.xchange_scalars(geom, theta_itf_i, theta_itf_j)
   ptopo.xchange_scalars(geom, pressure_itf_i, pressure_itf_j)

   # Common Rusanov fluxes
   for itf in range(nb_interfaces_horiz):

      elem_L = itf
      elem_R = itf + 1

      # Direction x1

      eig_L[:] = numpy.abs( u1_itf_i[elem_L, 1, :] ) + numpy.sqrt( metric.H_contra_11_itf_i[:, itf] * heat_capacity_ratio * pressure_itf_i[elem_L, 1, :] / rho_itf_i[elem_L, 1, :] )
      eig_L[:] = numpy.abs( u1_itf_i[elem_R, 0, :] ) + numpy.sqrt( metric.H_contra_11_itf_i[:, itf] * heat_capacity_ratio * pressure_itf_i[elem_R, 0, :] / rho_itf_i[elem_R, 0, :] )

      eig[:] = numpy.maximum(eig_L, eig_R)

      # --- Continuity equation

      flux_L[:] = metric.sqrtG_itf_i[:, itf] * rho_itf_i[elem_L, 1, :] * u1_itf_i[elem_L, 1, :]
      flux_R[:] = metric.sqrtG_itf_i[:, itf] * rho_itf_i[elem_R, 0, :] * u1_itf_i[elem_R, 0, :]

      flux_Eq0_itf_i[elem_L, :, 1] = 0.5 * ( flux_L  + flux_R - eig * metric.sqrtG_itf_i[:, itf] * ( rho_itf_i[elem_R, 0, :] - rho_itf_i[elem_L, 1, :] ) )
      flux_Eq0_itf_i[elem_R, :, 0] = flux_Eq0_itf_i[elem_L, :, 1]

      # --- u1 equation

      flux_L[:] = metric.sqrtG_itf_i[:, itf] * ( rho_itf_i[elem_L, 1, :] * u1_itf_i[elem_L, 1, :]**2 + metric.H_contra_11_itf_i[:, itf] * pressure_itf_i[elem_L, 1, :]**2 )
      flux_R[:] = metric.sqrtG_itf_i[:, itf] * ( rho_itf_i[elem_R, 0, :] * u1_itf_i[elem_R, 0, :]**2 + metric.H_contra_11_itf_i[:, itf] * pressure_itf_i[elem_R, 0, :]**2 )

      flux_Eq1_itf_i[elem_L, :, 1] = 0.5 * ( flux_L  + flux_R - eig * metric.sqrtG_itf_i[:, itf] * ( rho_itf_i[elem_R, 0, :] * u1_itf_i[elem_R, 0, :] - rho_itf_i[elem_L, 1, :] * u1_itf_i[elem_L, 1, :] ) )
      flux_Eq1_itf_i[elem_R, :, 0] = flux_Eq1_itf_i[elem_L, :, 1]

      # --- u2 equation

      flux_L[:] = metric.sqrtG_itf_i[:, itf] * ( rho_itf_i[elem_L, 1, :] * u2_itf_i[elem_L, 1, :] * u1_itf_i[elem_L, 1, :] + metric.H_contra_21_itf_i[:, itf] * pressure_itf_i[elem_L, 1, :]**2 )
      flux_R[:] = metric.sqrtG_itf_i[:, itf] * ( rho_itf_i[elem_R, 0, :] * u2_itf_i[elem_R, 0, :] * u1_itf_i[elem_R, 0, :] + metric.H_contra_21_itf_i[:, itf] * pressure_itf_i[elem_R, 0, :]**2 )

      flux_Eq2_itf_i[elem_L, :, 1] = 0.5 * ( flux_L  + flux_R - eig * metric.sqrtG_itf_i[:, itf] * ( rho_itf_i[elem_R, 0, :] * u2_itf_i[elem_R, 0, :] - rho_itf_i[elem_L, 1, :] * u2_itf_i[elem_L, 1, :] ) )
      flux_Eq2_itf_i[elem_R, :, 0] = flux_Eq2_itf_i[elem_L, :, 1]

      # --- u3 equation

      flux_L[:] = metric.sqrtG_itf_i[:, itf] * ( rho_itf_i[elem_L, 1, :] * u3_itf_i[elem_L, 1, :] * u1_itf_i[elem_L, 1, :] + metric.H_contra_31_itf_i[:, itf] * pressure_itf_i[elem_L, 1, :]**2 )
      flux_R[:] = metric.sqrtG_itf_i[:, itf] * ( rho_itf_i[elem_R, 0, :] * u3_itf_i[elem_R, 0, :] * u1_itf_i[elem_R, 0, :] + metric.H_contra_31_itf_i[:, itf] * pressure_itf_i[elem_R, 0, :]**2 )

      flux_Eq3_itf_i[elem_L, :, 1] = 0.5 * ( flux_L  + flux_R - eig * metric.sqrtG_itf_i[:, itf] * ( rho_itf_i[elem_R, 0, :] * u3_itf_i[elem_R, 0, :] - rho_itf_i[elem_L, 1, :] * u3_itf_i[elem_L, 1, :] ) )
      flux_Eq3_itf_i[elem_R, :, 0] = flux_Eq3_itf_i[elem_L, :, 1]

      # --- thermodynamics equation

      flux_L[:] = metric.sqrtG_itf_i[:, itf] * ( rho_itf_i[elem_L, 1, :] * theta_itf_i[elem_L, 1, :] * u1_itf_i[elem_L, 1, :] )
      flux_R[:] = metric.sqrtG_itf_i[:, itf] * ( rho_itf_i[elem_R, 0, :] * theta_itf_i[elem_R, 0, :] * u1_itf_i[elem_R, 0, :] )

      flux_Eq4_itf_i[elem_L, :, 1] = 0.5 * ( flux_L  + flux_R - eig * metric.sqrtG_itf_i[:, itf] * ( rho_itf_i[elem_R, 0, :] * theta_itf_i[elem_R, 0, :] - rho_itf_i[elem_L, 1, :] * theta_itf_i[elem_L, 1, :] ) )
      flux_Eq4_itf_i[elem_R, :, 0] = flux_Eq4_itf_i[elem_L, :, 1]

      # Direction x2

      eig_L[:] = numpy.abs( u2_itf_j[elem_L, 1, :] ) + numpy.sqrt( metric.H_contra_22_itf_j[itf, :] * heat_capacity_ratio * pressure_itf_j[elem_L, 1, :] / rho_itf_j[elem_L, 1, :] )
      eig_L[:] = numpy.abs( u2_itf_j[elem_R, 0, :] ) + numpy.sqrt( metric.H_contra_22_itf_j[itf, :] * heat_capacity_ratio * pressure_itf_j[elem_R, 0, :] / rho_itf_j[elem_R, 0, :] )

      eig[:] = numpy.maximum(eig_L, eig_R)

      # --- Continuity equation

      flux_L[:] = metric.sqrtG_itf_j[itf, :] * rho_itf_j[elem_L, 1, :] * u2_itf_j[elem_L, 1, :]
      flux_R[:] = metric.sqrtG_itf_j[itf, :] * rho_itf_j[elem_R, 0, :] * u2_itf_j[elem_R, 0, :]

      flux_Eq0_itf_j[elem_L, 1, :] = 0.5 * ( flux_L + flux_R - eig * metric.sqrtG_itf_j[itf, :] * ( rho_itf_j[elem_R, 0, :] - rho_itf_j[elem_L, 1, :] ) )
      flux_Eq0_itf_j[elem_R, 0, :] = flux_Eq0_itf_j[elem_L, 1, :]

      # --- u1 equation

      flux_L[:] = metric.sqrtG_itf_j[itf, :] * ( rho_itf_j[elem_L, 1, :] * u1_itf_j[elem_L, 1, :] * u2_itf_j[elem_L, 1, :] + metric.H_contra_12_itf_j[itf, :] * rho_itf_j[elem_L, 1, :]**2 )
      flux_R[:] = metric.sqrtG_itf_j[itf, :] * ( rho_itf_j[elem_R, 0, :] * u1_itf_j[elem_R, 0, :] * u2_itf_j[elem_R, 0, :] + metric.H_contra_12_itf_j[itf, :] * rho_itf_j[elem_R, 0, :]**2 )

      flux_Eq1_itf_j[elem_L, 1, :] = 0.5 * ( flux_L + flux_R - eig * metric.sqrtG_itf_j[itf, :] * ( rho_itf_j[elem_R, 0, :] * u1_itf_j[elem_R, 0, :] - rho_itf_j[elem_L, 1, :] * u1_itf_j[elem_L, 1, :]) )
      flux_Eq1_itf_j[elem_R, 0, :] = flux_Eq1_itf_j[elem_L, 1, :]

      # --- u2 equation

      flux_L[:] = metric.sqrtG_itf_j[itf, :] * ( rho_itf_j[elem_L, 1, :] * u2_itf_j[elem_L, 1, :]**2 + metric.H_contra_22_itf_j[itf, :] * rho_itf_j[elem_L, 1, :]**2 )
      flux_R[:] = metric.sqrtG_itf_j[itf, :] * ( rho_itf_j[elem_R, 0, :] * u2_itf_j[elem_R, 0, :]**2 + metric.H_contra_22_itf_j[itf, :] * rho_itf_j[elem_R, 0, :]**2 )

      flux_Eq2_itf_j[elem_L, 1, :] = 0.5 * ( flux_L + flux_R - eig * metric.sqrtG_itf_j[itf, :] * ( rho_itf_j[elem_R, 0, :] * u2_itf_j[elem_R, 0, :] - rho_itf_j[elem_L, 1, :] * u2_itf_j[elem_L, 1, :]) )
      flux_Eq2_itf_j[elem_R, 0, :] = flux_Eq2_itf_j[elem_L, 1, :]

      # --- u3 equation

      flux_L[:] = metric.sqrtG_itf_j[itf, :] * ( rho_itf_j[elem_L, 1, :] * u3_itf_j[elem_L, 1, :] * u2_itf_j[elem_L, 1, :] + metric.H_contra_32_itf_j[itf, :] * rho_itf_j[elem_L, 1, :]**2 )
      flux_R[:] = metric.sqrtG_itf_j[itf, :] * ( rho_itf_j[elem_R, 0, :] * u3_itf_j[elem_R, 0, :] * u2_itf_j[elem_R, 0, :] + metric.H_contra_32_itf_j[itf, :] * rho_itf_j[elem_R, 0, :]**2 )

      flux_Eq3_itf_j[elem_L, 1, :] = 0.5 * ( flux_L + flux_R - eig * metric.sqrtG_itf_j[itf, :] * ( rho_itf_j[elem_R, 0, :] * u3_itf_j[elem_R, 0, :] - rho_itf_j[elem_L, 1, :] * u3_itf_j[elem_L, 1, :]) )
      flux_Eq3_itf_j[elem_R, 0, :] = flux_Eq3_itf_j[elem_L, 1, :]

      # --- thermodynamics equation

      flux_L[:] = metric.sqrtG_itf_j[itf, :] * ( rho_itf_j[elem_L, 1, :] * theta_itf_j[elem_L, 1, :] * u2_itf_j[elem_L, 1, :] )
      flux_R[:] = metric.sqrtG_itf_j[itf, :] * ( rho_itf_j[elem_R, 0, :] * theta_itf_j[elem_R, 0, :] * u2_itf_j[elem_R, 0, :] )

      flux_Eq4_itf_j[elem_L, 1, :] = 0.5 * ( flux_L + flux_R - eig * metric.sqrtG_itf_j[itf, :] * ( rho_itf_j[elem_R, 0, :] * theta_itf_j[elem_R, 0, :] - rho_itf_j[elem_L, 1, :] * theta_itf_j[elem_L, 1, :]) )
      flux_Eq4_itf_j[elem_R, 0, :] = flux_Eq4_itf_j[elem_L, 1, :]

   # Compute the derivatives
   for elem in range(nb_elements_horiz):
      epais = elem * nbsolpts + numpy.arange(nbsolpts)
   
      # --- Direction x1
      df1_dx1[idx_rho][:,epais]        = flux_Eq0_x1[:, epais] @ mtrx.diff_solpt_tr + flux_Eq0_itf_i[elem+offset, :, :] @ mtrx.correction_tr
      df1_dx1[idx_rho_u1][:,epais]     = flux_Eq1_x1[:, epais] @ mtrx.diff_solpt_tr + flux_Eq1_itf_i[elem+offset, :, :] @ mtrx.correction_tr
      df1_dx1[idx_rho_u2][:,epais]     = flux_Eq2_x1[:, epais] @ mtrx.diff_solpt_tr + flux_Eq2_itf_i[elem+offset, :, :] @ mtrx.correction_tr
      df1_dx1[idx_rho_u3][:,epais]     = flux_Eq3_x1[:, epais] @ mtrx.diff_solpt_tr + flux_Eq3_itf_i[elem+offset, :, :] @ mtrx.correction_tr
      df1_dx1[idx_rho_theta][:,epais]  = flux_Eq4_x1[:, epais] @ mtrx.diff_solpt_tr + flux_Eq4_itf_i[elem+offset, :, :] @ mtrx.correction_tr

      # --- Direction x2

      df2_dx2[idx_rho, epais, :]       = mtrx.diff_solpt @ flux_Eq0_x2[epais, :] + mtrx.correction @ flux_Eq0_itf_j[elem+offset, :, :]
      df2_dx2[idx_rho_u1, epais, :]    = mtrx.diff_solpt @ flux_Eq1_x2[epais, :] + mtrx.correction @ flux_Eq1_itf_j[elem+offset, :, :]
      df2_dx2[idx_rho_u2, epais, :]    = mtrx.diff_solpt @ flux_Eq2_x2[epais, :] + mtrx.correction @ flux_Eq2_itf_j[elem+offset, :, :]
      df2_dx2[idx_rho_u3, epais, :]    = mtrx.diff_solpt @ flux_Eq3_x2[epais, :] + mtrx.correction @ flux_Eq3_itf_j[elem+offset, :, :]
      df2_dx2[idx_rho_theta, epais, :] = mtrx.diff_solpt @ flux_Eq4_x2[epais, :] + mtrx.correction @ flux_Eq4_itf_j[elem+offset, :, :]
   
   # Add coriolis and metric terms

   # TODO : metric horizontale en 2D ???

#   forcing[idx_rho_u1,:,:] = 2.0 * ( metric.christoffel_1_01[0, :, :] * rho * u1 + metric.christoffel_1_02[0, :, :] * rho * u2 + metric.christoffel_1_03[0, :, :] * rho * u3) \
#         + metric.christoffel_1_11[0, :, :] * rho * u1**2 + 2.0 * metric.christoffel_1_12[0, :, :] * rho * u1 * u2 \
#         + 2.0 * metric.christoffel_1_13[0, :, :] * rho * u1 * u3 + metric.christoffel_1_22[0, :, :] * rho * u2**2 \
#         + 2.0 * metric.christoffel_1_23[0, :, :] * rho * u2 * u3 + metric.christoffel_1_33[0, :, :] * rho * u3**2
#
#   forcing[idx_rho_u2,:,:] = 2.0 * (metric.christoffel_2_01[0, :, :] * rho * u1 + metric.christoffel_2_02[0, :, :] * rho * u2 + metric.christoffel_2_03[0, :, :] * rho * u3) \
#         + metric.christoffel_2_11[0, :, :] * rho * u1**2 + 2.0 * metric.christoffel_2_12[0, :, :] * rho * u1 * u2 \
#         + 2.0 * metric.christoffel_2_13[0, :, :] * rho * u1 * u3 + metric.christoffel_2_22[0, :, :] * rho * u2**2 \
#         + 2.0 * metric.christoffel_2_23[0, :, :] * rho * u2 * u3 + metric.christoffel_2_33[0, :, :] * rho * u3**2

   forcing[idx_rho_u1,:,:] = 2.0 * ( metric.christoffel_1_01[0, :, :] * rho * u1 + metric.christoffel_1_02[0, :, :] * rho * u2 ) \
         + metric.christoffel_1_11[0, :, :] * rho * u1**2 \
         + 2.0 * metric.christoffel_1_12[0, :, :] * rho * u1 * u2 \
         + metric.christoffel_1_22[0, :, :] * rho * u2**2 

   forcing[idx_rho_u2,:,:] = 2.0 * (metric.christoffel_2_01[0, :, :] * rho * u1 + metric.christoffel_2_02[0, :, :] * rho * u2 ) \
         + metric.christoffel_2_11[0, :, :] * rho * u1**2 \
         + 2.0 * metric.christoffel_2_12[0, :, :] * rho * u1 * u2 \
         + metric.christoffel_2_22[0, :, :] * rho * u2**2 


   forcing[idx_rho_u3,:,:] = 2.0 * (metric.christoffel_3_01[0, :, :] * rho * u1 + metric.christoffel_3_02[0, :, :] * rho * u2 ) \
         + metric.christoffel_3_11[0, :, :] * rho * u1**2 \
         + 2.0 * metric.christoffel_3_12[0, :, :] * rho * u1 * u2 \
         + metric.christoffel_3_22[0, :, :] * rho * u2**2 

   # Assemble the right-hand sides
   for var in range(5):
      rhs[var] = metric.inv_sqrtG[0, :, :] * -( df1_dx1[var] + df2_dx2[var] ) - forcing[var]

   rhs[3] = 0.

   # TODO
#   if filter_rhs:
#      for var in range(3):
#         rhs[var,:,:] = apply_filter(rhs[var,:,:], mtrx, nb_elements_horiz, nbsolpts)

   return rhs
