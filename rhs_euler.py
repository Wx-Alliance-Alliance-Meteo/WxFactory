import numpy

from definitions import idx_rho_u1, idx_rho_u2, idx_rho_w, idx_rho, idx_rho_theta, gravity, p0, Rd, cpd, cvd
from dgfilter import apply_filter

# TODO : reviser les parametres : nbsolpts, nb_elements devraient être dans le geom
def rhs_euler(Q, geom, mtrx, metric, topo, ptopo, nbsolpts: int, nb_elements_horiz: int, nb_elements_vert: int, case_number: int, filter_rhs: bool = False):

   type_vec = Q.dtype
   nb_equations = Q.shape[0]
   idx_first_tracer = 5

   nb_interfaces_horiz = nb_elements_horiz + 1
   nb_pts_horiz = nb_elements_horiz * nbsolpts
   nb_vertical_levels = nb_elements_vert * nbsolpts

   # Result
   rhs = numpy.empty_like(Q, dtype=type_vec)

   # Work arrays
   flux_x1 = numpy.empty_like(Q, dtype=type_vec)
   flux_x2 = numpy.empty_like(Q, dtype=type_vec)
   flux_x3 = numpy.empty_like(Q, dtype=type_vec)

   df1dx1 = numpy.empty_like(Q, dtype=type_vec)
   df2dx2 = numpy.empty_like(Q, dtype=type_vec)
#   df3_dx3 = numpy.empty_like(Q)

   forcing = numpy.empty_like(Q, dtype=type_vec)

   variables_itf_i  = numpy.empty((nb_equations, nb_vertical_levels, nb_elements_horiz + 2, 2, nb_pts_horiz), dtype=type_vec)
   flux_x1_itf_i    = numpy.empty((nb_equations, nb_vertical_levels, nb_elements_horiz + 2, nb_pts_horiz, 2), dtype=type_vec)

   variables_itf_j  = numpy.empty((nb_equations, nb_vertical_levels, nb_elements_horiz + 2, 2, nb_pts_horiz), dtype=type_vec)
   flux_x2_itf_j    = numpy.empty_like(variables_itf_j)

#   itf_variable_k  = numpy.empty((nb_equations, nb_elements_vert + 2, 2, nb_pts_horiz, nb_pts_horiz), dtype=type_vec)
#   itf_flux_k      = numpy.empty_like(itf_variable_k)
#   itf_diffusion_k = numpy.empty_like(itf_variable_k)

#   eig_L          = numpy.empty((nb_vertical_levels, nb_pts_horiz), dtype=type_vec)
#   eig_R          = numpy.empty((nb_vertical_levels, nb_pts_horiz), dtype=type_vec)
#   eig            = numpy.empty((nb_vertical_levels, nb_pts_horiz), dtype=type_vec)
   
   # flux_L         = numpy.empty(nbsolpts * nb_elements_horiz, dtype=type_vec)
   # flux_R         = numpy.empty(nbsolpts * nb_elements_horiz, dtype=type_vec)

   # Offset due to the halo
   offset = 1

   blocking = False

   # Interpolate to the element interface
   for elem in range(nb_elements_horiz):
      epais = elem * nbsolpts + numpy.arange(nbsolpts)
      pos = elem + offset

      # --- Direction x1
      variables_itf_i[:, :, pos, 0, :] = Q[:, :, :, epais] @ mtrx.extrap_west
      variables_itf_i[:, :, pos, 1, :] = Q[:, :, :, epais] @ mtrx.extrap_east

      # --- Direction x2
      variables_itf_j[:, :, pos, 0, :] = mtrx.extrap_south @ Q[:, :, epais, :]
      variables_itf_j[:, :, pos, 1, :] = mtrx.extrap_north @ Q[:, :, epais, :]

   # TODO
   # --- Direction x3

   # Initiate transfers
   all_request = ptopo.xchange_Euler_interfaces(geom, variables_itf_i, variables_itf_j, blocking=blocking)

   # Unpack dynamical variables
   rho = Q[idx_rho,:, :, :]
   u1  = Q[idx_rho_u1, :, :, :] / rho
   u2  = Q[idx_rho_u2, :, :, :] / rho
   w   = Q[idx_rho_w, :, :, :]  / rho
   theta = Q[idx_rho_theta, :, :, :]  / rho

   # Scale u3 to the local coordinate system
   u3 = w * 2. / geom.Δx1 # TODO : terrain following

#   pressure = p0 * (Q[idx_rho_theta, :, :, :] * Rd / p0)**(cpd/cvd)

   # Compute the fluxes
   flux_x1[idx_rho,       :, :, :] = metric.sqrtG * Q[idx_rho_u1, :, :, :]
#   flux_x1[idx_rho_u1,    :, :, :] = metric.sqrtG * (Q[idx_rho_u1, :, :, :] * u1 + metric.H_contra_11 * pressure)
#   flux_x1[idx_rho_u2,    :, :, :] = metric.sqrtG * (Q[idx_rho_u2, :, :, :] * u1 + metric.H_contra_21 * pressure)
#   flux_x1[idx_rho_u3,    :, :, :] = metric.sqrtG * (Q[idx_rho_u3, :, :, :] * u1 + metric.H_contra_31 * pressure)
   flux_x1[idx_rho_theta, :, :, :] = metric.sqrtG * Q[idx_rho_theta, :, :, :] * u1

   flux_x2[idx_rho,       :, :, :] = metric.sqrtG * Q[idx_rho_u2, :, :, :]
#   flux_x2[idx_rho_u1,    :, :, :] = metric.sqrtG * (Q[idx_rho_u1, :, :, :] * u2 + metric.H_contra_12 * pressure)
#   flux_x2[idx_rho_u2,    :, :, :] = metric.sqrtG * (Q[idx_rho_u2, :, :, :] * u2 + metric.H_contra_22 * pressure)
#   flux_x2[idx_rho_u3,    :, :, :] = metric.sqrtG * (Q[idx_rho_u3, :, :, :] * u2 + metric.H_contra_32 * pressure)
   flux_x2[idx_rho_theta, :, :, :] = metric.sqrtG * Q[idx_rho_theta, :, :, :] * u2

   flux_x3[idx_rho,       :, :, :] = metric.sqrtG * rho * u3
#   flux_x3[idx_rho_u1,    :, :, :] = metric.sqrtG * (Q[idx_rho_u1, :, :, :] * u3 + metric.H_contra_13 * pressure)
#   flux_x3[idx_rho_u2,    :, :, :] = metric.sqrtG * (Q[idx_rho_u2, :, :, :] * u3 + metric.H_contra_23 * pressure)
#   flux_x3[idx_rho_u3,    :, :, :] = metric.sqrtG * (Q[idx_rho_u3, :, :, :] * u3 + metric.H_contra_33 * pressure)
   flux_x3[idx_rho_theta, :, :, :] = metric.sqrtG * Q[idx_rho_theta, :, :, :] * u3

   # Tracers
   flux_x1[idx_first_tracer:,:, :, :] = metric.sqrtG * u1 * Q[idx_first_tracer:, :, :, :]
   flux_x2[idx_first_tracer:,:, :, :] = metric.sqrtG * u2 * Q[idx_first_tracer:, :, :, :]
#   flux_x3[idx_first_tracer:,:, :, :] = metric.sqrtG * u3 * Q[idx_first_tracer:, :, :, :]

   # Compute the derivatives, corrections for the boundaries will be added later
   for elem in range(nb_elements_horiz):
      epais = elem * nbsolpts + numpy.arange(nbsolpts)

      # --- Direction x1

      df1dx1[:,:,:,epais] = flux_x1[:,:,:,epais] @ mtrx.diff_solpt_tr

      # --- Direction x2

      df2dx2[:,:,epais,:] = mtrx.diff_solpt @ flux_x2[:,:,epais,:]

   # Finish transfers
   all_request.wait() # TODO : Vincent

   # Common Rusanov fluxes
   for itf in range(nb_interfaces_horiz):

      elem_L = itf
      elem_R = itf + 1

      # --- Direction x1

      rho_L = variables_itf_i[idx_rho, :, elem_L, 1, :]
      rho_R = variables_itf_i[idx_rho, :, elem_R, 0, :]
      u1_L = variables_itf_i[idx_rho_u1, :, elem_L, 1, :] / rho_L
      u1_R = variables_itf_i[idx_rho_u1, :, elem_R, 0, :] / rho_R
      rho_tracers_L = variables_itf_i[idx_first_tracer:, :, elem_L, 1, :]
      rho_tracers_R = variables_itf_i[idx_first_tracer:, :, elem_R, 0, :]

      eig = numpy.maximum(numpy.abs(u1_L), numpy.abs(u1_R))

      flux_L = metric.sqrtG_itf_i[:, itf] * u1_L * rho_tracers_L
      flux_R = metric.sqrtG_itf_i[:, itf] * u1_R * rho_tracers_R

      flux_x1_itf_i[idx_first_tracer:, :, elem_L, :, 1] = 0.5 * ( flux_L  + flux_R - eig * metric.sqrtG_itf_i[:, itf] * ( rho_tracers_R - rho_tracers_L ) )
      flux_x1_itf_i[idx_first_tracer:, :, elem_R, :, 0] = flux_x1_itf_i[idx_first_tracer:, :, elem_L, :, 1]

      # --- Direction x2

      rho_L = variables_itf_j[idx_rho, :, elem_L, 1, :]
      rho_R = variables_itf_j[idx_rho, :, elem_R, 0, :]
      u2_L = variables_itf_j[idx_rho_u2, :, elem_L, 1, :] / rho_L
      u2_R = variables_itf_j[idx_rho_u2, :, elem_R, 0, :] / rho_R
      rho_tracer_L = variables_itf_j[idx_first_tracer:, :, elem_L, 1, :]
      rho_tracer_R = variables_itf_j[idx_first_tracer:, :, elem_R, 0, :]

      eig = numpy.maximum(numpy.abs(u2_L), numpy.abs(u2_R))

      flux_L[:] = metric.sqrtG_itf_j[itf, :] * u2_L * rho_tracer_L
      flux_R[:] = metric.sqrtG_itf_j[itf, :] * u2_R * rho_tracer_R

      flux_x2_itf_j[idx_first_tracer:, :, elem_L, 1, :] = 0.5 * ( flux_L + flux_R - eig * metric.sqrtG_itf_j[itf, :] * ( rho_tracers_R - rho_tracers_L ) )
      flux_x2_itf_j[idx_first_tracer:, :, elem_R, 0, :] = flux_x2_itf_j[idx_first_tracer:, :, elem_L, 1, :]


   # Add corrections to the derivatives
   for elem in range(nb_elements_horiz):
      epais = elem * nbsolpts + numpy.arange(nbsolpts)

      # --- Direction x1

      df1dx1[:,:,:,epais] += flux_x1_itf_i[:, :, elem+offset, :, :] @ mtrx.correction_tr

      # --- Direction x2

      df2dx2[:,:,epais,:] += mtrx.correction @ flux_x2_itf_j[:, :, elem+offset, :, :]

   # Add coriolis, metric and terms due to varying bottom topography
#   forcing[idx_h,:,:] = 0.0
#
   # Note: christoffel_1_22 is zero
#   forcing[idx_hu1,:,:] = 2.0 * ( metric.christoffel_1_01 * h * u1 + metric.christoffel_1_02 * h * u2) \
#         + metric.christoffel_1_11 * h * u1**2 + 2.0 * metric.christoffel_1_12 * h * u1 * u2 \
#         + gravity * h * ( metric.H_contra_11 * topo.dzdx1 + metric.H_contra_12 * topo.dzdx2)
#
   # Note: metric.christoffel_2_11 is zero
#   forcing[idx_hu2,:,:] = 2.0 * (metric.christoffel_2_01 * h * u1 + metric.christoffel_2_02 * h * u2) \
#         + 2.0 * metric.christoffel_2_12 * h * u1 * u2 + metric.christoffel_2_22 * h * u2**2 \
#         + gravity * h * ( metric.H_contra_21 * topo.dzdx1 + metric.H_contra_22 * topo.dzdx2)


   # Assemble the right-hand sides
   rhs = - metric.inv_sqrtG * ( df1dx1 + df2dx2 ) - forcing

   # TODO : debug
   rhs[:idx_first_tracer, :, :, :] = 0.

#   if filter_rhs:
#      for var in range(3):
#         rhs[var,:,:] = apply_filter(rhs[var,:,:], mtrx, nb_elements_horiz, nbsolpts)

   return rhs
