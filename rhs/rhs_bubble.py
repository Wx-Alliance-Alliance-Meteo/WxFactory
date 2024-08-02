import numpy

from common.definitions import idx_2d_rho, idx_2d_rho_u, idx_2d_rho_w, idx_2d_rho_theta,  \
                               p0, Rd, cpd, cvd, heat_capacity_ratio, gravity

def rhs_bubble(Q, geom, mtrx, nbsolpts, nb_elements_x, nb_elements_z):

   flux_x1 = numpy.empty_like(Q)
   flux_x3 = numpy.empty_like(Q)

   # --- Unpack physical variables
   rho      = Q[idx_2d_rho,:,:]
   uu       = Q[idx_2d_rho_u,:,:] / rho
   ww       = Q[idx_2d_rho_w,:,:] / rho
   pressure = p0 * numpy.exp((cpd/cvd) * numpy.log((Rd/p0)*Q[idx_2d_rho_theta, :, :]))

   # --- Compute the fluxes
   flux_x1[idx_2d_rho,:,:]       = Q[idx_2d_rho_u,:,:]
   flux_x1[idx_2d_rho_u,:,:]     = Q[idx_2d_rho_u,:,:] * uu + pressure
   flux_x1[idx_2d_rho_w,:,:]     = Q[idx_2d_rho_u,:,:] * ww
   flux_x1[idx_2d_rho_theta,:,:] = Q[idx_2d_rho_theta,:,:] * uu

   flux_x3[idx_2d_rho,:,:]       = Q[idx_2d_rho_w,:,:]
   flux_x3[idx_2d_rho_u,:,:]     = Q[idx_2d_rho_w,:,:] * uu
   flux_x3[idx_2d_rho_w,:,:]     = Q[idx_2d_rho_w,:,:] * ww + pressure
   flux_x3[idx_2d_rho_theta,:,:] = Q[idx_2d_rho_theta,:,:] * ww

   # --- Interpolate to the element interface
   var_itf_x = Q @ mtrx.extrap_x
   var_itf_z = Q @ mtrx.extrap_z

   # --- Interface pressure
   pressure_itf_x = p0 * (var_itf_x[idx_2d_rho_theta] * Rd / p0)**(cpd / cvd)
   sound_itf_x = numpy.sqrt(heat_capacity_ratio * pressure_itf_x / var_itf_x[idx_2d_rho])
   mach_itf_x = var_itf_x[idx_2d_rho_u] / (var_itf_x[idx_2d_rho] * sound_itf_x)

   pressure_itf_z = p0 * (var_itf_z[idx_2d_rho_theta] * Rd / p0)**(cpd / cvd)
   sound_itf_z = numpy.sqrt(heat_capacity_ratio * pressure_itf_z / var_itf_z[idx_2d_rho])
   mach_itf_z = var_itf_z[idx_2d_rho_w] / (var_itf_z[idx_2d_rho] * sound_itf_z)

   # --- Common AUSM fluxes
   common_flux_z = numpy.empty_like(var_itf_z)

   bot_itf  = numpy.arange(nbsolpts)
   top_itf = numpy.arange(nbsolpts, 2*nbsolpts)

   # ------ Vertical
   a_T = sound_itf_z[nb_elements_x:, bot_itf]
   a_B = sound_itf_z[:-nb_elements_x, top_itf]

   M_T = mach_itf_z[nb_elements_x:, bot_itf]
   M_B = mach_itf_z[:-nb_elements_x, top_itf]

   M   = 0.25 * ((M_B + 1.)**2 - (M_T - 1.)**2)

   common_flux_z[:, nb_elements_x:, bot_itf] = (var_itf_z[:, :-nb_elements_x, top_itf] * numpy.maximum(0., M) * a_B) + \
                                             (var_itf_z[:, nb_elements_x:, bot_itf] * numpy.minimum(0., M) * a_T)
   common_flux_z[idx_2d_rho_w, nb_elements_x:, bot_itf] = 0.5 * ((1. + M_B) * pressure_itf_z[:-nb_elements_x, top_itf] + \
                                                               (1. - M_T) * pressure_itf_z[nb_elements_x:, bot_itf]).T
   common_flux_z[:, :-nb_elements_x, top_itf] = common_flux_z[:, nb_elements_x:, bot_itf]

   # Zero flux BC at bottom and top boundaries
   # except for (vertical) momentum, where we use extrapolated pressure at boundary
   common_flux_z[:,  :nb_elements_x, bot_itf] = 0.0
   common_flux_z[:, -nb_elements_x:, top_itf] = 0.0
   common_flux_z[idx_2d_rho_w,  :nb_elements_x, bot_itf] = pressure_itf_z[ :nb_elements_x, bot_itf].T
   common_flux_z[idx_2d_rho_w, -nb_elements_x:, top_itf] = pressure_itf_z[-nb_elements_x:, top_itf].T

   # ------ Horizontal
   common_flux_x = numpy.empty_like(var_itf_x)
   west_itf  = bot_itf
   east_itf = top_itf

   a_W = sound_itf_x[:-1, east_itf]
   a_E = sound_itf_x[1:, west_itf]

   M_W = mach_itf_x[:-1, east_itf]
   M_E = mach_itf_x[1:, west_itf]

   M = 0.25 * ((M_W + 1.)**2 - (M_E - 1.)**2)

   common_flux_x[:, 1:, west_itf] = (var_itf_x[:, :-1, east_itf] * numpy.maximum(0., M) * a_W) + \
                                  (var_itf_x[:, 1:, west_itf] * numpy.minimum(0., M) * a_E)
   common_flux_x[idx_2d_rho_u, 1:, west_itf] = 0.5 * ((1. + M_W) * pressure_itf_x[:-1, east_itf] + \
                                                    (1. - M_E) * pressure_itf_x[1:, west_itf]).T
   common_flux_x[:, :-1, east_itf] = common_flux_x[:, 1:, west_itf]

   # Boundary conditions
   n0 = nb_elements_x - 1
   jump = nb_elements_x
   common_flux_x[:,   ::jump, west_itf] = 0.0
   common_flux_x[:, n0::jump, east_itf] = 0.0
   common_flux_x[idx_2d_rho_u,   ::jump, west_itf] = pressure_itf_x[  ::jump, west_itf].T
   common_flux_x[idx_2d_rho_u, n0::jump, east_itf] = pressure_itf_x[n0::jump, east_itf].T


   df1_dx1 = (flux_x1 @ mtrx.derivative_x + common_flux_x @ mtrx.correction_WE) * (2.0 / geom.Δx1) 
   df3_dx3 = (flux_x3 @ mtrx.derivative_z + common_flux_z @ mtrx.correction_DU) * (2.0 / geom.Δx3)

   # --- Assemble the right-hand sides
   rhs = - ( df1_dx1 + df3_dx3 )

   rhs[idx_2d_rho_w,:,:] -= Q[idx_2d_rho,:,:] * gravity

   return rhs
