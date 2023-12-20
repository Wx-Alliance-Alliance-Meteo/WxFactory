"""
A bunch of flux functions for our RHS functions
"""
from typing import Callable, Tuple

import numpy

from common.definitions import heat_capacity_ratio, p0, Rd, \
                               idx_2d_rho, idx_2d_rho_u, idx_2d_rho_w, idx_2d_rho_theta

FluxFunction2D = Callable[[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray,
                           numpy.ndarray, numpy.ndarray],
                          Tuple[numpy.ndarray, numpy.ndarray]]

def ausm_2d_fv(Q, ifaces_var, ifaces_pres, ifaces_flux, kfaces_var, kfaces_pres, kfaces_flux):
   del Q
   # --- Common AUSM fluxes --- vertical

   # Left
   a_L = numpy.sqrt(heat_capacity_ratio * kfaces_pres[:-1, 1, :] / kfaces_var[idx_2d_rho, :-1, 1, :])
   M_L = kfaces_var[idx_2d_rho_w, :-1, 1, :] / (kfaces_var[idx_2d_rho, :-1, 1, :] * a_L)

   # Right
   a_R = numpy.sqrt(heat_capacity_ratio * kfaces_pres[1:, 0, :] / kfaces_var[idx_2d_rho, 1:, 0, :])
   M_R = kfaces_var[idx_2d_rho_w, 1:, 0, :] / (kfaces_var[idx_2d_rho, 1:, 0, :] * a_R)

   # Mid
   M = 0.25 * (( M_L + 1.)**2 - (M_R - 1.)**2)

   kfaces_flux[:, 1:, 0, :] = \
      (kfaces_var[:, :-1, 1, :] * numpy.maximum(0., M) * a_L) + (kfaces_var[:, 1: , 0, :] * numpy.minimum(0., M) * a_R)
   kfaces_flux[idx_2d_rho_w, 1:, 0, :] += \
      0.5 * ((1. + M_L) * kfaces_pres[:-1,1,:] + (1. - M_R) * kfaces_pres[1:,0,:])
   kfaces_flux[:, :-1, 1, :] = kfaces_flux[:, 1:, 0, :]

   # --- Common AUSM fluxes --- horizontal

   # Left state
   a_L = numpy.sqrt(heat_capacity_ratio * ifaces_pres[:-1, :, 1] / ifaces_var[idx_2d_rho, :-1, :, 1])
   M_L = ifaces_var[idx_2d_rho_u, :-1, :, 1] / (ifaces_var[idx_2d_rho, :-1, :, 1] * a_L)

   # Right state
   a_R = numpy.sqrt(heat_capacity_ratio * ifaces_pres[1:, :, 0] / ifaces_var[idx_2d_rho, 1:, :, 0])
   M_R = ifaces_var[idx_2d_rho_u, 1:, :, 0] / ( ifaces_var[idx_2d_rho, 1:, :, 0] * a_R)

   M = 0.25 * ((M_L + 1.)**2 - (M_R - 1.)**2)

   ifaces_flux[:, 1:, :, 0] = \
      (ifaces_var[:, :-1, :, 1] * numpy.maximum(0., M) * a_L) + (ifaces_var[:, 1:, :, 0] * numpy.minimum(0., M) * a_R)
   ifaces_flux[idx_2d_rho_u, 1:, :, 0] += \
      0.5 * ((1. + M_L) * ifaces_pres[:-1, :, 1] + (1. - M_R) * ifaces_pres[1:, :, 0])
   ifaces_flux[:, :-1, :, 1] = ifaces_flux[:, 1:, :, 0]

   return (ifaces_flux, kfaces_flux)

tot = 0
def upwind_2d_fv(Q, ifaces_var, ifaces_pres, ifaces_flux, kfaces_var, kfaces_pres, kfaces_flux):

   del ifaces_pres, kfaces_pres

   icomm_flux = ifaces_flux.copy()
   kcomm_flux = kfaces_flux.copy()

   icomm_flux[:, 1:, :, 0]  = numpy.where(ifaces_var[idx_2d_rho_u, :-1, :, 1] + ifaces_var[idx_2d_rho_u, 1:, :, 0] > 0,
                                          ifaces_flux[:, :-1, :, 1],
                                          ifaces_flux[:, 1:, :, 0])
   # icomm_flux[:, 1:, :, 0] = 0.5 * (ifaces_flux[:, :-1, :, 1] + ifaces_flux[:, 1:, :, 0])
   icomm_flux[:, :-1, :, 1] = icomm_flux[:, 1:, :, 0]

   kcomm_flux[:, 1:, 0, :]  = numpy.where(kfaces_var[idx_2d_rho_w, :-1, 1, :] + kfaces_var[idx_2d_rho_w, 1:, 0, :] > 0,
                                           kfaces_flux[:, :-1, 1, :],
                                           kfaces_flux[:, 1:, 0, :])
   # kcomm_flux[:, 1:, 0, :] = 0.5 * (kfaces_flux[:, :-1, 1, :] + kfaces_flux[:, 1:, 0, :])
   kcomm_flux[:, :-1, 1, :] = kcomm_flux[:, 1:, 0, :]

   return icomm_flux, kcomm_flux

def rusanov_2d_fv(Q, ifaces_var, ifaces_pres, ifaces_flux, kfaces_var, kfaces_pres, kfaces_flux):
   # --- Common Rusanov fluxes

   icomm_flux = ifaces_flux.copy()
   kcomm_flux = kfaces_flux.copy()

   # print(f'Q shape = {Q.shape}')

   # Along x3
   eig_L = numpy.abs(kfaces_var[idx_2d_rho_w, :-1, 1, :] / kfaces_var[idx_2d_rho, :-1, 1, :]) \
         + numpy.sqrt(heat_capacity_ratio * kfaces_pres[:-1, 1, :] / kfaces_var[idx_2d_rho, :-1, 1, :])

   eig_R = numpy.abs(kfaces_var[idx_2d_rho_w, 1:, 0, :] / kfaces_var[idx_2d_rho, 1:, 0, :]) \
         + numpy.sqrt(heat_capacity_ratio * kfaces_pres[ 1:, 0, :] / kfaces_var[idx_2d_rho, 1:, 0, :])

   kcomm_flux[:, 1:, 0, :] = 0.5 * (kfaces_flux[:, :-1, 1, :] + kfaces_flux[:, 1:, 0, :] \
         - numpy.maximum(numpy.abs(eig_L), numpy.abs(eig_R)) * (kfaces_var[:, 1:, 0, :] - kfaces_var[:, :-1, 1, :]))
   kcomm_flux[:, :-1, 1, :] = kcomm_flux[:, 1:, 0, :]

   # Along x1
   eig_L = numpy.abs(ifaces_var[idx_2d_rho_u, :-1, :, 1] / ifaces_var[idx_2d_rho, :-1, :, 1]) \
         + numpy.sqrt(heat_capacity_ratio * ifaces_pres[:-1, :, 1] / ifaces_var[idx_2d_rho, :-1, :, 1])
   eig_R = numpy.abs(ifaces_var[idx_2d_rho_u, 1:, :, 0] / ifaces_var[idx_2d_rho, 1:, :, 0]) \
         + numpy.sqrt(heat_capacity_ratio * ifaces_pres[1:, :, 0] / ifaces_var[idx_2d_rho, 1:,:,0])

   icomm_flux[:, 1:, :, 0] = 0.5 * (ifaces_flux[:, :-1, :, 1] + ifaces_flux[:, 1:, :, 0] \
         - numpy.maximum(numpy.abs(eig_L), numpy.abs(eig_R)) * (ifaces_var[:, 1:, :, 0] - ifaces_var[:, :-1, :, 1]))
   icomm_flux[:, :-1, :, 1] = icomm_flux[:, 1:, :, 0]

   return icomm_flux, kcomm_flux

def roe_2d_fv(Q, ifaces_var, ifaces_pres, ifaces_flux, kfaces_var, kfaces_pres, kfaces_flux):
   """L^2-Roe flux function (Osswald et al.)
   """

   #TODO Treat 4th var (theta) like rho

   del ifaces_pres, kfaces_pres

   num_var, num_elem_x, num_elem_y = Q.shape       # Only 1 pt per element since we're doing FV
   i_r, i_u, i_w, i_t = idx_2d_rho, idx_2d_rho_u, idx_2d_rho_w, idx_2d_rho_theta

   if num_var != 4: raise ValueError(f'We\'re supposed to have 4 variables!')

   icomm_flux = ifaces_flux.copy()
   kcomm_flux = kfaces_flux.copy()


   unpacked_var = ifaces_var.copy()
   unpacked_var[i_u] /= unpacked_var[i_r]
   unpacked_var[i_w] /= unpacked_var[i_r]
   #unpacked_var[i_t] /= unpacked_var[i_r]     # Not used

   density_hori = unpacked_var[i_r]
   u_hori       = unpacked_var[i_u]
   w_hori       = unpacked_var[i_w]

   pressure_hori    = p0 * (ifaces_var[i_t] * Rd / p0) ** heat_capacity_ratio
   energy_hori      = pressure_hori / (heat_capacity_ratio - 1.0) + (u_hori ** 2 + w_hori ** 2) * density_hori * 0.5
   enthalpy_hori    = (density_hori * energy_hori + pressure_hori) / density_hori
   sound_speed_hori = numpy.sqrt(heat_capacity_ratio * pressure_hori / density_hori)
   mach_hori        = numpy.sqrt(u_hori ** 2 + w_hori ** 2) / sound_speed_hori

   # Roe averages
   density_roe_h = numpy.sqrt(density_hori[:, 0] * density_hori[:, 1])
   sqrt_rho_l    = numpy.sqrt(density_hori[:, 0])
   sqrt_rho_r    = numpy.sqrt(density_hori[:, 1])
   u_roe_hori    = (sqrt_rho_l * u_hori[:, 0] + sqrt_rho_r * u_hori[:, 1]) / (sqrt_rho_l + sqrt_rho_r)
   w_roe_hori    = (sqrt_rho_l * w_hori[:, 0] + sqrt_rho_r * w_hori[:, 1]) / (sqrt_rho_l + sqrt_rho_r)
   v2_roe_hori   = u_roe_hori ** 2 + w_roe_hori ** 2
   enthalpy_roe_hori    = (sqrt_rho_l * enthalpy_hori[:, 0] + sqrt_rho_r * enthalpy_hori[:, 1]) / \
                          (sqrt_rho_l + sqrt_rho_r)
   sound_speed_roe_hori = numpy.sqrt((heat_capacity_ratio - 1.0) * (enthalpy_roe_hori - 0.5 * v2_roe_hori))

   lam = numpy.empty((4, num_elem_x, num_elem_y))
   lam[i_r] = numpy.abs(u_roe_hori - sound_speed_roe_hori)
   lam[i_u] = numpy.abs(u_roe_hori)
   lam[i_w] = lam[i_u]
   lam[i_t] = numpy.abs(u_roe_hori + sound_speed_roe_hori)

   diff_density  = density_hori[:, 1] - density_hori[:, 0]
   diff_u        = u_hori[:, 1] - u_hori[:, 0]
   diff_w        = w_hori[:, 1] - w_hori[:, 0]
   diff_pressure = pressure_hori[:, 1] - pressure_hori[:, 0]

   # Entropy fix
   delta_lambda_1 = numpy.maximum((u_hori[:, 1] - sound_speed_hori[:, 1]) - (u_hori[:, 0] - sound_speed_hori[:, 0]),
                                  0.0)
   delta_lambda_2 = numpy.maximum((u_hori[:, 1] + sound_speed_hori[:, 1]) - (u_hori[:, 0] + sound_speed_hori[:, 0]),
                                  0.0)
   lambda_density = lam[i_r]
   lambda_density[lambda_density < 2*delta_lambda_1] = lambda_density**2 / (4 * delta_lambda_1) + delta_lambda_1
   lambda_theta = lam[i_t]
   lambda_theta[lambda_theta < 2*delta_lambda_2] = lambda_theta**2 / (4 * delta_lambda_2) + delta_lambda_2

   shock_fix = False
   if shock_fix:
      # Shock fix
      # TODO implement?
      pass
   else:
      # Low Mach fix
      factor = numpy.minimum(1.0, numpy.maximum(mach_hori[:, 0], mach_hori[:, 1]))
      diff_u *= factor
      diff_w *= factor

   alpha = numpy.empty_like(lam)
   alpha[i_r] = (diff_pressure / sound_speed_roe_hori - density_roe_h * diff_u) / (2.0 * sound_speed_roe_hori)
   alpha[i_u] = diff_density - diff_pressure / (sound_speed_roe_hori ** 2)
   alpha[i_w] = density_roe_h * diff_w
   alpha[i_t] = (diff_pressure / sound_speed_roe_hori + density_roe_h * diff_u) * (2.0 * sound_speed_roe_hori)

   roe_vec = numpy.empty((4, 4, num_elem_x, num_elem_y))
   roe_vec[i_r, i_r] = 1.0
   roe_vec[i_r, i_u] = u_roe_hori - sound_speed_roe_hori
   roe_vec[i_r, i_w] = w_roe_hori                           # Perpendicular to boundary
   roe_vec[i_r, i_t] = enthalpy_roe_hori - sound_speed_roe_hori * u_roe_hori

   roe_vec[i_u, i_r] = 1.0
   roe_vec[i_u, i_u] = u_roe_hori
   roe_vec[i_u, i_w] = w_roe_hori
   roe_vec[i_u, i_t] = 0.5 * v2_roe_hori

   roe_vec[i_w, i_r] = 0.0
   roe_vec[i_w, i_u] = 0.0
   roe_vec[i_w, i_w] = 1.0
   roe_vec[i_w, i_t] = w_roe_hori

   roe_vec[3, i_r] = 1.0
   roe_vec[3, i_u] = u_roe_hori + sound_speed_roe_hori
   roe_vec[3, i_w] = w_roe_hori
   roe_vec[3, i_t] = enthalpy_roe_hori + sound_speed_roe_hori * u_roe_hori

   factor_left  = u_hori[:, 0]
   factor_right = u_hori[:, 1]

   # Left/right physical fluxes
   flux = numpy.empty_like(icomm_flux[:, :, 0])
   flux[i_r] = factor_left
   flux[i_u] = factor_left * u_hori[:, 0] + pressure_hori[:, 0]
   flux[i_w] = factor_left * w_hori[:, 0]
   flux[i_t] = factor_left * enthalpy_hori[:, 0]   # Holds the total energy density E for now (rather than virtual potential temperature)

   flux[i_r] += factor_right
   flux[i_u] += factor_right * u_hori[:, 1] + pressure_hori
   flux[i_w] += factor_right * w_hori[:, 1]
   flux[i_t] += factor_right * enthalpy_hori[:, 1]

   # Numerical diffusion
   diffusion = numpy.zeros_like(flux)
   fac = alpha * lam
   # TODO Matrix-vector multiply instead? Would need to reorder the indices though
   for i in range(4):
      for j in range(4):
         diffusion[j] += fac[i] * roe_vec[i, j]

   flux[i_r] -= diffusion[i_r]
   u = flux[i_w] - diffusion [i_u] # Should copy i_w entries
   flux[i_w] = flux[i_u] - diffusion[i_w]
   flux[i_u] = u
   flux[i_t] -= diffusion[i_t]

   # Convert energy to virtual potential temperature

   # ---------------- vertical -------------------------

   density_vert = kfaces_var[idx_2d_rho]
   u_vert       = kfaces_var[idx_2d_rho_w] / density_vert
   w_vert       = kfaces_var[idx_2d_rho_u] / density_vert

   pressure_vert = p0 * (kfaces_var[idx_2d_rho_theta] * Rd / p0) ** heat_capacity_ratio
   energy_vert      = pressure_vert / (heat_capacity_ratio - 1.0) + (u_vert ** 2 + w_vert ** 2) * density_vert * 0.5
   enthalpy_vert    = (energy_vert + pressure_vert) / density_vert
   sound_speed_vert = numpy.sqrt(heat_capacity_ratio * pressure_vert / density_vert)
   mach_vert        = numpy.sqrt(u_vert ** 2 + w_vert ** 2) / sound_speed_vert

   return icomm_flux, kcomm_flux
