"""
A bunch of flux functions for our RHS functions
"""
from typing import Callable, Tuple

import numpy

from common.definitions import cpd, cvd, heat_capacity_ratio, p0, Rd, \
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

   del ifaces_pres, kfaces_pres

   icomm_flux = ifaces_flux.copy()
   kcomm_flux = kfaces_flux.copy()

   density_hori = ifaces_var[idx_2d_rho]
   u_hori       = ifaces_var[idx_2d_rho_u] / density_hori
   w_hori       = ifaces_var[idx_2d_rho_w] / density_hori

   pressure_hori    = p0 * (ifaces_var[idx_2d_rho_theta] * Rd / p0) ** heat_capacity_ratio
   energy_hori      = pressure_hori / (heat_capacity_ratio - 1.0) + (u_hori ** 2 + w_hori ** 2) * density_hori * 0.5
   enthalpy_hori    = (energy_hori + pressure_hori) / density_hori
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

   lambda_density = numpy.abs(u_roe_hori - sound_speed_roe_hori)
   lambda_u       = numpy.abs(u_roe_hori)
   lambda_theta   = numpy.abs(u_roe_hori + sound_speed_roe_hori)

   diff_density  = density_hori[:, 1] - density_hori[:, 0]
   diff_u        = u_hori[:, 1] - u_hori[:, 0]
   diff_w        = w_hori[:, 1] - w_hori[:, 0]
   diff_pressure = pressure_hori[:, 1] - pressure_hori[:, 0]

   # Entropy fix
   delta_lambda_1 = numpy.maximum((u_hori[:, 1] - sound_speed_hori[:, 1]) - (u_hori[:, 0] - sound_speed_hori[:, 0]),
                                  0.0)
   delta_lambda_2 = numpy.maximum((u_hori[:, 1] + sound_speed_hori[:, 1]) - (u_hori[:, 0] + sound_speed_hori[:, 0]),
                                  0.0)
   lambda_density[lambda_density < 2*delta_lambda_1] = lambda_density**2 / (4 * delta_lambda_1) + delta_lambda_1
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

   alpha_pressure = (diff_pressure / sound_speed_roe_hori - density_roe_h * diff_u) / (2.0 * sound_speed_roe_hori)
   alpha_u        = diff_density - diff_pressure / (sound_speed_roe_hori ** 2)
   alpha_w        = density_roe_h * diff_w
   alpha_theta    = (diff_pressure / sound_speed_roe_hori + density_roe_h * diff_u) * (2.0 * sound_speed_roe_hori)

   roe_vec = numpy.empty((4, 4))
   roe_vec[0, 0] = 1.0
   roe_vec[0, 1] = u_roe_hori - sound_speed_roe_hori
   roe_vec[0, 2] = w_roe_hori                           # Perpendicular to boundary
   roe_vec[0, 3] = enthalpy_roe_hori - sound_speed_roe_hori * u_roe_hori

   roe_vec[1, 0] = 1.0
   roe_vec[1, 1] = u_roe_hori
   roe_vec[1, 2] = w_roe_hori
   roe_vec[1, 3] = 0.5 * v2_roe_hori

   roe_vec[2, 0] = 0.0
   roe_vec[2, 1] = 0.0
   roe_vec[2, 2] = 1.0
   roe_vec[2, 3] = w_roe_hori

   roe_vec[3, 0] = 1.0
   roe_vec[3, 1] = u_roe_hori + sound_speed_roe_hori
   roe_vec[3, 2] = w_roe_hori
   roe_vec[3, 3] = enthalpy_roe_hori + sound_speed_roe_hori * u_roe_hori

   factor_left  = u_hori[:, 0]
   factor_right = u_hori[:, 1]

   flux = numpy.empty_like(icomm_flux[:, :, 0])
   flux[idx_2d_rho] = factor_left + factor_right
   flux[idx_2d_rho_u] = factor_left * u_hori[:, 0] + pressure_hori[:, 0]
   flux[idx_2d_rho_w] = factor_left * w_hori[:, 0]
   flux[idx_2d_rho_theta]= factor
   # //left and right physical fluxes
   # fac = ul[IRHO] * vnl;
   # f[IRHO]   = fac;
   # f[IRHOVX] = fac * vnl + pl;
   # f[IRHOVY] = fac * vtl;
   # f[IRHOE]  = fac * Hl;

   # fac = ur[IRHO] * vnr;
   # f[IRHO]   += fac;
   # f[IRHOVX] += fac * vnr + pr;
   # f[IRHOVY] += fac * vtr;
   # f[IRHOE]  += fac * Hr;



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
