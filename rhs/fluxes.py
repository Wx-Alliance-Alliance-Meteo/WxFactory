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
