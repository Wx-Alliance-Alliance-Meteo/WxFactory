"""
A bunch of flux functions for our RHS functions
"""
from typing import Callable, Tuple

import numpy
import pdb

from common.definitions import cpd, cvd, heat_capacity_ratio, p0, Rd, \
                               idx_2d_rho, idx_2d_rho_u, idx_2d_rho_w, idx_2d_rho_theta, \
                               idx_rho, idx_rho_u1, idx_rho_u2, idx_rho_w

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


def AUSM_3d_vert(variables_itf_k, pressure_itf_k, w_itf_k, metric, nb_interfaces_vert, advection_only, \
                    flux_x3_itf_k, wflux_adv_x3_itf_k, wflux_pres_x3_itf_k):
   """Compute common Rusanov vertical fluxes"""
   nb_equations = variables_itf_k.shape[0]
   for itf in range(nb_interfaces_vert):

      elem_D = itf
      elem_U = itf + 1
      
      # Left state
      a_D = numpy.sqrt(metric.H_contra_33_itf_k[itf,:,:] * heat_capacity_ratio * pressure_itf_k[:, elem_D, 1, :] / variables_itf_k[idx_rho, :, elem_D, 1, :])
      M_D = variables_itf_k[idx_rho_w, :, elem_D, 1, :] / (variables_itf_k[idx_rho, :, elem_D, 1, :] * a_D)

      # Right state
      a_U = numpy.sqrt(metric.H_contra_33_itf_k[itf,:,:] * heat_capacity_ratio * pressure_itf_k[:, elem_U, 0, :] / variables_itf_k[idx_rho, :, elem_U, 0, :])
      M_U = variables_itf_k[idx_rho_w, :, elem_U, 0, :] / (variables_itf_k[idx_rho, :, elem_U, 0, :] * a_U)

      M = 0.25 * (( M_D + 1.)**2 - (M_U - 1.)**2)

      # Common AUSM flux
      for eqn in range (nb_equations):
         flux_x3_itf_k[eqn, :, elem_D, 1, :] = (metric.sqrtG_itf_k[itf, :, :] * variables_itf_k[eqn, :, elem_D, 1, :] * numpy.maximum(0., M) * a_D) + (metric.sqrtG_itf_k[itf, :, :] * variables_itf_k[eqn, :, elem_U, 0, :] * numpy.minimum(0., M) * a_U)

      # ... and now add the pressure contribution
      flux_x3_itf_k[idx_rho_u1, :, elem_D, 1, :] += metric.sqrtG_itf_k[itf,:, :] * metric.H_contra_31_itf_k[itf,:, :] * 0.5 * ( (1 + M_D) * pressure_itf_k[:, elem_D, 1, :] + (1 - M_U) * pressure_itf_k[:, elem_U, 0, :] )
      flux_x3_itf_k[idx_rho_u2, :, elem_D, 1, :] += metric.sqrtG_itf_k[itf,:, :] * metric.H_contra_32_itf_k[itf,:, :] * 0.5 * ( (1 + M_D) * pressure_itf_k[:, elem_D, 1, :] + (1 - M_U) * pressure_itf_k[:, elem_U, 0, :] )
      flux_x3_itf_k[idx_rho_w, :, elem_D, 1, :]  += metric.sqrtG_itf_k[itf,:, :] * metric.H_contra_33_itf_k[itf,:, :] * 0.5 * ( (1 + M_D) * pressure_itf_k[:, elem_D, 1, :] + (1 - M_U) * pressure_itf_k[:, elem_U, 0, :] )

      flux_x3_itf_k[:, :, elem_U, 0, :] = flux_x3_itf_k[:, :, elem_D, 1, :]

      # Separating advective and pressure fluxes for rho-w
      wflux_adv_x3_itf_k[:,elem_D,1,:]  = (metric.sqrtG_itf_k[itf,:, :] * variables_itf_k[idx_rho_w, :, elem_D, 1, :] * numpy.maximum(0, M) * a_D  +  metric.sqrtG_itf_k[itf,:, :] * variables_itf_k[idx_rho_w, :, elem_U, 0, :] * numpy.minimum(0., M) * a_U)
      wflux_adv_x3_itf_k[:,elem_U,0,:]  = wflux_adv_x3_itf_k[:,elem_D,1,:]
      wflux_pres_x3_itf_k[:,elem_D,1,:] = metric.sqrtG_itf_k[itf, :, :] * metric.H_contra_33_itf_k[itf, :, :] * 0.5 * ( (1 + M_D) * pressure_itf_k[:, elem_D, 1, :] + (1 - M_D) * pressure_itf_k[:, elem_U, 0, :] ) / pressure_itf_k[:,elem_D,1,:]
      wflux_pres_x3_itf_k[:,elem_U,0,:] = metric.sqrtG_itf_k[itf, :, :] * metric.H_contra_33_itf_k[itf, :, :] * 0.5 * ( (1 + M_D) * pressure_itf_k[:, elem_D, 1, :] + (1 - M_D) * pressure_itf_k[:, elem_U, 0, :] ) / pressure_itf_k[:,elem_U,0,:]




def AUSM_3d_hori_i(u1_itf_i, variables_itf_i, pressure_itf_i, metric, nb_interfaces_hori, advection_only, \
                      flux_x1_itf_i, wflux_adv_x1_itf_i, wflux_pres_x1_itf_i):

   nb_equations = variables_itf_i.shape[0]
   for itf in range(nb_interfaces_hori):

      elem_L = itf
      elem_R = itf + 1

      # Left state
      a_L = numpy.sqrt(metric.H_contra_11_itf_i[:,:,itf] * heat_capacity_ratio * pressure_itf_i[:, elem_L, 1, :] / variables_itf_i[idx_rho, :, elem_L, 1, :])
      M_L = variables_itf_i[idx_rho_u1, :, elem_L, 1, :] / (variables_itf_i[idx_rho, :, elem_L, 1, :] * a_L)

      # Right state
      a_R = numpy.sqrt(metric.H_contra_11_itf_i[:,:,itf] * heat_capacity_ratio * pressure_itf_i[:, elem_R, 0, :] / variables_itf_i[idx_rho, :, elem_R, 0, :])
      M_R = variables_itf_i[idx_rho_u1, :, elem_R, 0, :] / (variables_itf_i[idx_rho, :, elem_R, 0, :] * a_R)

      M = 0.25 * (( M_L + 1.)**2 - (M_R - 1.)**2)


      # Common AUSM flux
      for eqn in range (nb_equations):
         flux_x1_itf_i[eqn, :, elem_L, :, 1] = (metric.sqrtG_itf_i[:, :, itf] * variables_itf_i[eqn, :, elem_L, 1, :] * numpy.maximum(0., M) * a_L) + (metric.sqrtG_itf_i[:, :, itf] * variables_itf_i[eqn, :, elem_R, 0, :] * numpy.minimum(0., M) * a_R)

      # ... and now add the pressure contribution
      flux_x1_itf_i[idx_rho_u1, :, elem_L, :, 1] += metric.sqrtG_itf_i[:, :, itf] * metric.H_contra_11_itf_i[:, :, itf] * 0.5 * ( (1 + M_L) * pressure_itf_i[:, elem_L, 1, :] + (1 - M_R) * pressure_itf_i[:, elem_R, 0, :] )
      flux_x1_itf_i[idx_rho_u2, :, elem_L, :, 1] += metric.sqrtG_itf_i[:, :, itf] * metric.H_contra_12_itf_i[:, :, itf] * 0.5 * ( (1 + M_L) * pressure_itf_i[:, elem_L, 1, :] + (1 - M_R) * pressure_itf_i[:, elem_R, 0, :] )
      flux_x1_itf_i[idx_rho_w, :, elem_L, :, 1]  += metric.sqrtG_itf_i[:, :, itf] * metric.H_contra_13_itf_i[:, :, itf] * 0.5 * ( (1 + M_L) * pressure_itf_i[:, elem_L, 1, :] + (1 - M_R) * pressure_itf_i[:, elem_R, 0, :] )

      flux_x1_itf_i[:, :, elem_R, :, 0] = flux_x1_itf_i[:, :, elem_L, :, 1]


      # Separating advective and pressure fluxes for rho-w
      wflux_adv_x1_itf_i[:,elem_L,:,1]  = (metric.sqrtG_itf_i[:, :, itf] * variables_itf_i[idx_rho_w, :, elem_L, 1, :] * numpy.maximum(0, M) * a_L  +  metric.sqrtG_itf_i[:, :, itf] * variables_itf_i[idx_rho_w, :, elem_R, 0, :] * numpy.minimum(0., M) * a_R)
      wflux_adv_x1_itf_i[:,elem_R,:,0]  = wflux_adv_x1_itf_i[:,elem_L,:,1]
      wflux_pres_x1_itf_i[:,elem_L,:,1] = metric.sqrtG_itf_i[:, :, itf] * metric.H_contra_13_itf_i[:, :, itf] * 0.5 * ( (1 + M_L) * pressure_itf_i[:, elem_L, 1, :] + (1 - M_L) * pressure_itf_i[:, elem_R, 0, :] ) / pressure_itf_i[:,elem_L,1,:]
      wflux_pres_x1_itf_i[:,elem_R,:,0] = metric.sqrtG_itf_i[:, :, itf] * metric.H_contra_13_itf_i[:, :, itf] * 0.5 * ( (1 + M_L) * pressure_itf_i[:, elem_L, 1, :] + (1 - M_L) * pressure_itf_i[:, elem_R, 0, :] ) / pressure_itf_i[:,elem_R,0,:]


def AUSM_3d_hori_j(u2_itf_j, variables_itf_j, pressure_itf_j, metric, nb_interfaces_hori, advection_only, \
                      flux_x2_itf_j, wflux_adv_x2_itf_j, wflux_pres_x2_itf_j):
   nb_equations = variables_itf_j.shape[0]
   for itf in range(nb_interfaces_hori):

      elem_L = itf
      elem_R = itf + 1

      # Left state
      a_L = numpy.sqrt(metric.H_contra_22_itf_j[:,itf,:] * heat_capacity_ratio * pressure_itf_j[:, elem_L, 1, :] / variables_itf_j[idx_rho, :, elem_L, 1, :])
      M_L = variables_itf_j[idx_rho_u2, :, elem_L, 1, :] / (variables_itf_j[idx_2d_rho, :, elem_L, 1, :] * a_L)

      # Right state
      a_R = numpy.sqrt(metric.H_contra_11_itf_j[:,itf,:] * heat_capacity_ratio * pressure_itf_j[:, elem_R, 0, :] / variables_itf_j[idx_rho, :, elem_R, 0, :])
      M_R = variables_itf_j[idx_rho_u2, :, elem_R, 0, :] / (variables_itf_j[idx_rho, :, elem_R, 0, :] * a_R)

      M = 0.25 * (( M_L + 1.)**2 - (M_R - 1.)**2)

      # Common AUSM flux
      for eqn in range (nb_equations):
         flux_x2_itf_j[eqn, :, elem_L, 1, :] = (metric.sqrtG_itf_j[:, itf, :] * variables_itf_j[eqn, :, elem_L, 1, :] * numpy.maximum(0., M) * a_L) + (metric.sqrtG_itf_j[:, itf, :] * variables_itf_j[eqn, :, elem_R, 0, :] * numpy.minimum(0., M) * a_R)

      # ... and now add the pressure contribution
      flux_x2_itf_j[idx_rho_u1, :, elem_L, 1, :] += metric.sqrtG_itf_j[:, itf, :] * metric.H_contra_21_itf_j[:, itf, :] * 0.5 * ( (1 + M_L) * pressure_itf_j[:, elem_L, 1, :] + (1 - M_R) * pressure_itf_j[:, elem_R, 0, :] )
      flux_x2_itf_j[idx_rho_u2, :, elem_L, 1, :] += metric.sqrtG_itf_j[:, itf, :] * metric.H_contra_22_itf_j[:, itf, :] * 0.5 * ( (1 + M_L) * pressure_itf_j[:, elem_L, 1, :] + (1 - M_R) * pressure_itf_j[:, elem_R, 0, :] )
      flux_x2_itf_j[idx_rho_w, :, elem_L, 1, :]  += metric.sqrtG_itf_j[:, itf, :] * metric.H_contra_23_itf_j[:, itf, :] * 0.5 * ( (1 + M_L) * pressure_itf_j[:, elem_L, 1, :] + (1 - M_R) * pressure_itf_j[:, elem_R, 0, :] )

      flux_x2_itf_j[:, :, elem_R, 0, :] = flux_x2_itf_j[:, :, elem_L, 1, :]

      # Separating advective and pressure fluxes for rho-w
      wflux_adv_x2_itf_j[:,elem_L,1,:]  = (metric.sqrtG_itf_j[:, itf, :] * variables_itf_j[idx_rho_w, :, elem_L, 1, :] * numpy.maximum(0, M) * a_L  +  metric.sqrtG_itf_j[:, itf, :] * variables_itf_j[idx_rho_w, :, elem_R, 0, :] * numpy.minimum(0., M) * a_R)
      wflux_adv_x2_itf_j[:,elem_R,0,:]  = wflux_adv_x2_itf_j[:,elem_L,1,:]
      wflux_pres_x2_itf_j[:,elem_L,1,:] = metric.sqrtG_itf_j[:, itf, :] * metric.H_contra_23_itf_j[:, itf, :] * 0.5 * ( (1 + M_L) * pressure_itf_j[:, elem_L, 1, :] + (1 - M_L) * pressure_itf_j[:, elem_R, 0, :] ) / pressure_itf_j[:,elem_L,1,:]
      wflux_pres_x2_itf_j[:,elem_R,0,:] = metric.sqrtG_itf_j[:, itf, :] * metric.H_contra_23_itf_j[:, itf, :] * 0.5 * ( (1 + M_L) * pressure_itf_j[:, elem_L, 1, :] + (1 - M_L) * pressure_itf_j[:, elem_R, 0, :] ) / pressure_itf_j[:,elem_R,0,:]

