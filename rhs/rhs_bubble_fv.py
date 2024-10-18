from typing import Any, Callable, Union

import numpy

from common.definitions import idx_2d_rho, idx_2d_rho_u, idx_2d_rho_w, idx_2d_rho_theta,  \
                               p0, Rd, cpd, cvd, gravity
from geometry           import Cartesian2D
from .fluxes            import FluxFunction2D


# @profile
def rhs_bubble_fv(Q: numpy.ndarray[Any, numpy.dtype[Union[numpy.float64,numpy.complex128]]],
                  geom: Cartesian2D,
                  nb_elements_x: int,
                  nb_elements_z: int,
                  compute_flux: FluxFunction2D):

   nb_equations = Q.shape[0] # Number of constituent Euler equations.  Probably 6.

   df1_dx1 = numpy.empty_like(Q)
   df3_dx3 = numpy.empty_like(Q)

   kfaces_flux = numpy.empty((nb_equations, nb_elements_z, 2, nb_elements_x))
   kfaces_var  = numpy.empty((nb_equations, nb_elements_z, 2, nb_elements_x))

   ifaces_flux = numpy.empty((nb_equations, nb_elements_x, nb_elements_z, 2))
   ifaces_var  = numpy.empty((nb_equations, nb_elements_x, nb_elements_z, 2))

   flux_x1 = numpy.empty_like(Q)
   flux_x3 = numpy.empty_like(Q)

   # --- Unpack physical variables
   rho      = Q[idx_2d_rho,:,:]
   uu       = Q[idx_2d_rho_u,:,:] / rho
   ww       = Q[idx_2d_rho_w,:,:] / rho
   pressure = p0 * (Q[idx_2d_rho_theta,:,:] * Rd / p0)**(cpd / cvd)

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
   kfaces_var[:,:,0,:] = Q[:,:,:]
   kfaces_var[:,:,1,:] = Q[:,:,:]

   kfaces_flux[:,:,0,:] = flux_x3[:,:,:]
   kfaces_flux[:,:,1,:] = flux_x3[:,:,:]

   ifaces_var[:,:,:,0] = Q[:,:,:].transpose((0, 2, 1))
   ifaces_var[:,:,:,1] = ifaces_var[:,:,:,0]

   ifaces_flux[:,:,:,0] = flux_x1[:,:,:].transpose((0, 2, 1))
   ifaces_flux[:,:,:,1] = ifaces_flux[:, :, :, 0]

   # --- Interface pressure
   ifaces_pres = p0 * numpy.exp((cpd / cvd) * numpy.log((Rd / p0) * ifaces_var[idx_2d_rho_theta]))
   kfaces_pres = p0 * numpy.exp((cpd / cvd) * numpy.log((Rd / p0) * kfaces_var[idx_2d_rho_theta]))

   # --- Boundary treatement

   # zeros flux BCs everywhere ...
   kfaces_flux[:, 0, 0, :] = 0.0
   kfaces_flux[:,-1, 1, :] = 0.0

   ifaces_flux[:, 0, :, 0] = 0.0
   ifaces_flux[:,-1, :, 1] = 0.0

   # except for momentum eqs where pressure is extrapolated to BCs.
   kfaces_flux[idx_2d_rho_w, 0, 0, :] = kfaces_pres[ 0, 0, :]
   kfaces_flux[idx_2d_rho_w,-1, 1, :] = kfaces_pres[-1, 1, :]

   ifaces_flux[idx_2d_rho_u, 0, :, 0] = ifaces_pres[ 0, :, 0]  # TODO : pour les cas théoriques seulement ...
   ifaces_flux[idx_2d_rho_u,-1, :, 1] = ifaces_pres[-1, :, 1]

   ifaces_flux, kfaces_flux = compute_flux(Q, ifaces_var, ifaces_pres, ifaces_flux,
                                           kfaces_var, kfaces_pres, kfaces_flux)

   # --- Compute the derivatives
   df3_dx3[:, :, :] = (kfaces_flux[:, :, 1, :] - kfaces_flux[:, :, 0, :]) / geom.Δx3
   df1_dx1[:, :, :] = (ifaces_flux[:, :, :, 1] - ifaces_flux[:, :, :, 0]).transpose((0, 2, 1)) / geom.Δx1

   # --- Assemble the right-hand sides
   rhs = -(df1_dx1 + df3_dx3)

   rhs[idx_2d_rho_w,:,:] -= Q[idx_2d_rho,:,:] * gravity

   return rhs
