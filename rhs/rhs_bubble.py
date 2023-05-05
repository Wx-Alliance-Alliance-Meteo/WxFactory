import numpy
import cupy

from common.definitions import *

# This global variable is set to True if running on GPU. It can also be imported by other modules
is_gpu = False

@profile
def rhs_bubble(Q, geom, mtrx, nbsolpts, nb_elements_x, nb_elements_z):

   datatype = Q.dtype
   nb_equations = Q.shape[0] # Number of constituent Euler equations.  Probably 6.

   nb_interfaces_x = nb_elements_x + 1
   nb_interfaces_z = nb_elements_z + 1

   if is_gpu:       # If running on GPU, use cupy arrays
      Q = cupy.asarray(Q)
      if type(mtrx.extrap_down) != cupy.ndarray:
         # This is to mainly avoid re-allocating memory on GPU since these arrays are constant throughout
         # the simulation and are not modified. Since we are changing the objects of mtrx directly, they should
         # be updated outside the function scope as well.
         mtrx.extrap_down = cupy.asarray(mtrx.extrap_down)
         mtrx.extrap_up   = cupy.asarray(mtrx.extrap_up)
         mtrx.extrap_west = cupy.asarray(mtrx.extrap_west)
         mtrx.extrap_east = cupy.asarray(mtrx.extrap_east)
         mtrx.correction  = cupy.asarray(mtrx.correction)
         mtrx.diff_solpt = cupy.asarray(mtrx.diff_solpt)

   # use the appropriate array module
   xp = cupy.get_array_module(Q)

   flux_x1 = xp.empty_like(Q, dtype=datatype)
   flux_x3 = xp.empty_like(Q, dtype=datatype)

   df1_dx1 = xp.empty_like(Q, dtype=datatype)
   df3_dx3 = xp.empty_like(Q, dtype=datatype)

   kfaces_flux = xp.empty((nb_equations, nb_elements_z, 2, nbsolpts*nb_elements_x), dtype=datatype)
   kfaces_var  = xp.empty((nb_equations, nb_elements_z, 2, nbsolpts*nb_elements_x), dtype=datatype)

   ifaces_flux = xp.empty((nb_equations, nb_elements_x, nbsolpts*nb_elements_z, 2), dtype=datatype)
   ifaces_var  = xp.empty((nb_equations, nb_elements_x, nbsolpts*nb_elements_z, 2), dtype=datatype)

    # Start timer after we have allocated memory either on CPU or GPU
   """ tic = time.time() """
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
   # TODO: This could be computed concurrently when running on GPU. It might not make a huge difference on CPU though.
   # Could be a good test case for CUDA streams.
   for elem in range(nb_elements_z):
      epais = elem * nbsolpts + xp.arange(nbsolpts)

      kfaces_var[:,elem,0,:] = mtrx.extrap_down @ Q[:,epais,:]
      kfaces_var[:,elem,1,:] = mtrx.extrap_up @ Q[:,epais,:]

    # TODO: Same comment as above
   for elem in range(nb_elements_x):
      epais = elem * nbsolpts + xp.arange(nbsolpts)

      ifaces_var[:,elem,:,0] = Q[:,:,epais] @ mtrx.extrap_west
      ifaces_var[:,elem,:,1] = Q[:,:,epais] @ mtrx.extrap_east

   # TODO: Another area where operations could be done concurrently
   # --- Interface pressure
   ifaces_pres = p0 * (ifaces_var[idx_2d_rho_theta] * Rd / p0)**(cpd / cvd)
   kfaces_pres = p0 * (kfaces_var[idx_2d_rho_theta] * Rd / p0)**(cpd / cvd)

   # --- Bondary treatement

   # zeros flux BCs everywhere ...
   kfaces_flux[:,0,0,:]  = 0.0
   kfaces_flux[:,-1,1,:] = 0.0

   # Skip periodic faces
   if not geom.xperiodic:
      ifaces_flux[:, 0,:,0] = 0.0
      ifaces_flux[:,-1,:,1] = 0.0

   # except for momentum eqs where pressure is extrapolated to BCs.
   kfaces_flux[idx_2d_rho_w, 0, 0,:] = kfaces_pres[0,0,:]
   kfaces_flux[idx_2d_rho_w,-1,1,:]  = kfaces_pres[-1,1,:]

   ifaces_flux[idx_2d_rho_u, 0,:,0] = ifaces_pres[0,:,0]  # TODO : pour les cas théoriques seulement ...
   ifaces_flux[idx_2d_rho_u,-1,:,1] = ifaces_pres[-1,:,1]

   # --- Common AUSM fluxes
   for itf in range(1, nb_interfaces_z - 1):

      left  = itf - 1
      right = itf

      # Left state
      a_L   = xp.sqrt( heat_capacity_ratio * kfaces_pres[left,1,:] / kfaces_var[idx_2d_rho,left,1,:] )
      M_L   = kfaces_var[idx_2d_rho_w,left,1,:] / ( kfaces_var[idx_2d_rho,left,1,:] * a_L )

      # Right state
      a_R   = xp.sqrt( heat_capacity_ratio * kfaces_pres[right,0,:] / kfaces_var[idx_2d_rho,right,0,:] )
      M_R   = kfaces_var[idx_2d_rho_w,right,0,:] / ( kfaces_var[idx_2d_rho,right,0,:] * a_R )

      M = 0.25 * (( M_L + 1.)**2 - (M_R - 1.)**2)

      kfaces_flux[:,right,0,:] = (kfaces_var[:,left,1,:] * xp.maximum(0., M) * a_L) + (kfaces_var[:,right,0,:] * xp.minimum(0., M) * a_R)
      kfaces_flux[idx_2d_rho_w,right,0,:] += 0.5 * ( (1. + M_L) * kfaces_pres[left,1,:] + (1. - M_R) * kfaces_pres[right,0,:] )
   
      kfaces_flux[:,left,1,:] = kfaces_flux[:,right,0,:]

   if geom.xperiodic:
      ifaces_var[:, 0, :, :] = ifaces_var[:, -1, :, :]
      ifaces_pres[0, :, :]=ifaces_pres[-1, :, :]

   for itf in range(1, nb_interfaces_x - 1):

      left  = itf - 1
      right = itf

      # Left state

      if (geom.xperiodic and left == 0):
         left = -1

      a_L   = xp.sqrt( heat_capacity_ratio * ifaces_pres[left,:,1] / ifaces_var[idx_2d_rho,left,:,1] )
      M_L   = ifaces_var[idx_2d_rho_u,left,:,1] / ( ifaces_var[idx_2d_rho,left,:,1] * a_L )

      # Right state
      a_R   = xp.sqrt( heat_capacity_ratio * ifaces_pres[right,:,0] / ifaces_var[idx_2d_rho,right,:,0] )
      M_R   = ifaces_var[idx_2d_rho_u,right,:,0] / ( ifaces_var[idx_2d_rho,right,:,0] * a_R )

      M = 0.25 * ((M_L + 1.)**2 - (M_R - 1.)**2)

      ifaces_flux[:,right,:,0] = (ifaces_var[:,left,:,1] * xp.maximum(0., M) * a_L) + (ifaces_var[:,right,:,0] * xp.minimum(0., M) * a_R)
      ifaces_flux[idx_2d_rho_u,right,:,0] += 0.5 * ( (1. + M_L) * ifaces_pres[left,:,1] + (1. - M_R) * ifaces_pres[right,:,0] )

      ifaces_flux[:,left,:,1] = ifaces_flux[:,right,:,0]

   if geom.xperiodic:
      ifaces_flux[:, 0, :, :] = ifaces_flux[:, -1, :, :]

   # --- Compute the derivatives
   # TODO: Check carefully if the computation of the derivatives can be done concurrently
   for elem in range(nb_elements_z):
      epais = elem * nbsolpts + xp.arange(nbsolpts)

      df3_dx3[:,epais,:] = ( mtrx.diff_solpt @ flux_x3[:,epais,:] + mtrx.correction @ kfaces_flux[:,elem,:,:] ) * 2.0/geom.Δx3

    # TODO: Same comment as above
   for elem in range(nb_elements_x):
      epais = elem * nbsolpts + xp.arange(nbsolpts)

      df1_dx1[:,:,epais] = ( flux_x1[:,:,epais] @ mtrx.diff_solpt.T + ifaces_flux[:,elem,:,:] @ mtrx.correction.T ) * 2.0/geom.Δx1

   # --- Assemble the right-hand sides
   rhs = - ( df1_dx1 + df3_dx3 )

   rhs[idx_2d_rho_w,:,:] -= Q[idx_2d_rho,:,:] * gravity

   # Stop the timer once we have finished all the computations
   """
   toc = time.time()
   print(f'Time to compute fluxes and derivatives: {toc - tic} s')
   """
   if is_gpu:
       rhs = rhs.get()

   return rhs
