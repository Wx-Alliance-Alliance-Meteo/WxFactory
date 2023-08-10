from mpi4py import MPI
import numpy

from common.definitions import idx_2d_rho, idx_2d_rho_u, idx_2d_rho_w, idx_2d_rho_theta,  \
                               p0, Rd, cpd, cvd, heat_capacity_ratio, gravity

def rhs_bubble(Q, geom, mtrx, nbsolpts, nb_elements_x, nb_elements_z):

   datatype = Q.dtype
   nb_equations = Q.shape[0] # Number of constituent Euler equations.  Probably 6.

   nb_interfaces_x = nb_elements_x + 1
   nb_interfaces_z = nb_elements_z + 1

   flux_x1 = numpy.empty_like(Q, dtype=datatype)
   flux_x3 = numpy.empty_like(Q, dtype=datatype)

   df1_dx1 = numpy.empty_like(Q, dtype=datatype)
   df3_dx3 = numpy.empty_like(Q, dtype=datatype)

   kfaces_flux = numpy.empty((nb_equations, nb_elements_z, 2, nbsolpts*nb_elements_x), dtype=datatype)
   kfaces_var  = numpy.empty((nb_equations, nb_elements_z, 2, nbsolpts*nb_elements_x), dtype=datatype)

   ifaces_flux = numpy.empty((nb_equations, nb_elements_x + 2, nbsolpts*nb_elements_z, 2), dtype=datatype)
   ifaces_var  = numpy.ones((nb_equations, nb_elements_x + 2, nbsolpts*nb_elements_z, 2), dtype=datatype)

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
   for elem in range(nb_elements_z):
      epais = elem * nbsolpts + numpy.arange(nbsolpts)

      kfaces_var[:,elem,0,:] = mtrx.extrap_down @ Q[:,epais,:]
      kfaces_var[:,elem,1,:] = mtrx.extrap_up @ Q[:,epais,:]

   for elem in range(nb_elements_x):
      epais = elem * nbsolpts + numpy.arange(nbsolpts)

      ifaces_var[:,elem+1,:,0] = Q[:,:,epais] @ mtrx.extrap_west
      ifaces_var[:,elem+1,:,1] = Q[:,:,epais] @ mtrx.extrap_east

   if MPI.COMM_WORLD.size == 1:
      ifaces_var[:, -1, :, :] = ifaces_var[:,  1, :, :]
      ifaces_var[:,  0, :, :] = ifaces_var[:, -2, :, :]
   else:
      rank = MPI.COMM_WORLD.rank
      next = (rank + 1) % MPI.COMM_WORLD.size
      prev = (rank - 1) % MPI.COMM_WORLD.size

      buffer = numpy.empty_like(ifaces_var[:, 0, :, :])
      if rank % 2 == 0:
         # To/from right
         buffer[:] = ifaces_var[:, -2, :, :]; MPI.COMM_WORLD.Send(buffer, dest=next, tag=rank)
         MPI.COMM_WORLD.Recv(buffer, source=next, tag=rank); ifaces_var[:, -1, :, :] = buffer

         # To/from left
         buffer[:] = ifaces_var[:, 1, :, :]; MPI.COMM_WORLD.Send(buffer, dest=prev, tag=rank)
         MPI.COMM_WORLD.Recv(buffer, source=prev, tag=rank); ifaces_var[:, 0, :, :] = buffer

      else:
         # From/to left
         MPI.COMM_WORLD.Recv(buffer, source=prev, tag=prev); ifaces_var[:, 0, :, :] = buffer
         buffer[:] = ifaces_var[:, 1, :, :]; MPI.COMM_WORLD.Send(buffer, dest=prev, tag=prev)

         # From/to right
         MPI.COMM_WORLD.Recv(buffer, source=next, tag=next); ifaces_var[:, -1, :, :] = buffer
         buffer[:] = ifaces_var[:, -2, :, :]; MPI.COMM_WORLD.Send(buffer, dest=next, tag=next)

   # --- Interface pressure
   ifaces_pres = p0 * (ifaces_var[idx_2d_rho_theta] * Rd / p0)**(cpd / cvd)
   kfaces_pres = p0 * (kfaces_var[idx_2d_rho_theta] * Rd / p0)**(cpd / cvd)

   # --- Boundary treatement

   # zeros flux BCs everywhere ...
   kfaces_flux[:,0,0,:]  = 0.0
   kfaces_flux[:,-1,1,:] = 0.0

   # Skip periodic faces
   if not geom.xperiodic:
      if MPI.COMM_WORLD.rank == 0:
         ifaces_flux[:, 1,:,0] = 0.0
      if MPI.COMM_WORLD.rank == MPI.COMM_WORLD.size - 1:
         ifaces_flux[:,-2,:,1] = 0.0

   # except for momentum eqs where pressure is extrapolated to BCs.
   kfaces_flux[idx_2d_rho_w, 0, 0, :] = kfaces_pres[ 0, 0, :]
   kfaces_flux[idx_2d_rho_w,-1, 1, :] = kfaces_pres[-1, 1, :]

   if MPI.COMM_WORLD.rank == 0:
      ifaces_flux[idx_2d_rho_u,  1, :, 0] = ifaces_pres[ 1, :, 0]  # TODO : pour les cas théoriques seulement ...
   if MPI.COMM_WORLD.rank == MPI.COMM_WORLD.size - 1:
      ifaces_flux[idx_2d_rho_u, -2, :, 1] = ifaces_pres[-2, :, 1]

   # ifaces_flux[idx_2d_rho_u,  0, :, :] = ifaces_flux[idx_2d_rho_u, -2, : , :]
   # ifaces_flux[idx_2d_rho_u, -1, :, :] = ifaces_flux[idx_2d_rho_u,  1, : , :]

   # --- Vertical AUSM fluxes
   for itf in range(1, nb_interfaces_z - 1):

      left  = itf - 1
      right = itf

      # Left state
      a_L = numpy.sqrt(heat_capacity_ratio * kfaces_pres[left, 1, :] / kfaces_var[idx_2d_rho, left, 1, :])
      M_L = kfaces_var[idx_2d_rho_w, left, 1, :] / (kfaces_var[idx_2d_rho, left, 1, :] * a_L)

      # Right state
      a_R = numpy.sqrt(heat_capacity_ratio * kfaces_pres[right, 0, :] / kfaces_var[idx_2d_rho, right, 0, :])
      M_R = kfaces_var[idx_2d_rho_w, right, 0, :] / (kfaces_var[idx_2d_rho, right, 0, :] * a_R)

      M = 0.25 * (( M_L + 1.)**2 - (M_R - 1.)**2)

      kfaces_flux[:,right,0,:] = (kfaces_var[:,left,1,:] * numpy.maximum(0., M) * a_L) + \
                                 (kfaces_var[:,right,0,:] * numpy.minimum(0., M) * a_R)
      kfaces_flux[idx_2d_rho_w,right,0,:] += 0.5 * ((1. + M_L) * kfaces_pres[left,1,:] + \
                                                    (1. - M_R) * kfaces_pres[right,0,:])

      kfaces_flux[:,left,1,:] = kfaces_flux[:,right,0,:]

   # --- Horizontal AUSM fluxes
   start = 1
   end = nb_interfaces_x + 1
   if (not geom.xperiodic) and MPI.COMM_WORLD.rank == MPI.COMM_WORLD.size - 1: end -= 1
   if (not geom.xperiodic) and MPI.COMM_WORLD.rank == 0: start += 1
   for itf in range(start, end):

      left  = itf - 1
      right = itf

      # Left state
      a_L = numpy.sqrt(heat_capacity_ratio * ifaces_pres[left, :, 1] / ifaces_var[idx_2d_rho, left, :, 1])
      M_L = ifaces_var[idx_2d_rho_u, left, :, 1] / (ifaces_var[idx_2d_rho, left, :, 1] * a_L)

      # Right state
      a_R = numpy.sqrt(heat_capacity_ratio * ifaces_pres[right, :, 0] / ifaces_var[idx_2d_rho, right, :, 0])
      M_R = ifaces_var[idx_2d_rho_u, right, :, 0] / ( ifaces_var[idx_2d_rho, right, :, 0] * a_R)

      M = 0.25 * ((M_L + 1.)**2 - (M_R - 1.)**2)

      ifaces_flux[:,right,:,0] = (ifaces_var[:,left,:,1] * numpy.maximum(0., M) * a_L) + \
                                 (ifaces_var[:,right,:,0] * numpy.minimum(0., M) * a_R)
      ifaces_flux[idx_2d_rho_u,right,:,0] += 0.5 * ((1. + M_L) * ifaces_pres[left,:,1] + \
                                                    (1. - M_R) * ifaces_pres[right,:,0])

      ifaces_flux[:,left,:,1] = ifaces_flux[:,right,:,0]

   # --- Compute the derivatives
   for elem in range(nb_elements_z):
      epais = elem * nbsolpts + numpy.arange(nbsolpts)

      if geom.bottom_layer_delta > 0:
         if elem < geom.nb_elements_bottom_layer:
            df3_dx3[:, epais, :] = (mtrx.diff_solpt @ flux_x3[:, epais, :] + \
                                    mtrx.correction @ kfaces_flux[:, elem, :, :]) * (2.0 / geom.bottom_layer_delta)
         else:
            df3_dx3[:, epais, :] = (mtrx.diff_solpt @ flux_x3[:, epais, :] + \
                                    mtrx.correction @ kfaces_flux[:, elem, :, :]) * (2.0 / geom.Δx3)
      else:
         df3_dx3[:, epais, :] = (mtrx.diff_solpt @ flux_x3[:, epais, :] + mtrx.correction @ kfaces_flux[:, elem, :,:]) \
                                * 2.0 / geom.Δx3

   for elem in range(nb_elements_x):
      epais = elem * nbsolpts + numpy.arange(nbsolpts)

      df1_dx1[:,:,epais] = (flux_x1[:,:,epais] @ mtrx.diff_solpt.T + ifaces_flux[:,elem+1,:,:] @ mtrx.correction.T) * \
                           2.0/geom.Δx1

   # --- Assemble the right-hand sides
   rhs = - ( df1_dx1 + df3_dx3 )

   rhs[idx_2d_rho_w,:,:] -= Q[idx_2d_rho,:,:] * gravity

   # TODO : Add sources terms for Brikman penalization
   # It may be better to do this elementwise...
   if len(geom.chiMask) > 1:
      etac = 1 # 1e-1
      for item in geom.chiMask:
         k=item[0]
         i=item[1]
         if item in geom.chiMaskBoundary:
            nrmluw = geom.terrainNormalXcmp[k, i] * df1_dx1[idx_2d_rho_u, k, i] + \
                     geom.terrainNormalZcmp[k, i] * df3_dx3[idx_2d_rho_w, k, i]
            rhs[idx_2d_rho_u, k, i] = -(1 / etac) * nrmluw * geom.terrainNormalXcmp[k, i]
            rhs[idx_2d_rho_w, k, i] = -(1 / etac) * nrmluw * geom.terrainNormalZcmp[k, i]

         else:
            rhs[idx_2d_rho_u, k, i] = 0
            rhs[idx_2d_rho_w, k, i] = 0

   return rhs
