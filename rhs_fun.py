import numpy

from constants import *

def rhs_fun(Q, geom, mtrx, metric, nbsolpts, nb_elements_x1, nb_elements_x2, α):

   type_vec = type(Q[1,1,1])

   nb_interfaces_x1 = nb_elements_x1 - 1
   nb_interfaces_x2 = nb_elements_x2 - 1

   flux_x1          = numpy.zeros_like(Q, dtype=type_vec)
   flux_x2          = numpy.zeros_like(Q, dtype=type_vec)

   df1_dx1          = numpy.zeros_like(Q, dtype=type_vec)
   df2_dx2          = numpy.zeros_like(Q, dtype=type_vec)
   dQ_dx1           = numpy.zeros_like(Q, dtype=type_vec)
   dQ_dx2           = numpy.zeros_like(Q, dtype=type_vec)
   Q_xx             = numpy.zeros_like(Q, dtype=type_vec)
   Q_yy             = numpy.zeros_like(Q, dtype=type_vec)
   upwind_diffusion = numpy.zeros_like(Q, dtype=type_vec)

   jfaces_flux      = numpy.zeros((2, nbsolpts*nb_elements_x1, nb_equations, nb_elements_x2), dtype=type_vec)
   jfaces_var       = numpy.zeros((2, nbsolpts*nb_elements_x1, nb_equations, nb_elements_x2), dtype=type_vec)
   jfaces_diffusion = numpy.zeros((2, nbsolpts*nb_elements_x1, nb_equations, nb_elements_x2), dtype=type_vec)

   ifaces_flux      = numpy.zeros((nbsolpts*nb_elements_x2, 2, nb_equations, nb_elements_x1), dtype=type_vec)
   ifaces_var       = numpy.zeros((nbsolpts*nb_elements_x2, 2, nb_equations, nb_elements_x1), dtype=type_vec)
   ifaces_diffusion = numpy.zeros((nbsolpts*nb_elements_x2, 2, nb_equations, nb_elements_x1), dtype=type_vec)

   jfaces_pres = numpy.zeros((2, nbsolpts*nb_elements_x1, nb_elements_x2), dtype=type_vec)
   ifaces_pres = numpy.zeros((nbsolpts*nb_elements_x2, 2, nb_elements_x1), dtype=type_vec)

   # Unpack physical variables
   h  = Q[:,:,0]
   u1 = Q[:,:,1] / Q[:,:,0]
   u2 = Q[:,:,2] / Q[:,:,0]

   # Compute the fluxes
   flux_x1[:,:,0] = Q[:,:,1]
   flux_x1[:,:,1] = Q[:,:,1] * u1 + 0.5 * metric.H_contra_11 * gravity * Q[:,:,0]**2
   flux_x1[:,:,2] = Q[:,:,1] * u2 + 0.5 * metric.H_contra_12 * gravity * Q[:,:,0]**2

   flux_x2[:,:,0] = Q[:,:,2]
   flux_x2[:,:,1] = Q[:,:,2] * u1 + 0.5 * metric.H_contra_21 * gravity * Q[:,:,0]**2
   flux_x2[:,:,2] = Q[:,:,2] * u2 + 0.5 * metric.H_contra_22 * gravity * Q[:,:,0]**2

   # Interpolate to the element interface
   for elem in range(nb_elements_x2):
      epais = elem * nbsolpts + numpy.arange(nbsolpts)

      for eq in range(nb_equations):
         jfaces_flux[0,:,eq,elem] = mtrx.lcoef @ flux_x2[epais,:,eq]
         jfaces_flux[1,:,eq,elem] = mtrx.rcoef @ flux_x2[epais,:,eq]

         jfaces_var[0,:,eq,elem] = mtrx.lcoef @ Q[epais,:,eq]
         jfaces_var[1,:,eq,elem] = mtrx.rcoef @ Q[epais,:,eq]

   for elem in range(nb_elements_x1):
      epais = elem * nbsolpts + numpy.arange(nbsolpts)

      for eq in range(nb_equations):
         ifaces_flux[:,0,eq,elem] = flux_x1[:,epais,eq] @ mtrx.lcoef
         ifaces_flux[:,1,eq,elem] = flux_x1[:,epais,eq] @ mtrx.rcoef

         ifaces_var[:,0,eq,elem] = Q[:,epais,eq] @ mtrx.lcoef
         ifaces_var[:,1,eq,elem] = Q[:,epais,eq] @ mtrx.rcoef

   print('TODO : halo exchange');exit(0)

   # Bondary treatement
   # zeros flux BCs everywhere ...
   jfaces_flux[0,:,:,0]  = 0.0
   jfaces_flux[1,:,:,-1] = 0.0

   ifaces_flux[:,0,:,0]  = 0.0
   ifaces_flux[:,1,:,-1] = 0.0

   jfaces_diffusion[0,:,:,0]  = 0.0
   jfaces_diffusion[1,:,:,-1] = 0.0

   ifaces_diffusion[:,0,:,0]  = 0.0
   ifaces_diffusion[:,1,:,-1] = 0.0

   # ... except for momentum eqs where pressure is extrapolated to BCs.
   jfaces_flux[0,:,RHO_W,0]  = jfaces_pres[0,:,0]
   jfaces_flux[1,:,RHO_W,-1] = jfaces_pres[1,:,-1]

   ifaces_flux[:,0,RHO_U,0]   = ifaces_pres[:,0,0]  # TODO : theo seulement ...
   ifaces_flux[:,1,RHO_U,-1] = ifaces_pres[:,1,-1]

   # Common Rusanov fluxes
   for itf in range(nb_interfaces_x2):

      eig_L = numpy.abs(jfaces_var[1,:,RHO_W,itf] / jfaces_var[1,:,RHO,itf]) \
            + numpy.sqrt( heat_capacity_ratio * jfaces_pres[1,:,itf] / jfaces_var[1,:,RHO,itf]  )

      eig_R = numpy.abs(jfaces_var[0,:,RHO_W,itf+1] / jfaces_var[0,:,RHO,itf+1]) \
            + numpy.sqrt( heat_capacity_ratio * jfaces_pres[0,:,itf+1] / jfaces_var[0,:,RHO,itf+1])

      for eq in range(nb_equations):
         jfaces_flux[0,:,eq,itf+1] = 0.5 * ( jfaces_flux[1,:,eq,itf] + jfaces_flux[0,:,eq,itf+1] \
               - numpy.maximum(numpy.abs(eig_L), numpy.abs(eig_R)) * ( jfaces_var[0,:,eq,itf+1] - jfaces_var[1,:,eq,itf] ) )

      jfaces_flux[1,:,:,itf] = jfaces_flux[0,:,:,itf+1]

      jfaces_diffusion[0,:,:,itf+1] = 0.5 * ( jfaces_var[1,:,:,itf] + jfaces_var[0,:,:,itf+1] )
      jfaces_diffusion[1,:,:,itf]   = jfaces_diffusion[0,:,:,itf+1]

   for itf in range(nb_interfaces_x1):

      eig_L = numpy.abs(ifaces_var[:,1,RHO_U,itf] / ifaces_var[:,1,RHO,itf]) \
            + numpy.sqrt( heat_capacity_ratio * ifaces_pres[:,1,itf] / ifaces_var[:,1,RHO,itf] )
      eig_R = numpy.abs(ifaces_var[:,0,RHO_U,itf+1] / ifaces_var[:,0,RHO,itf+1]) \
            + numpy.sqrt( heat_capacity_ratio * ifaces_pres[:,0,itf+1] / ifaces_var[:,0,RHO,itf+1] )

      for eq in range(nb_equations):
         ifaces_flux[:,0,eq,itf+1] = 0.5 * ( ifaces_flux[:,1,eq,itf] + ifaces_flux[:,0,eq,itf+1] \
               - numpy.maximum(numpy.abs(eig_L), numpy.abs(eig_R)) * ( ifaces_var[:,0,eq,itf+1] - ifaces_var[:,1,eq,itf] ) )

      ifaces_flux[:,1,:,itf] = ifaces_flux[:,0,:,itf+1]

      ifaces_diffusion[:,0,:,itf+1] = 0.5 * ( ifaces_var[:,1,:,itf] + ifaces_var[:,0,:,itf+1] )
      ifaces_diffusion[:,1,:,itf]   = ifaces_diffusion[:,0,:,itf+1]

   # Compute the derivatives
   for elem in range(nb_elements_x2):
      epais = elem * nbsolpts + numpy.arange(nbsolpts)

      for eq in range(nb_equations):
         df2_dx2[epais,:,eq] = ( mtrx.diff_solpt @ flux_x2[epais,:,eq] + mtrx.correction @ jfaces_flux[:,:,eq,elem] ) * 2.0/geom.Δz
         dQ_dx2[epais,:,eq]  = ( mtrx.diff_solpt @ Q[epais,:,eq] + mtrx.correction @ jfaces_diffusion[:,:,eq,elem] ) * 2.0/geom.Δz

   for elem in range(nb_elements_x1):
      epais = elem * nbsolpts + numpy.arange(nbsolpts)

      for eq in range(nb_equations):
         df1_dx1[:,epais,eq] = ( flux_x1[:,epais,eq] @ mtrx.diff_solpt.T + ifaces_flux[:,:,eq,elem] @ mtrx.correction.T ) * 2.0/geom.Δx
         dQ_dx1[:,epais,eq]  = ( Q[:,epais,eq] @ mtrx.diff_solpt.T + ifaces_diffusion[:,:,eq,elem] @ mtrx.correction.T ) * 2.0/geom.Δx

   # Interpolate first derivative of diffusion to the interface
   for elem in range(nb_elements_x2):
      epais = elem * nbsolpts + numpy.arange(nbsolpts)

      for eq in range(nb_equations):
         jfaces_var[0,:,eq,elem] = mtrx.lcoef @ dQ_dx2[epais,:,eq]
         jfaces_var[1,:,eq,elem] = mtrx.rcoef @ dQ_dx2[epais,:,eq]

   for elem in range(nb_elements_x1):
      epais = elem * nbsolpts + numpy.arange(nbsolpts)

      for eq in range(nb_equations):
         ifaces_var[:,0,eq,elem] = dQ_dx1[:,epais,eq] @ mtrx.lcoef
         ifaces_var[:,1,eq,elem] = dQ_dx1[:,epais,eq] @ mtrx.rcoef

   # Communication at cell interfaces with central flux
   for itf in range(nb_interfaces_x2):
      jfaces_diffusion[0,:,:,itf+1] = 0.5 * ( jfaces_var[1,:,:,itf] + jfaces_var[0,:,:,itf+1] )
      jfaces_diffusion[1,:,:,itf]   = jfaces_diffusion[0,:,:,itf+1]

   for itf in range(nb_interfaces_x1):
      ifaces_diffusion[:,0,:,itf+1] = 0.5 * ( ifaces_var[:,1,:,itf] + ifaces_var[:,0,:,itf+1] )
      ifaces_diffusion[:,1,:,itf]   = ifaces_diffusion[:,0,:,itf+1]

   # Bondary treatement (this should be equivalent to a zero diffusion coefficient at the boundary)
   jfaces_diffusion[0,:,:,0]  = 0.0
   jfaces_diffusion[1,:,:,-1] = 0.0

   ifaces_diffusion[:,0,:,0]  = 0.0
   ifaces_diffusion[:,1,:,-1] = 0.0
#
   # Finalize the diffusion operator
   for elem in range(nb_elements_x2):
      epais = elem * nbsolpts + numpy.arange(nbsolpts)

      for eq in range(nb_equations):
         Q_yy[epais,:,eq] = ( mtrx.diff_solpt @ dQ_dx2[epais,:,eq] + mtrx.correction @ jfaces_diffusion[:,:,eq,elem] ) * 2.0/geom.Δz

   for elem in range(nb_elements_x1):
      epais = elem * nbsolpts + numpy.arange(nbsolpts)

      for eq in range(nb_equations):
         Q_xx[:,epais,eq] = ( dQ_dx1[:,epais,eq] @ mtrx.diff_solpt.T + ifaces_diffusion[:,:,eq,elem] @ mtrx.correction.T ) * 2.0/geom.Δx

   for eq in range(nb_equations):
      upwind_diffusion[:,:,eq] = (α * 0.5 * geom.Δx / nbsolpts) * numpy.abs(u1) * Q_xx[:,:,eq] + (α * 0.5 * geom.Δz / nbsolpts) * numpy.abs(u2) * Q_yy[:,:,eq]

   # Assemble the right-hand sides
   rhs = - df1_dx1 - df2_dx2 + upwind_diffusion

   rhs[:,:,RHO_W] = rhs[:,:,RHO_W] - Q[:,:,RHO] * gravity

   return rhs
