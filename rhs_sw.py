import numpy

from definitions import *
from xchange import *

def rhs_sw(Q, geom, mtrx, metric, cube_face, nbsolpts, nb_elements_x1, nb_elements_x2, α):

   type_vec = type(Q[0, 0, 0])

   nb_interfaces_x1 = nb_elements_x1 + 1
   nb_interfaces_x2 = nb_elements_x2 + 1

   flux_x1          = numpy.zeros_like(Q, dtype=type_vec)
   flux_x2          = numpy.zeros_like(Q, dtype=type_vec)

   df1_dx1          = numpy.zeros_like(Q, dtype=type_vec)
   df2_dx2          = numpy.zeros_like(Q, dtype=type_vec)
   dQ_dx1           = numpy.zeros_like(Q, dtype=type_vec)
   dQ_dx2           = numpy.zeros_like(Q, dtype=type_vec)
   Q_xx             = numpy.zeros_like(Q, dtype=type_vec)
   Q_yy             = numpy.zeros_like(Q, dtype=type_vec)
   upwind_diffusion = numpy.zeros_like(Q, dtype=type_vec)

   flux_itf_j      = numpy.zeros((nb_equations, nb_interfaces_x2, 2, nbsolpts*nb_elements_x1), dtype=type_vec)
   var_itf_j       = numpy.zeros((nb_equations, nb_interfaces_x2, 2, nbsolpts*nb_elements_x1), dtype=type_vec)
   diffusion_itf_j = numpy.zeros((nb_equations, nb_interfaces_x2, 2, nbsolpts*nb_elements_x1), dtype=type_vec)

   flux_itf_i      = numpy.zeros((nb_equations, nb_interfaces_x1, 2, nbsolpts*nb_elements_x2), dtype=type_vec)
   var_itf_i       = numpy.zeros((nb_equations, nb_interfaces_x1, 2, nbsolpts*nb_elements_x2), dtype=type_vec)
   diffusion_itf_i = numpy.zeros((nb_equations, nb_interfaces_x1, 2, nbsolpts*nb_elements_x2), dtype=type_vec)

   # Unpack physical variables
   h  = Q[idx_h, :, :]
   u1 = Q[idx_hu1, :, :] / Q[idx_h, :, :]
   u2 = Q[idx_hu2, :, :] / Q[idx_h, :, :]

   # Interpolate to the element interface
   for elem in range(nb_elements_x2):
      epais = elem * nbsolpts + numpy.arange(nbsolpts)
      itf_l = elem
      itf_r = elem + 1

      var_itf_j[idx_h, itf_l, 1, :] = mtrx.lcoef @ h[epais, :]
      var_itf_j[idx_h, itf_r, 0, :] = mtrx.rcoef @ h[epais, :]

      var_itf_j[idx_u1, itf_l, 1, :] = mtrx.lcoef @ u1[epais, :]
      var_itf_j[idx_u1, itf_r, 0, :] = mtrx.rcoef @ u1[epais, :]

      var_itf_j[idx_u2, itf_l, 1, :] = mtrx.lcoef @ u2[epais, :]
      var_itf_j[idx_u2, itf_r, 0, :] = mtrx.rcoef @ u2[epais, :]

   for elem in range(nb_elements_x1):
      epais = elem * nbsolpts + numpy.arange(nbsolpts)
      itf_l = elem
      itf_r = elem + 1

      var_itf_i[idx_h, itf_l, 1, :] = h[:, epais] @ mtrx.lcoef
      var_itf_i[idx_h, itf_r, 0, :] = h[:, epais] @ mtrx.rcoef

      var_itf_i[idx_u1, itf_l, 1, :] = u1[:, epais] @ mtrx.lcoef
      var_itf_i[idx_u1, itf_r, 0, :] = u1[:, epais] @ mtrx.rcoef

      var_itf_i[idx_u2, itf_l, 1, :] = u2[:, epais] @ mtrx.lcoef
      var_itf_i[idx_u2, itf_r, 0, :] = u2[:, epais] @ mtrx.rcoef

   xchange_halo(geom, var_itf_i, var_itf_j, cube_face)

#   import graphx
#   graphx.plot_uv(geom, u1, u2)

#   print(h[:,0])
#   print("////////////")
#   print(var_itf_i[0, 0, 0, :])
#   print("-------------")
#   print(h[:,0] - var_itf_i[0, 0, 0, :])

#   print(h[:,-1])
#   print("////////////")
#   print(var_itf_i[0, -1, 1, :])
#   print("-------------")
#   print(h[:,-1] - var_itf_i[0, -1, 1, :])

#   print(h[0,:])
#   print("////////////")
#   print(var_itf_j[0, 0, 0, :])
#   print("-------------")
#   print(h[0,:] - var_itf_j[0, 0, 0, :])

#   print(h[-1,:])
#   print("////////////")
#   print(var_itf_j[0, -1, 1, :])
#   print("-------------")
#   print(h[-1,:] - var_itf_j[0, -1, 1, :])

#   print(u1[:,0])
#   print("////////////")
#   print(var_itf_i[1, 0, 0, :])
#   print("-------------")
#   print(u1[:,0] - var_itf_i[1, 0, 0, :])

#   print(u1[:,-1])
#   print("////////////")
#   print(var_itf_i[1, -1, 1, :])
#   print("-------------")
#   print(u1[:,-1] - var_itf_i[1, -1, 1, :])

#   print(u1[0,:])
#   print("////////////")
#   print(var_itf_j[1, 0, 0, :])
#   print("-------------")
#   print(u1[0,:] - var_itf_j[1, 0, 0, :])

#   print(u1[-1,:])
#   print("////////////")
#   print(var_itf_j[1, -1, 1, :])
#   print("-------------")
#   print(u1[-1,:] - var_itf_j[1, -1, 1, :])

#   print(u2[:,0])
#   print("////////////")
#   print(var_itf_i[2, 0, 0, :])
#   print("-------------")
#   print(u2[:,0] - var_itf_i[2, 0, 0, :])

#   print(u2[:,-1])
#   print("////////////")
#   print(var_itf_i[2, -1, 1, :])
#   print("-------------")
#   print(u2[:,-1] - var_itf_i[2, -1, 1, :])

#   print(u2[0,:])
#   print("////////////")
#   print(var_itf_j[2, 0, 0, :])
#   print("-------------")
#   print(u2[0,:] - var_itf_j[2, 0, 0, :])

#   print(u1[-1,:])
#   print("////////////")
#   print(var_itf_j[1, -1, 1, :])
#   print("-------------")
#   print(u1[-1,:] - var_itf_j[1, -1, 1, :])

   print("SW, rendu icitte")
   exit(0)

   # Compute the fluxes
   flux_x1[idx_h, :, :]   = Q[idx_hu1, :, :]
   flux_x1[idx_hu1, :, :] = Q[idx_hu1, :, :] * u1 + 0.5 * metric.H_contra_11 * gravity * Q[idx_h, :, :]**2
   flux_x1[idx_hu2, :, :] = Q[idx_hu1, :, :] * u2 + 0.5 * metric.H_contra_12 * gravity * Q[idx_h, :, :]**2

   flux_x2[idx_h, :, :]   = Q[idx_hu2, :, :]
   flux_x2[idx_hu1, :, :] = Q[idx_hu2, :, :] * u1 + 0.5 * metric.H_contra_21 * gravity * Q[idx_h, :, :]**2
   flux_x2[idx_hu2, :, :] = Q[idx_hu2, :, :] * u2 + 0.5 * metric.H_contra_22 * gravity * Q[idx_h, :, :]**2

   # Common Rusanov fluxes
   for itf in range(nb_interfaces_x2):

      eig_L = numpy.abs(var_itf_j[1,:,RHO_W,itf] / var_itf_j[1,:,RHO,itf]) \
            + numpy.sqrt( heat_capacity_ratio * itf_j_pres[1,:,itf] / var_itf_j[1,:,RHO,itf]  )

      eig_R = numpy.abs(var_itf_j[0,:,RHO_W,itf+1] / var_itf_j[0,:,RHO,itf+1]) \
            + numpy.sqrt( heat_capacity_ratio * itf_j_pres[0,:,itf+1] / var_itf_j[0,:,RHO,itf+1])

      for eq in range(nb_equations):
         flux_itf_j[0,:,eq,itf+1] = 0.5 * ( flux_itf_j[1,:,eq,itf] + flux_itf_j[0,:,eq,itf+1] \
               - numpy.maximum(numpy.abs(eig_L), numpy.abs(eig_R)) * ( var_itf_j[0,:,eq,itf+1] - var_itf_j[1,:,eq,itf] ) )

      flux_itf_j[1,:,:,itf] = flux_itf_j[0,:,:,itf+1]

      diffusion_itf_j[0,:,:,itf+1] = 0.5 * ( var_itf_j[1,:,:,itf] + var_itf_j[0,:,:,itf+1] )
      diffusion_itf_j[1,:,:,itf]   = diffusion_itf_j[0,:,:,itf+1]

   for itf in range(nb_interfaces_x1):

      eig_L = numpy.abs(var_itf_i[:,1,RHO_U,itf] / var_itf_i[:,1,RHO,itf]) \
            + numpy.sqrt( heat_capacity_ratio * itf_i_pres[:,1,itf] / var_itf_i[:,1,RHO,itf] )
      eig_R = numpy.abs(var_itf_i[:,0,RHO_U,itf+1] / var_itf_i[:,0,RHO,itf+1]) \
            + numpy.sqrt( heat_capacity_ratio * itf_i_pres[:,0,itf+1] / var_itf_i[:,0,RHO,itf+1] )

      for eq in range(nb_equations):
         flux_itf_i[:,0,eq,itf+1] = 0.5 * ( flux_itf_i[:,1,eq,itf] + flux_itf_i[:,0,eq,itf+1] \
               - numpy.maximum(numpy.abs(eig_L), numpy.abs(eig_R)) * ( var_itf_i[:,0,eq,itf+1] - var_itf_i[:,1,eq,itf] ) )

      flux_itf_i[:,1,:,itf] = flux_itf_i[:,0,:,itf+1]

      diffusion_itf_i[:,0,:,itf+1] = 0.5 * ( var_itf_i[:,1,:,itf] + var_itf_i[:,0,:,itf+1] )
      diffusion_itf_i[:,1,:,itf]   = diffusion_itf_i[:,0,:,itf+1]

   # Compute the derivatives
   for elem in range(nb_elements_x2):
      epais = elem * nbsolpts + numpy.arange(nbsolpts)

      for eq in range(nb_equations):
         df2_dx2[epais,:,eq] = ( mtrx.diff_solpt @ flux_x2[epais,:,eq] + mtrx.correction @ flux_itf_j[:,:,eq,elem] ) * 2.0/geom.Δz
         dQ_dx2[epais,:,eq]  = ( mtrx.diff_solpt @ Q[epais,:,eq] + mtrx.correction @ diffusion_itf_j[:,:,eq,elem] ) * 2.0/geom.Δz

   for elem in range(nb_elements_x1):
      epais = elem * nbsolpts + numpy.arange(nbsolpts)

      for eq in range(nb_equations):
         df1_dx1[:,epais,eq] = ( flux_x1[:,epais,eq] @ mtrx.diff_solpt.T + flux_itf_i[:,:,eq,elem] @ mtrx.correction.T ) * 2.0/geom.Δx
         dQ_dx1[:,epais,eq]  = ( Q[:,epais,eq] @ mtrx.diff_solpt.T + diffusion_itf_i[:,:,eq,elem] @ mtrx.correction.T ) * 2.0/geom.Δx

   # Interpolate first derivative of diffusion to the interface
   for elem in range(nb_elements_x2):
      epais = elem * nbsolpts + numpy.arange(nbsolpts)

      for eq in range(nb_equations):
         var_itf_j[0,:,eq,elem] = mtrx.lcoef @ dQ_dx2[epais,:,eq]
         var_itf_j[1,:,eq,elem] = mtrx.rcoef @ dQ_dx2[epais,:,eq]

   for elem in range(nb_elements_x1):
      epais = elem * nbsolpts + numpy.arange(nbsolpts)

      for eq in range(nb_equations):
         var_itf_i[:,0,eq,elem] = dQ_dx1[:,epais,eq] @ mtrx.lcoef
         var_itf_i[:,1,eq,elem] = dQ_dx1[:,epais,eq] @ mtrx.rcoef

   # Communication at cell interfaces with central flux
   for itf in range(nb_interfaces_x2):
      diffusion_itf_j[0,:,:,itf+1] = 0.5 * ( var_itf_j[1,:,:,itf] + var_itf_j[0,:,:,itf+1] )
      diffusion_itf_j[1,:,:,itf]   = diffusion_itf_j[0,:,:,itf+1]

   for itf in range(nb_interfaces_x1):
      diffusion_itf_i[:,0,:,itf+1] = 0.5 * ( var_itf_i[:,1,:,itf] + var_itf_i[:,0,:,itf+1] )
      diffusion_itf_i[:,1,:,itf]   = diffusion_itf_i[:,0,:,itf+1]

   # Bondary treatement (this should be equivalent to a zero diffusion coefficient at the boundary)
   diffusion_itf_j[0,:,:,0]  = 0.0
   diffusion_itf_j[1,:,:,-1] = 0.0

   diffusion_itf_i[:,0,:,0]  = 0.0
   diffusion_itf_i[:,1,:,-1] = 0.0

   # Finalize the diffusion operator
   for elem in range(nb_elements_x2):
      epais = elem * nbsolpts + numpy.arange(nbsolpts)

      for eq in range(nb_equations):
         Q_yy[epais,:,eq] = ( mtrx.diff_solpt @ dQ_dx2[epais,:,eq] + mtrx.correction @ diffusion_itf_j[:,:,eq,elem] ) * 2.0/geom.Δz

   for elem in range(nb_elements_x1):
      epais = elem * nbsolpts + numpy.arange(nbsolpts)

      for eq in range(nb_equations):
         Q_xx[:,epais,eq] = ( dQ_dx1[:,epais,eq] @ mtrx.diff_solpt.T + diffusion_itf_i[:,:,eq,elem] @ mtrx.correction.T ) * 2.0/geom.Δx

   for eq in range(nb_equations):
      upwind_diffusion[:,:,eq] = (α * 0.5 * geom.Δx / nbsolpts) * numpy.abs(u1) * Q_xx[:,:,eq] + (α * 0.5 * geom.Δz / nbsolpts) * numpy.abs(u2) * Q_yy[:,:,eq]

   # Assemble the right-hand sides
   rhs = - df1_dx1 - df2_dx2 + upwind_diffusion

   rhs[:,:,RHO_W] = rhs[:,:,RHO_W] - Q[:,:,RHO] * gravity

   return rhs
