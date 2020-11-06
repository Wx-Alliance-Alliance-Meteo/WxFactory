import numpy
import cupy

from definitions import idx_h, idx_hu1, idx_hu2, idx_u1, idx_u2, gravity
from dgfilter import apply_filter
from timer import TimerGroup

# Preload some CuPy stuff
dummy1_d = cupy.array([1])
dummy2 = numpy.array([2])
dummy2_d = cupy.asarray(dummy2)
dummy3_d = dummy1_d @ dummy2_d
dummy3 = cupy.asnumpy(cupy.linalg.norm(dummy3_d))
# Done preloading


def compute_derivatives(nb_elements, nb_sol_pts, offset,
                        flux_x1, flux_itf_i, flux_x2, flux_itf_j,
                        dx1, dx2,
                        diff_sol_pt, diff_sol_pt_tr, correction, correction_tr,
                        timer):
   """
   Compute the 'derivatives' part of the right-hand side

   :param nb_elements:     Number of elements in one direction
   :param nb_sol_pts:      Number of solution points in an element (= order)
   :param offset:          Size of the halo (on one side)
   :param flux_x1:         Flux along axis 1 at every solution point
   :param flux_itf_i:      Flux along axis 1, across element interfaces
   :param flux_x2:         Flux along axis 2 at every solution point
   :param flux_itf_j:      Flux along axis 2, across element interfaces
   :param dx1:             Element size along axis 1
   :param dx2:             Element size along axis 2
   :param diff_sol_pt:     Weights matrix for computing Lagrange interpolation derivatives
   :param diff_sol_pt_tr:  Transpose of diff_sol_pt
   :param correction:
   :param correction_tr:
   :param timer:           To know how much time it took

   :return:                2 arrays, each containing the derivative of the 3 fields along an axis
   """

   timer.start()

   datatype = flux_itf_i[0].dtype
   nb_dof = nb_elements * nb_sol_pts
   df1_dx1 = numpy.empty((3, nb_dof, nb_dof), dtype = datatype)
   df2_dx2 = numpy.empty((3, nb_dof, nb_dof), dtype = datatype)

   for elem in range(nb_elements):
      slice = elem * nb_sol_pts + numpy.arange(nb_sol_pts)

      # --- Direction x1
      df1_dx1[idx_h]  [:,slice] = ( flux_x1[0][:,slice] @ diff_sol_pt_tr + flux_itf_i[0][elem+offset,:,:] @ correction_tr ) * 2.0 / dx1
      df1_dx1[idx_hu1][:,slice] = ( flux_x1[1][:,slice] @ diff_sol_pt_tr + flux_itf_i[1][elem+offset,:,:] @ correction_tr ) * 2.0 / dx1
      df1_dx1[idx_hu2][:,slice] = ( flux_x1[2][:,slice] @ diff_sol_pt_tr + flux_itf_i[2][elem+offset,:,:] @ correction_tr ) * 2.0 / dx1

      # --- Direction x2
      df2_dx2[idx_h,  slice,:] = ( diff_sol_pt @ flux_x2[0][slice,:] + correction @ flux_itf_j[0][elem+offset,:,:] ) * 2.0 / dx2
      df2_dx2[idx_hu1,slice,:] = ( diff_sol_pt @ flux_x2[1][slice,:] + correction @ flux_itf_j[1][elem+offset,:,:] ) * 2.0 / dx2
      df2_dx2[idx_hu2,slice,:] = ( diff_sol_pt @ flux_x2[2][slice,:] + correction @ flux_itf_j[2][elem+offset,:,:] ) * 2.0 / dx2
   timer.stop()

   return df1_dx1, df2_dx2

def compute_derivatives_alt(nb_elements, nb_sol_pts, offset,
                            flux_x1, flux_itf_i, flux_x2, flux_itf_j,
                            dx1, dx2, diff_sol_pt, correction, timer):
   timer.start()

   datatype = flux_itf_i[0].dtype
   nb_dof = nb_elements * nb_sol_pts

   # Transpose before, for better use of cache (for derivative along axis 1)
   fx1 = [flux_x1[i].T for i in range(3)]
   fix1 = [flux_itf_i[i].transpose((0, 2, 1)) for i in range(3)]

   df1_dx1 = numpy.empty((3, nb_dof, nb_dof), dtype = datatype)
   df2_dx2 = numpy.empty((3, nb_dof, nb_dof), dtype = datatype)
   for elem in range(nb_elements):
      slice = elem * nb_sol_pts + numpy.arange(nb_sol_pts)
      pos = elem + offset

      # --- Direction x1
      df1_dx1[idx_h]  [slice,:] = ( diff_sol_pt @ fx1[0][slice,:] + correction @ fix1[0][pos,:,:] ) * 2.0 / dx1
      df1_dx1[idx_hu1][slice,:] = ( diff_sol_pt @ fx1[1][slice,:] + correction @ fix1[1][pos,:,:] ) * 2.0 / dx1
      df1_dx1[idx_hu2][slice,:] = ( diff_sol_pt @ fx1[2][slice,:] + correction @ fix1[2][pos,:,:] ) * 2.0 / dx1

      # --- Direction x2
      df2_dx2[idx_h,  slice,:] = ( diff_sol_pt @ flux_x2[0][slice,:] + correction @ flux_itf_j[0][pos,:,:] ) * 2.0 / dx2
      df2_dx2[idx_hu1,slice,:] = ( diff_sol_pt @ flux_x2[1][slice,:] + correction @ flux_itf_j[1][pos,:,:] ) * 2.0 / dx2
      df2_dx2[idx_hu2,slice,:] = ( diff_sol_pt @ flux_x2[2][slice,:] + correction @ flux_itf_j[2][pos,:,:] ) * 2.0 / dx2

   # Don't forget to transpose back for axis 1
   result = numpy.array([df1_dx1[i].T for i in range(3)]), df2_dx2

   timer.stop()

   return result


def compute_derivatives_gpu(nb_elements, nb_sol_pts, offset,
                            flux_x1, flux_itf_i, flux_x2, flux_itf_j,
                            dx1, dx2, diff_sol_pt, correction, timer, rank):

   timer.start()

   datatype = flux_itf_i[0].dtype
   nb_dof = nb_elements * nb_sol_pts

   df1_dx1 = cupy.empty((3, nb_dof, nb_dof), dtype = datatype)
   df2_dx2 = cupy.empty((3, nb_dof, nb_dof), dtype = datatype)

   # Transpose before, for better use of cache (for derivative along axis 1)
   fx1  = cupy.array([flux_x1[i].T for i in range(3)])
   fix1 = cupy.array([flux_itf_i[i].transpose((0, 2, 1)) for i in range(3)])
   fx2  = cupy.array(flux_x2)
   fix2 = cupy.array(flux_itf_j)

   diff = cupy.array(diff_sol_pt)
   corr = cupy.array(correction)

   for elem in range(nb_elements):
      slice = elem * nb_sol_pts + numpy.arange(nb_sol_pts)
      pos = elem + offset

      # --- Direction x1
      df1_dx1[idx_h]  [slice,:] = ( diff @ fx1[0][slice,:] + corr @ fix1[0][pos,:,:] ) * 2.0 / dx1
      df1_dx1[idx_hu1][slice,:] = ( diff @ fx1[1][slice,:] + corr @ fix1[1][pos,:,:] ) * 2.0 / dx1
      df1_dx1[idx_hu2][slice,:] = ( diff @ fx1[2][slice,:] + corr @ fix1[2][pos,:,:] ) * 2.0 / dx1

      # --- Direction x2
      df2_dx2[idx_h,  slice,:] = ( diff @ fx2[0][slice,:] + corr @ fix2[0][pos,:,:] ) * 2.0 / dx2
      df2_dx2[idx_hu1,slice,:] = ( diff @ fx2[1][slice,:] + corr @ fix2[1][pos,:,:] ) * 2.0 / dx2
      df2_dx2[idx_hu2,slice,:] = ( diff @ fx2[2][slice,:] + corr @ fix2[2][pos,:,:] ) * 2.0 / dx2


   # Copy to CPU
   x1_h = cupy.asnumpy(df1_dx1)
   x2_h = cupy.asnumpy(df2_dx2)


   # Don't forget to transpose back for axis 1
   result = numpy.array([x1_h[i].T for i in range(3)]), x2_h

   timer.stop()

   return result


def rhs_sw(Q, geom, mtrx, metric, topo, ptopo, nbsolpts, nb_elements_horiz, case_number, filter_rhs=False, timers = None):

   timers = timers if timers is not None else TimerGroup(5, 0.0)
   timers[0].start()

   type_vec = type(Q[0, 0, 0])

   shallow_water_equations = ( case_number > 1 )

   nb_interfaces_horiz = nb_elements_horiz + 1

   forcing = numpy.zeros_like(Q, dtype=type_vec)
   rhs = numpy.zeros_like(Q, dtype=type_vec)

   flux_Eq0_itf_j = numpy.zeros((nb_elements_horiz+2, 2, nbsolpts*nb_elements_horiz), dtype=type_vec)
   flux_Eq1_itf_j = numpy.zeros((nb_elements_horiz+2, 2, nbsolpts*nb_elements_horiz), dtype=type_vec)
   flux_Eq2_itf_j = numpy.zeros((nb_elements_horiz+2, 2, nbsolpts*nb_elements_horiz), dtype=type_vec)
   h_itf_j        = numpy.zeros((nb_elements_horiz+2, 2, nbsolpts*nb_elements_horiz), dtype=type_vec)
   u1_itf_j       = numpy.zeros((nb_elements_horiz+2, 2, nbsolpts*nb_elements_horiz), dtype=type_vec)
   u2_itf_j       = numpy.zeros((nb_elements_horiz+2, 2, nbsolpts*nb_elements_horiz), dtype=type_vec)

   flux_Eq0_itf_i = numpy.zeros((nb_elements_horiz+2, nbsolpts*nb_elements_horiz, 2), dtype=type_vec)
   flux_Eq1_itf_i = numpy.zeros((nb_elements_horiz+2, nbsolpts*nb_elements_horiz, 2), dtype=type_vec)
   flux_Eq2_itf_i = numpy.zeros((nb_elements_horiz+2, nbsolpts*nb_elements_horiz, 2), dtype=type_vec)
   h_itf_i        = numpy.zeros((nb_elements_horiz+2, 2, nbsolpts*nb_elements_horiz), dtype=type_vec)
   u1_itf_i       = numpy.zeros((nb_elements_horiz+2, 2, nbsolpts*nb_elements_horiz), dtype=type_vec)
   u2_itf_i       = numpy.zeros((nb_elements_horiz+2, 2, nbsolpts*nb_elements_horiz), dtype=type_vec)

   eig_L          = numpy.zeros(nbsolpts*nb_elements_horiz, dtype=type_vec)
   eig_R          = numpy.zeros(nbsolpts*nb_elements_horiz, dtype=type_vec)
   eig            = numpy.zeros(nbsolpts*nb_elements_horiz, dtype=type_vec)

   flux_L         = numpy.zeros(nbsolpts*nb_elements_horiz, dtype=type_vec)
   flux_R         = numpy.zeros(nbsolpts*nb_elements_horiz, dtype=type_vec)

   # Unpack physical variables
   h = Q[idx_h, :, :]
   hsquared = Q[idx_h, :, :]**2

   if shallow_water_equations:
      u1 = Q[idx_hu1,:,:] / h
      u2 = Q[idx_hu2,:,:] / h
   else:
      u1 = Q[idx_u1, :, :]
      u2 = Q[idx_u2, :, :]

   # Compute the fluxes
   flux_Eq0_x1 = h * metric.sqrtG * u1
   flux_Eq0_x2 = h * metric.sqrtG * u2

   flux_Eq1_x1 = metric.sqrtG * ( Q[idx_hu1,:,:] * u1 + 0.5 * gravity * metric.H_contra_11 * hsquared )
   flux_Eq1_x2 = metric.sqrtG * ( Q[idx_hu1,:,:] * u2 + 0.5 * gravity * metric.H_contra_12 * hsquared )

   flux_Eq2_x1 = metric.sqrtG * ( Q[idx_hu2,:,:] * u1 + 0.5 * gravity * metric.H_contra_21 * hsquared )
   flux_Eq2_x2 = metric.sqrtG * ( Q[idx_hu2,:,:] * u2 + 0.5 * gravity * metric.H_contra_22 * hsquared )

   # Offset due to the halo
   offset = 1

   HH = h + topo.hsurf

   # Interpolate to the element interface
   for elem in range(nb_elements_horiz):
      epais = elem * nbsolpts + numpy.arange(nbsolpts)

      pos = elem + offset

      # --- Direction x1

      h_itf_i[pos, 0, :] = HH[:, epais] @ mtrx.extrap_west
      h_itf_i[pos, 1, :] = HH[:, epais] @ mtrx.extrap_east

      u1_itf_i[pos, 0, :] = u1[:, epais] @ mtrx.extrap_west
      u1_itf_i[pos, 1, :] = u1[:, epais] @ mtrx.extrap_east

      u2_itf_i[pos, 0, :] = u2[:, epais] @ mtrx.extrap_west
      u2_itf_i[pos, 1, :] = u2[:, epais] @ mtrx.extrap_east

      # --- Direction x2

      h_itf_j[pos, 0, :] = mtrx.extrap_south @ HH[epais, :]
      h_itf_j[pos, 1, :] = mtrx.extrap_north @ HH[epais, :]

      u1_itf_j[pos, 0, :] = mtrx.extrap_south @ u1[epais, :]
      u1_itf_j[pos, 1, :] = mtrx.extrap_north @ u1[epais, :]

      u2_itf_j[pos, 0, :] = mtrx.extrap_south @ u2[epais, :]
      u2_itf_j[pos, 1, :] = mtrx.extrap_north @ u2[epais, :]

   timers[0].stop()

   timers[1].start()
   ptopo.xchange_scalars(geom, h_itf_i, h_itf_j)
   ptopo.xchange_vectors(geom, u1_itf_i, u2_itf_i, u1_itf_j, u2_itf_j)
   timers[1].stop()

   timers[2].start()
   # Common Rusanov fluxes
   timers[3].start()
   for itf in range(nb_interfaces_horiz):

      elem_L = itf
      elem_R = itf + 1

      h_itf_i[elem_L, 1, :] -= topo.hsurf_itf_i[elem_L, :, 1]
      h_itf_i[elem_R, 0, :] -= topo.hsurf_itf_i[elem_R, :, 0]

      h_itf_j[elem_L, 1, :] -= topo.hsurf_itf_j[elem_L, 1, :]
      h_itf_j[elem_R, 0, :] -= topo.hsurf_itf_j[elem_R, 0, :]

      # Direction x1

      if shallow_water_equations:
         eig_L[:] = numpy.abs( u1_itf_i[elem_L, 1, :] ) + numpy.sqrt( gravity * h_itf_i[elem_L, 1, :] * metric.H_contra_11_itf_i[:, itf] )
         eig_R[:] = numpy.abs( u1_itf_i[elem_R, 0, :] ) + numpy.sqrt( gravity * h_itf_i[elem_R, 0, :] * metric.H_contra_11_itf_i[:, itf] )
      else:
         eig_L[:] = numpy.abs( u1_itf_i[elem_L, 1, :] )
         eig_R[:] = numpy.abs( u1_itf_i[elem_R, 0, :] )

      eig[:] = numpy.maximum(eig_L, eig_R)

      # --- Continuity equation

      flux_L[:] = metric.sqrtG_itf_i[:, itf] * h_itf_i[elem_L, 1, :] * u1_itf_i[elem_L, 1, :]
      flux_R[:] = metric.sqrtG_itf_i[:, itf] * h_itf_i[elem_R, 0, :] * u1_itf_i[elem_R, 0, :]

      flux_Eq0_itf_i[elem_L, :, 1] = 0.5 * ( flux_L  + flux_R - eig * metric.sqrtG_itf_i[:, itf] * ( h_itf_i[elem_R, 0, :] - h_itf_i[elem_L, 1, :] ) )
      flux_Eq0_itf_i[elem_R, :, 0] = flux_Eq0_itf_i[elem_L, :, 1]

      # --- u1 equation

      flux_L[:] = metric.sqrtG_itf_i[:, itf] * ( h_itf_i[elem_L, 1, :] * u1_itf_i[elem_L, 1, :]**2 \
            + 0.5 * gravity * metric.H_contra_11_itf_i[:, itf] * h_itf_i[elem_L, 1, :]**2 )
      flux_R[:] = metric.sqrtG_itf_i[:, itf] * ( h_itf_i[elem_R, 0, :] * u1_itf_i[elem_R, 0, :]**2 \
            + 0.5 * gravity * metric.H_contra_11_itf_i[:, itf] * h_itf_i[elem_R, 0, :]**2 )

      flux_Eq1_itf_i[elem_L, :, 1] = 0.5 * ( flux_L  + flux_R - eig * metric.sqrtG_itf_i[:, itf] \
            * ( h_itf_i[elem_R, 0, :] * u1_itf_i[elem_R, 0, :] - h_itf_i[elem_L, 1, :] * u1_itf_i[elem_L, 1, :] ) )
      flux_Eq1_itf_i[elem_R, :, 0] = flux_Eq1_itf_i[elem_L, :, 1]

      # --- u2 equation

      flux_L[:] = metric.sqrtG_itf_i[:, itf] * ( h_itf_i[elem_L, 1, :] * u2_itf_i[elem_L, 1, :] * u1_itf_i[elem_L, 1, :] \
            + 0.5 * gravity * metric.H_contra_21_itf_i[:, itf] * h_itf_i[elem_L, 1, :]**2 )
      flux_R[:] = metric.sqrtG_itf_i[:, itf] * ( h_itf_i[elem_R, 0, :] * u2_itf_i[elem_R, 0, :] * u1_itf_i[elem_R, 0, :] \
            + 0.5 * gravity * metric.H_contra_21_itf_i[:, itf] * h_itf_i[elem_R, 0, :]**2 )

      flux_Eq2_itf_i[elem_L, :, 1] = 0.5 * ( flux_L  + flux_R - eig * metric.sqrtG_itf_i[:, itf] \
            * ( h_itf_i[elem_R, 0, :] * u2_itf_i[elem_R, 0, :] - h_itf_i[elem_L, 1, :] * u2_itf_i[elem_L, 1, :] ) )
      flux_Eq2_itf_i[elem_R, :, 0] = flux_Eq2_itf_i[elem_L, :, 1]

      # Direction x2

      if shallow_water_equations:
         eig_L[:] = numpy.abs( u2_itf_j[elem_L, 1, :] ) + numpy.sqrt( gravity * h_itf_j[elem_L, 1, :] * metric.H_contra_22_itf_j[itf, :] )
         eig_R[:] = numpy.abs( u2_itf_j[elem_R, 0, :] ) + numpy.sqrt( gravity * h_itf_j[elem_R, 0, :] * metric.H_contra_22_itf_j[itf, :] )
      else:
         eig_L[:] = numpy.abs( u2_itf_j[elem_L, 1, :] )
         eig_R[:] = numpy.abs( u2_itf_j[elem_R, 0, :] )

      eig[:] = numpy.maximum(eig_L, eig_R)

      # --- Continuity equation

      flux_L[:] = metric.sqrtG_itf_j[itf, :] * h_itf_j[elem_L, 1, :] * u2_itf_j[elem_L, 1, :]
      flux_R[:] = metric.sqrtG_itf_j[itf, :] * h_itf_j[elem_R, 0, :] * u2_itf_j[elem_R, 0, :]

      flux_Eq0_itf_j[elem_L, 1, :] = 0.5 * ( flux_L + flux_R - eig * metric.sqrtG_itf_j[itf, :] * ( h_itf_j[elem_R, 0, :] - h_itf_j[elem_L, 1, :] ) )
      flux_Eq0_itf_j[elem_R, 0, :] = flux_Eq0_itf_j[elem_L, 1, :]

      # --- u1 equation

      flux_L[:] = metric.sqrtG_itf_j[itf, :] * ( h_itf_j[elem_L, 1, :] * u1_itf_j[elem_L, 1, :] * u2_itf_j[elem_L, 1, :] \
            + 0.5 * gravity * metric.H_contra_12_itf_j[itf, :] * h_itf_j[elem_L, 1, :]**2 )
      flux_R[:] = metric.sqrtG_itf_j[itf, :] * ( h_itf_j[elem_R, 0, :] * u1_itf_j[elem_R, 0, :] * u2_itf_j[elem_R, 0, :] \
            + 0.5 * gravity * metric.H_contra_12_itf_j[itf, :] * h_itf_j[elem_R, 0, :]**2 )

      flux_Eq1_itf_j[elem_L, 1, :] = 0.5 * ( flux_L + flux_R - eig * metric.sqrtG_itf_j[itf, :] \
            * ( h_itf_j[elem_R, 0, :] * u1_itf_j[elem_R, 0, :] - h_itf_j[elem_L, 1, :] * u1_itf_j[elem_L, 1, :]) )
      flux_Eq1_itf_j[elem_R, 0, :] = flux_Eq1_itf_j[elem_L, 1, :]

      # --- u2 equation

      flux_L[:] = metric.sqrtG_itf_j[itf, :] * ( h_itf_j[elem_L, 1, :] * u2_itf_j[elem_L, 1, :]**2 \
            + 0.5 * gravity * metric.H_contra_22_itf_j[itf, :] * h_itf_j[elem_L, 1, :]**2 )
      flux_R[:] = metric.sqrtG_itf_j[itf, :] * ( h_itf_j[elem_R, 0, :] * u2_itf_j[elem_R, 0, :]**2 \
            + 0.5 * gravity * metric.H_contra_22_itf_j[itf, :] * h_itf_j[elem_R, 0, :]**2 )

      flux_Eq2_itf_j[elem_L, 1, :] = 0.5 * ( flux_L + flux_R - eig * metric.sqrtG_itf_j[itf, :] \
            * ( h_itf_j[elem_R, 0, :] * u2_itf_j[elem_R, 0, :] - h_itf_j[elem_L, 1, :] * u2_itf_j[elem_L, 1, :]) )
      flux_Eq2_itf_j[elem_R, 0, :] = flux_Eq2_itf_j[elem_L, 1, :]
   timers[3].stop()

   # Compute the derivatives
   df1_dx1, df2_dx2 = compute_derivatives(
      nb_elements_horiz, nbsolpts, offset,
      [flux_Eq0_x1, flux_Eq1_x1, flux_Eq2_x1], [flux_Eq0_itf_i, flux_Eq1_itf_i, flux_Eq2_itf_i],
      [flux_Eq0_x2, flux_Eq1_x2, flux_Eq2_x2], [flux_Eq0_itf_j, flux_Eq1_itf_j, flux_Eq2_itf_j],
      geom.Δx1, geom.Δx2, mtrx.diff_solpt, mtrx.diff_solpt_tr, mtrx.correction, mtrx.correction_tr, timers[4])

   # test_d1, test_d2 = compute_derivatives_alt(
   #    nb_elements_horiz, nbsolpts, offset,
   #    [flux_Eq0_x1, flux_Eq1_x1, flux_Eq2_x1], [flux_Eq0_itf_i, flux_Eq1_itf_i, flux_Eq2_itf_i],
   #    [flux_Eq0_x2, flux_Eq1_x2, flux_Eq2_x2], [flux_Eq0_itf_j, flux_Eq1_itf_j, flux_Eq2_itf_j],
   #    geom.Δx1, geom.Δx2, mtrx.diff_solpt, mtrx.correction, timers[5])
   #
   # norm1 = numpy.linalg.norm(df1_dx1 - test_d1)
   # norm2 = numpy.linalg.norm(df2_dx2 - test_d2)
   #
   # if norm1 > 1e-12 or norm2 > 1e-12:
   #    print('Got a big difference! ({} and {})'.format(norm1, norm2))
   #    raise ValueError

   # gpu_d1, gpu_d2 = compute_derivatives_gpu(
   #    nb_elements_horiz, nbsolpts, offset,
   #    [flux_Eq0_x1, flux_Eq1_x1, flux_Eq2_x1], [flux_Eq0_itf_i, flux_Eq1_itf_i, flux_Eq2_itf_i],
   #    [flux_Eq0_x2, flux_Eq1_x2, flux_Eq2_x2], [flux_Eq0_itf_j, flux_Eq1_itf_j, flux_Eq2_itf_j],
   #    geom.Δx1, geom.Δx2, mtrx.diff_solpt, mtrx.correction, timers[6], ptopo.rank)
   #
   # norm1 = numpy.linalg.norm(df1_dx1 - gpu_d1) / numpy.linalg.norm(df1_dx1)
   # norm2 = numpy.linalg.norm(df2_dx2 - gpu_d2) / numpy.linalg.norm(df2_dx2)
   #
   # if norm1 > 1e-12 or norm2 > 1e-12:
   #    print('Got a big relative difference (GPU)! ({} and {})'.format(
   #       norm1, norm2))
   #    raise ValueError


   # Add coriolis, metric and terms due to varying bottom topography
   forcing[idx_h,:,:] = 0.0

   forcing[idx_hu1,:,:] = 2.0 * ( metric.christoffel_1_01 * h * u1 + metric.christoffel_1_02 * h * u2) \
         + metric.christoffel_1_11 * h * u1**2 + 2.0 * metric.christoffel_1_12 * h * u1 * u2 \
         + gravity * h * ( metric.H_contra_11 * topo.dzdx1 + metric.H_contra_12 * topo.dzdx2)

   forcing[idx_hu2,:,:] = 2.0 * (metric.christoffel_2_01 * h * u1 + metric.christoffel_2_02 * h * u2) \
         + 2.0 * metric.christoffel_2_21 * h * u2 * u1 + metric.christoffel_2_22 * h * u2**2 \
         + gravity * h * ( metric.H_contra_21 * topo.dzdx1 + metric.H_contra_22 * topo.dzdx2)

   # Assemble the right-hand sides
   for var in range(3):
      rhs[var] = metric.inv_sqrtG * ( - df1_dx1[var] - df2_dx2[var] ) - forcing[var]

   if not shallow_water_equations:
      rhs[idx_hu1,:,:] = 0.0
      rhs[idx_hu2,:,:] = 0.0

   if filter_rhs:
      rhs[0,:,:] = apply_filter(rhs[0,:,:], mtrx, nb_elements_horiz, nbsolpts)
      rhs[1,:,:] = apply_filter(rhs[1,:,:], mtrx, nb_elements_horiz, nbsolpts)
      rhs[2,:,:] = apply_filter(rhs[2,:,:], mtrx, nb_elements_horiz, nbsolpts)

   timers[2].stop()

   return rhs
