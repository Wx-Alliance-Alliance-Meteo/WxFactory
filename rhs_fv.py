import numpy

from definitions import idx_h, idx_hu1, idx_hu2, gravity

def rhs_sw_fv(Q, geom, mtrx, metric, topo, ptopo,
              nbsolpts: int, nb_elements_horiz: int, case_number: int,
              filter_rhs: bool = False):

   datatype = Q.dtype
   num_elements = Q.shape[1]
   # print(f'Num_elements: {num_elements}')

   h    = Q[idx_h, :, :]
   h_u1 = Q[idx_hu1, :, :]
   h_u2 = Q[idx_hu2, :, :]
   u1   = h_u1 / h
   u2   = h_u2 / h

   h_squared = h**2
   h_topo    = h + topo.hsurf

   halo_offset = 1

   flux_e0_x1 = h * metric.sqrtG * u1
   flux_e0_x2 = h * metric.sqrtG * u2

   flux_e1_x1 = metric.sqrtG * (h_u1 * u1 + 0.5 * gravity * metric.H_contra_11 * h_squared)
   flux_e1_x2 = metric.sqrtG * (h_u1 * u2 + 0.5 * gravity * metric.H_contra_12 * h_squared)

   flux_e2_x1 = metric.sqrtG * (h_u2 * u1 + 0.5 * gravity * metric.H_contra_21 * h_squared)
   flux_e2_x2 = metric.sqrtG * (h_u2 * u2 + 0.5 * gravity * metric.H_contra_22 * h_squared)

   # --- Compute flux at interfaces
   flux_itf = numpy.zeros((3, 2, 2, num_elements + 1, num_elements), dtype=datatype)
   f0_itf_ns = flux_itf[0, 0, :, :, :]
   f0_itf_ew = flux_itf[0, 1, :, :, :]
   f1_itf_ns = flux_itf[1, 0, :, :, :]
   f1_itf_ew = flux_itf[1, 1, :, :, :]
   f2_itf_ns = flux_itf[2, 0, :, :, :]
   f2_itf_ew = flux_itf[2, 1, :, :, :]

   # e0
   f0_itf_ew[0, :-1, :] = flux_e0_x1[:, :]    # west-east, first component
   f0_itf_ew[1, :-1, :] = flux_e0_x2[:, :]    # west-east, second component
   f0_itf_ns[0, :-1, :] = flux_e0_x1[:, :].T  # south-north
   f0_itf_ns[1, :-1, :] = flux_e0_x2[:, :].T  # south-north

   # e1
   f1_itf_ew[0, :-1, :] = flux_e1_x1[:, :]    # west-east
   f1_itf_ew[1, :-1, :] = flux_e1_x2[:, :]    # west-east
   f1_itf_ns[0, :-1, :] = flux_e1_x1[:, :].T  # south-north
   f1_itf_ns[1, :-1, :] = flux_e1_x2[:, :].T  # south-north

   # e2
   f2_itf_ew[0, :-1, :] = flux_e2_x1[:, :]    # west-east
   f2_itf_ew[1, :-1, :] = flux_e2_x2[:, :]    # west-east
   f2_itf_ns[0, :-1, :] = flux_e2_x1[:, :].T  # south-north
   f2_itf_ns[1, :-1, :] = flux_e2_x2[:, :].T  # south-north

   for i in range(3):
      for j in range(2):
         for k in range(2):
            flux_itf[i, j, k, 1:, :] += flux_itf[i, j, k, :-1, :]

   for iEq in range(3):
      f_n, f_s, f_w, f_e = ptopo.xchange_simple_vectors(
         geom.X[0, :], geom.Y[:, 0],
         flux_itf[iEq, 0, 0, -1, :], flux_itf[iEq, 0, 1, -1, :],  # North
         flux_itf[iEq, 0, 0,  0, :], flux_itf[iEq, 0, 1,  0, :],  # South
         flux_itf[iEq, 1, 0,  0, :], flux_itf[iEq, 1, 1,  0, :],  # West
         flux_itf[iEq, 1, 0, -1, :], flux_itf[iEq, 1, 1, -1, :]   # East
      )

      flux_itf[iEq, 0, 0, -1, :] += f_n[0]
      flux_itf[iEq, 0, 1, -1, :] += f_n[1]
      flux_itf[iEq, 0, 0,  0, :] += f_s[0]
      flux_itf[iEq, 0, 1,  0, :] += f_s[1]
      flux_itf[iEq, 1, 0,  0, :] += f_w[0]
      flux_itf[iEq, 1, 1,  0, :] += f_w[1]
      flux_itf[iEq, 1, 0, -1, :] += f_e[0]
      flux_itf[iEq, 1, 1, -1, :] += f_e[1]

   flux_itf[:] *= 0.5

   #TODO Should divide these by elem size ?
   df0_dx1 = f0_itf_ew[0, 1:, :] - f0_itf_ew[0, :-1, :]
   df1_dx1 = f1_itf_ew[0, 1:, :] - f1_itf_ew[0, :-1, :]
   df2_dx1 = f2_itf_ew[0, 1:, :] - f2_itf_ew[0, :-1, :]

   df0_dx2 = (f0_itf_ns[1, 1:, :] - f0_itf_ns[1, :-1, :]).T
   df1_dx2 = (f1_itf_ns[1, 1:, :] - f1_itf_ns[1, :-1, :]).T
   df2_dx2 = (f2_itf_ns[1, 1:, :] - f2_itf_ns[1, :-1, :]).T

   forcing_0 = numpy.zeros((num_elements, num_elements))

   # Note: christoffel_1_22 is zero
   forcing_1 = 2.0 * (metric.christoffel_1_01 * h * u1 + metric.christoffel_1_02 * h * u2) \
               + metric.christoffel_1_11 * h * u1**2 + 2.0 * metric.christoffel_1_12 * h * u1 * u2 \
               + gravity * h * (metric.H_contra_11 * topo.dzdx1 + metric.H_contra_12 * topo.dzdx2)

   # Note: metric.christoffel_2_11 is zero
   forcing_2 = 2.0 * (metric.christoffel_2_01 * h * u1 + metric.christoffel_2_02 * h * u2) \
               + 2.0 * metric.christoffel_2_12 * h * u1 * u2 + metric.christoffel_2_22 * h * u2**2 \
               + gravity * h * (metric.H_contra_21 * topo.dzdx1 + metric.H_contra_22 * topo.dzdx2)

   rhs = numpy.empty_like(Q)
   rhs[idx_h]   = metric.inv_sqrtG * -(df0_dx1 + df0_dx2) - forcing_0
   rhs[idx_hu1] = metric.inv_sqrtG * -(df1_dx1 + df1_dx2) - forcing_1
   rhs[idx_hu2] = metric.inv_sqrtG * -(df2_dx1 + df2_dx2) - forcing_2

   return rhs
