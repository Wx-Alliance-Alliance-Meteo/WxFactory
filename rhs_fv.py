import numpy

from definitions  import idx_h, idx_hu1, idx_hu2, gravity
from graphx       import plot_field, plot_array, plot_vector_field

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
   # flux_itf = numpy.zeros((3, 2, 2, num_elements + 1, num_elements), dtype=datatype)

   f0_itf_ew = numpy.zeros((num_elements, num_elements + 1), dtype=datatype)
   f1_itf_ew = numpy.zeros_like(f0_itf_ew)
   f2_itf_ew = numpy.zeros_like(f0_itf_ew)

   f0_itf_ns = numpy.zeros((num_elements + 1, num_elements), dtype=datatype)
   f1_itf_ns = numpy.zeros_like(f0_itf_ns)
   f2_itf_ns = numpy.zeros_like(f0_itf_ns)

   f0_itf_ew[:, :-1]  = flux_e0_x1[:, :]
   f0_itf_ew[:, 1:]  += flux_e0_x1[:, :]

   f1_itf_ew[:, :-1]  = flux_e1_x1[:, :]
   f1_itf_ew[:, 1:]  += flux_e1_x1[:, :]

   f2_itf_ew[:, :-1]  = flux_e2_x1[:, :]
   f2_itf_ew[:, 1:]  += flux_e2_x1[:, :]

   f0_itf_ns[:-1, :]  = flux_e0_x2[:, :]
   f0_itf_ns[1:, :]  += flux_e0_x2[:, :]

   f1_itf_ns[:-1, :]  = flux_e1_x2[:, :]
   f1_itf_ns[1:, :]  += flux_e1_x2[:, :]

   f2_itf_ns[:-1, :]  = flux_e2_x2[:, :]
   f2_itf_ns[1:, :]  += flux_e2_x2[:, :]

   # f0_itf_ew[:, 1:] += flux_e0_x1[:, :]
   # f1_itf_ew[:, 1:] += f1_itf_ew[:, :-1]
   # f2_itf_ew[:, 1:] += f2_itf_ew[:, :-1]

   # f0_itf_ns[1:, :] += f0_itf_ns[:-1, :]
   # f1_itf_ns[1:, :] += f1_itf_ns[:-1, :]
   # f2_itf_ns[1:, :] += f2_itf_ns[:-1, :]

   X = geom.X[0, :]
   Y = geom.Y[:, 0]

   f0_n, f0_s, f0_w, f0_e = ptopo.xchange_simple_vectors(
      X, Y,
      flux_e0_x1[-1, :], flux_e0_x2[-1, :],  # North
      flux_e0_x1[0, :],  flux_e0_x2[0, :],   # South
      flux_e0_x1[:, 0],  flux_e0_x2[:, 0],   # West
      flux_e0_x1[:, -1], flux_e0_x2[:, -1]   # East
   )

   f0_itf_ns[-1, :] += f0_n[1]
   f0_itf_ns[0, :]  += f0_s[1]
   f0_itf_ew[:, 0]  += f0_w[0]
   f0_itf_ew[:, -1] += f0_e[0]

   # if ptopo.rank == 3:
   #    flux_e1_x2[0, :] = numpy.flip(flux_e1_x2[0, :])

   f1_n, f1_s, f1_w, f1_e = ptopo.xchange_simple_vectors(
      X, Y,
      flux_e1_x1[-1, :], flux_e1_x2[-1, :],  # North
      flux_e1_x1[0, :],  flux_e1_x2[0, :],   # South
      flux_e1_x1[:, 0],  flux_e1_x2[:, 0],   # West
      flux_e1_x1[:, -1], flux_e1_x2[:, -1]   # East
   )

   f1_itf_ns[-1, :] += f1_n[1]
   f1_itf_ns[0, :]  += f1_s[1]
   f1_itf_ew[:, 0]  = f1_w[0] * 2.0
   # f1_itf_ew[:, 0]  *= 2.0
   f1_itf_ew[:, -1] += f1_e[0]

   f2_n, f2_s, f2_w, f2_e = ptopo.xchange_simple_vectors(
      X, Y,
      flux_e2_x1[-1, :], flux_e2_x2[-1, :],  # North
      flux_e2_x1[0, :],  flux_e2_x2[0, :],   # South
      flux_e2_x1[:, 0],  flux_e2_x2[:, 0],   # West
      flux_e2_x1[:, -1], flux_e2_x2[:, -1]   # East
   )

   f2_itf_ns[-1, :] += f2_n[1]
   f2_itf_ns[0, :]  += f2_s[1]
   f2_itf_ew[:, 0]  += f2_w[0]
   f2_itf_ew[:, -1] += f2_e[0]

   f0_itf_ew *= 0.5
   f1_itf_ew *= 0.5
   f2_itf_ew *= 0.5
   f0_itf_ns *= 0.5
   f1_itf_ns *= 0.5
   f2_itf_ns *= 0.5

   #TODO Should divide these by elem size ?
   df0_dx1 = f0_itf_ew[:, 1:] - f0_itf_ew[:, :-1]
   df1_dx1 = f1_itf_ew[:, 1:] - f1_itf_ew[:, :-1]
   df2_dx1 = f2_itf_ew[:, 1:] - f2_itf_ew[:, :-1]

   df0_dx2 = f0_itf_ns[1:, :] - f0_itf_ns[:-1, :]
   df1_dx2 = f1_itf_ns[1:, :] - f1_itf_ns[:-1, :]
   df2_dx2 = f2_itf_ns[1:, :] - f2_itf_ns[:-1, :]

   # use_west  = f1_itf_ew[:, 1:] > f1_itf_ew[:, :-1]
   # df0_dx1 = f0_itf_ew[:, 1:]
   # df0_dx1[use_west] = f0_itf_ew[:, :-1][use_west]
   # df1_dx1 = f1_itf_ew[:, 1:]
   # df1_dx1[use_west] = f1_itf_ew[:, :-1][use_west]
   # df2_dx1 = f2_itf_ew[:, 1:]
   # df2_dx1[use_west] = f2_itf_ew[:, :-1][use_west]
   #
   # use_south = f2_itf_ns[1:, :] > f2_itf_ns[:-1, :]
   # df0_dx2 = f0_itf_ns[1:, :]
   # df0_dx2[use_south] = f0_itf_ns[:-1, :][use_south]
   # df1_dx2 = f1_itf_ns[1:, :]
   # df1_dx2[use_south] = f1_itf_ns[:-1, :][use_south]
   # df2_dx2 = f2_itf_ns[1:, :]
   # df2_dx2[use_south] = f2_itf_ns[:-1, :][use_south]

   forcing_0 = numpy.zeros((num_elements, num_elements))

   # plot_array(flux_e0_x1, filename='flux_e0_x1.png')
   # plot_array(f0_itf_ew[:, :-1], filename='f_itf_ew.png')

   plot_array(f0_itf_ew[:, :-1], filename='f0_itf_w_fv.png')
   plot_array(f1_itf_ew[:, :-1], filename='f1_itf_w_fv.png')
   plot_array(f2_itf_ew[:, :-1], filename='f2_itf_w_fv.png')
   # plot_array(f0_itf_ew[:, 1:], filename='f_itf_e_fv.png')

   plot_array(flux_e1_x1, filename='f1_x1_fv.png')
   plot_array(flux_e1_x2, filename='f1_x2_fv.png')

   # plot_array(f0_itf_ns[:-1, :], filename='f_itf_ns.png')
   # plot_array(df0_dx1, filename='f0_itf.png')
   # plot_array(df0_dx1)
   # plot_field(geom, f0_itf_ew[:, 1:])
   # plot_field(geom, df1_dx1, filename='df0_dx1_fv.png')
   # plot_field(geom, df0_dx1)
   # plot_field(geom, f0_itf_ew[:, :-1])

   # plot_field(geom, df0_dx1.real)
   # plot_array(df0_dx1)

   # plot_vector_field(geom, u1, u2)
   raise ValueError

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
