import numpy

c  = 343.0
c2 = c*c

# last_check = None
# last_eval = None
# last_evec = None
# last_einv = None

def perform_checks(jac, eigenval, eigenvec, eigenvec_inv):

   # global last_check, last_eval, last_evec, last_einv
   # if last_check is not None:
   #    if (jac.shape == last_check.shape):
   #       for a, b in zip([last_check, last_eval, last_evec, last_einv], [jac, eigenval, eigenvec, eigenvec_inv]):
   #          diff = b - a
   #          diff_norm = numpy.linalg.norm(diff)
   #          last_norm = numpy.linalg.norm(a)
   #          print(f'Diff {diff_norm:.2e} (rel {diff_norm/last_norm:.2e}), last {last_norm:.2e}')
   # last_check = jac
   # last_eval = eigenval
   # last_evec = eigenvec
   # last_einv = eigenvec_inv

   def check_elem(elem):
      f = jac[elem]
      eval = eigenval[elem]
      evec = eigenvec[elem]
      evec_i = eigenvec_inv[elem]

      if numpy.linalg.norm(f) == 0: return

      result = evec @ evec_i
      diff   = numpy.eye(5) - result
      diff_norm = numpy.linalg.norm(diff) #/ numpy.linalg.norm(numpy.eye(5))
      if diff_norm > 1e-13:
         numpy.set_printoptions(precision=6)
         raise ValueError(
            f'Error in eigenvectors for elem {elem}. error = {diff_norm:.3e} '
            f'\neigenvectors:\n{evec}'
            f'\ninverse:\n{evec_i}'
            f'\nresult:\n{result}'
            f'\ndiff:\n{diff}'
            f'\nflux jacobian\n{f}')

      result_jac = evec @ numpy.diag(eval) @ evec_i
      diff_jac   = result_jac - f
      diff_norm_jac = numpy.linalg.norm(diff_jac) / numpy.linalg.norm(f)
      if diff_norm_jac > 1e-15:
         numpy.set_printoptions(precision=9)
         raise ValueError(
            f'Error in flux jacobians for elem {elem}. error = {diff_norm_jac:.3e} '
            f'\n{evec}'
            f'\ninverse:\n{evec_i}'
            f'\neigenvalues:\n{eval}'
            f'\nR*Lambda:\n{evec @ numpy.diag(eval)}'
            f'\nresult:\n{result_jac}'
            f'\njacobian:\n{f}'
            f'\ndiff:\n{diff}')

   shape = jac.shape
   # print(f'Checking shape {shape}')
   if len(shape) == 5:
      for i in range(shape[0]):
         for j in range(shape[1]):
            for k in range(shape[2]):
               check_elem((i, j, k))
   elif len(shape) == 4:
      for i in range(shape[0]):
         for j in range(shape[1]):
            check_elem((i, j))
   else:
      raise ValueError(f'Check not implemented for this shape {shape}')

def eigenstuff_x(vars, metric_h):
   u1, u2, u3, theta = vars
   h11, h21, h31     = metric_h

   shape = u1.shape
   flux_j       = numpy.zeros(shape + (5, 5))
   eig_vals     = numpy.empty(shape + (5,))
   eig_vecs     = numpy.zeros(shape + (5, 5))
   eig_vecs_inv = numpy.zeros(shape + (5, 5))

   h11_sqrt = numpy.sqrt(h11)

   s = numpy.index_exp[:, :]
   if (len(shape) > 2): s += numpy.index_exp[:]

   flux_j[s + (0, 1)] = 1.0
   flux_j[s + (1, 0)] = h11 * c2 - u1 * u1
   flux_j[s + (1, 1)] = 2.0 * u1
   flux_j[s + (2, 0)] = h21 * c2 - u1 * u2
   flux_j[s + (2, 1)] = u2
   flux_j[s + (2, 2)] = u1
   flux_j[s + (3, 0)] = h31 * c2 - u1 * u3
   flux_j[s + (3, 1)] = u3
   flux_j[s + (3, 3)] = u1
   flux_j[s + (4, 0)] = -theta * u1
   flux_j[s + (4, 1)] = theta
   flux_j[s + (4, 4)] = u1

   eig_vals[s + (0,)] = u1
   eig_vals[s + (1,)] = u1
   eig_vals[s + (2,)] = u1
   eig_vals[s + (3,)] = u1 + c * h11_sqrt
   eig_vals[s + (4,)] = u1 - c * h11_sqrt

   eig_vecs[s + (0, 3)] = 1.0 / theta
   eig_vecs[s + (0, 4)] = eig_vecs[s + (0, 3)]
   eig_vecs[s + (1, 3)] = (u1 + c * h11_sqrt) / theta
   eig_vecs[s + (1, 4)] = (u1 - c * h11_sqrt) / theta
   eig_vecs[s + (2, 0)] = 1.0
   eig_vecs[s + (2, 3)] =  (c * h21 + u2 * h11_sqrt) / (theta * h11_sqrt)
   eig_vecs[s + (2, 4)] = -(c * h21 - u2 * h11_sqrt) / (theta * h11_sqrt)
   eig_vecs[s + (3, 1)] = 1.0
   eig_vecs[s + (3, 3)] =  (c * h31 + u3 * h11_sqrt) / (theta * h11_sqrt)
   eig_vecs[s + (3, 4)] = -(c * h31 - u3 * h11_sqrt) / (theta * h11_sqrt)
   eig_vecs[s + (4, 2)] = 1.0
   eig_vecs[s + (4, 3)] = 1.0
   eig_vecs[s + (4, 4)] = 1.0

   eig_vecs_inv[s + (0, 0)] = -(h11 * u2 - h21 * u1) / h11
   eig_vecs_inv[s + (0, 1)] = - h21 / h11
   eig_vecs_inv[s + (0, 2)] = 1.0
   eig_vecs_inv[s + (1, 0)] = -(h11 * u3 - h31 * u1) / h11
   eig_vecs_inv[s + (1, 1)] = - h31 / h11
   eig_vecs_inv[s + (1, 3)] = 1.0
   eig_vecs_inv[s + (2, 0)] = -theta
   eig_vecs_inv[s + (2, 4)] = 1.0
   eig_vecs_inv[s + (3, 0)] = -(theta * (u1 - c * h11_sqrt)) / (2.0 * c * h11_sqrt)
   eig_vecs_inv[s + (3, 1)] =  theta / (2.0 * c * h11_sqrt)
   eig_vecs_inv[s + (4, 0)] =  (theta * (u1 + c * h11_sqrt)) / (2.0 * c * h11_sqrt)
   eig_vecs_inv[s + (4, 1)] = -theta / (2.0 * c * h11_sqrt)

   # perform_checks(flux_j, eig_vals, eig_vecs, eig_vecs_inv)

   return flux_j, eig_vals, eig_vecs, eig_vecs_inv

def eigenstuff_y(vars, metric_h):
   u1, u2, u3, theta = vars
   h12, h22, h32     = metric_h

   shape = u1.shape
   flux_j       = numpy.zeros(shape + (5, 5))
   eig_vals     = numpy.zeros(shape + (5,))
   eig_vecs     = numpy.zeros(shape + (5, 5))
   eig_vecs_inv = numpy.zeros(shape + (5, 5))

   h22_sqrt = numpy.sqrt(h22)

   s = numpy.index_exp[:, :]
   if (len(shape) > 2): s += numpy.index_exp[:]

   flux_j[s + (0, 2)] = 1.0
   flux_j[s + (1, 0)] = h12 * c2 - u2 * u1
   flux_j[s + (1, 1)] = u2
   flux_j[s + (1, 2)] = u1
   flux_j[s + (2, 0)] = h22 * c2 - u2 * u2
   flux_j[s + (2, 2)] = 2.0 * u2
   flux_j[s + (3, 0)] = h32 * c2 - u2 * u3
   flux_j[s + (3, 2)] = u3
   flux_j[s + (3, 3)] = u2
   flux_j[s + (4, 0)] = -theta * u2
   flux_j[s + (4, 2)] = theta
   flux_j[s + (4, 4)] = u2

   eig_vals[s + (0,)] = u2
   eig_vals[s + (1,)] = u2
   eig_vals[s + (2,)] = u2
   eig_vals[s + (3,)] = u2 + c * h22_sqrt
   eig_vals[s + (4,)] = u2 - c * h22_sqrt

   eig_vecs[s + (0, 3)] = 1.0 / theta
   eig_vecs[s + (0, 4)] = eig_vecs[s + (0, 3)]
   eig_vecs[s + (1, 0)] = 1.0
   eig_vecs[s + (1, 3)] =  (c * h12 + u1 * h22_sqrt) / (theta * h22_sqrt)
   eig_vecs[s + (1, 4)] = -(c * h12 - u1 * h22_sqrt) / (theta * h22_sqrt)
   eig_vecs[s + (2, 3)] = (u2 + c * h22_sqrt) / theta
   eig_vecs[s + (2, 4)] = (u2 - c * h22_sqrt) / theta
   eig_vecs[s + (3, 1)] = 1.0
   eig_vecs[s + (3, 3)] =  (c * h32 + u3 * h22_sqrt) / (theta * h22_sqrt)
   eig_vecs[s + (3, 4)] = -(c * h32 - u3 * h22_sqrt) / (theta * h22_sqrt)
   eig_vecs[s + (4, 2)] = 1.0
   eig_vecs[s + (4, 3)] = 1.0
   eig_vecs[s + (4, 4)] = 1.0

   eig_vecs_inv[s + (0, 0)] = (h12 * u2 - h22 * u1) / h22
   eig_vecs_inv[s + (0, 1)] = 1.0
   eig_vecs_inv[s + (0, 2)] = - h12 / h22
   eig_vecs_inv[s + (1, 0)] = -(h22 * u3 - h32 * u2) / h22
   eig_vecs_inv[s + (1, 2)] = - h32 / h22
   eig_vecs_inv[s + (1, 3)] = 1.0
   eig_vecs_inv[s + (2, 0)] = -theta
   eig_vecs_inv[s + (2, 4)] = 1.0
   eig_vecs_inv[s + (3, 0)] = -(theta * (u2 - c * h22_sqrt)) / (2.0 * c * h22_sqrt)
   eig_vecs_inv[s + (3, 2)] =  theta / (2.0 * c * h22_sqrt)
   eig_vecs_inv[s + (4, 0)] =  (theta * (u2 + c * h22_sqrt)) / (2.0 * c * h22_sqrt)
   eig_vecs_inv[s + (4, 2)] = -theta / (2.0 * c * h22_sqrt)

   # perform_checks(flux_j, eig_vals, eig_vecs, eig_vecs_inv)
   return flux_j, eig_vals, eig_vecs, eig_vecs_inv

def eigenstuff_z(vars, metric_h):
   u1, u2, u3, theta = vars
   h13, h23, h33     = metric_h

   shape = u1.shape
   flux_j       = numpy.zeros(shape + (5, 5))
   eig_vals     = numpy.empty(shape + (5,))
   eig_vecs     = numpy.zeros(shape + (5, 5))
   eig_vecs_inv = numpy.zeros(shape + (5, 5))

   h33_sqrt = numpy.sqrt(h33)

   flux_j[:, :, :, 0, 3] = 1.0
   flux_j[:, :, :, 1, 0] = h13 * c2 - u3 * u1
   flux_j[:, :, :, 1, 1] = u3
   flux_j[:, :, :, 1, 3] = u1
   flux_j[:, :, :, 2, 0] = h23 * c2 - u3 * u2
   flux_j[:, :, :, 2, 2] = u3
   flux_j[:, :, :, 2, 3] = u2
   flux_j[:, :, :, 3, 0] = h33 * c2 - u3 * u3
   flux_j[:, :, :, 3, 3] = 2.0 * u3
   flux_j[:, :, :, 4, 0] = -theta * u3
   flux_j[:, :, :, 4, 3] = theta
   flux_j[:, :, :, 4, 4] = u3

   eig_vals[:, :, :, 0] = u3
   eig_vals[:, :, :, 1] = u3
   eig_vals[:, :, :, 2] = u3
   eig_vals[:, :, :, 3] = u3 + c * h33_sqrt
   eig_vals[:, :, :, 4] = u3 - c * h33_sqrt

   eig_vecs[:, :, :, 0, 3] = 1.0 / theta
   eig_vecs[:, :, :, 0, 4] = eig_vecs[:, :, :, 0, 3]
   eig_vecs[:, :, :, 1, 0] = 1.0
   eig_vecs[:, :, :, 1, 3] =  (c * h13 + u1 * h33_sqrt) / (theta * h33_sqrt)
   eig_vecs[:, :, :, 1, 4] = -(c * h13 - u1 * h33_sqrt) / (theta * h33_sqrt)
   eig_vecs[:, :, :, 2, 1] = 1.0
   eig_vecs[:, :, :, 2, 3] =  (c * h23 + u2 * h33_sqrt) / (theta * h33_sqrt)
   eig_vecs[:, :, :, 2, 4] = -(c * h23 - u2 * h33_sqrt) / (theta * h33_sqrt)
   eig_vecs[:, :, :, 3, 3] = (u3 + c * h33_sqrt) / theta
   eig_vecs[:, :, :, 3, 4] = (u3 - c * h33_sqrt) / theta
   eig_vecs[:, :, :, 4, 2] = 1.0
   eig_vecs[:, :, :, 4, 3] = 1.0
   eig_vecs[:, :, :, 4, 4] = 1.0

   eig_vecs_inv[:, :, :, 0, 0] = (h13 * u3 - h33 * u1) / h33
   eig_vecs_inv[:, :, :, 0, 1] = 1.0
   eig_vecs_inv[:, :, :, 0, 3] = - h13 / h33
   eig_vecs_inv[:, :, :, 1, 0] = (h23 * u3 - h33 * u2) / h33
   eig_vecs_inv[:, :, :, 1, 2] = 1.0
   eig_vecs_inv[:, :, :, 1, 3] = - h23 / h33
   eig_vecs_inv[:, :, :, 2, 0] = -theta
   eig_vecs_inv[:, :, :, 2, 4] = 1.0
   eig_vecs_inv[:, :, :, 3, 0] = -(theta * (u3 - c * h33_sqrt)) / (2.0 * c * h33_sqrt)
   eig_vecs_inv[:, :, :, 3, 3] =  theta / (2.0 * c * h33_sqrt)
   eig_vecs_inv[:, :, :, 4, 0] =  (theta * (u3 + c * h33_sqrt)) / (2.0 * c * h33_sqrt)
   eig_vecs_inv[:, :, :, 4, 3] = -theta / (2.0 * c * h33_sqrt)

   # perform_checks(flux_j, eig_vals, eig_vecs, eig_vecs_inv)
   return flux_j, eig_vals, eig_vecs, eig_vecs_inv
