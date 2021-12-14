import numpy
from time import time

from definitions import idx_rho, idx_rho_u1, idx_rho_u2, idx_rho_w, idx_rho_theta
from euler_eigen import c, eigenstuff_x, eigenstuff_y, eigenstuff_z, perform_checks

class SymmetricGaussSeidel:
   def __init__(self, Q_0, metric, pseudo_dt, ptopo, geom, eta) -> None:
      self.full_shape         = Q_0.shape
      self.nb_elem_horizontal = self.full_shape[2]
      self.nb_elem_vertical   = self.full_shape[1]

      # START LARGE
      self.sound_fraction = 0.5 # Large (0.2+) means stable, small (0.1-) means fast. Must be between 0 and 1
      self.eta    = eta
      self.dx     = numpy.array([2.0, 2.0, geom.Δx3]) * 5000.0
      # self.dx     = numpy.array([geom.Δx1, geom.Δx2, geom.Δx3]) * 10000.0
      # self.volume = self.dx[0] * self.dx[1] * self.dx[2]
      # self.ds     = [self.dx[1] * self.dx[2], self.dx[0] * self.dx[2], self.dx[0] * self.dx[1]]
      self.dt     = 1.0                  # TODO put as parameter?

      # self.volume = 100.0
      # self.ds = [1.0, 1.0, 1.0]

      self.pseudo_dt = pseudo_dt
      self.metric = metric

      # print(f'Volume {self.volume}, dx {self.dx}, ds {self.ds}')
      print(f'SGS precond with eta = {self.eta}, dx = {self.dx}, dt = {self.dt}, deltas {geom.Δx1}, {geom.Δx2}, {geom.Δx3}')

      ##############################
      # Unpack dynamical variables
      reduced_var = numpy.empty_like(Q_0)

      rho_full                   = Q_0[idx_rho]
      reduced_var[idx_rho_u1]    = Q_0[idx_rho_u1] / rho_full
      reduced_var[idx_rho_u2]    = Q_0[idx_rho_u2] / rho_full
      reduced_var[idx_rho_w]     = Q_0[idx_rho_w]  / rho_full
      reduced_var[idx_rho_theta] = Q_0[idx_rho_theta] / rho_full

      u1    = reduced_var[idx_rho_u1]
      u2    = reduced_var[idx_rho_u2]
      u3    = reduced_var[idx_rho_w]
      theta = reduced_var[idx_rho_theta]

      #TODO test with H_contra == Identity

      #TODO Set flux to zero for ground/sky boundary layers

      halo = numpy.empty((4, 5, u1.shape[1], u1.shape[2]))
      field_exchange = ptopo.xchange_sgs_fields(geom, reduced_var, halo[0], halo[1], halo[2], halo[3])

      t0 = time()

      ###############################################################
      # Compute flux jacobian operators at every relevant position

      halo_shape = (u1.shape[0] + 2, u1.shape[1] + 2, u1.shape[2] + 2)

      flux_jacobians   = numpy.zeros((3,3) + halo_shape + (5,5))
      eigenvalues      = numpy.zeros((3,3) + halo_shape + (5,))
      eigenvectors     = numpy.empty_like(flux_jacobians)
      eigenvectors_inv = numpy.empty_like(flux_jacobians)

      # x+
      s = numpy.s_[0, 2, 1:-1, 1:-1, 1:-1]
      flux_jacobians[s], eigenvalues[s], eigenvectors[s], eigenvectors_inv[s] = \
         eigenstuff_x((u1, u2, u3, theta), (metric.H_contra_11_itf_i[:, 1:], metric.H_contra_21_itf_i[:, 1:], metric.H_contra_31_itf_i[:, 1:]))

      # y+
      s = numpy.s_[1, 2, 1:-1, 1:-1, 1:-1]
      flux_jacobians[s], eigenvalues[s], eigenvectors[s], eigenvectors_inv[s] = \
         eigenstuff_y((u1, u2, u3, theta), (metric.H_contra_12_itf_j[1:, :], metric.H_contra_22_itf_j[1:, :], metric.H_contra_32_itf_j[1:, :]))

      # z+
      s = numpy.s_[2, 2, 1:-1, 1:-1, 1:-1]
      flux_jacobians[s], eigenvalues[s], eigenvectors[s], eigenvectors_inv[s] = \
         eigenstuff_z((u1, u2, u3, theta), (metric.H_contra_13, metric.H_contra_23, metric.H_contra_33))

      # x-
      s = numpy.s_[0, 0, 1:-1, 1:-1, 1:-1]
      flux_jacobians[s], eigenvalues[s], eigenvectors[s], eigenvectors_inv[s] = \
         eigenstuff_x((u1, u2, u3, theta), (metric.H_contra_11_itf_i[:, :-1], metric.H_contra_21_itf_i[:, :-1], metric.H_contra_31_itf_i[:, :-1]))

      # y-
      s = numpy.s_[1, 0, 1:-1, 1:-1, 1:-1]
      flux_jacobians[s], eigenvalues[s], eigenvectors[s], eigenvectors_inv[s] = \
         eigenstuff_y((u1, u2, u3, theta), (metric.H_contra_12_itf_j[:-1, :], metric.H_contra_22_itf_j[:-1, :], metric.H_contra_32_itf_j[:-1, :]))
      
      # z- = z+ 
      flux_jacobians[2, 0], eigenvalues[2, 0], eigenvectors[2, 0], eigenvectors_inv[2, 0] = \
         flux_jacobians[2, 2], eigenvalues[2, 2], eigenvectors[2, 2], eigenvectors_inv[2, 2]

      # x-mid
      s = numpy.s_[0, 1, 1:-1, 1:-1, 1:-1]
      flux_jacobians[s], eigenvalues[s], eigenvectors[s], eigenvectors_inv[s] = \
         eigenstuff_x((u1, u2, u3, theta), (metric.H_contra_11, metric.H_contra_21, metric.H_contra_31))

      # y-mid
      s = numpy.s_[1, 1, 1:-1, 1:-1, 1:-1]
      flux_jacobians[s], eigenvalues[s], eigenvectors[s], eigenvectors_inv[s] = \
         eigenstuff_y((u1, u2, u3, theta), (metric.H_contra_12, metric.H_contra_22, metric.H_contra_32))

      # z-mid = z+ = z-
      flux_jacobians[2, 1], eigenvalues[2, 1], eigenvectors[2, 1], eigenvectors_inv[2, 1] = \
         flux_jacobians[2, 2], eigenvalues[2, 2], eigenvectors[2, 2], eigenvectors_inv[2, 2]

      field_exchange.wait()

      # West border (left of current tile, right of left element, positive along x)
      s = numpy.s_[0, 2, 1:-1, 0, 1:-1]
      flux_jacobians[s], eigenvalues[s], eigenvectors[s], eigenvectors_inv[s] = \
         eigenstuff_x((halo[2, idx_rho_u1], halo[2, idx_rho_u2], halo[2, idx_rho_w], halo[2, idx_rho_theta]),
                     (metric.H_contra_11_itf_i[:, 0], metric.H_contra_21_itf_i[:, 0], metric.H_contra_31_itf_i[:, 0]))

      # South border (bottom of current tile, top of bottom element, positive along y)
      s = numpy.s_[1, 2, 1:-1, 1:-1, 0]
      flux_jacobians[s], eigenvalues[s], eigenvectors[s], eigenvectors_inv[s] = \
         eigenstuff_y((halo[1, idx_rho_u1], halo[1, idx_rho_u2], halo[1, idx_rho_w], halo[1, idx_rho_theta]),
                     (metric.H_contra_12_itf_j[0, :], metric.H_contra_22_itf_j[0, :], metric.H_contra_32_itf_j[0, :]))

      # East border (right of current tile, left of right element, negative along x)
      s = numpy.s_[0, 0, 1:-1, -1, 1:-1]
      flux_jacobians[s], eigenvalues[s], eigenvectors[s], eigenvectors_inv[s] = \
         eigenstuff_x((halo[3, idx_rho_u1], halo[3, idx_rho_u2], halo[3, idx_rho_w], halo[3, idx_rho_theta]),
                     (metric.H_contra_11_itf_i[:, -1], metric.H_contra_21_itf_i[:, -1], metric.H_contra_31_itf_i[:, -1]))

      # North border (top of current tile, bottom of top element, negative along y)
      s = numpy.s_[1, 0, 1:-1, 1:-1, -1]
      flux_jacobians[s], eigenvalues[s], eigenvectors[s], eigenvectors_inv[s] = \
         eigenstuff_y((halo[0, idx_rho_u1], halo[0, idx_rho_u2], halo[0, idx_rho_w], halo[0, idx_rho_theta]),
                     (metric.H_contra_12_itf_j[-1, :], metric.H_contra_22_itf_j[-1, :], metric.H_contra_32_itf_j[-1, :]))

      # perform_checks(flux_jacobians[s], eigenvalues[s], eigenvectors[s], eigenvectors[s])

      t1 = time()

      # Verify the computations
      for dim in range(3):
         for pos in range(3):
            # print(f'dim {dim}, pos {pos}')
            # s = numpy.s_[dim, pos, 1:-1, 1:-1, 1:-1]
            s = numpy.s_[dim, pos, :, :, :]
            perform_checks(flux_jacobians[s], eigenvalues[s], eigenvectors[s], eigenvectors_inv[s])

      ####################################################
      # Adjust eigenvalues:
      #   - apply threshold to avoid values close to 0
      #   - split into positive and negative

      cutoff   = c * self.sound_fraction

      eigenvalues_plus  = numpy.where(numpy.abs(eigenvalues) < cutoff, numpy.copysign(0.5 * (cutoff + eigenvalues*eigenvalues/cutoff), eigenvalues), eigenvalues)
      eigenvalues_minus = eigenvalues_plus.copy()

      eigenvalues_plus  = numpy.maximum(eigenvalues_plus, 0.0)
      eigenvalues_minus = numpy.minimum(eigenvalues_minus, 0.0)

      #################################################
      # Compute the positive/negative flux jacobians
      self.flux_jacobians_plus  = numpy.zeros_like(flux_jacobians)
      self.flux_jacobians_minus = numpy.zeros_like(flux_jacobians)
      t2 = time()
      # A vectorized/hidden loop implementation would be useful here...
      # f_jacobian_plus [:, :, :, :] = eigenvectors[:, :, :, :] @ numpy.diag(eigen_plus [:, :, :, :]) @ eigenvectors_inv[:, :, :, :]
      # f_jacobian_minus[:, :, :, :] = eigenvectors[:, :, :, :] @ numpy.diag(eigen_minus[:, :, :, :]) @ eigenvectors_inv[:, :, :, :]
      for dim in range(3):
         for pos in range(3):
            for k in range(self.nb_elem_vertical):
               for i in range(self.nb_elem_horizontal):
                  for j in range(self.nb_elem_horizontal):
                     s = numpy.s_[dim, pos, k, i, j]
                     self.flux_jacobians_plus [s] = eigenvectors[s] @ numpy.diag(eigenvalues_plus [s]) @ eigenvectors_inv[s]
                     self.flux_jacobians_minus[s] = eigenvectors[s] @ numpy.diag(eigenvalues_minus[s]) @ eigenvectors_inv[s]
      t3 = time()

      self.D_stored = numpy.empty((self.nb_elem_vertical, self.nb_elem_horizontal, self.nb_elem_horizontal, 5, 5))
      self.D_inv_stored = numpy.empty_like(self.D_stored)

      for k in range(self.nb_elem_vertical):
         for i in range(self.nb_elem_horizontal):
            for j in range(self.nb_elem_horizontal):
               self.D_stored[k, i, j] = self.D((k, i, j))
               self.D_inv_stored[k, i, j] = self.D_inv((k, i, j))

      t4 = time()

      # print(f'Computed stuff in {t1 - t0:.2e} s, {t3 - t2:.2e} s')

   def L(self, elem1, elem2, dim):
      # v  = self.volume
      # dx = self.ds[dim]
      dx = self.dx[dim]
      dt = self.pseudo_dt[elem1]
      f_elem1 = (dim, 1, elem1[0] + 1, elem1[1] + 1, elem1[2] + 1)
      f_elem2 = (dim, 1, elem2[0] + 1, elem2[1] + 1, elem2[2] + 1)
      j_elem  = 0.5 *  (self.flux_jacobians_plus[f_elem1] + self.flux_jacobians_plus[f_elem2])
      return numpy.zeros_like(j_elem)
      return ((-self.eta * dt) / dx) * j_elem

   def U(self, elem1, elem2, dim):
      # v  = self.volume
      # dx = self.ds[dim]
      dx = self.dx[dim]
      dt = self.pseudo_dt[elem1]
      f_elem1 = (dim, 1, elem1[0] + 1, elem1[1] + 1, elem1[2] + 1)
      f_elem2 = (dim, 1, elem2[0] + 1, elem2[1] + 1, elem2[2] + 1)
      j_elem  = 0.5 * (self.flux_jacobians_minus[f_elem1] + self.flux_jacobians_minus[f_elem2])
      return numpy.zeros_like(j_elem)
      return ((self.eta * dt) / dx) * j_elem

   def D(self, elem):
      f_elem = (elem[0] + 1, elem[1] + 1, elem[2] + 1)
      # v  = self.volume
      dt = self.pseudo_dt[elem]
      I  = numpy.eye(5)

      t1 = I
      t2 = (2.0 * self.eta * dt) / (1.0 * self.dt) * I
      t3 = self.eta * dt * \
            ((1.0/self.dx[0])*(self.flux_jacobians_plus[(0,1)+f_elem] - self.flux_jacobians_minus[(0,1)+f_elem]) +
             (1.0/self.dx[1])*(self.flux_jacobians_plus[(1,1)+f_elem] - self.flux_jacobians_minus[(1,1)+f_elem]) +
             (1.0/self.dx[2])*(self.flux_jacobians_plus[(2,1)+f_elem] - self.flux_jacobians_minus[(2,1)+f_elem]))
      D = t1 + t2 + t3
      # print(f't2 = \n{t2}, \nt3 =\n{t3}')
      return D

   def D_inv(self, elem):
      return numpy.linalg.inv(self.D(elem))

   def __call__(self, vec):
      return self.apply_sgs(vec)

   def backward_step(self, rhs):
      def reverse_range(n):
         return range(n-1, -1, -1)
      sol = rhs.copy()
      for k in reverse_range(self.nb_elem_vertical):
         for i in reverse_range(self.nb_elem_horizontal):
            for j in reverse_range(self.nb_elem_horizontal):
               value = sol[0:5, k, i, j].copy()
               if (k < self.nb_elem_vertical - 1):   value -= self.U((k, i, j), (k+1, i, j), 2) @ sol[0:5, k+1, i, j]
               if (i < self.nb_elem_horizontal - 1): value -= self.U((k, i, j), (k, i+1, j), 0) @ sol[0:5, k, i+1, j]
               if (j < self.nb_elem_horizontal - 1): value -= self.U((k, i, j), (k, i, j+1), 1) @ sol[0:5, k, i, j+1]
               # sol[0:5, k, i, j] = self.D_inv((k, i, j)) @ value
               sol[0:5, k, i, j] = self.D_inv_stored[k, i, j] @ value
      return sol

   def diagonal_step(self, rhs):
      sol = rhs.copy()
      for k in range(self.nb_elem_vertical):
         for i in range(self.nb_elem_horizontal):
            for j in range(self.nb_elem_horizontal):
               # sol[0:5, k, i, j] = self.D((k, i, j)) @ sol[0:5, k, i, j]
               sol[0:5, k, i, j] = self.D_stored[k, i, j] @ sol[0:5, k, i, j]
      return sol

   def forward_step(self, rhs):
      sol = rhs.copy()
      for k in range(self.nb_elem_vertical):
         for i in range(self.nb_elem_horizontal):
            for j in range(self.nb_elem_horizontal):
               value = sol[0:5, k, i, j].copy()
               # if (k > 0): value -= L((k, i, j), (k-1, i, j), 2) @ sol[0:5, k-1, i, j]
               # if (i > 0): value -= L((k, i, j), (k, i-1, j), 0) @ sol[0:5, k, i-1, j]
               # if (j > 0): value -= L((k, i, j), (k, i, j-1), 1) @ sol[0:5, k, i, j-1]
               if (k > 0): value -= self.L((k-1, i, j), (k, i, j), 2) @ sol[0:5, k-1, i, j]
               if (i > 0): value -= self.L((k, i-1, j), (k, i, j), 0) @ sol[0:5, k, i-1, j]
               if (j > 0): value -= self.L((k, i, j-1), (k, i, j), 1) @ sol[0:5, k, i, j-1]
               # sol[0:5, k, i, j] = self.D_inv((k, i, j)) @ value
               sol[0:5, k, i, j] = self.D_inv_stored[k, i, j] @ value
      return sol

   def apply_sgs(self, vec):
      """
      Solve (D+U)D^-1(D+L)x = b
      1. Backward substitute to solve (D+U)y = b
      2. Solve diagonal D^-1 z = y
      3. Forward substitute to solve (D+L)x = z
      """

      b = vec.reshape(self.full_shape) #* self.metric.sqrtG
      y = self.backward_step(b)
      z = self.diagonal_step(y)
      x = self.forward_step(z) #/ self.metric.sqrtG

      sol = x
               
      if False:
         diff = sol - b
         diff_norm = numpy.linalg.norm(diff[0:5])
         orig_norm = numpy.linalg.norm(b[0:5])
         new_norm = numpy.linalg.norm(sol[0:5])
         print (f'diff norm = {diff_norm:.2e} (rel {diff_norm/orig_norm:.4e}), b norm: {orig_norm:.2e}, result norm: {new_norm:.2e}')

      # return vec

      return numpy.ravel(sol)
