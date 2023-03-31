import numpy

from .bamphi_backward_error_analysis import backward_error_analysis

class StepState:
   def __init__(self):
      self.taus       = numpy.empty(0, dtype=float)    # Delta-t's between time steps
      self.sizes      = numpy.empty(0, dtype=int)
      self.tolerances = numpy.empty(0, dtype=float)
      self.ell        = numpy.empty(0, dtype=int)     # TODO what is that?
      self.x          = []     # TODO what is that too?
      self.x_complex  = []      # Map that indicates which x's are complex (1) and which are real (-1)
      self.s          = numpy.empty(0, dtype=int)
      self.divided_differences = []   # TODO what is that one?
      self.F          = []
      self.ts         = numpy.empty(0, dtype=float)
      self.jumps      = numpy.empty(0, dtype=int)

      self.latest_id  = -1

   def get_tau(self):
      return self.taus[self.latest_id]

   def set_s(self, s):
      self.s[self.latest_id] = s

   def get_s(self):
      return self.s[self.latest_id]

   def get_ts(self):
      return self.ts[self.latest_id]

   def set_ts(self, ts):
      self.ts[self.latest_id] = ts

   def get_p(self):
      return self.sizes[self.latest_id]

   def get_ell(self):
      return self.ell[self.latest_id]

   def get_tol(self):
      return self.tolerances[self.latest_id]

   def get_x(self):
      return self.x[self.latest_id]

   def get_divided_differences(self):
      return self.divided_differences[self.latest_id]

   def set_divided_differences(self, dd):
      self.divided_differences[self.latest_id] = dd

   def get_big_f(self):
      return self.F[self.latest_id]

   def set_big_f(self, F):
      self.F[self.latest_id] = F

   def get_x_complex(self):
      return self.x_complex[self.latest_id]

   def set_x_complex(self, new_map):
      self.x_complex[self.latest_id] = new_map

   def get_jump(self):
      return self.jumps[self.latest_id]

   def set_jump(self, jump):
      self.jumps[self.latest_id] = jump

   def add_item(self, tau, p, tol, max_degree, points):
      ids = numpy.where((self.taus == tau) * (self.sizes == p) * (self.tolerances == tol))[0]
      if ids.size == 0:
         self.latest_id = self.taus.size

         self.taus       = numpy.append(self.taus, tau)
         self.sizes      = numpy.append(self.sizes, p)
         self.tolerances = numpy.append(self.tolerances, tol)

         self.ell = numpy.append(self.ell, max(max_degree - p - points.num_points, 1))

         self.divided_differences.append(None)
         self.F.append(None)
         self.s = numpy.append(self.s, -1)

         if p > 0:
            zeros = numpy.zeros((p))
            mu = numpy.repeat(points.avg, self.ell[-1])
            self.x.append(numpy.append(zeros, numpy.append(points.rho, mu)))
         else:
            mu = numpy.repeat(points.avg, self.ell[-1] - 1)
            self.x.append(numpy.append(numpy.append(points.avg, points.rho), mu))
         # print(f'latest x: \n{self.x[-1]}')

         self.x_complex.append(None)

         return True

      else:
         self.latest_id = ids[0]

      return False

   def prepare_step(self, tau, p, tolerance, max_poly_degree, points, polygon):
      if self.add_item(tau, p, tolerance, max_poly_degree, points):
         backward_error_analysis(self, points, polygon)

         x_complex = numpy.where(numpy.isreal(self.get_x()), -1, 1)    # Do we really want == 0.0 ??
         x_complex = numpy.append(numpy.append(x_complex[0], x_complex), -x_complex[-1])
         self.set_x_complex(x_complex)

      if self.latest_id < 0:
         raise ValueError(f'self.latest_id is invalid! {self.latest_id}')
      elif self.latest_id < self.ts.size:
         self.ts[self.latest_id]    = self.taus[self.latest_id] / self.s[self.latest_id]
         self.jumps[self.latest_id] = 1
      elif self.latest_id == self.ts.size:
         self.ts    = numpy.append(self.ts, self.taus[self.latest_id] / self.s[self.latest_id])
         self.jumps = numpy.append(self.jumps, 1)
      else:
         raise ValueError(f'self.latest_id is too large! {self.latest_id}')

      new_jump = max(1, int(numpy.round( self.jumps[self.latest_id] * self.ts[self.latest_id] * self.s[self.latest_id] / self.taus[self.latest_id])))
      self.jumps[self.latest_id] = new_jump

   def update_d(self, jump):
      jump_limit = numpy.inf

      dd = self.get_divided_differences()
      while dd.shape[0] < jump:
         num_diffs = dd.shape[0]
         diff_size = dd.shape[1]
         cp1 = numpy.append(1.0, numpy.cumprod(num_diffs       * numpy.ones(diff_size - 1)))
         cp2 = numpy.append(1.0, numpy.cumprod((num_diffs + 1) * numpy.ones(diff_size - 1)))

         new_entry = dd[-1] * cp1
         new_entry = self.get_big_f().T @ new_entry
         new_entry = new_entry / cp2

         if (not numpy.all(numpy.isfinite(new_entry))):
            jump     = num_diffs
            jump_lim = num_diffs
            break

         dd = numpy.append(dd, [new_entry], axis=0)

      self.set_divided_differences(dd)

      return jump, jump_limit
