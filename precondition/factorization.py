from typing import Callable, Optional, Tuple
import os

import numpy
import scipy

from common.program_options  import Configuration
from scripts.eigenvalue_util import gen_matrix
from .preconditioner         import Preconditioner

import hashlib

class Factorization(Preconditioner):
   def __init__(self, dtype, shape: Tuple, param: Configuration) -> None:
      super().__init__(dtype, shape, param)
      self.assembled_mat = None
      self.factorization = None
      self.type = param.preconditioner

      self.output_dir = param.output_dir

      def str_hash(s):
         return int(hashlib.md5(s.encode()).hexdigest(), 16)

      values = (param.dt, param.case_number,
                str_hash(param.equations), str_hash(param.grid_type), str_hash(param.jacobian_method),
                param.nbsolpts, param.nb_elements_horizontal, param.nb_elements_vertical)

      matrix_hash = values.__hash__() & 0xffffffffffff
      # matrix_hash = int(hash_obj.hexdigest()) & 0xffffffffffff
      self.matrix_file = os.path.join(param.output_dir, f'mat_{matrix_hash:012x}.npz')

   def prepare(self, matvec: Callable[[numpy.ndarray], numpy.ndarray]) -> None:
      if self.assembled_mat is None:
         try:
            self.assembled_mat = scipy.sparse.load_npz(self.matrix_file)
         except OSError:
            self.assembled_mat = gen_matrix(matvec, self.matrix_file, compressed=True)

         if self.type == 'lu':
            self.factorization = scipy.sparse.linalg.splu(self.assembled_mat)
         elif self.type == 'ilu':
            self.factorization = scipy.sparse.linalg.spilu(self.assembled_mat, drop_tol=1e-5, fill_factor=50.0)

   def __apply__(self, vec: numpy.ndarray, x0: Optional[numpy.ndarray] = None, verbose: Optional[int] = None) \
         -> numpy.ndarray:

      if self.factorization is not None:
         return self.factorization.solve(vec)

      raise ValueError(f'Looks like this factorization-based preconditioner does *not* have a factorization...')
