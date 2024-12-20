import hashlib
import os
from typing import Callable, Optional, Tuple

import numpy
from mpi4py import MPI
import scipy

from common import Configuration
from solvers.eigenvalue_util import gen_matrix
from .preconditioner import Preconditioner


class Factorization(Preconditioner):
    def __init__(self, dtype, shape: Tuple, param: Configuration) -> None:
        super().__init__(dtype, shape, param)
        self.assembled_mat = None
        self.factorization = None
        self.type = param.preconditioner

        self.output_dir = param.output_dir

        def str_hash(s):
            return int(hashlib.md5(s.encode()).hexdigest(), 16)

        values = (
            param.dt,
            param.case_number,
            MPI.COMM_WORLD.size,
            str_hash(param.equations),
            str_hash(param.grid_type),
            str_hash(param.jacobian_method),
            param.num_solpts,
            param.num_elements_horizontal,
            param.num_elements_vertical,
        )

        matrix_hash = values.__hash__() & 0xFFFFFFFFFFFF
        # matrix_hash = int(hash_obj.hexdigest()) & 0xffffffffffff
        self.matrix_file = os.path.join(param.output_dir, f"mat_{matrix_hash:012x}.{MPI.COMM_WORLD.rank}.npz")

    def prepare(self, matvec: Callable[[numpy.ndarray], numpy.ndarray]) -> None:
        if self.assembled_mat is None:
            try:
                self.assembled_mat = scipy.sparse.load_npz(self.matrix_file)
            except (FileNotFoundError, OSError):
                pass

            if self.assembled_mat is None:
                self.assembled_mat = gen_matrix(matvec, self.matrix_file, compressed=True, local=True)

            if self.type == "lu":
                self.factorization = scipy.sparse.linalg.splu(self.assembled_mat)
            elif self.type == "ilu":
                self.factorization = scipy.sparse.linalg.spilu(self.assembled_mat, drop_tol=1e-5, fill_factor=50.0)

    def __apply__(
        self, vec: numpy.ndarray, x0: Optional[numpy.ndarray] = None, verbose: Optional[int] = None
    ) -> numpy.ndarray:

        if self.factorization is not None:
            return self.factorization.solve(vec)

        raise ValueError(f"Looks like this factorization-based preconditioner does *not* have a factorization...")
