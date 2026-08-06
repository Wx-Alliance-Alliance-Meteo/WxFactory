import torch
from torch import Tensor

from tests.unit.mpi_test import run_test_on_x_process
from tests.unit.wx_test import WxTestCase
from wx_factory.context import Context
from wx_factory.solvers.fgmres import fgmres


class FgmresMpiTestCases(WxTestCase):
    tolerance: float
    matrix_size_multiplier: int

    def setUp(self) -> None:
        super().setUp()
        self.tolerance = 1e-7
        self.matrix_size_multiplier = 24

    def test_fgmres_mpi_2_processes(self):
        comm = run_test_on_x_process(self, 2)
        context = Context(comm, "cpu")
        comm2 = comm.Split(comm.rank)
        context2 = Context(comm2, "cpu")

        size: int = comm.size * self.matrix_size_multiplier

        full_matrix = torch.empty((size, size), dtype=float)
        full_vector = torch.empty(size, dtype=float)

        for i in range(size):
            full_vector[i] = i
            for j in range(size):
                full_matrix[i, j] = j + size * i

        from_index: int = comm.rank * self.matrix_size_multiplier
        to_index: int = (comm.rank + 1) * self.matrix_size_multiplier
        matrix = full_matrix[:, from_index:to_index].clone()
        vector = full_vector[from_index:to_index].clone()

        def full_matvec_handle(v: Tensor) -> Tensor:
            return full_matrix @ v

        def partial_matvec_handle(v: Tensor) -> Tensor:
            return matrix @ v

        x1, *_ = fgmres(partial_matvec_handle, vector, tol=self.tolerance, context=context)
        x2, *_ = fgmres(full_matvec_handle, full_vector, tol=self.tolerance, context=context2)

        """diff: float = torch.linalg.norm(x1 - x2[0, from_index:to_index]).item()

        norm: float = torch.linalg.norm(x1).item()

        abs_diff = abs(diff)

        relative_diff = abs_diff / norm
        
        self.assertLessEqual(relative_diff, self.tolerance, 'The MPI implementation didn\' gave a result close to the non MPI implementation')"""

        comm2.Disconnect()
        comm.Disconnect()
