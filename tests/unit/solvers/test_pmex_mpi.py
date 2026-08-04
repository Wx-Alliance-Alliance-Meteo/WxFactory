import torch
from torch import Tensor

from wx_factory.context import Context
from wx_factory.solvers import pmex

from mpi_test import run_test_on_x_process
from wx_test import WxTestCase


class PmexMpiTestCases(WxTestCase):
    tolerance: float
    matrix_size_multiplier: int

    def setUp(self) -> None:
        super().setUp()
        self.tolerance = 1e-7
        self.matrix_size_multiplier = 20

    def test_pmex_mpi_2_processes(self):
        comm = run_test_on_x_process(self, 2)
        context = Context(comm, "cpu")
        comm2 = comm.Split(comm.rank)
        context2 = Context(comm2, "cpu")

        def matvec_handle(v: Tensor) -> Tensor:
            return v

        size = comm.size * self.matrix_size_multiplier

        full_matrix = torch.empty((size, size), dtype=float)

        for i in range(size):
            for j in range(size):
                full_matrix[i, j] = j + size * i

        from_index = comm.rank * self.matrix_size_multiplier
        to_index = (comm.rank + 1) * self.matrix_size_multiplier
        matrix = full_matrix[:, from_index:to_index].copy()

        w1, _ = pmex(self.context.tensor([1.0]), matvec_handle, matrix, self.tolerance, context=context)
        w2, _ = pmex(self.context.tensor([1.0]), matvec_handle, full_matrix, self.tolerance, context=context2)

        diff = torch.linalg.norm(w1 - w2[0, from_index:to_index]).item()
        norm = torch.linalg.norm(w1).item()

        abs_diff = abs(diff)

        relative_diff = abs_diff / norm

        self.assertLessEqual(
            relative_diff,
            self.tolerance,
            "The MPI implementation didn' gave a result close to the non MPI implementation",
        )

        comm2.Disconnect()
        comm.Disconnect()
