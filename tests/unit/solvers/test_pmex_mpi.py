import unittest

from numpy import ndarray

from device import CpuDevice
from solvers import pmex

from mpi_test import run_test_on_x_process


class PmexMpiTestCases(unittest.TestCase):
    tolerance: float
    matrix_size_multiplier: int

    def setUp(self) -> None:
        super().setUp()
        self.tolerance = 1e-7
        self.matrix_size_multiplier = 20

    def test_pmex_mpi_2_processes(self):
        comm = run_test_on_x_process(self, 2)
        device = CpuDevice(comm)
        comm2 = comm.Split(comm.rank)
        device2 = CpuDevice(comm2)

        def matvec_handle(v: ndarray) -> ndarray:
            return v

        size: int = comm.size * self.matrix_size_multiplier

        full_matrix: ndarray = device.xp.empty((size, size), dtype=float)

        for i in range(size):
            for j in range(size):
                full_matrix[i, j] = j + size * i

        from_index: int = comm.rank * self.matrix_size_multiplier
        to_index: int = (comm.rank + 1) * self.matrix_size_multiplier
        matrix: ndarray = full_matrix[:, from_index:to_index].copy()

        w1: ndarray
        w2: ndarray

        w1, _ = pmex([1.0], matvec_handle, matrix, self.tolerance, device=device)
        w2, _ = pmex([1.0], matvec_handle, full_matrix, self.tolerance, device=device2)

        diff = device.xp.linalg.norm(w1 - w2[0, from_index:to_index]).item()
        norm = device.xp.linalg.norm(w1).item()

        abs_diff = abs(diff)

        relative_diff = abs_diff / norm

        self.assertLessEqual(
            relative_diff,
            self.tolerance,
            "The MPI implementation didn' gave a result close to the non MPI implementation",
        )

        comm2.Disconnect()
        comm.Disconnect()
