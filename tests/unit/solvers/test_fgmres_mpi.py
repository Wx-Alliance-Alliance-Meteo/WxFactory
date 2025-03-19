import unittest

from numpy import ndarray

from device import CpuDevice
from solvers.fgmres import fgmres

from mpi_test import run_test_on_x_process


class FgmresMpiTestCases(unittest.TestCase):
    tolerance: float
    matrix_size_multiplier: int

    def setUp(self) -> None:
        super().setUp()
        self.tolerance = 1e-7
        self.matrix_size_multiplier = 24

    def test_fgmres_mpi_2_processes(self):
        comm = run_test_on_x_process(self, 2)
        device = CpuDevice(comm)
        comm2 = comm.Split(comm.rank)
        device2 = CpuDevice(comm2)

        size: int = comm.size * self.matrix_size_multiplier

        full_matrix: ndarray = device.xp.empty((size, size), dtype=float)
        full_vector: ndarray = device.xp.empty(size, dtype=float)

        for i in range(size):
            full_vector[i] = i
            for j in range(size):
                full_matrix[i, j] = j + size * i

        from_index: int = comm.rank * self.matrix_size_multiplier
        to_index: int = (comm.rank + 1) * self.matrix_size_multiplier
        matrix: ndarray = full_matrix[:, from_index:to_index].copy()
        vector: ndarray = full_vector[from_index:to_index].copy()

        def full_matvec_handle(v: ndarray) -> ndarray:
            return full_matrix @ v

        def partial_matvec_handle(v: ndarray) -> ndarray:
            return matrix @ v

        x1, *_ = fgmres(partial_matvec_handle, vector, tol=self.tolerance, device=device)
        x2, *_ = fgmres(full_matvec_handle, full_vector, tol=self.tolerance, device=device2)

        """diff: float = self.cpu_device.xp.linalg.norm(x1 - x2[0, from_index:to_index]).item()

        norm: float = self.cpu_device.xp.linalg.norm(x1).item()

        abs_diff = abs(diff)

        relative_diff = abs_diff / norm
        
        self.assertLessEqual(relative_diff, self.tolerance, 'The MPI implementation didn\' gave a result close to the non MPI implementation')"""

        comm2.Disconnect()
        comm.Disconnect()
