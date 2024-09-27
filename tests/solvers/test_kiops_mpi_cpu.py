import cpu_test
from solvers.kiops import kiops
from mpi_test import run_test_on_x_process
from numpy import ndarray


class KiopsMpiCpuTestCases(cpu_test.CpuTestCases):
    tolerance: float
    matrix_size_multiplier: int

    def setUp(self) -> None:
        super().setUp()
        self.tolerance = 1e-7
        self.matrix_size_multiplier = 20

    def test_pmex_mpi_2_processes(self):
        comm = run_test_on_x_process(self, 2)
        comm2 = comm.Split(comm.rank)

        def matvec_handle(v: ndarray) -> ndarray: return v

        size: int = comm.size * self.matrix_size_multiplier

        full_matrix: ndarray = self.cpu_device.xp.empty((size, size), dtype=float)

        for i in range(size):
            for j in range(size):
                full_matrix[i, j] = j + size * i

        from_index: int = comm.rank * self.matrix_size_multiplier
        to_index: int = (comm.rank + 1) * self.matrix_size_multiplier
        matrix: ndarray = full_matrix[:, from_index:to_index].copy()

        w1: ndarray
        w2: ndarray 

        w1, _ = kiops([1.0], matvec_handle, matrix, self.tolerance, device=self.cpu_device, comm=comm)
        w2, _ = kiops([1.0], matvec_handle, full_matrix, self.tolerance, device=self.cpu_device, comm=comm2)

        norm1: float = self.cpu_device.xp.linalg.norm(w1).item()
        norm2: float = self.cpu_device.xp.linalg.norm(w2[0, from_index:to_index]).item()

        abs_diff = abs(norm2 - norm1)

        relative_diff = abs_diff / norm2
        
        self.assertLessEqual(relative_diff, self.tolerance, 'The MPI implementation didn\' gave a result close to the non MPI implementation')

        comm2.Disconnect()
        comm.Disconnect()


