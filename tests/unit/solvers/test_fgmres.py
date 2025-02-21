import random
import unittest

from numpy import ndarray
import scipy
import scipy.sparse.linalg

from device import CpuDevice, CudaDevice
from solvers.fgmres import fgmres

import ndarray_generator
import cuda_test


class FgmresComparisonTestCases(cuda_test.CudaTestCases):
    tolerance: float
    rand: random.Random

    cpu_vector: ndarray
    gpu_vector: ndarray

    cpu_A_matrix: ndarray
    gpu_A_matrix: ndarray

    def setUp(self) -> None:
        super().setUp()

        self.cpu_device = CpuDevice()

        seed: int = 5646459
        initial_vector_size: int = 64
        rand_min: float = -1000.0
        rand_max: float = 1000.0

        self.tolerance = 1e-7
        self.rand = random.Random(seed)

        [self.cpu_vector, self.gpu_vector] = ndarray_generator.generate_vectors(
            initial_vector_size, self.rand, rand_min, rand_max, [self.cpu_device, self.gpu_device]
        )

        [self.cpu_A_matrix, self.gpu_A_matrix] = ndarray_generator.generate_matrixes(
            (initial_vector_size, initial_vector_size),
            self.rand,
            rand_min,
            rand_max,
            [self.cpu_device, self.gpu_device],
        )

    def cpu_matvec(self, v: ndarray) -> ndarray:
        return self.cpu_A_matrix @ v

    def gpu_matvec(self, v: ndarray) -> ndarray:
        return self.gpu_A_matrix @ v

    def test_compare_cpu_to_gpu(self):
        x1: ndarray
        x2: ndarray

        x1, norm_r1, norm_b1, niter1, flag1, residuals1 = fgmres(
            self.cpu_matvec, self.cpu_vector, tol=self.tolerance, device=self.cpu_device
        )
        x2, norm_r2, norm_b2, niter2, flag2, residuals2 = fgmres(
            self.gpu_matvec, self.gpu_vector, tol=self.tolerance, device=self.gpu_device
        )

        diff: float = self.cpu_device.xp.linalg.norm(x1 - self.gpu_device.to_host(x2)).item()

        x1_value: float = self.cpu_device.xp.linalg.norm(x1).item()
        x2_value: float = self.gpu_device.xp.linalg.norm(x2).item()

        abs_diff: float = abs(diff)

        relative_diff_x1: float = abs(abs_diff / x1_value)
        relative_diff_x2: float = abs(abs_diff / x2_value)

        self.assertLessEqual(relative_diff_x1, self.tolerance, "Fgmres didn't give a value close to the cpu value")
        self.assertLessEqual(relative_diff_x2, self.tolerance, "Fgmres didn't give a value close to the gpu value")


class FgmresScipyTestCases(unittest.TestCase):
    tolerance: float
    rand: random.Random

    cpu_vector: ndarray
    cpu_A_matrix: ndarray

    def setUp(self):
        super().setUp()

        self.cpu_device = CpuDevice()

        seed: int = 5646459
        self.tolerance = 1e-7
        self.rand = random.Random(seed)

        seed: int = 5646459
        initial_vector_size: int = 64
        rand_min: float = -1000.0
        rand_max: float = 1000.0

        self.tolerance = 1e-7
        self.rand = random.Random(seed)

        [self.cpu_vector] = ndarray_generator.generate_vectors(
            initial_vector_size, self.rand, rand_min, rand_max, [self.cpu_device]
        )

        [self.cpu_A_matrix] = ndarray_generator.generate_matrixes(
            (initial_vector_size, initial_vector_size), self.rand, rand_min, rand_max, [self.cpu_device]
        )

    def matvec(self, v: ndarray) -> ndarray:
        return self.cpu_A_matrix @ v

    def test_compare_implementation_to_scipy(self):
        x1: ndarray
        x2: ndarray

        x1, norm_r1, norm_b1, niter1, flag1, residuals1 = fgmres(
            self.matvec, self.cpu_vector, tol=self.tolerance, restart=20, device=self.cpu_device
        )
        x2, info = scipy.sparse.linalg.gmres(self.cpu_A_matrix, self.cpu_vector, atol=self.tolerance, restart=20)

        diff: float = self.cpu_device.xp.linalg.norm(x2 - x1).item()

        absolute_diff: float = abs(diff)

        relative_diff: float = abs(absolute_diff / self.cpu_device.xp.linalg.norm(x2).item())

        self.assertLessEqual(relative_diff, self.tolerance)

    def test_compare_implementation_to_scipy_and_residual(self):
        x1: ndarray
        x2: ndarray
        initial_vector_size: int = 64

        self.cpu_A_matrix = self.cpu_device.xp.eye(initial_vector_size, dtype=float)
        self.cpu_vector = self.cpu_device.xp.array(range(1, initial_vector_size + 1))
        self.cpu_A_matrix[initial_vector_size - 1, 0] = 1

        x1, norm_r1, norm_b1, niter1, flag1, residuals1 = fgmres(
            self.matvec, self.cpu_vector, tol=self.tolerance, restart=20, device=self.cpu_device
        )
        x2, info = scipy.sparse.linalg.gmres(self.cpu_A_matrix, self.cpu_vector, atol=self.tolerance, restart=20)

        diff: float = self.cpu_device.xp.linalg.norm(x2 - x1).item()
        resedual: float = self.cpu_device.xp.linalg.norm(self.cpu_A_matrix @ x1 - self.cpu_vector).item()

        absolute_diff: float = abs(diff)
        absolute_residual: float = abs(resedual)

        relative_diff: float = abs(absolute_diff / self.cpu_device.xp.linalg.norm(x2).item())
        relative_residual: float = abs(absolute_residual / self.cpu_device.xp.linalg.norm(self.cpu_vector).item())

        self.assertLessEqual(relative_diff, self.tolerance)
        self.assertLessEqual(relative_residual, self.tolerance)


class FgmresEdgeCasesTestCases(unittest.TestCase):
    tolerance: float
    rand: random.Random

    def setUp(self):
        super().setUp()

        self.cpu_device = CpuDevice()

        seed: int = 5646459
        self.tolerance = 1e-7
        self.rand = random.Random(seed)

    def test_fgmres_throw_when_b_is_smaller_or_equal_to_restart(self):
        bad_limit_vector_size: int = 20
        limit_vector_size: int = bad_limit_vector_size + 1
        bad_vector_size: int = bad_limit_vector_size - 1

        rand_min: float = -1000.0
        rand_max: float = 1000.0

        A_matrix: ndarray
        b: ndarray

        [A_matrix] = ndarray_generator.generate_matrixes(
            (limit_vector_size, limit_vector_size), self.rand, rand_min, rand_max, [self.cpu_device]
        )
        [b] = ndarray_generator.generate_vectors(limit_vector_size, self.rand, rand_min, rand_max, [self.cpu_device])

        def matvec(v: ndarray) -> ndarray:
            return A_matrix @ v

        # this one should not throw
        fgmres(matvec, b, tol=self.tolerance, restart=bad_limit_vector_size, device=self.cpu_device)

        [A_matrix] = ndarray_generator.generate_matrixes(
            (bad_limit_vector_size, bad_limit_vector_size), self.rand, rand_min, rand_max, [self.cpu_device]
        )
        [b] = ndarray_generator.generate_vectors(
            bad_limit_vector_size, self.rand, rand_min, rand_max, [self.cpu_device]
        )

        with self.assertRaises(ValueError):
            fgmres(matvec, b, tol=self.tolerance, restart=bad_limit_vector_size, device=self.cpu_device)

        [A_matrix] = ndarray_generator.generate_matrixes(
            (bad_vector_size, bad_vector_size), self.rand, rand_min, rand_max, [self.cpu_device]
        )
        [b] = ndarray_generator.generate_vectors(bad_vector_size, self.rand, rand_min, rand_max, [self.cpu_device])

        with self.assertRaises(ValueError):
            fgmres(matvec, b, tol=self.tolerance, restart=bad_limit_vector_size, device=self.cpu_device)
