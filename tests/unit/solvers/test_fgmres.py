import random

from mpi4py import MPI
import numpy
from numpy.typing import NDArray
import scipy
import scipy.sparse.linalg
import torch
from torch import Tensor

from wx_factory.device import Device
from wx_factory.solvers.fgmres import fgmres

import tests.unit.array_generator as array_generator
from wx_test import WxTestCase


class FgmresScipyTestCases(WxTestCase):
    tolerance: float
    rand: random.Random

    cpu_vector: Tensor
    cpu_A_matrix: Tensor

    def setUp(self):
        super().setUp()

        self.cpu_device = Device(MPI.COMM_WORLD, "cpu")

        seed: int = 5646459
        self.tolerance = 1e-7
        self.rand = random.Random(seed)

        seed: int = 5646459
        initial_vector_size: int = 64
        rand_min: float = -1000.0
        rand_max: float = 1000.0

        self.tolerance = 1e-7
        self.rand = random.Random(seed)

        [self.cpu_vector] = array_generator.generate_vectors(
            initial_vector_size, self.rand, rand_min, rand_max, [self.cpu_device]
        )

        [self.cpu_A_matrix] = array_generator.generate_matrices(
            (initial_vector_size, initial_vector_size), self.rand, rand_min, rand_max, [self.cpu_device]
        )

    def matvec(self, v: NDArray) -> NDArray:
        return self.cpu_A_matrix @ v

    def test_compare_implementation_to_scipy(self):

        x1, norm_r1, norm_b1, niter1, flag1, residuals1 = fgmres(
            self.matvec, self.cpu_vector, tol=self.tolerance, restart=20, device=self.cpu_device
        )
        x2, info = scipy.sparse.linalg.gmres(
            self.cpu_A_matrix.numpy(), self.cpu_vector.numpy(), atol=self.tolerance, restart=20
        )

        # wx fgmres returns a device tensor; scipy returns a numpy array. Compare on the host.
        diff = numpy.linalg.norm(x2 - x1.cpu().numpy()).item()

        absolute_diff = abs(diff)
        relative_diff = abs(absolute_diff / numpy.linalg.norm(x2).item())

        self.assertLessEqual(relative_diff, self.tolerance)

    def test_compare_implementation_to_scipy_and_residual(self):
        initial_vector_size: int = 64

        self.cpu_A_matrix = torch.eye(initial_vector_size, dtype=float)
        self.cpu_vector = torch.tensor(range(1, initial_vector_size + 1), dtype=float)
        self.cpu_A_matrix[initial_vector_size - 1, 0] = 1

        x1, norm_r1, norm_b1, niter1, flag1, residuals1 = fgmres(
            self.matvec, self.cpu_vector, tol=self.tolerance, restart=20, device=self.cpu_device
        )
        x2, info = scipy.sparse.linalg.gmres(
            self.cpu_A_matrix.numpy(), self.cpu_vector.numpy(), atol=self.tolerance, restart=20
        )

        residual: float = torch.linalg.norm(self.cpu_A_matrix @ x1 - self.cpu_vector).item()

        # wx fgmres returns a device tensor; scipy returns a numpy array. Compare on the host.
        x1_host = self.cpu_device.to_host(x1)
        diff: float = numpy.linalg.norm(x2 - x1_host).item()

        absolute_diff: float = abs(diff)
        absolute_residual: float = abs(residual)

        relative_diff: float = abs(absolute_diff / numpy.linalg.norm(x2).item())
        relative_residual: float = abs(absolute_residual / torch.linalg.norm(self.cpu_vector).item())

        self.assertLessEqual(relative_diff, self.tolerance)
        self.assertLessEqual(relative_residual, self.tolerance)


class FgmresEdgeCasesTestCases(WxTestCase):
    tolerance: float
    rand: random.Random

    def setUp(self):
        super().setUp()

        self.cpu_device = Device(MPI.COMM_WORLD, "cpu")

        seed: int = 5646459
        self.tolerance = 1e-7
        self.rand = random.Random(seed)

    def test_fgmres_throw_when_b_is_smaller_or_equal_to_restart(self):
        bad_limit_vector_size: int = 20
        limit_vector_size: int = bad_limit_vector_size + 1
        bad_vector_size: int = bad_limit_vector_size - 1

        rand_min: float = -1000.0
        rand_max: float = 1000.0

        [A_matrix] = array_generator.generate_matrices(
            (limit_vector_size, limit_vector_size), self.rand, rand_min, rand_max, [self.cpu_device]
        )
        [b] = array_generator.generate_vectors(limit_vector_size, self.rand, rand_min, rand_max, [self.cpu_device])

        def matvec(v: Tensor) -> Tensor:
            return A_matrix @ v

        # this one should not throw
        fgmres(matvec, b, tol=self.tolerance, restart=bad_limit_vector_size, device=self.cpu_device)

        [A_matrix] = array_generator.generate_matrices(
            (bad_limit_vector_size, bad_limit_vector_size), self.rand, rand_min, rand_max, [self.cpu_device]
        )
        [b] = array_generator.generate_vectors(bad_limit_vector_size, self.rand, rand_min, rand_max, [self.cpu_device])

        with self.assertRaises(ValueError):
            fgmres(matvec, b, tol=self.tolerance, restart=bad_limit_vector_size, device=self.cpu_device)

        [A_matrix] = array_generator.generate_matrices(
            (bad_vector_size, bad_vector_size), self.rand, rand_min, rand_max, [self.cpu_device]
        )
        [b] = array_generator.generate_vectors(bad_vector_size, self.rand, rand_min, rand_max, [self.cpu_device])

        with self.assertRaises(ValueError):
            fgmres(matvec, b, tol=self.tolerance, restart=bad_limit_vector_size, device=self.cpu_device)
