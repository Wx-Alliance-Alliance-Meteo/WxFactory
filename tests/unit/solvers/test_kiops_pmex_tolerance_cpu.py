import random
import unittest

from numpy import ndarray

from device import CpuDevice
from solvers.kiops import kiops
from solvers.pmex import pmex

import ndarray_generator


class KiopsPmexToleranceCpuTestCases(unittest.TestCase):
    tolerance: float
    rand: random.Random

    kiops_matrix: ndarray
    pmex_matrix: ndarray

    def setUp(self) -> None:
        super().setUp()

        self.cpu_device = CpuDevice()

        seed: int = 5646459
        initial_matrix_size: int = 64
        rand_min: float = -1000.0
        rand_max: float = 1000.0

        self.show_debug_print = False

        self.tolerance = 1e-7
        self.rand = random.Random(seed)

        [self.kiops_matrix, self.pmex_matrix] = ndarray_generator.generate_matrixes(
            (initial_matrix_size, initial_matrix_size),
            self.rand,
            rand_min,
            rand_max,
            [self.cpu_device, self.cpu_device],
        )

    def test_compare_kiops_pmex(self):
        def matvec_handle(v: ndarray) -> ndarray:
            return v

        w1: ndarray
        w2: ndarray

        w1, stats1 = kiops([1.0], matvec_handle, self.kiops_matrix, self.tolerance, device=self.cpu_device)
        w2, stats2 = pmex([1.0], matvec_handle, self.pmex_matrix, self.tolerance, device=self.cpu_device)

        shape: tuple[int, int] = w1.shape

        self.assertEqual(len(w1.shape), 2, "Kiops didn't return a matrix")
        self.assertEqual(len(w2.shape), 2, "Pmex didn't return a matrix")

        self.failIf(not (w2.shape[0] == shape[0] and w2.shape[1] == shape[1]), "Both matrix should be the same size")

        diff: float = self.cpu_device.xp.linalg.norm(w1 - w2).item()

        w1_value: float = self.cpu_device.xp.linalg.norm(w1).item()
        w2_value: float = self.cpu_device.xp.linalg.norm(w2).item()

        abs_diff: float = abs(diff)

        relative_diff_w1: float = abs(abs_diff / w1_value)
        relative_diff_w2: float = abs(abs_diff / w2_value)

        self.assertLessEqual(relative_diff_w1, self.tolerance, f"Kiops didn't give a close result")
        self.assertLessEqual(relative_diff_w2, self.tolerance, f"Pmex didn't give a close result")
