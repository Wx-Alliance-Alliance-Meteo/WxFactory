import random

import array_generator
import torch
from mpi4py import MPI
from torch import Tensor
from wx_test import WxTestCase

from wx_factory.context import Context
from wx_factory.solvers.kiops import kiops
from wx_factory.solvers.pmex import pmex


class KiopsPmexToleranceCpuTestCases(WxTestCase):
    tolerance: float
    rand: random.Random

    def setUp(self) -> None:
        super().setUp()

        self.cpu_context = Context(self.comm, "cpu")

        seed: int = 5646459
        initial_matrix_size: int = 64
        rand_min: float = -1000.0
        rand_max: float = 1000.0

        self.show_debug_print = False

        self.tolerance = 1e-7
        self.rand = random.Random(seed)

        [self.kiops_matrix, self.pmex_matrix] = array_generator.generate_matrices(
            (initial_matrix_size, initial_matrix_size),
            self.rand,
            rand_min,
            rand_max,
            [self.cpu_context, self.cpu_context],
        )

    def test_compare_kiops_pmex(self):
        def matvec_handle(v: Tensor) -> Tensor:
            return v

        w1, _ = kiops(
            self.cpu_context.tensor([1.0]), matvec_handle, self.kiops_matrix, self.tolerance, context=self.cpu_context
        )
        w2, _ = pmex(
            self.cpu_context.tensor([1.0]), matvec_handle, self.pmex_matrix, self.tolerance, context=self.cpu_context
        )

        shape = w1.shape

        self.assertEqual(len(w1.shape), 2, "Kiops didn't return a matrix")
        self.assertEqual(len(w2.shape), 2, "Pmex didn't return a matrix")

        self.assertEqual(w2.shape, shape, "Both matrix should be the same size")

        diff: float = torch.linalg.norm(w1 - w2).item()

        w1_value: float = torch.linalg.norm(w1).item()
        w2_value: float = torch.linalg.norm(w2).item()

        abs_diff: float = abs(diff)

        relative_diff_w1: float = abs(abs_diff / w1_value)
        relative_diff_w2: float = abs(abs_diff / w2_value)

        self.assertLessEqual(relative_diff_w1, self.tolerance, "Kiops didn't give a close result")
        self.assertLessEqual(relative_diff_w2, self.tolerance, "Pmex didn't give a close result")
