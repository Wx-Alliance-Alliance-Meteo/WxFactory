from solvers.kiops import kiops
from solvers.pmex import pmex
import random
from numpy import ndarray

import cpu_test

class KiopsPmexToleranceCpuTestCases(cpu_test.CpuTestCases):
    tolerance: float
    rand: random.Random

    kiops_matrix: ndarray
    pmex_matrix: ndarray

    def setUp(self) -> None:
        super().setUp()

        seed: int = 5646459
        initial_matrix_size: int = 64
        rand_min: float = -1000.0
        rand_max: float = 1000.0

        self.show_debug_print = True

        self.tolerance = 1e-7
        self.rand = random.Random(seed)

        self.kiops_matrix: ndarray = self.cpu_device.xp.zeros((initial_matrix_size, initial_matrix_size), dtype=float)
        self.pmex_matrix: ndarray = self.cpu_device.xp.zeros((initial_matrix_size, initial_matrix_size), dtype=float)

        for it1 in range(initial_matrix_size):
            for it2 in range(initial_matrix_size):
                nb: float = self.rand.uniform(rand_min, rand_max)
                self.kiops_matrix[it1, it2] = nb
                self.pmex_matrix[it1, it2] = nb
    
    def test_compare_kiops_pmex(self):
        def matvec_handle(v: ndarray) -> ndarray: return v

        w1: ndarray
        w2: ndarray

        w1, stats1 = kiops([1.], matvec_handle, self.kiops_matrix, self.tolerance, device=self.cpu_device)
        w2, stats2 = pmex([1.], matvec_handle, self.pmex_matrix, self.tolerance, device=self.cpu_device)

        shape: tuple[int, int] = w1.shape

        self.assertEqual(len(w1.shape), 2, 'Kiops didn\'t return a matrix')
        self.assertEqual(len(w2.shape), 2, 'Pmex didn\'t return a matrix')

        self.failIf(not (w2.shape[0] == shape[0] and w2.shape[1] == shape[1]), 'Both matrix should be the same size')

        w1_value: float = self.cpu_device.xp.linalg.norm(w1).item()
        w2_value: float = self.cpu_device.xp.linalg.norm(w2).item()

        abs_diff: float = abs(w1_value - w2_value)

        relative_diff_w1: float = abs(abs_diff / w1_value)
        relative_diff_w2: float = abs(abs_diff / w2_value)

        if self.show_debug_print:
            c_name = 'KiopsPmexToleranceCpuTestCases'
            m_name = 'test_compare_kiops_pmex'
            print(f'In {c_name}.{m_name}, absolute difference is {abs_diff}')
            print(f'In {c_name}.{m_name}, relative difference from cpu is {relative_diff_w1} and relative difference from gpu is {relative_diff_w2}')

        self.assertLessEqual(relative_diff_w1, self.tolerance, f'Kiops didn\'t give a close result')
        self.assertLessEqual(relative_diff_w2, self.tolerance, f'Pmex didn\'t give a close result')
