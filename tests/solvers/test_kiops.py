from solvers.kiops import kiops
import random
from numpy import ndarray

import cuda_test
import ndarray_generator

class KiopsComparisonTestCases(cuda_test.CudaTestCases):
    tolerance: float
    rand: random.Random

    cpu_matrix: ndarray
    gpu_matrix: ndarray

    def setUp(self) -> None:
        super().setUp()
        seed: int = 5646459
        initial_matrix_size: int = 64
        rand_min: float = -1000.0
        rand_max: float = 1000.0

        self.tolerance = 1e-7
        self.rand = random.Random(seed)

        [self.cpu_matrix, self.gpu_matrix] = ndarray_generator.generate_matrixes(
            (initial_matrix_size, initial_matrix_size), self.rand, rand_min, rand_max, [self.cpu_device, self.gpu_device])
    
    def test_compare_cpu_to_gpu(self):
        def matvec_handle(v: ndarray) -> ndarray: return v

        w1: ndarray
        w2: ndarray

        w1, stats1 = kiops([1.], matvec_handle, self.cpu_matrix, self.tolerance, device=self.cpu_device)
        w2, stats2 = kiops([1.], matvec_handle, self.gpu_matrix, self.tolerance, device=self.gpu_device)

        shape: tuple[int, int] = w1.shape

        self.assertEqual(len(w1.shape), 2, 'Kiops didn\'t return a matrix with the cpu device')
        self.assertEqual(len(w2.shape), 2, 'Kiops didn\'t return a matrix with the gpu device')

        self.failIf(not (w2.shape[0] == shape[0] and w2.shape[1] == shape[1]), 'Both matrix should be the same size')

        diff: float = self.cpu_device.xp.linalg.norm(w1 - self.gpu_device.to_host(w2)).item()

        w1_value: float = self.cpu_device.xp.linalg.norm(w1).item()
        w2_value: float = self.gpu_device.xp.linalg.norm(w2).item()

        abs_diff: float = abs(diff)

        relative_diff_w1: float = abs(abs_diff / w1_value)
        relative_diff_w2: float = abs(abs_diff / w2_value)

        self.assertLessEqual(relative_diff_w1, self.tolerance, f'Kiops didn\'t give a close result to the cpu value')
        self.assertLessEqual(relative_diff_w2, self.tolerance, f'Kiops didn\'t give a close result to the gpu value')
