import numpy
import torch
from scipy.linalg import expm

from tests.unit.mpi_test import MpiTestCase
from wx_factory.solvers.kiops import kiops
from wx_factory.solvers.pmex import pmex


class ExponentialMpiTestCase(MpiTestCase):
    def test_distributed_phi_combination_against_dense_reference(self):
        size = 24
        matrix = -torch.diag(torch.linspace(0.0, 3.0, size, dtype=torch.float64))
        matrix += 0.4 * torch.diag(torch.ones(size - 1, dtype=torch.float64), diagonal=1)
        generator = torch.Generator().manual_seed(62)
        vectors = torch.randn((3, size), generator=generator, dtype=torch.float64)
        # Unequal rank contributions exercise the global Krylov reductions.
        vectors[:, size // 2 :] *= 100
        augmented = torch.zeros((size + 2, size + 2), dtype=torch.float64)
        augmented[:size, :size] = matrix
        augmented[:size, size:] = vectors[[2, 1]].T
        augmented[size, size + 1] = 1
        exponential = torch.from_numpy(expm(augmented.numpy()))
        reference = exponential[:size, :size] @ vectors[0] + exponential[:size, size + 1]
        local = slice(self.comm.rank * size // self.comm.size, (self.comm.rank + 1) * size // self.comm.size)

        def matvec(vector):
            full = torch.from_numpy(numpy.concatenate(self.comm.allgather(vector.cpu().numpy())))
            return matrix[local] @ full

        for solver in (kiops, pmex):
            with self.subTest(solver=solver.__name__):
                result, _ = solver(
                    torch.tensor([1.0], dtype=torch.float64), matvec, vectors[:, local], 1e-10, context=self.context
                )
                torch.testing.assert_close(result[0], reference[local], rtol=1e-9, atol=1e-9)
