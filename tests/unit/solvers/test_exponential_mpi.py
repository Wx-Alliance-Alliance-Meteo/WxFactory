from unittest.mock import patch

import numpy
import torch
from scipy.linalg import expm

from tests.unit.mpi_test import MpiTestCase
from wx_factory.integrators.butcher import METHODS
from wx_factory.solvers.exode import ExodeStartup, exode
from wx_factory.solvers.kiops import kiops
from wx_factory.solvers.pmex import PmexStartup, pmex


class ExponentialMpiTestCase(MpiTestCase):
    def test_distributed_phi_combination_against_dense_reference(self):
        size = 24
        matrix = -torch.diag(torch.linspace(0.0, 3.0, size, dtype=torch.float64, device="cpu"))
        matrix += 0.4 * torch.diag(torch.ones(size - 1, dtype=torch.float64, device="cpu"), diagonal=1)
        generator = torch.Generator().manual_seed(62)
        vectors = torch.randn((3, size), generator=generator, dtype=torch.float64, device="cpu")
        # Unequal rank contributions exercise the global Krylov reductions.
        vectors[:, size // 2 :] *= 100
        augmented = torch.zeros((size + 2, size + 2), dtype=torch.float64, device="cpu")
        augmented[:size, :size] = matrix
        augmented[:size, size:] = vectors[[2, 1]].T
        augmented[size, size + 1] = 1
        exponential = torch.from_numpy(expm(augmented.numpy()))
        reference = exponential[:size, :size] @ vectors[0] + exponential[:size, size + 1]
        local = slice(self.comm.rank * size // self.comm.size, (self.comm.rank + 1) * size // self.comm.size)
        matrix = matrix.to(self.context.torch_device)
        vectors = vectors.to(self.context.torch_device)
        reference = reference.to(self.context.torch_device)

        def matvec(vector):
            full = torch.from_numpy(numpy.concatenate(self.comm.allgather(vector.cpu().numpy()))).to(matrix.device)
            return matrix[local] @ full

        for solver in (kiops, pmex):
            with self.subTest(solver=solver.__name__):
                startup = PmexStartup(0.1 * (self.comm.rank + 1), None if self.comm.rank == 0 else 15)
                kwargs = {"startup": startup} if solver is pmex else {}
                result, _ = solver(
                    torch.tensor([1.0], dtype=torch.float64),
                    matvec,
                    vectors[:, local],
                    1e-10,
                    context=self.context,
                    **kwargs,
                )
                if solver is pmex:
                    self.assertAlmostEqual(startup.first_accepted, 0.1)
                    self.assertEqual(startup.first_dimension, 15)
                torch.testing.assert_close(result[0], reference[local], rtol=1e-9, atol=1e-9)

        method = METHODS["DP5(4)"]

        def check_initial_step(*args, **kwargs):
            steps = self.comm.allgather(kwargs["first_step"])
            self.assertEqual(min(steps), max(steps))
            return method(*args, **kwargs)

        with patch.dict(METHODS, {"DP5(4)": check_initial_step}):
            result, _ = exode(
                1.0,
                matvec,
                vectors[:, local],
                method="DP5(4)",
                controller="H211B",
                startup=ExodeStartup(0.1 * (self.comm.rank + 1)),
                rtol=1e-11,
                atol=1e-12,
                context=self.context,
            )
        torch.testing.assert_close(result, reference[local], rtol=1e-9, atol=1e-9)
