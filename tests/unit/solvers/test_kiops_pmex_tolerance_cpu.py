import torch
from scipy.linalg import expm
from wx_test import WxTestCase

from wx_factory.context import Context
from wx_factory.solvers.kiops import kiops
from wx_factory.solvers.pmex import pmex


class KiopsPmexToleranceCpuTestCases(WxTestCase):
    def test_pmex_rejects_nonfinite_operator_action(self):
        context = Context(self.comm, "cpu")
        inputs = torch.tensor([[0.0, 0.0], [1.0, 2.0]], dtype=torch.float32)
        with self.assertRaisesRegex(FloatingPointError, "non-finite"):
            pmex(
                torch.tensor([1.0]),
                lambda vector: torch.full_like(vector, float("nan")) if self.comm.rank == 0 else vector,
                inputs,
                1.2e-5,
                context=context,
            )

    def test_phi_combination_against_independent_matrix_exponential(self):
        context = Context(self.comm, "cpu")
        generator = torch.Generator().manual_seed(62)
        size = 24
        matrix = -torch.diag(torch.linspace(0.0, 3.0, size, dtype=torch.float64))
        matrix += 0.4 * torch.diag(torch.ones(size - 1, dtype=torch.float64), diagonal=1)
        vectors = torch.randn((3, size), generator=generator, dtype=torch.float64)

        # exp([[A, b2, b1], [0, 0, 1], [0, 0, 0]]) gives
        # exp(A)b0 + phi1(A)b1 + phi2(A)b2 without inverting A.
        augmented = torch.zeros((size + 2, size + 2), dtype=torch.float64)
        augmented[:size, :size] = matrix
        augmented[:size, size:] = vectors[[2, 1]].T
        augmented[size, size + 1] = 1
        reference = torch.from_numpy(expm(augmented.numpy()))
        expected = reference[:size, :size] @ vectors[0] + reference[:size, size + 1]
        for solver in (kiops, pmex):
            with self.subTest(solver=solver.__name__):
                actual, _ = solver(
                    torch.tensor([1.0], dtype=torch.float64),
                    lambda vector: matrix @ vector,
                    vectors,
                    1e-10,
                    context=context,
                )
                torch.testing.assert_close(actual[0], expected, rtol=1e-9, atol=1e-10)
