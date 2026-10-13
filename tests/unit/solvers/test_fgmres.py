import torch

from tests.unit.wx_test import WxTestCase
from wx_factory.context import Context
from wx_factory.solvers.fgmres import fgmres


class FgmresTestCases(WxTestCase):
    def test_restarted_solve_matches_dense_solution_and_residual(self):
        context = Context(self.comm, "cpu")
        size = 32
        matrix = torch.diag(torch.linspace(1.0, 4.0, size, dtype=torch.float64))
        matrix += 0.5 * torch.diag(torch.ones(size - 1, dtype=torch.float64), diagonal=1)
        forcing = torch.linspace(-1.0, 1.0, size, dtype=torch.float64)
        reference = torch.linalg.solve(matrix, forcing)
        for restart in (4, 20):
            with self.subTest(restart=restart):
                solution, _, _, _, flag, _ = fgmres(
                    lambda vector: matrix @ vector, forcing, tol=1e-10, restart=restart, context=context
                )
                self.assertEqual(flag, 0)
                self.assertLess((matrix @ solution - forcing).norm().item() / forcing.norm().item(), 1e-10)
                torch.testing.assert_close(solution, reference, rtol=1e-9, atol=1e-10)


class FgmresEdgeCasesTestCases(WxTestCase):
    def test_fgmres_throw_when_b_is_smaller_or_equal_to_restart(self):
        context = Context(self.comm, "cpu")
        for size in (19, 20):
            with self.subTest(size=size), self.assertRaises(ValueError):
                fgmres(lambda vector: vector, torch.ones(size, dtype=torch.float64), restart=20, context=context)
        forcing = torch.ones(21, dtype=torch.float64)
        solution, _, _, _, flag, _ = fgmres(lambda vector: vector, forcing, restart=20, context=context)
        self.assertEqual(flag, 0)
        torch.testing.assert_close(solution, forcing, rtol=0, atol=1e-14)
