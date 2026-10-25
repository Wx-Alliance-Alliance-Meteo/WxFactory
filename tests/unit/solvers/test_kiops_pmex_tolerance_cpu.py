import torch
from scipy.linalg import expm
from wx_test import WxTestCase

from wx_factory.context import Context
from wx_factory.solvers.kiops import kiops
from wx_factory.solvers.pmex import PmexStartup, pmex


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

    def test_pmex_startup_keeps_first_accepted_pair_and_preserves_action(self):
        context = Context(self.comm, "cpu")
        matrix = torch.diag(torch.tensor([-1.0, -2.0], dtype=torch.float64))
        inputs = torch.tensor([[0.0, 0.0], [1.0, 2.0]], dtype=torch.float64)
        startup = PmexStartup(0.3, 3)
        for time in (1.0, 0.1):
            with self.subTest(time=time):
                actual, stats = pmex(
                    [time],
                    lambda v: matrix @ v,
                    inputs,
                    1e-12,
                    m_init=3,
                    mmin=3,
                    mmax=3,
                    context=context,
                    startup=startup,
                )
                expected = 1 - torch.exp(time * matrix.diag())
                torch.testing.assert_close(actual[0], expected, rtol=1e-12, atol=1e-12)
                self.assertAlmostEqual(startup.first_accepted, min(time, 0.3))
                self.assertEqual(startup.first_dimension, 3)
                self.assertEqual(stats[0], 4 if time == 1.0 else 1)

        def fail(vector):
            raise RuntimeError("operator failed")

        with self.assertRaisesRegex(RuntimeError, "operator failed"):
            pmex([1.0], fail, inputs, context=context, startup=startup)
        self.assertAlmostEqual(startup.first_accepted, 0.1)
        self.assertEqual(startup.first_dimension, 3)

    def test_pmex_reduces_basis_from_cap_without_losing_accuracy(self):
        context = Context(self.comm, "cpu")
        eigenvalues = -torch.linspace(0.1, 30, 48, dtype=torch.float64)
        inputs = torch.stack((torch.zeros_like(eigenvalues), torch.ones_like(eigenvalues)))
        startup = PmexStartup(0.25, 32)
        actual, stats = pmex(
            [1.0],
            lambda v: eigenvalues * v,
            inputs,
            1e-8,
            m_init=6,
            mmin=6,
            mmax=16,
            context=context,
            startup=startup,
        )
        expected = torch.expm1(eigenvalues) / eigenvalues
        torch.testing.assert_close(actual[0], expected, rtol=1e-8, atol=1e-10)
        self.assertEqual(startup.first_dimension, 16)
        self.assertGreater(stats[0], 1)
        self.assertLess(stats[2], 16 * stats[0])

    def test_pmex_substeps_match_dense_exponential(self):
        context = Context(self.comm, "cpu")
        matrix = torch.diag(torch.tensor([-1.0, -2.0], dtype=torch.float64))
        vectors = torch.tensor([[0.1, 0.2], [1.0, 2.0], [0.3, -0.4], [-0.2, 0.5]], dtype=torch.float64)
        for count in (1, 3, 4):
            augmented = torch.zeros((count + 1, count + 1), dtype=torch.float64)
            augmented[:2, :2] = matrix
            augmented[:2, 2:] = vectors[1:count].flip(0).T
            for index in range(2, count):
                augmented[index, index + 1] = 1
            initial = torch.zeros(count + 1, dtype=torch.float64)
            initial[:2] = vectors[0]
            if count > 1:
                initial[-1] = 1
            for time in (1.0, -0.7):
                with self.subTest(count=count, time=time):
                    expected = torch.from_numpy(expm(time * augmented.numpy())) @ initial
                    actual, stats = pmex(
                        [time],
                        lambda v: matrix @ v,
                        vectors[:count],
                        1e-12,
                        m_init=5,
                        mmin=5,
                        mmax=5,
                        context=context,
                        startup=PmexStartup(0.2),
                    )
                    self.assertGreater(stats[0], 1)
                    torch.testing.assert_close(actual[0], expected[:2], rtol=1e-12, atol=1e-12)

    def test_pmex_float32_action_for_stable_nonnormal_matrix(self):
        context = Context(self.comm, "cpu")
        generator = torch.Generator().manual_seed(718)
        size = 24
        basis, _ = torch.linalg.qr(torch.randn(size, size, generator=generator, dtype=torch.float64))
        scales = torch.logspace(0, 1, size, dtype=torch.float64)
        balanced = basis @ torch.diag(-torch.linspace(0.1, 80, size, dtype=torch.float64)) @ basis.T
        matrix = (scales[:, None] * balanced / scales[None, :]).float()
        forcing = (scales * torch.randn(size, generator=generator, dtype=torch.float64)).float()
        inputs = torch.stack((torch.zeros_like(forcing), forcing))

        augmented = torch.zeros(size + 1, size + 1, dtype=torch.float64)
        augmented[:size, :size] = matrix.double() / scales[:, None] * scales[None, :]
        augmented[:size, size] = forcing.double() / scales
        expected = torch.from_numpy(expm(augmented.numpy()))[:size, size]
        actual, _ = pmex(
            torch.tensor([1.0]),
            lambda vector: matrix @ vector,
            inputs,
            1e-5,
            m_init=10,
            mmin=10,
            mmax=size,
            context=context,
        )
        torch.testing.assert_close(actual[0].double() / scales, expected, rtol=2e-5, atol=2e-6)

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
