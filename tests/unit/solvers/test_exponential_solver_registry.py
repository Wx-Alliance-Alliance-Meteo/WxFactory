from types import SimpleNamespace
from unittest.mock import patch

from tests.unit.wx_test import WxTestCase
from wx_factory.integrators.rosexp2 import RosExp2
from wx_factory.solvers.exponential_solver import (
    EXPONENTIAL_SOLVER_REGISTRY,
    ExponentialSolverRequest,
    register_exponential_solver,
    resolve_exponential_solver,
)


class ExponentialSolverRegistryTestCases(WxTestCase):
    def setUp(self):
        self.device = SimpleNamespace(comm=SimpleNamespace(rank=1))
        self.request = ExponentialSolverRequest(
            [1.0],
            lambda value: value,
            object(),
            tolerance=1e-6,
            krylov_mmax=42,
            device=self.device,
            krylov_minit=7,
            krylov_mmin=5,
            announce=False,
        )

    def test_builtin_solvers_are_registered(self):
        self.assertEqual({"pmex", "pmex_ne", "kiops", "exode"}, set(EXPONENTIAL_SOLVER_REGISTRY))

    def test_unknown_solver_lists_registered_names(self):
        with self.assertRaisesRegex(ValueError, "Registered solvers.*exode.*kiops.*pmex"):
            resolve_exponential_solver("missing")

    def test_duplicate_registration_raises(self):
        with self.assertRaisesRegex(ValueError, "already registered"):
            register_exponential_solver("pmex")(lambda request: request)

    @patch("wx_factory.solvers.exponential_solver.pmex")
    def test_pmex_adapter_translates_common_request(self, pmex_mock):
        pmex_mock.return_value = ("value", (2, 3, 11, 4, 5e-7, 19, 0))

        result = resolve_exponential_solver("pmex")(self.request)

        self.assertEqual("value", result.value)
        self.assertEqual(11, result.iterations)
        self.assertEqual(19, result.final_krylov_size)
        pmex_mock.assert_called_once_with(
            self.request.tau_out,
            self.request.operator,
            self.request.vectors,
            tol=1e-6,
            mmax=42,
            task1=False,
            device=self.device,
            m_init=7,
            mmin=5,
        )

    @patch("wx_factory.solvers.exponential_solver.pmex")
    def test_pmex_ne_supplies_its_default_minimum(self, pmex_mock):
        pmex_mock.return_value = ("value", (1, 0, 8, 1, 1e-8, 16, 0))
        request = ExponentialSolverRequest(
            [1.0],
            self.request.operator,
            self.request.vectors,
            1e-6,
            42,
            self.device,
            announce=False,
        )

        resolve_exponential_solver("pmex_ne")(request)

        self.assertEqual(16, pmex_mock.call_args.kwargs["mmin"])

    def test_exode_rejects_multiple_output_times(self):
        request = ExponentialSolverRequest(
            [0.5, 1.0],
            self.request.operator,
            self.request.vectors,
            1e-6,
            42,
            self.device,
            announce=False,
        )
        with self.assertRaisesRegex(ValueError, "exactly one output time"):
            resolve_exponential_solver("exode")(request)

    def test_rosexp2_resolves_configured_solver_during_construction(self):
        config = SimpleNamespace(
            tolerance=1e-6,
            gmres_restart=20,
            krylov_mmax=42,
            exponential_solver="missing",
            exode_method="BS3(2)",
            exode_controller="deadbeat",
            verbose_solver=0,
        )
        with self.assertRaisesRegex(ValueError, "not registered"):
            RosExp2(config, lambda value: value, lambda value: value, device=self.device)
