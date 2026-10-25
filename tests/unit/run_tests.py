#!/usr/bin/env python3

import argparse
import os
import re
import sys
import traceback
import unittest
import warnings

main_project_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(main_project_dir)

from wx_test import WxTestRunner

from tests.unit.common.test_angle24 import Angle24TestCase
from tests.unit.common.test_config_hints import ConfigHintsTestCases
from tests.unit.common.test_configuration import ConfigurationTestCases
from tests.unit.common.test_panel_conversions import PanelConversionTestCase
from tests.unit.common.test_sort_fields import SortFieldsByDependencyTestCases
from tests.unit.common.test_tensor_math import SpatialContractionTestCase
from tests.unit.geometry.test_cartesian_metric5d import CartesianMetric5DTestCase
from tests.unit.geometry.test_exponential_filter import CartesianFilterTestCase, ExponentialFilterTestCase
from tests.unit.geometry.test_geometry_registry import GeometryRegistryTestCases
from tests.unit.geometry.test_metric5d import Metric5DTestCase, TabulatedSurfaceTestCase
from tests.unit.geometry.test_operator_identities import OperatorIdentityTestCase
from tests.unit.init.test_cartesian import CartesianInitializationTestCase
from tests.unit.integrators.test_epi import EpiTestCase
from tests.unit.integrators.test_implicit import ImplicitIntegratorTestCase
from tests.unit.integrators.test_splitting import SplittingTestCase
from tests.unit.integrators.test_srerk import SrerkTestCase
from tests.unit.jacobian.test_cartesian_euler import CartesianEulerJacobianTestCase
from tests.unit.jacobian.test_euler import EulerJacobianTestCase
from tests.unit.jacobian.test_euler_partitions import EulerPartitionTestCase
from tests.unit.jacobian.test_finite_difference import FiniteDifferenceJacobianTestCases
from tests.unit.output.test_integration_comparison import IntegrationComparisonTestCase
from tests.unit.output.test_output_registry import OutputRegistryTestCases
from tests.unit.output.test_state import StateTestCases
from tests.unit.precondition.test_preconditioner_registry import (
    PreconditionerRegistryTestCases,
)
from tests.unit.restart.test_restart import Euler2DRestartTestCase
from tests.unit.rhs.test_cartesian_euler import CartesianEulerTestCase
from tests.unit.rhs.test_euler import EulerTestCase
from tests.unit.rhs.test_euler_operator import EulerOperatorTestCase
from tests.unit.rhs.test_rhs_registry import (
    PartitionedIntegratorTestCases,
    RhsBundleTestCases,
    RhsRegistryTestCases,
)
from tests.unit.rhs.test_shallow_water import ShallowWaterInterfaceTestCase
from tests.unit.solvers.test_exode import ExodeTestCase
from tests.unit.solvers.test_exponential_solver_registry import (
    ExponentialSolverRegistryTestCases,
)
from tests.unit.solvers.test_fgmres import (
    FgmresEdgeCasesTestCases,
    FgmresTestCases,
)
from tests.unit.solvers.test_kiops_pmex_tolerance_cpu import (
    KiopsPmexToleranceCpuTestCases,
)
from tests.unit.step_hooks.test_step_hook_registry import StepHookRegistryTestCases


def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    traceback.print_stack()
    print(f"{filename}:{lineno}: {category.__name__}: {message}")


def add_test(suite: unittest.TestSuite, test: unittest.TestCase, test_re: re.Pattern | None):
    if test_re is None or test_re.search(str(test)) is not None:
        suite.addTest(test)


def load_tests(test_name: str):
    test_re = re.compile(test_name, re.IGNORECASE)
    suite = unittest.TestSuite()
    for test_case in (
        IntegrationComparisonTestCase,
        SpatialContractionTestCase,
        EulerJacobianTestCase,
        EulerPartitionTestCase,
        ImplicitIntegratorTestCase,
        SrerkTestCase,
        EpiTestCase,
        SplittingTestCase,
        EulerOperatorTestCase,
        EulerTestCase,
        ShallowWaterInterfaceTestCase,
        Metric5DTestCase,
        TabulatedSurfaceTestCase,
        CartesianEulerTestCase,
        CartesianMetric5DTestCase,
        CartesianInitializationTestCase,
        CartesianEulerJacobianTestCase,
        CartesianFilterTestCase,
        FgmresEdgeCasesTestCases,
        FgmresTestCases,
        KiopsPmexToleranceCpuTestCases,
        FiniteDifferenceJacobianTestCases,
        ExponentialFilterTestCase,
        OperatorIdentityTestCase,
        ExponentialSolverRegistryTestCases,
        ExodeTestCase,
        StateTestCases,
        Euler2DRestartTestCase,
        ConfigurationTestCases,
        PanelConversionTestCase,
        ConfigHintsTestCases,
        SortFieldsByDependencyTestCases,
        Angle24TestCase,
        RhsBundleTestCases,
        PartitionedIntegratorTestCases,
        RhsRegistryTestCases,
        PreconditionerRegistryTestCases,
        GeometryRegistryTestCases,
        OutputRegistryTestCases,
        StepHookRegistryTestCases,
    ):
        for name in unittest.defaultTestLoader.getTestCaseNames(test_case):
            add_test(suite, test_case(name), test_re)
    return suite


if __name__ == "__main__":
    warnings.showwarning = warn_with_traceback
    warnings.simplefilter("always")

    parser = argparse.ArgumentParser(description="Run the suite of single-process tests.")
    parser.add_argument(
        "test_name",
        nargs="?",
        default="",
        type=str,
        help="Will only run tests whose name or type matches this regular expression.",
    )
    parser.add_argument("--no-buffer", action="store_true", help="Print all test output to terminal")
    parser.add_argument("--failfast", action="store_true", help="Stop running tests after 1 failure")
    args = parser.parse_args()

    runner = WxTestRunner(buffer=not args.no_buffer, verbosity=0, failfast=args.failfast)
    result = runner.run(load_tests(args.test_name))
    if not result.wasSuccessful():
        failed_tests = "\n  ".join(
            [f"{r[0]}" for r in result.errors + result.failures] + [f"{r}" for r in result.unexpectedSuccesses]
        )
        print(f"failed tests: \n  {failed_tests}")
        raise SystemExit(-1)
