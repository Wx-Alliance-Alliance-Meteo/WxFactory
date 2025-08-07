#!/usr/bin/env python3

import argparse
import os
import re
import sys
from typing import Optional
import unittest

main_project_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
main_module_dir = os.path.join(main_project_dir, "wx_factory")
sys.path.append(main_project_dir)
sys.path.append(main_module_dir)

from tests.unit.common.test_configuration import ConfigurationTestCases
from tests.unit.common.test_angle24 import Angle24TestCase
from tests.unit.compiler.test_compilation import CompilationTestCases, CompilationGPUTestCases
from tests.unit.output.test_state import StateTestCases
from tests.unit.restart.test_restart import Euler2DRestartTestCase
from tests.unit.solvers.test_pmex import PmexComparisonTestCases
from tests.unit.solvers.test_kiops import KiopsComparisonTestCases
from tests.unit.solvers.test_fgmres import FgmresComparisonTestCases, FgmresScipyTestCases, FgmresEdgeCasesTestCases
from tests.unit.solvers.test_kiops_pmex_tolerance_cpu import KiopsPmexToleranceCpuTestCases
from tests.unit.solvers.test_kiops_pmex_tolerance_gpu import KiopsPmexToleranceGpuTestCases


def add_test(suite: unittest.TestSuite, test: unittest.TestCase, test_re: Optional[re.Pattern]):
    if test_re is None or test_re.search(str(test)) is not None:
        suite.addTest(test)


def load_tests(test_name):
    """Create a test suite with cases we want to run."""

    test_re = re.compile(test_name, re.IGNORECASE)
    suite = unittest.TestSuite()

    add_test(suite, CompilationTestCases("test_cpp_kernels_compilation"), test_re)
    add_test(suite, CompilationGPUTestCases("test_cuda_kernels_compilation"), test_re)
    add_test(suite, CompilationTestCases("test_cpp_compilation_twice"), test_re)

    add_test(suite, PmexComparisonTestCases("test_compare_cpu_to_gpu"), test_re)
    add_test(suite, KiopsComparisonTestCases("test_compare_cpu_to_gpu"), test_re)
    add_test(suite, FgmresComparisonTestCases("test_compare_cpu_to_gpu"), test_re)

    add_test(suite, FgmresEdgeCasesTestCases("test_fgmres_throw_when_b_is_smaller_or_equal_to_restart"), test_re)

    add_test(suite, FgmresScipyTestCases("test_compare_implementation_to_scipy"), test_re)
    add_test(suite, FgmresScipyTestCases("test_compare_implementation_to_scipy_and_residual"), test_re)

    add_test(suite, KiopsPmexToleranceCpuTestCases("test_compare_kiops_pmex"), test_re)
    add_test(suite, KiopsPmexToleranceGpuTestCases("test_compare_kiops_pmex"), test_re)

    add_test(suite, StateTestCases("test_save_load_works"), test_re)
    add_test(suite, StateTestCases("test_load_old_state"), test_re)
    add_test(suite, Euler2DRestartTestCase("test_gen_restart"), test_re)
    add_test(suite, Euler2DRestartTestCase("test_read_restart"), test_re)

    add_test(suite, ConfigurationTestCases("test_load_configuration_with_schema_default"), test_re)
    add_test(suite, ConfigurationTestCases("test_load_configuration_with_valid_values"), test_re)
    add_test(suite, ConfigurationTestCases("test_load_configuration_with_invalid_values"), test_re)
    add_test(suite, ConfigurationTestCases("test_load_configuration_with_dependency"), test_re)
    add_test(suite, ConfigurationTestCases("test_load_configuration_with_expression"), test_re)

    add_test(suite, Angle24TestCase("test_cyclic"), test_re)
    add_test(suite, Angle24TestCase("test_rounding"), test_re)

    return suite


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solve NWP problems with WxFactory!")
    parser.add_argument(
        "test_name",
        nargs="?",
        default="",
        type=str,
        help="Will only run tests whose name or type matches this regular expression.",
    )
    args = parser.parse_args()

    runner = unittest.TextTestRunner(buffer=True, verbosity=1)
    result = runner.run(load_tests(args.test_name))
    if not result.wasSuccessful():
        failed_tests = "\n  ".join([f"{r[0]}" for r in result.errors + result.unexpectedSuccesses + result.failures])
        print(f"failed tests: \n  {failed_tests}")
        raise SystemExit(-1)
