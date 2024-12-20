#!/usr/bin/env python3

import os
import sys
import unittest

main_project_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
main_module_dir = os.path.join(main_project_dir, "wx_factory")
sys.path.append(main_project_dir)
sys.path.append(main_module_dir)

from tests.solvers.test_pmex import PmexComparisonTestCases
from tests.solvers.test_kiops import KiopsComparisonTestCases
from tests.solvers.test_fgmres import FgmresComparisonTestCases, FgmresScipyTestCases, FgmresEdgeCasesTestCases
from tests.solvers.test_kiops_pmex_tolerance_cpu import KiopsPmexToleranceCpuTestCases
from tests.solvers.test_kiops_pmex_tolerance_gpu import KiopsPmexToleranceGpuTestCases
from tests.output.test_state import StateTestCases
from tests.common.test_configuration import ConfigurationTestCases
from tests.compiler.test_compilation import CompilationTestCases, CompilationGPUTestCases


def load_tests():
    suite = unittest.TestSuite()
    suite.addTest(PmexComparisonTestCases("test_compare_cpu_to_gpu"))
    suite.addTest(KiopsComparisonTestCases("test_compare_cpu_to_gpu"))
    suite.addTest(FgmresComparisonTestCases("test_compare_cpu_to_gpu"))

    suite.addTest(FgmresEdgeCasesTestCases("test_fgmres_throw_when_b_is_smaller_or_equal_to_restart"))

    suite.addTest(FgmresScipyTestCases("test_compare_implementation_to_scipy"))
    suite.addTest(FgmresScipyTestCases("test_compare_implementation_to_scipy_and_residual"))

    suite.addTest(KiopsPmexToleranceCpuTestCases("test_compare_kiops_pmex"))
    suite.addTest(KiopsPmexToleranceGpuTestCases("test_compare_kiops_pmex"))

    suite.addTest(StateTestCases("test_save_load_works"))
    suite.addTest(ConfigurationTestCases("test_load_configuration_with_schema_default"))
    suite.addTest(ConfigurationTestCases("test_load_configuration_with_valid_values"))
    suite.addTest(ConfigurationTestCases("test_load_configuration_with_dependancy"))

    suite.addTest(CompilationTestCases("test_cpp_kernels_compilation"))
    suite.addTest(CompilationGPUTestCases("test_cuda_kernels_compilation"))
    suite.addTest(CompilationTestCases("test_cpp_compilation_twice"))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(load_tests())
