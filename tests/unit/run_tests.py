#!/usr/bin/env python3

import argparse
import os
import re
import sys
from typing import Optional
import unittest

from wx_test import WxTestSuite

main_project_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(main_project_dir)

from tests.unit.common.test_configuration import ConfigurationTestCases
from tests.unit.common.test_config_hints import ConfigHintsTestCases
from tests.unit.common.test_sort_fields import SortFieldsByDependencyTestCases
from tests.unit.common.test_angle24 import Angle24TestCase
from tests.unit.rhs.test_rhs_registry import RhsBundleTestCases, RhsRegistryTestCases
from tests.unit.precondition.test_preconditioner_registry import PreconditionerRegistryTestCases
from tests.unit.geometry.test_geometry_registry import GeometryRegistryTestCases
from tests.unit.output.test_output_registry import OutputRegistryTestCases
from tests.unit.step_hooks.test_step_hook_registry import StepHookRegistryTestCases
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


def load_tests(test_name: str, skip_compile: bool):
    """Create a test suite with cases we want to run."""

    test_re = re.compile(test_name, re.IGNORECASE)
    suite = WxTestSuite()

    if not skip_compile:
        add_test(suite, CompilationTestCases("test_cpp_kernels_compilation"), test_re)
        add_test(suite, CompilationGPUTestCases("test_cuda_kernels_compilation"), test_re)
        # add_test(suite, CompilationTestCases("test_cpp_compilation_twice"), test_re)

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

    add_test(suite, ConfigHintsTestCases("test_type_hints_are_up_to_date"), test_re)

    add_test(suite, SortFieldsByDependencyTestCases("test_no_dependencies_preserves_order"), test_re)
    add_test(suite, SortFieldsByDependencyTestCases("test_dependent_field_placed_after_target"), test_re)
    add_test(suite, SortFieldsByDependencyTestCases("test_multi_level_chain_is_ordered"), test_re)
    add_test(suite, SortFieldsByDependencyTestCases("test_no_field_is_emitted_more_than_once"), test_re)
    add_test(suite, SortFieldsByDependencyTestCases("test_unresolvable_dependency_is_kept_not_dropped"), test_re)
    add_test(suite, SortFieldsByDependencyTestCases("test_dependency_cycle_does_not_hang_and_keeps_fields"), test_re)
    add_test(suite, SortFieldsByDependencyTestCases("test_duplicate_name_raises"), test_re)

    add_test(suite, Angle24TestCase("test_cyclic"), test_re)
    add_test(suite, Angle24TestCase("test_rounding"), test_re)

    add_test(suite, RhsBundleTestCases("test_full_and_shape_are_stored"), test_re)
    add_test(suite, RhsBundleTestCases("test_missing_partitions_raise_when_called"), test_re)
    add_test(suite, RhsBundleTestCases("test_provided_partitions_are_used"), test_re)
    add_test(suite, RhsRegistryTestCases("test_expected_combinations_are_registered"), test_re)
    add_test(suite, RhsRegistryTestCases("test_unknown_discretization_raises"), test_re)
    add_test(suite, RhsRegistryTestCases("test_unregistered_combination_raises_helpful_error"), test_re)
    add_test(suite, RhsRegistryTestCases("test_resolve_dispatches_to_registered_factory"), test_re)
    add_test(suite, RhsRegistryTestCases("test_duplicate_registration_raises"), test_re)

    add_test(suite, PreconditionerRegistryTestCases("test_none_resolves_to_no_preconditioner"), test_re)
    add_test(suite, PreconditionerRegistryTestCases("test_no_builtin_preconditioners"), test_re)
    add_test(suite, PreconditionerRegistryTestCases("test_unknown_preconditioner_raises_helpful_error"), test_re)
    add_test(suite, PreconditionerRegistryTestCases("test_resolve_dispatches_to_registered_factory"), test_re)
    add_test(suite, PreconditionerRegistryTestCases("test_duplicate_registration_raises"), test_re)
    add_test(suite, PreconditionerRegistryTestCases("test_base_prepare_is_a_noop"), test_re)

    add_test(suite, GeometryRegistryTestCases("test_expected_combinations_are_registered"), test_re)
    add_test(suite, GeometryRegistryTestCases("test_unregistered_combination_raises_helpful_error"), test_re)
    add_test(suite, GeometryRegistryTestCases("test_resolve_dispatches_to_registered_factory"), test_re)
    add_test(suite, GeometryRegistryTestCases("test_grid_file_forces_cubed_sphere_2d"), test_re)
    add_test(suite, GeometryRegistryTestCases("test_duplicate_registration_raises"), test_re)

    add_test(suite, OutputRegistryTestCases("test_expected_combinations_are_registered"), test_re)
    add_test(suite, OutputRegistryTestCases("test_format_independent_family_uses_default_entry"), test_re)
    add_test(suite, OutputRegistryTestCases("test_format_specific_dispatch"), test_re)
    add_test(suite, OutputRegistryTestCases("test_unknown_combination_raises_helpful_error"), test_re)
    add_test(suite, OutputRegistryTestCases("test_duplicate_registration_raises"), test_re)

    add_test(suite, StepHookRegistryTestCases("test_builtin_hooks_registered_with_expected_phases"), test_re)
    add_test(suite, StepHookRegistryTestCases("test_resolve_only_returns_requested_phase"), test_re)
    add_test(suite, StepHookRegistryTestCases("test_schar_does_not_apply_to_non_cubesphere"), test_re)
    add_test(suite, StepHookRegistryTestCases("test_dcmip_dispatches_on_case_number"), test_re)
    add_test(suite, StepHookRegistryTestCases("test_no_hooks_when_nothing_applies"), test_re)
    add_test(suite, StepHookRegistryTestCases("test_provider_returning_none_is_skipped"), test_re)
    add_test(suite, StepHookRegistryTestCases("test_unknown_phase_raises"), test_re)
    add_test(suite, StepHookRegistryTestCases("test_duplicate_registration_raises"), test_re)

    return suite


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the suite of single-process tests.")
    parser.add_argument(
        "test_name",
        nargs="?",
        default="",
        type=str,
        help="Will only run tests whose name or type matches this regular expression.",
    )
    parser.add_argument("--no-buffer", action="store_true", help="Print all test output to terminal")
    parser.add_argument("--skip-compile", action="store_true", help="Skip compilation test cases (they are slow)")
    args = parser.parse_args()

    runner = unittest.TextTestRunner(buffer=not args.no_buffer, verbosity=0)
    result = runner.run(load_tests(args.test_name, args.skip_compile))
    if not result.wasSuccessful():
        failed_tests = "\n  ".join([f"{r[0]}" for r in result.errors + result.unexpectedSuccesses + result.failures])
        print(f"failed tests: \n  {failed_tests}")
        raise SystemExit(-1)
