#!/usr/bin/env python3

import argparse
import os
import re
import sys
from typing import Optional
from unittest import TestCase, TestSuite

from mpi_test import MpiRunner, MpiTestSuite

main_project_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
main_module_dir = os.path.join(main_project_dir, "wx_factory")
sys.path.append(main_project_dir)
sys.path.append(main_module_dir)

from tests.unit.common.test_process_topology import ExchangeTest, GatherScatterTest
from tests.unit.operators.test_extrap import OperatorsExtrapEuler3DTestCase
from tests.unit.pde.test_rusanov_3d import PdeRusanov3DTestCase
from tests.unit.restart.test_restart import ShallowWaterRestartTestCase, Euler3DRestartTestCase
from tests.unit.rhs.test_side_by_side import RhsSideBySideEuler3DTestCase
from tests.unit.solvers.test_pmex_mpi import PmexMpiTestCases
from tests.unit.solvers.test_kiops_mpi import KiopsMpiTestCases
from tests.unit.solvers.test_fgmres_mpi import FgmresMpiTestCases


def add_test(suite: TestSuite, test: TestCase, test_re: Optional[re.Pattern]):
    if test_re is None or test_re.search(str(test)) is not None:
        suite.addTest(test)


def load_tests(test_name: str):
    suite = MpiTestSuite()

    test_re = re.compile(test_name, re.IGNORECASE)
    add_test(suite, ShallowWaterRestartTestCase(6, "test_read_restart"), test_re)
    add_test(suite, Euler3DRestartTestCase(6, "test_read_restart"), test_re)

    add_test(suite, ShallowWaterRestartTestCase(24, "test_read_restart"), test_re)
    add_test(suite, Euler3DRestartTestCase(24, "test_read_restart"), test_re)

    add_test(suite, ShallowWaterRestartTestCase(24, "test_multisize"), test_re)
    add_test(suite, Euler3DRestartTestCase(24, "test_multisize"), test_re)

    add_test(suite, GatherScatterTest(6, "gather_scatter_2d"), test_re)
    add_test(suite, GatherScatterTest(6, "gather_scatter_elem_2d"), test_re)
    add_test(suite, GatherScatterTest(6, "gather_scatter_3d"), test_re)
    add_test(suite, GatherScatterTest(6, "gather_scatter_elem_3d"), test_re)
    add_test(suite, GatherScatterTest(6, "gather_scatter_elem_4d"), test_re)

    add_test(suite, GatherScatterTest(24, "gather_scatter_2d"), test_re)
    add_test(suite, GatherScatterTest(24, "gather_scatter_elem_2d"), test_re)
    add_test(suite, GatherScatterTest(24, "gather_scatter_3d"), test_re)
    add_test(suite, GatherScatterTest(24, "gather_scatter_elem_3d"), test_re)
    add_test(suite, GatherScatterTest(24, "gather_scatter_elem_4d"), test_re)

    add_test(suite, GatherScatterTest(54, "gather_scatter_2d", optional=True), test_re)
    add_test(suite, GatherScatterTest(54, "gather_scatter_elem_2d", optional=True), test_re)
    add_test(suite, GatherScatterTest(54, "gather_scatter_3d", optional=True), test_re)
    add_test(suite, GatherScatterTest(54, "gather_scatter_elem_3d", optional=True), test_re)
    add_test(suite, GatherScatterTest(54, "gather_scatter_elem_4d", optional=True), test_re)

    add_test(suite, GatherScatterTest(24, "fail_wrong_num_proc"), test_re)  # This test needs at least 24 procs
    add_test(suite, GatherScatterTest(6, "fail_not_square"), test_re)
    add_test(suite, GatherScatterTest(6, "fail_not_cube"), test_re)
    add_test(suite, GatherScatterTest(6, "fail_wrong_num_dim"), test_re)

    add_test(suite, ExchangeTest("vector2d_1d_shape1d"), test_re)
    add_test(suite, ExchangeTest("vector2d_1d_shape2d"), test_re)
    add_test(suite, ExchangeTest("vector2d_2d_shape1d"), test_re)
    add_test(suite, ExchangeTest("vector2d_2d_shape3d"), test_re)
    add_test(suite, ExchangeTest("vector3d_1d_shape1d"), test_re)
    add_test(suite, ExchangeTest("vector3d_1d_shape2d"), test_re)
    add_test(suite, ExchangeTest("vector3d_3d_shape1d"), test_re)
    add_test(suite, ExchangeTest("vector3d_4d_shape3d"), test_re)
    add_test(suite, ExchangeTest("scalar_1d_shape1d"), test_re)
    add_test(suite, ExchangeTest("scalar_1d_shape2d"), test_re)
    add_test(suite, ExchangeTest("scalar_1d_shape3d"), test_re)
    add_test(suite, ExchangeTest("scalar_2d_shape1d"), test_re)
    add_test(suite, ExchangeTest("scalar_2d_shape2d"), test_re)

    add_test(suite, PmexMpiTestCases("test_pmex_mpi_2_processes"), test_re)
    add_test(suite, KiopsMpiTestCases("test_kiops_mpi_2_processes"), test_re)

    add_test(suite, PdeRusanov3DTestCase(6, "test_rusanov_kernel_cpu"), test_re)
    add_test(suite, PdeRusanov3DTestCase(6, "test_rusanov_kernel_gpu"), test_re)
    add_test(suite, PdeRusanov3DTestCase(24, "test_rusanov_kernel_cpu"), test_re)
    add_test(suite, PdeRusanov3DTestCase(24, "test_rusanov_kernel_gpu"), test_re)

    add_test(suite, OperatorsExtrapEuler3DTestCase(6, "test_extrap_kernel_cpu"), test_re)
    add_test(suite, OperatorsExtrapEuler3DTestCase(6, "test_extrap_kernel_gpu"), test_re)
    add_test(suite, OperatorsExtrapEuler3DTestCase(24, "test_extrap_kernel_cpu", optional=True), test_re)
    add_test(suite, OperatorsExtrapEuler3DTestCase(24, "test_extrap_kernel_gpu", optional=True), test_re)

    add_test(suite, RhsSideBySideEuler3DTestCase(6, "test_rhs_side_by_side"), test_re)

    # TODO : This test needs more works on the data division between processes
    # suite.addTest(FgmresMpiTestCases('test_fgmres_mpi_2_processes'))

    return suite


def trace_run(runner):
    import sys
    import trace
    import mpi4py

    # define Trace object: trace line numbers at runtime, exclude some modules
    tracer = trace.Trace(
        ignoredirs=[sys.prefix, sys.exec_prefix],
        ignoremods=[
            "inspect",
            "contextlib",
            "_bootstrap",
            "_weakrefset",
            "abc",
            "posixpath",
            "genericpath",
            "textwrap",
        ],
        trace=1,
        count=0,
    )

    # by default trace goes to stdout
    # redirect to a different file for each processes
    sys.stdout = open(f"trace_{mpi4py.MPI.COMM_WORLD.rank:04d}.txt", "w")

    tracer.runfunc(runner.run, load_tests())


def regular_run(runner, args):
    result = runner.run(load_tests(args.test_name))

    if not result.wasSuccessful():
        raise SystemExit(-1)


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

    runner = MpiRunner(buffer=True, verbosity=0)

    # trace_run(runner)
    regular_run(runner, args)
