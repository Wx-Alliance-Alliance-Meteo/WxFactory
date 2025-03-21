#!/usr/bin/env python3

import os
import sys
import unittest

from mpi_test import MpiRunner

main_project_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
main_module_dir = os.path.join(main_project_dir, "wx_factory")
sys.path.append(main_project_dir)
sys.path.append(main_module_dir)

from tests.unit.common.test_process_topology import ProcessTopologyTest, GatherScatterTest
from tests.unit.restart.test_restart import ShallowWaterRestartTestCase, Euler3DRestartTestCase
from tests.unit.solvers.test_pmex_mpi import PmexMpiTestCases
from tests.unit.solvers.test_kiops_mpi import KiopsMpiTestCases
from tests.unit.solvers.test_fgmres_mpi import FgmresMpiTestCases


def load_tests():
    suite = unittest.TestSuite()

    suite.addTest(ShallowWaterRestartTestCase(6, "test_read_restart"))
    suite.addTest(Euler3DRestartTestCase(6, "test_read_restart"))

    suite.addTest(ShallowWaterRestartTestCase(24, "test_read_restart"))
    suite.addTest(Euler3DRestartTestCase(24, "test_read_restart"))

    suite.addTest(ShallowWaterRestartTestCase(24, "test_multisize"))
    suite.addTest(Euler3DRestartTestCase(24, "test_multisize"))

    suite.addTest(GatherScatterTest(6, "gather_scatter_2d"))
    suite.addTest(GatherScatterTest(6, "gather_scatter_elem_2d"))
    suite.addTest(GatherScatterTest(6, "gather_scatter_3d"))
    suite.addTest(GatherScatterTest(6, "gather_scatter_elem_3d"))
    suite.addTest(GatherScatterTest(6, "gather_scatter_elem_4d"))

    suite.addTest(GatherScatterTest(24, "gather_scatter_2d"))
    suite.addTest(GatherScatterTest(24, "gather_scatter_elem_2d"))
    suite.addTest(GatherScatterTest(24, "gather_scatter_3d"))
    suite.addTest(GatherScatterTest(24, "gather_scatter_elem_3d"))
    suite.addTest(GatherScatterTest(24, "gather_scatter_elem_4d"))

    suite.addTest(GatherScatterTest(54, "gather_scatter_2d", optional=True))
    suite.addTest(GatherScatterTest(54, "gather_scatter_elem_2d", optional=True))
    suite.addTest(GatherScatterTest(54, "gather_scatter_3d", optional=True))
    suite.addTest(GatherScatterTest(54, "gather_scatter_elem_3d", optional=True))
    suite.addTest(GatherScatterTest(54, "gather_scatter_elem_4d", optional=True))

    suite.addTest(GatherScatterTest(24, "fail_wrong_num_proc"))  # This test needs at least 24 procs
    suite.addTest(GatherScatterTest(6, "fail_not_square"))
    suite.addTest(GatherScatterTest(6, "fail_not_cube"))
    suite.addTest(GatherScatterTest(6, "fail_wrong_num_dim"))

    suite.addTest(ProcessTopologyTest("vector2d_1d_shape1d"))
    suite.addTest(ProcessTopologyTest("vector2d_1d_shape2d"))
    suite.addTest(ProcessTopologyTest("vector2d_2d_shape1d"))
    suite.addTest(ProcessTopologyTest("vector2d_2d_shape3d"))
    suite.addTest(ProcessTopologyTest("vector3d_1d_shape1d"))
    suite.addTest(ProcessTopologyTest("vector3d_1d_shape2d"))
    suite.addTest(ProcessTopologyTest("vector3d_3d_shape1d"))
    suite.addTest(ProcessTopologyTest("vector3d_4d_shape3d"))
    suite.addTest(ProcessTopologyTest("scalar_1d_shape1d"))
    suite.addTest(ProcessTopologyTest("scalar_1d_shape2d"))
    suite.addTest(ProcessTopologyTest("scalar_1d_shape3d"))
    suite.addTest(ProcessTopologyTest("scalar_2d_shape1d"))
    suite.addTest(ProcessTopologyTest("scalar_2d_shape2d"))

    suite.addTest(PmexMpiTestCases("test_pmex_mpi_2_processes"))
    suite.addTest(KiopsMpiTestCases("test_kiops_mpi_2_processes"))

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


def regular_run(runner):
    result = runner.run(load_tests())

    if not result.wasSuccessful():
        sys.exit(-1)


if __name__ == "__main__":
    runner = MpiRunner(buffer=True)

    # trace_run(runner)
    regular_run(runner)
