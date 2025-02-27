#!/usr/bin/env python3

import os
import sys
import unittest

from mpi_test import MpiRunner

main_project_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
main_module_dir = os.path.join(main_project_dir, "wx_factory")
sys.path.append(main_project_dir)
sys.path.append(main_module_dir)

from tests.unit.common.test_process_topology import ProcessTopologyTest
from tests.unit.solvers.test_pmex_mpi import PmexMpiTestCases
from tests.unit.solvers.test_kiops_mpi import KiopsMpiTestCases
from tests.unit.solvers.test_fgmres_mpi import FgmresMpiTestCases


def load_tests():
    suite = unittest.TestSuite()

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
    # suite.addTest(FgmresMpiTestCases('test_fgmres_mpi_2_processes')) # TODO : This test needs more works on the data division between processes

    return suite


if __name__ == "__main__":
    runner = MpiRunner()
    result = runner.run(load_tests())

    if not result.wasSuccessful():
        sys.exit(-1)
