#!/usr/bin/env python3

import os
import sys
import unittest
from mpi_test import MpiRunner

main_project_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
sys.path.append(main_project_dir)

from tests.common.test_process_topology import ProcessTopologyTest
from tests.solvers.test_pmex_mpi import PmexMpiTestCases
from tests.solvers.test_kiops_mpi import KiopsMpiTestCases
from tests.solvers.test_fgmres_mpi import FgmresMpiTestCases

def load_tests():
    suite = unittest.TestSuite()
    suite.addTest(ProcessTopologyTest('test1'))
    suite.addTest(ProcessTopologyTest('test2'))
    suite.addTest(PmexMpiTestCases('test_pmex_mpi_2_processes'))
    suite.addTest(KiopsMpiTestCases('test_kiops_mpi_2_processes'))
    suite.addTest(FgmresMpiTestCases('test_fgmres_mpi_2_processes'))

    return suite

if __name__ == '__main__':
    runner = MpiRunner()
    runner.run(load_tests())
