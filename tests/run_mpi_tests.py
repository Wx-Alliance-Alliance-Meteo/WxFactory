#!/usr/bin/env python3

import os
import sys
import unittest
from mpi_test import MpiRunner

main_project_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
sys.path.append(main_project_dir)

from tests.common.test_process_topology import ProcessTopologyTest
from tests.solvers.test_pmex_mpi_cpu import PmexMpiCpuTestCases
from tests.solvers.test_pmex_mpi_gpu import PmexMpiGpuTestCases
from tests.solvers.test_kiops_mpi_cpu import KiopsMpiCpuTestCases
from tests.solvers.test_kiops_mpi_gpu import KiopsMpiGpuTestCases

def load_tests():
    suite = unittest.TestSuite()
    suite.addTest(ProcessTopologyTest('test1'))
    suite.addTest(ProcessTopologyTest('test2'))
    suite.addTest(PmexMpiCpuTestCases('test_pmex_mpi_2_processes'))
    suite.addTest(PmexMpiGpuTestCases('test_pmex_mpi_2_processes'))
    suite.addTest(KiopsMpiCpuTestCases('test_pmex_mpi_2_processes'))
    suite.addTest(KiopsMpiGpuTestCases('test_pmex_mpi_2_processes'))
    return suite

if __name__ == '__main__':
    runner = MpiRunner()
    runner.run(load_tests())
