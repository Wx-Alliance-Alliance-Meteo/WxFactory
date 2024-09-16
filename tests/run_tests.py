#!/usr/bin/env python3

import os
import sys
import unittest

main_project_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
sys.path.append(main_project_dir)

from tests.process_topology import ProcessTopologyTest
from tests.solvers.test_pmex import PmexTestCases
from tests.solvers.test_kiops import KiopsTestCases
from tests.solvers.test_kiops_pmex_tolerance_cpu import KiopsPmexToleranceCpuTestCases
from tests.solvers.test_kiops_pmex_tolerance_gpu import KiopsPmexToleranceGpuTestCases

def load_tests():
    suite = unittest.TestSuite()
    suite.addTest(PmexTestCases('test_compare_cpu_to_gpu'))
    suite.addTest(KiopsTestCases('test_compare_cpu_to_gpu'))
    suite.addTest(KiopsPmexToleranceCpuTestCases('test_compare_kiops_pmex'))
    suite.addTest(KiopsPmexToleranceGpuTestCases('test_compare_kiops_pmex'))
    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(load_tests())
