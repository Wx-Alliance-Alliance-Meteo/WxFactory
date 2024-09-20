#!/usr/bin/env python3

import os
import sys
import unittest

main_project_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
sys.path.append(main_project_dir)

from tests.common.test_process_topology import ProcessTopologyTest

def load_tests():
    suite = unittest.TestSuite()
    suite.addTest(ProcessTopologyTest('test1'))
    suite.addTest(ProcessTopologyTest('test2'))
    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(load_tests())
