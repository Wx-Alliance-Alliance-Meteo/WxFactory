#!/usr/bin/env python3

import unittest
from mpi4py import MPI


class FakeTest(unittest.TestCase):
    def test1(self):
        self.assertEqual(0, MPI.COMM_WORLD.rank, 'you are not the chosen one (or zero in that case)')
        raise Exception('hello there')
    
    def test2(self):
        self.assertTrue(True)
    
    def test3(self):
        self.skipTest('')



import os
import sys
from mpi_test import MpiRunner

main_project_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
sys.path.append(main_project_dir)

def load_tests():
    suite = unittest.TestSuite()
    suite.addTest(FakeTest('test1'))
    suite.addTest(FakeTest('test2'))
    suite.addTest(FakeTest('test3'))
    return suite

if __name__ == '__main__':
    runner = MpiRunner()
    runner.run(load_tests())
