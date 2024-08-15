#!/usr/bin/env python3

import os
import sys
import unittest

main_project_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
sys.path.append(main_project_dir)

from process_topology import ProcessTopologyTest

if __name__ == '__main__':
    unittest.main()
