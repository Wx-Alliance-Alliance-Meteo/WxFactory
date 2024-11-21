#!/bin/bash -l
mpirun -n 4 python tests/run_integration_tests.py small_cartesian2d_problem
