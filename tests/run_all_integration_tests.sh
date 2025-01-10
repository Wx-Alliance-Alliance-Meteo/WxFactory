#!/bin/bash
# mpirun -n 2 python tests/run_integration_tests.py small_cartesian2d_problem || exit -1
./tests/run_integration_tests.py small_cartesian2d_problem || exit -1
