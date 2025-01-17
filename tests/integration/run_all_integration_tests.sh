#!/bin/bash
# mpirun -n 2 python tests/integration/run_integration_tests.py small_cartesian2d_problem || exit -1
./tests/integration/run_integration_tests.py small_cartesian2d_problem || exit -1
