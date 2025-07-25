#!/bin/bash

MPIRUN=${MPIRUN:-mpirun}

# ${MPIRUN} -n 2 python tests/integration/run_integration_tests.py small_cartesian2d_problem || exit -1
${MPIRUN} -n 1 ./tests/integration/run_integration_tests.py small_cartesian2d_problem || exit -1
${MPIRUN} -n 24 ./tests/integration/run_integration_tests.py small_sw_case5 || exit -1
${MPIRUN} -n 54 ./tests/integration/run_integration_tests.py small_dcmip21 || exit -1
