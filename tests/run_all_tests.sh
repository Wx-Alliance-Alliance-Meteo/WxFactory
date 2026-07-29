#!/usr/bin/env bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
WX_DIR=${SCRIPT_DIR}/..

export PYTHONPATH=${PYTHONPATH}:${WX_DIR}

cd ${WX_DIR}

${WX_DIR}/scripts/run.sh -n 1 ${WX_DIR}/tests/unit/run_tests.py || exit -1
${WX_DIR}/scripts/run.sh -n 6 ${PYTHON} ${WX_DIR}/tests/unit/run_mpi_tests.py || exit -1
${WX_DIR}/tests/integration/run_all_integration_tests.sh || exit -1
