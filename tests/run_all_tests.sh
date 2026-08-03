#!/usr/bin/env bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
WX_DIR=${SCRIPT_DIR}/..

export PYTHONPATH=${PYTHONPATH}:${WX_DIR}

failfast=""

while [ $# -gt 0 ]; do
    case "${1}" in 
        --failfast ) failfast="--failfast";     shift  ;;
        * ) echo "Unrecognized option ${1}";    exit 1 ;;
    esac
done

cd ${WX_DIR}

${WX_DIR}/scripts/run.sh -n 1 ${WX_DIR}/tests/unit/run_tests.py ${failfast} || exit -1
${WX_DIR}/scripts/run.sh -n 6 ${PYTHON} ${WX_DIR}/tests/unit/run_mpi_tests.py ${failfast} || exit -1
${WX_DIR}/tests/integration/run_all_integration_tests.sh ${failfast} || exit -1
