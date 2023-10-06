#!/usr/bin/env bash

SCRIPT_NAME=$(basename ${BASH_SOURCE[0]})
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
GEF_DIR=$(dirname ${SCRIPT_DIR})
BASE_WORK_DIR=${GEF_DIR}/work
BASE_JOB_SCRIPT=${SCRIPT_DIR}/launch_gef.job
BASE_LOAD_ENV=${SCRIPT_DIR}/load_env.sh

# echo "SCRIPT_NAME = ${SCRIPT_NAME}"
# echo "SCRIPT_DIR  = ${SCRIPT_DIR}"
# echo "GEF_DIR     = ${GEF_DIR}"

function echo_usage() {
    echo "${SCRIPT_NAME}:"
    echo "  Launch a GEF configuration on a cluster."
    echo
    echo "Usage: ${SCRIPT_NAME} [config_file] [[options]]"
    echo
    echo "  Make sure to provide a file called \"load_env.sh\" in the same directory"
    echo "  as this script to load the execution environment you want to use for running GEF."
    echo "  That file will be sourced before running GEF."
    echo
    echo "  options:"
    echo "    [config_file]   Path of the GEF configuration you want to run"
    echo "    -h|--help       Print this message"
    echo "    -c|--cpu        Number of cores you want to use for running"
    echo "    -m|--mach       Name of the machine on which you want to run"
    echo "    -w|--wtime      Number of minutes"
}

function validate_config() {
    if test ! -f ${1}; then
        echo "${1} does not seem to be a valid file"
        echo_usage
        exit
    fi
}

BASE_CONFIG=""
NUM_PES=1
MACH=""
WTIME=180

test $# -lt 1 && echo_usage && exit

# Parse options
while [ $# -gt 0 ]; do
    case ${1} in
    -h | --help)
        echo_usage
        exit
        ;;
    -c | --cpu)
        NUM_PES=${2}
        shift 1
        ;;
    -m | --mach)
        MACH=${2}
        shift 1
        ;;
    -w | --wtime)
        WTIME=${2}
        shift 1
        ;;
    *)
        validate_config ${1}
        BASE_CONFIG=${1}
        ;;
    esac
    shift 1
done

if [ "x" == "x${BASE_CONFIG}" ]; then
    echo "Please specify a (valid) config file"
    exit
fi

if [ ! -f ${BASE_JOB_SCRIPT} ]; then
    echo "${BASE_JOB_SCRIPT} does not seem to exist. We do need a job script."
    exit
fi

if [ ! -f ${BASE_LOAD_ENV} ]; then
    echo "Must provide a 'load_env.sh' file (placed in the same directory as this script)"
    exit
fi

JOB_NAME="no_name"
for i in {0..1000}; do
    d=$(date +%Y%m%d_%H%M%S)
    JOB_NAME=gef.$d.$i
    WORK_DIR=${BASE_WORK_DIR}/$d.$i
    test -d ${WORK_DIR} || break
done
mkdir -pv ${WORK_DIR} || exit -1

CONFIG_FILE=${WORK_DIR}/$(basename ${BASE_CONFIG})
cp -v ${BASE_CONFIG} ${CONFIG_FILE}

JOB_SCRIPT=${WORK_DIR}/$(basename ${BASE_JOB_SCRIPT})
cp -v ${BASE_JOB_SCRIPT} ${JOB_SCRIPT}

LOAD_ENV_FILE=${WORK_DIR}/load_env.sh
cp -v ${BASE_LOAD_ENV} ${LOAD_ENV_FILE}

sed \
    -e 's|^NUM_PES=.*|NUM_PES='${NUM_PES}'|' \
    -e 's|^EXEC_DIR=.*|EXEC_DIR='${GEF_DIR}'|' \
    -e 's|^WORK_DIR=.*|WORK_DIR='${WORK_DIR}'|' \
    -e 's|^CONFIG_FILE=.*|CONFIG_FILE='${CONFIG_FILE}'|' \
    -i ${JOB_SCRIPT}

# cat ${JOB_SCRIPT}

LAUNCH_NUM_CPU=${NUM_PES}
if [ ${LAUNCH_NUM_CPU} -lt 80 ]; then
    LAUNCH_NUM_CPU=80
fi

LAUNCH_COMMAND="ord_soumet -cpus ${LAUNCH_NUM_CPU} -w ${WTIME} -jn ${JOB_NAME} -mpi -jobfile ${JOB_SCRIPT} -listing ${WORK_DIR} -cm 2000M -waste 100"
if [ "x" != "x${MACH}" ]; then
    LAUNCH_COMMAND="${LAUNCH_COMMAND} -mach ${MACH}"
fi

echo "LAUNCH_COMMAND = ${LAUNCH_COMMAND}"

${LAUNCH_COMMAND}
